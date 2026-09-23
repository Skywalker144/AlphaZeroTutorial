import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .mcts import MCTS
from .metrics import MetricsTracker
from .network import MuZeroNet, scale_gradient
from .alphazero_parallel import ParallelSelfPlayer
from .replay_buffer import ReplayBuffer
from .scheduler import SelfPlayScheduler
from .utils import auto_device, random_augment_batch


@dataclass
class CollectStats:
    """Aggregates for the games collected inside a single iteration."""

    game_lens: list = field(default_factory=list)
    sample_lens: list = field(default_factory=list)
    winners: list = field(default_factory=list)

    def add(self, game_len, sample_len, winner):
        self.game_lens.append(game_len)
        self.sample_lens.append(sample_len)
        self.winners.append(winner)

    @property
    def rows(self):
        return sum(self.sample_lens)


class Reporter:
    """Every console message emitted during training lives here.

    Configuration (all optional, read from `args`):

    verbose
        Full output when True. When False only two lines per iteration are
        kept: the `=== Iter ... ===` header and the final `[Train]` result.
        Per-game selfplay progress, the `[Stats]` line, buffer/skip notices
        and checkpoint paths are suppressed.
    log_every
        Print a selfplay progress line every `log_every` games. 0 (default)
        keeps the old cadence of `games // 5`.
    show_progress
        Whether to render the tqdm bar for the training loop.
    """

    def __init__(self, verbose=True, log_every=0, show_progress=True):
        self.verbose = verbose
        self.log_every = max(0, int(log_every))
        self.show_progress = show_progress
        self._pbar = None

    # -- iteration header -------------------------------------------------

    def iteration_started(self, i, games, game_count, total_samples, window_size):
        print(f"\n=== Iter {i} | collect {games} games ===", flush=True)
        if self.verbose:
            print(
                f"[Stats] total_games={game_count} "
                f"total_samples={total_samples} window={window_size}",
                flush=True,
            )

    # -- selfplay ---------------------------------------------------------

    def selfplay_progress(self, i, done, total, t0, stats):
        if not self.verbose:
            return
        every = self.log_every or max(1, total // 5)
        if done % every and done != total:
            return
        dt = max(1e-9, time.time() - t0)
        rows_per_s = stats.rows / dt
        gw = max(2, len(str(total)))
        head = f"[SelfPlay] Iter={i} Games={done:0{gw}d}/{total} Rows/s={rows_per_s:.1f}"
        if stats.game_lens:
            arr = np.asarray(stats.game_lens, dtype=np.float64)
            black, draw, white = MetricsTracker.winrate_summary(stats.winners)
            head += (
                f" GameLen:Avg={arr.mean():.1f} Min={int(arr.min())} "
                f"Max={int(arr.max())} Std={int(round(arr.std()))} "
                f"Rows={stats.rows} "
                f"BDW={int(round(black * 100)):02d}/"
                f"{int(round(draw * 100)):02d}/"
                f"{int(round(white * 100)):02d}"
            )
        else:
            head += " GameLen=N/A BDW=N/A"
        print(head, flush=True)

    # -- training ---------------------------------------------------------

    def train_skipped(self, i, total_samples, min_rows):
        if not self.verbose:
            return
        print(
            f"[Train] iter={i} skipped (buffer not ready: "
            f"{total_samples} < {min_rows})",
            flush=True,
        )

    def train_progress(self, i, train_steps):
        if not self.show_progress:
            return range(train_steps)
        self._pbar = tqdm(range(train_steps), desc=f"Train Iter={i}")
        return self._pbar

    def train_step_done(self, steps_done, total_loss):
        if self._pbar is not None:
            self._pbar.set_postfix(loss=f"{total_loss / steps_done:.4f}")

    def train_progress_finished(self):
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

    def train_done(self, i, steps, elapsed, total, policy, value):
        print(
            f"[Train] iter={i} steps={steps} t={elapsed:.1f}s | "
            f"total={total:.4f} policy={policy:.4f} value={value:.4f}",
            flush=True,
        )

    # -- lifecycle --------------------------------------------------------

    def checkpoint_saved(self, path):
        if self.verbose:
            print(f"Checkpoint saved to {path}")

    def model_saved(self, path):
        if self.verbose:
            print(f"Model saved to {path}")

    def checkpoint_loaded(self, path):
        if self.verbose:
            print(f"Checkpoint loaded from {path}")

    def no_checkpoint(self):
        if self.verbose:
            print("No checkpoint found, starting from scratch.")

    def interrupted(self):
        print("\nStopping... saving final checkpoint and plots", flush=True)


def _module_grad_norm(module):
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum())
    return total ** 0.5


class MuZero:
    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.device = auto_device()
        self.model = MuZeroNet(
            game.board_size,
            game.num_planes,
            num_blocks=args.get("num_blocks", 1),
            num_channels=args.get("num_channels", 32),
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.get("lr", 1e-3),
            weight_decay=args.get("weight_decay", 3e-4),
        )
        self.mcts = MCTS(game, args, self.model, self.device)
        self.parallel_player = (
            ParallelSelfPlayer(game, args, self.model, self.device)
            if args.get("parallel", True)
            else None
        )
        self.replay_buffer = ReplayBuffer(
            min_rows=args.get("min_rows", 30000),
            taper_window_exponent=args.get("taper_window_exponent", 0.675),
            expand_window_per_row=args.get("expand_window_per_row", 0.4),
            max_rows=args.get("max_rows", None),
        )
        self.replay_ratio = args.get("replay_ratio", 8)
        rows_needed_per_iteration = (
            args.get("train_steps", 200) * args.get("batch_size", 128)
        ) / self.replay_ratio
        self.scheduler = SelfPlayScheduler(
            bootstrap_games=args.get("bootstrap_games", 200),
            min_rows=args.get("min_rows", 30000),
            rows_needed_per_iteration=rows_needed_per_iteration,
            fallback_rows_per_game=float(game.board_size ** 2),
        )
        self.game_count = 0
        self.iteration = 0
        self.metrics = MetricsTracker()
        self.reporter = Reporter(
            verbose=args.get("verbose", True),
            log_every=args.get("log_every", 0),
            show_progress=args.get("show_progress", True),
        )

    def selfplay(self):
        memory = []
        state = self.game.get_initial_state()
        to_play = 1
        half_life = self.args.get("half_life", self.game.board_size)
        while not self.game.is_terminal(state, to_play):

            mcts_policy, _ = self.mcts.search(
                state, to_play,
                num_simulations=self.args.get("num_simulations", 1.7 * self.game.board_size ** 2)
            )

            if len(memory) + 1 < half_life:
                action = np.random.choice(len(mcts_policy), p=mcts_policy)
            else:
                action = int(np.argmax(mcts_policy))

            memory.append({
                "observation": self.game.encode_state(state, to_play),
                "player": to_play,
                "action": int(action),
                "mcts_policy": mcts_policy,
            })

            state = self.game.get_next_state(state, action, to_play)
            to_play = -to_play

        winner = self.game.get_winner(state, to_play)
        for step in memory:
            step["value_target"] = float(winner) * step["player"]
        return memory, winner, len(memory)

    def train_step(self):
        unroll_steps = self.args.get("unroll_steps", 5)
        batch = self.replay_buffer.sample(
            self.args.get("batch_size", 128),
            unroll_steps,
            self.game.board_size ** 2,
        )
        if not batch:
            return None

        batch = random_augment_batch(batch, self.game.board_size)
        observations = torch.tensor(
            np.array([s["observation"] for s in batch]), dtype=torch.float32, device=self.device
        )
        actions = torch.tensor(
            np.array([s["actions"] for s in batch]), dtype=torch.long, device=self.device
        )
        policy_targets = torch.tensor(
            np.array([s["policy_targets"] for s in batch]), dtype=torch.float32, device=self.device
        )
        value_targets = torch.tensor(
            np.array([s["value_targets"] for s in batch]), dtype=torch.float32, device=self.device
        )
        policy_mask = torch.tensor(
            np.array([s["policy_mask"] for s in batch]), dtype=torch.float32, device=self.device
        )

        batch_size = observations.shape[0]
        self.model.train()
        self.optimizer.zero_grad()

        hidden_state = self.model.representation(observations)
        policy_loss = 0.0
        value_loss = 0.0
        step_losses = []
        for k in range(unroll_steps + 1):
            policy_logits, value = self.model.prediction(hidden_state)
            step_policy = -torch.sum(
                policy_targets[:, k] * F.log_softmax(policy_logits, dim=1), dim=1
            )
            policy_term = torch.sum(policy_mask[:, k] * step_policy)
            value_term = F.mse_loss(value, value_targets[:, k], reduction="sum")
            # 遵循官方伪代码：根预测权重为 1，K 个循环预测各为 1/K。
            gradient_scale = 1.0 if k == 0 else 1.0 / unroll_steps
            policy_loss = policy_loss + scale_gradient(policy_term, gradient_scale)
            value_loss = value_loss + scale_gradient(value_term, gradient_scale)
            step_losses.append(((policy_term + value_term) / batch_size).item())
            if k < unroll_steps:
                # 当前预测使用未缩放的梯度；只有后续循环跨过此边界时乘 0.5。
                # s^0 到首次 dynamics 的连接保持不变，与官方 update_weights 一致。
                if k > 0:
                    hidden_state = scale_gradient(hidden_state, 0.5)
                hidden_state = self.model.dynamics(hidden_state, actions[:, k])

        policy_loss = policy_loss / batch_size
        value_loss = value_loss / batch_size
        total_loss = policy_loss + self.args.get("value_loss_scale", 1.0) * value_loss

        total_loss.backward()
        grad_norms = {
            "representation": _module_grad_norm(self.model.representation),
            "dynamics": _module_grad_norm(self.model.dynamics),
            "prediction": _module_grad_norm(self.model.prediction),
        }
        self.optimizer.step()
        self.model.eval()
        return {
            "total": total_loss.item(),
            "policy": policy_loss.item(),
            "value": value_loss.item(),
            "step_losses": step_losses,
            "grad_norms": grad_norms,
        }

    # -- learn loop -------------------------------------------------------

    def learn(self):
        num_iterations = self.args.get("num_iterations", None)
        save_interval = self.args.get("save_interval", 5)
        self.load_checkpoint()
        try:
            while num_iterations is None or self.iteration < num_iterations:
                i = self.iteration
                self._run_iteration(i)
                self.iteration = i + 1
                if i % save_interval == 0:
                    self.save_model(i)
                    self.save_checkpoint()
                self.plot_metrics()
        except KeyboardInterrupt:
            self.reporter.interrupted()
            self.save_checkpoint()
            self.plot_metrics()
            raise
        else:
            self.save_checkpoint()
            self.plot_metrics()

    def _run_iteration(self, i):
        games = self.scheduler.games_to_order(
            self.replay_buffer.total_samples_added
        )
        self.reporter.iteration_started(
            i,
            games,
            game_count=self.game_count,
            total_samples=self.replay_buffer.total_samples_added,
            window_size=self.replay_buffer.window_size(),
        )
        stats = self._collect_games(i, games)
        self.scheduler.record_iteration(games, stats.rows)
        self._train_iteration(i)

    def _collect_games(self, i, games):
        stats = CollectStats()
        started = time.time()
        for done, (game_data, winner, game_len) in enumerate(self._selfplay_games(games), 1):
            self.replay_buffer.add_game(game_data)
            self.game_count += 1
            self.metrics.record_game(self.game_count, winner, game_len, i)
            stats.add(game_len, len(game_data), winner)
            self.reporter.selfplay_progress(i, done, games, started, stats)
        return stats

    def _selfplay_games(self, games):
        """Yield (game, winner, game_len) per game, 串行或并行后端二选一。"""
        if self.parallel_player is not None:
            yield from self.parallel_player.run(games)
            return
        for _ in range(games):
            yield self.selfplay()

    def _train_iteration(self, i):
        if not self.replay_buffer.is_ready():
            self.reporter.train_skipped(
                i,
                self.replay_buffer.total_samples_added,
                self.replay_buffer.min_rows,
            )
            return

        train_steps = self.args.get("train_steps", 200)
        t0 = time.time()
        totals = {"total": 0.0, "policy": 0.0, "value": 0.0}
        step_losses_sum = []
        grad_sums = {"representation": 0.0, "dynamics": 0.0, "prediction": 0.0}
        steps_done = 0
        try:
            for _ in self.reporter.train_progress(i, train_steps):
                result = self.train_step()
                if not result:
                    continue
                steps_done += 1
                for key in totals:
                    totals[key] += result[key]
                if not step_losses_sum:
                    step_losses_sum = [0.0] * len(result["step_losses"])
                for index, value in enumerate(result["step_losses"]):
                    step_losses_sum[index] += value
                for key, value in result["grad_norms"].items():
                    grad_sums[key] += value
                self.reporter.train_step_done(steps_done, totals["total"])
        finally:
            self.reporter.train_progress_finished()

        if steps_done:
            losses = tuple(totals[key] / steps_done for key in ("total", "policy", "value"))
            mean_step_losses = [value / steps_done for value in step_losses_sum]
            mean_grads = {key: value / steps_done for key, value in grad_sums.items()}
            self.metrics.record_losses(*losses, mean_step_losses, mean_grads)
            self.reporter.train_done(i, steps_done, time.time() - t0, *losses)

    # -- io ---------------------------------------------------------------

    def plot_metrics(self):
        data_dir = self.args.get("data_dir", "data")
        self.metrics.plot(data_dir)

    def save_model(self, iteration):
        models_dir = os.path.join(self.args.get("data_dir", "data"), "models")
        os.makedirs(models_dir, exist_ok=True)
        path = os.path.join(models_dir, f"model_{iteration}.pth")
        torch.save(self.model.state_dict(), path)
        self.reporter.model_saved(path)

    def save_checkpoint(self):
        ckpt_dir = os.path.join(self.args.get("data_dir", "data"), "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, "checkpoint.pth")
        torch.save(
            {
                "iteration": self.iteration,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "game_count": self.game_count,
                "metrics": self.metrics.state(),
                "replay_buffer": self.replay_buffer.get_state(),
                "scheduler": self.scheduler.state(),
            },
            path,
        )
        self.reporter.checkpoint_saved(path)

    def load_checkpoint(self, filename=None):
        if filename is None:
            filename = os.path.join(
                self.args.get("data_dir", "data"), "checkpoints", "checkpoint.pth"
            )
        if not os.path.exists(filename):
            self.reporter.no_checkpoint()
            return False
        checkpoint = torch.load(filename, map_location=self.device, weights_only=False)
        self.iteration = checkpoint["iteration"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.game_count = checkpoint["game_count"]
        self.metrics.load_state(checkpoint["metrics"])
        self.replay_buffer.load_state(checkpoint["replay_buffer"])
        self.scheduler.load_state(checkpoint["scheduler"])
        self.reporter.checkpoint_loaded(filename)
        return True
