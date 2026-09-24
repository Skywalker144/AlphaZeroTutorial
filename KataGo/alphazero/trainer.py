import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .mcts import MCTS
from .alphazero_parallel import ParallelSelfPlayer
from .metrics import MetricsTracker
from .network import ResNet
from .replay_buffer import ReplayBuffer
from .scheduler import SelfPlayScheduler
from .utils import (
    apply_temperature,
    auto_device,
    chosen_move_temperature,
    finish_game_samples,
    lcb_play_selection,
    policy_surprise,
    random_augment_batch,
    reduced_search_limit,
    search_visit_counts,
)


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


class AlphaZero:
    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.device = auto_device()
        self.model = ResNet(
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
        turn_number = 0
        full_simulations, cheap_simulations = search_visit_counts(
            self.args, self.game.board_size
        )
        cheap_search_prob = self.args.get("cheap_search_prob", 0.75)
        win_loss_history = []
        root = None
        while not self.game.is_terminal(state, to_play):

            cheap = np.random.random() < cheap_search_prob
            if cheap:
                num_simulations = cheap_simulations
                weight = 0.0
            else:
                num_simulations, weight = reduced_search_limit(
                    self.args, win_loss_history, full_simulations, cheap_simulations
                )
            raw_policy, root_value, root = self.mcts.search(
                state, to_play,
                num_simulations=num_simulations,
                turn_number=turn_number,
                cheap=cheap,
                root=root if cheap else None,
            )
            win_loss_history.append(root_value * to_play)
            target_policy = lcb_play_selection(root, self.game.board_size ** 2, self.args)

            memory.append({
                "state": state,
                "to_play": to_play,
                "mcts_policy": target_policy,
                "weight": weight,
                "policy_surprise": policy_surprise(root.prior_policy, target_policy),
                "search_wdl": (root.wdl_sum / root.visits).copy(),
                "nn_wdl": root.nn_wdl.copy(),
            })

            temperature = chosen_move_temperature(
                self.args, turn_number, self.game.board_size
            )
            action = np.random.choice(
                len(raw_policy), p=apply_temperature(raw_policy, temperature)
            )

            root = self.mcts.advance(root, action)

            state = self.game.get_next_state(state, action, to_play)
            to_play = -to_play
            turn_number += 1

        winner = self.game.get_winner(state, to_play)
        samples = finish_game_samples(memory, winner, self.game, self.args)
        return samples, winner, turn_number

    def train_step(self):
        batch = self.replay_buffer.sample(self.args.get("batch_size", 128))
        if not batch:
            return None

        batch = random_augment_batch(batch, self.game.board_size)
        states = torch.tensor(
            np.array([s["encoded_state"] for s in batch]), dtype=torch.float32, device=self.device
        )
        policy_targets = torch.tensor(
            np.array([s["policy_target"] for s in batch]), dtype=torch.float32, device=self.device
        )
        opponent_targets = torch.tensor(
            np.array([s["opponent_policy"] for s in batch]), dtype=torch.float32, device=self.device
        )
        policy_targets_soft = torch.tensor(
            np.array([s["policy_target_soft"] for s in batch]), dtype=torch.float32, device=self.device
        )
        opponent_targets_soft = torch.tensor(
            np.array([s["opponent_policy_soft"] for s in batch]), dtype=torch.float32, device=self.device
        )
        opponent_weight = torch.tensor(
            np.array([s["opponent_weight"] for s in batch]), dtype=torch.float32, device=self.device
        )
        value_targets = torch.tensor(
            np.array([s["value_target"] for s in batch]), dtype=torch.float32, device=self.device
        )
        self.model.train()
        self.optimizer.zero_grad()
        policy_logits, value_logits = self.model(states)

        opponent_scale = self.args.get("opponent_policy_loss_scale", 0.15)
        soft_scale = self.args.get("soft_policy_weight_scale", 8.0)

        policy_player = self._policy_cross_entropy(policy_logits[:, 0], policy_targets)
        policy_opponent = self._policy_cross_entropy(
            policy_logits[:, 1], opponent_targets, opponent_weight
        )
        policy_soft = self._policy_cross_entropy(policy_logits[:, 2], policy_targets_soft)
        policy_opponent_soft = self._policy_cross_entropy(
            policy_logits[:, 3], opponent_targets_soft, opponent_weight
        )

        value_losses = -torch.sum(value_targets * F.log_softmax(value_logits, dim=1), dim=1)
        value_loss = value_losses.mean()

        weighted_player = policy_player
        weighted_opponent = opponent_scale * policy_opponent
        weighted_soft = soft_scale * policy_soft
        weighted_opponent_soft = opponent_scale * soft_scale * policy_opponent_soft
        policy_loss = weighted_player + weighted_opponent + weighted_soft + weighted_opponent_soft
        total_loss = policy_loss + self.args.get("value_loss_scale", 1.2) * value_loss

        total_loss.backward()
        self.optimizer.step()
        self.model.eval()
        return {
            "total": total_loss.item(),
            "policy": policy_loss.item(),
            "value": value_loss.item(),
            "policy_player": weighted_player.item(),
            "policy_opponent": weighted_opponent.item(),
            "policy_soft": weighted_soft.item(),
            "policy_opponent_soft": weighted_opponent_soft.item(),
        }

    @staticmethod
    def _policy_cross_entropy(policy_logits, target, weight=None):
        losses = -torch.sum(target * F.log_softmax(policy_logits, dim=1), dim=1)
        if weight is not None:
            losses = losses * weight
        return losses.mean()

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
        """Yield (samples, winner, game_len) per game, 串行或并行后端二选一。"""
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
        totals = {
            "total": 0.0,
            "policy": 0.0,
            "value": 0.0,
            "policy_player": 0.0,
            "policy_opponent": 0.0,
            "policy_soft": 0.0,
            "policy_opponent_soft": 0.0,
        }
        steps_done = 0
        try:
            for _ in self.reporter.train_progress(i, train_steps):
                res = self.train_step()
                if res:
                    for key in totals:
                        totals[key] += res[key]
                    steps_done += 1
                    self.reporter.train_step_done(steps_done, totals["total"])
        finally:
            self.reporter.train_progress_finished()

        if steps_done:
            means = {key: value / steps_done for key, value in totals.items()}
            self.metrics.record_losses(means["total"], means["policy"], means["value"], means)
            self.reporter.train_done(
                i,
                steps_done,
                time.time() - t0,
                means["total"],
                means["policy"],
                means["value"],
            )

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
