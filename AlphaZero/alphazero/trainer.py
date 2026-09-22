import itertools
import math
import os
import time
from collections import deque

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .mcts import MCTS
from .network import ResNet
from .replay_buffer import ReplayBuffer
from .utils import auto_device, random_augment_batch


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
        self.replay_buffer = ReplayBuffer(
            min_rows=args.get("min_rows", 20000),
            taper_window_exponent=args.get("taper_window_exponent", 0.675),
            expand_window_per_row=args.get("expand_window_per_row", 0.4),
            max_rows=args.get("max_rows", None),
        )
        self.replay_ratio = args.get("replay_ratio", 8)
        self.bootstrap_games = args.get("bootstrap_games", 200)
        self.game_count = 0
        self.losses = {"total": [], "policy": [], "value": []}
        self._target_cum = 0.0
        self._rpg_history = deque(maxlen=20)
        self._winrate_sample_every = args.get("winrate_sample_every", 10)
        stats_window = args.get("stats_window", 300)
        self._recent_game_lengths = deque(maxlen=stats_window)
        self._black_win_counts = deque(maxlen=stats_window)
        self._white_win_counts = deque(maxlen=stats_window)
        self.winrate_history = []
        self.game_length_history = []

    def _next_target_cum(self):
        needed = (
            self.args.get("train_steps", 500) * self.args.get("batch_size", 128)
        ) / self.replay_ratio
        if self._target_cum <= 0.0:
            self._target_cum = max(float(self.args.get("min_rows", 20000)), needed)
        else:
            self._target_cum += needed
        return self._target_cum

    def _rows_per_game(self):
        for g, r in reversed(self._rpg_history):
            if g > 0 and r > 0:
                return r / g
        return float(self.game.board_size ** 2)

    def _games_to_order(self):
        if not self._rpg_history:
            return self.bootstrap_games
        target_cum = self._next_target_cum()
        deficit = target_cum - self.replay_buffer.total_samples_added
        rpg = self._rows_per_game()
        return math.ceil(deficit / rpg) if deficit > 0 else 0

    def selfplay(self):
        memory = []
        state = self.game.get_initial_state()
        to_play = 1
        while not self.game.is_terminal(state, to_play):

            mcts_policy = self.mcts.search(state, to_play, self.args.get("num_simulations", 1.7 * self.game.board_size ** 2))

            memory.append({
                "state": state,
                "to_play": to_play,
                "mcts_policy": mcts_policy,
            })

            half_life = self.args.get("half_life", self.game.board_size)
            if len(memory) < half_life:
                action = np.random.choice(len(mcts_policy), p=mcts_policy)
            else:
                action = int(np.argmax(mcts_policy))

            state = self.game.get_next_state(state, action, to_play)
            to_play = -to_play

        winner = self.game.get_winner(state, to_play)
        self.last_game_result = (winner, len(memory))
        return [
            {
                "encoded_state": self.game.encode_state(sample["state"], sample["to_play"]),
                "policy_target": sample["mcts_policy"],
                "value_target": float(winner) * sample["to_play"],
            }
            for sample in memory
        ]

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
        value_targets = torch.tensor(
            np.array([s["value_target"] for s in batch]), dtype=torch.float32, device=self.device
        )

        self.model.train()
        self.optimizer.zero_grad()
        policy_logits, value = self.model(states)

        policy_loss = -torch.mean(torch.sum(policy_targets * F.log_softmax(policy_logits, dim=1), dim=1))
        value_loss = F.mse_loss(value.squeeze(1), value_targets)
        total_loss = policy_loss + value_loss

        total_loss.backward()
        self.optimizer.step()
        self.model.eval()
        return total_loss.item(), policy_loss.item(), value_loss.item()

    def _record_game_result(self, winner, game_len):
        self._recent_game_lengths.append(game_len)
        self.game_length_history.append((self.game_count, game_len))
        self._black_win_counts.append(1 if winner == 1 else 0)
        self._white_win_counts.append(1 if winner == -1 else 0)
        if self.game_count % self._winrate_sample_every == 0:
            total = len(self._black_win_counts)
            if total == 0:
                return
            b = float(np.sum(self._black_win_counts)) / total
            w = float(np.sum(self._white_win_counts)) / total
            self.winrate_history.append((self.game_count, b, w, 1.0 - b - w))

    def plot_metrics(self):
        try:
            logs_dir = os.path.join(self.args.get("data_dir", "data"), "logs")
            os.makedirs(logs_dir, exist_ok=True)

            if self.losses["total"]:
                plt.figure(figsize=(10, 6))
                plt.plot(self.losses["total"], label="Total Loss")
                plt.title("Total Training Loss")
                plt.xlabel("Training Iteration")
                plt.ylabel("Loss")
                plt.yscale("log")
                plt.legend()
                plt.grid(True, which="both")
                plt.savefig(os.path.join(logs_dir, "total_loss.png"), dpi=200)
                plt.close()

                plt.figure(figsize=(10, 6))
                for key in ("policy", "value"):
                    plt.plot(self.losses[key], label=key.replace("_", " ").title())
                plt.title("Loss Components")
                plt.xlabel("Training Iteration")
                plt.ylabel("Loss")
                plt.yscale("log")
                plt.legend()
                plt.grid(True, which="both")
                plt.savefig(os.path.join(logs_dir, "loss_components.png"), dpi=200)
                plt.close()

            if self.winrate_history:
                games, b_rates, w_rates, d_rates = zip(*self.winrate_history)
                plt.figure(figsize=(10, 6))
                plt.plot(games, b_rates, label="Black Win Rate", color="black")
                plt.plot(games, w_rates, label="White Win Rate", color="red")
                plt.plot(games, d_rates, label="Draw Rate", color="gray")
                plt.title("Win Rates (rolling window)")
                plt.xlabel("Game Count")
                plt.ylabel("Rate")
                plt.ylim(0, 1)
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(logs_dir, "win_rates.png"), dpi=200)
                plt.close()

            if self.game_length_history:
                games = [g for g, _ in self.game_length_history]
                lens = [l for _, l in self.game_length_history]
                plt.figure(figsize=(10, 6))
                plt.plot(games, lens, alpha=0.3, linewidth=0.5, label="Per-Game Length")
                window = min(50, len(lens))
                if window > 0:
                    avg_games = games[window - 1:]
                    avg_lens = [
                        float(np.mean(lens[i - window + 1: i + 1]))
                        for i in range(window - 1, len(lens))
                    ]
                    plt.plot(avg_games, avg_lens, color="blue", linewidth=2,
                             label=f"Avg (window {window})")
                plt.title("Game Length (steps per game)")
                plt.xlabel("Game Count")
                plt.ylabel("Steps")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(logs_dir, "game_length.png"), dpi=200)
                plt.close()
        except Exception as e:
            print(f"Plotting failed: {e}")

    def _print_selfplay(self, it, games_done, total_games, t0, game_lens, sample_lens, winners):
        dt = max(1e-9, time.time() - t0)
        total_rows = sum(sample_lens) if sample_lens else 0
        sps = total_rows / dt
        gw = max(2, len(str(total_games)))
        head = f"[SelfPlay] Iter={it} Games={games_done:0{gw}d}/{total_games} Sps={sps:.1f}"
        if game_lens:
            arr = np.asarray(game_lens, dtype=np.float64)
            n = len(winners)
            b = sum(1 for w in winners if w == 1) / n * 100
            w = sum(1 for w_ in winners if w_ == -1) / n * 100
            d = 100 - b - w
            head += (f" GameLen:Avg={arr.mean():.1f} Min={int(arr.min())} "
                     f"Max={int(arr.max())} Std={int(round(arr.std()))} "
                     f"Rows={total_rows} "
                     f"BDW={int(round(b)):02d}/{int(round(d)):02d}/{int(round(w)):02d}")
        else:
            head += " GameLen=N/A BDW=N/A"
        print(head, flush=True)

    def learn(self):
        num_iterations = self.args.get("num_iterations", None)
        train_steps = self.args.get("train_steps", 500)
        save_interval = self.args.get("save_interval", 10)
        try:
            for i in itertools.count(1):
                if num_iterations is not None and i > num_iterations:
                    break
                games = self._games_to_order()
                bootstrap = not self._rpg_history
                print(
                    f"\n=== Iter {i} | collect {games} games "
                    f"({'random bootstrap' if bootstrap else 'adaptive'}, "
                    f"RR target={self.replay_ratio}) ===",
                    flush=True,
                )
                print(
                    f"[Stats] total_games={self.game_count} "
                    f"total_samples={self.replay_buffer.total_samples_added}",
                    flush=True,
                )

                game_lens, sample_lens, winners = [], [], []
                started = time.time()
                report_every = max(1, games // 5)
                for gidx in range(games):
                    game_data = self.selfplay()
                    self.replay_buffer.add_game(game_data)
                    winner, game_len = self.last_game_result
                    sample_lens.append(len(game_data))
                    game_lens.append(game_len)
                    winners.append(winner)
                    self.game_count += 1
                    self._record_game_result(winner, game_len)
                    if (gidx + 1) % report_every == 0 or gidx + 1 == games:
                        self._print_selfplay(i, gidx + 1, games, started, game_lens, sample_lens, winners)
                self._rpg_history.append((games, sum(sample_lens)))

                if not self.replay_buffer.is_ready():
                    print(
                        f"[Train] iter={i} skipped (buffer not ready: "
                        f"{self.replay_buffer.total_samples_added} < "
                        f"{self.replay_buffer.min_rows})",
                        flush=True,
                    )
                else:
                    total_loss = policy_loss = value_loss = 0.0
                    steps_done = 0
                    t0 = time.time()
                    pbar = tqdm(range(train_steps), desc=f"Train Iter={i}")
                    for _ in pbar:
                        res = self.train_step()
                        if res:
                            total_loss += res[0]
                            policy_loss += res[1]
                            value_loss += res[2]
                            steps_done += 1
                            pbar.set_postfix(loss=f"{total_loss / steps_done:.4f}")
                    pbar.close()
                    if steps_done:
                        self.losses["total"].append(total_loss / steps_done)
                        self.losses["policy"].append(policy_loss / steps_done)
                        self.losses["value"].append(value_loss / steps_done)
                        print(
                            f"[Train] iter={i} steps={steps_done} "
                            f"t={time.time() - t0:.1f}s | "
                            f"total={self.losses['total'][-1]:.4f} "
                            f"policy={self.losses['policy'][-1]:.4f} "
                            f"value={self.losses['value'][-1]:.4f}",
                            flush=True,
                        )

                if i % save_interval == 0:
                    self.save_checkpoint(f"checkpoint_{i}.pth")
                    self.plot_metrics()
        except KeyboardInterrupt:
            print("\nStopping... saving final checkpoint and plots", flush=True)
            self.save_checkpoint("checkpoint_final.pth")
            self.plot_metrics()
            raise

    def save_checkpoint(self, filename):
        ckpt_dir = os.path.join(self.args.get("data_dir", "data"), "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, filename)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "game_count": self.game_count,
                "losses": self.losses,
                "replay_buffer": self.replay_buffer.get_state(),
                "target_cum": self._target_cum,
                "rpg_history": list(self._rpg_history),
            },
            path,
        )
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, filename=None):
        if filename is None:
            import glob

            data_dir = self.args.get("data_dir", "data")
            checkpoints = glob.glob(os.path.join(data_dir, "checkpoints", "*.pth"))
            if not checkpoints:
                print("No checkpoint found, starting from scratch.")
                return False
            filename = max(checkpoints, key=os.path.getmtime)

        checkpoint = torch.load(filename, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.game_count = checkpoint.get("game_count", 0)
        self.losses = checkpoint.get(
            "losses", {"total": [], "policy": [], "value": []}
        )
        if "replay_buffer" in checkpoint:
            self.replay_buffer.load_state(checkpoint["replay_buffer"])
        self._target_cum = checkpoint.get("target_cum", 0.0)
        self._rpg_history = deque(
            checkpoint.get("rpg_history", []), maxlen=20
        )
        print(f"Checkpoint loaded from {filename}")
        return True
