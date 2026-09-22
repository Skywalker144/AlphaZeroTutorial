import os

from .plots import render_training, winrate_history, write_metrics_csv


class MetricsTracker:
    def __init__(self, winrate_window=300, winrate_sample_every=10):
        self.winrate_window = winrate_window
        self.winrate_sample_every = max(1, winrate_sample_every)
        self.losses = {"total": [], "policy": [], "value": []}
        self.game_records = []

    def record_game(self, game_index, winner, length, iteration):
        self.game_records.append((game_index, winner, length, iteration))

    def record_losses(self, total, policy, value):
        self.losses["total"].append(total)
        self.losses["policy"].append(policy)
        self.losses["value"].append(value)

    @staticmethod
    def winrate_summary(winners):
        n = len(winners)
        if n == 0:
            return 0.0, 0.0, 0.0
        black = sum(1 for w in winners if w == 1) / n
        white = sum(1 for w in winners if w == -1) / n
        return black, 1.0 - black - white, white

    def winrate_history(self):
        return winrate_history(
            self.game_records, self.winrate_window, self.winrate_sample_every
        )

    def plot(self, out_dir):
        try:
            os.makedirs(out_dir, exist_ok=True)
            write_metrics_csv(out_dir, self.losses, self.game_records)
            render_training(
                out_dir,
                self.losses,
                self.game_records,
                self.winrate_window,
                self.winrate_sample_every,
            )
        except Exception as e:
            print(f"Plotting failed: {e}")

    def state(self):
        return {
            "losses": self.losses,
            "game_records": self.game_records,
        }

    def load_state(self, state):
        self.losses = state["losses"]
        self.game_records = state["game_records"]
