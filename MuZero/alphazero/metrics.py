import os

from .plots import render_training, write_metrics_csv


class MetricsTracker:
    def __init__(self):
        self.losses = {
            "total": [],
            "policy": [],
            "value": [],
            "step_losses": [],
            "grad_representation": [],
            "grad_dynamics": [],
            "grad_prediction": [],
        }
        self.game_records = []

    def record_game(self, game_index, winner, length, iteration):
        self.game_records.append((game_index, winner, length, iteration))

    def record_losses(self, total, policy, value, step_losses=None, grad_norms=None):
        self.losses["total"].append(total)
        self.losses["policy"].append(policy)
        self.losses["value"].append(value)
        self.losses["step_losses"].append(list(step_losses) if step_losses else [])
        grad_norms = grad_norms or {}
        self.losses["grad_representation"].append(grad_norms.get("representation", float("nan")))
        self.losses["grad_dynamics"].append(grad_norms.get("dynamics", float("nan")))
        self.losses["grad_prediction"].append(grad_norms.get("prediction", float("nan")))

    @staticmethod
    def winrate_summary(winners):
        n = len(winners)
        if n == 0:
            return 0.0, 0.0, 0.0
        black = sum(1 for w in winners if w == 1) / n
        white = sum(1 for w in winners if w == -1) / n
        return black, 1.0 - black - white, white

    def plot(self, out_dir):
        try:
            os.makedirs(out_dir, exist_ok=True)
            write_metrics_csv(out_dir, self.losses, self.game_records)
            render_training(out_dir, self.losses, self.game_records)
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
