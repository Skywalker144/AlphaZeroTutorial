import numpy as np


class ReplayBuffer:
    def __init__(
        self,
        min_rows=20000,
        taper_window_exponent=0.675,
        expand_window_per_row=0.4,
        max_rows=None,
    ):
        self.min_rows = max(int(min_rows), 1)
        self.taper_window_exponent = float(taper_window_exponent)
        self.expand_window_per_row = float(expand_window_per_row)
        self.max_rows = max(int(max_rows), 1) if max_rows is not None else None
        self.buffer = []
        self.total_samples_added = 0

    def __len__(self):
        return len(self.buffer)

    def window_size(self):
        e = self.taper_window_exponent
        unscaled = (self.total_samples_added ** e) - (self.min_rows ** e)
        scaled = unscaled / (e * self.min_rows ** (e - 1))
        desired = int(scaled * self.expand_window_per_row + self.min_rows)
        desired = max(desired, self.min_rows)
        if self.max_rows is not None:
            desired = min(desired, self.max_rows)
        return desired

    def add_game(self, game_memory):
        self.buffer.extend(game_memory)
        self.total_samples_added += len(game_memory)
        overflow = len(self.buffer) - self.window_size()
        if overflow > 0:
            del self.buffer[:overflow]

    def sample(self, batch_size):
        if not self.is_ready() or len(self.buffer) < batch_size:
            return []
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def is_ready(self):
        return self.total_samples_added >= self.min_rows

    def get_state(self):
        return {
            "buffer": list(self.buffer),
            "total_samples_added": self.total_samples_added,
        }

    def load_state(self, state):
        self.buffer = list(state.get("buffer", []))
        self.total_samples_added = state.get("total_samples_added", len(self.buffer))
