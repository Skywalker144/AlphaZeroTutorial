import numpy as np


def compute_desired_num_rows(
    num_usable_rows,
    min_rows,
    add_to_data_rows,
    taper_window_exponent,
    expand_window_per_row,
    taper_window_scale,
    max_rows,
):
    """KataGo shuffler window size: desired training-window rows given total run rows."""
    window_taper_offset = taper_window_scale if taper_window_scale is not None else min_rows
    power_law_x = num_usable_rows - min_rows + window_taper_offset + add_to_data_rows
    unscaled_power_law = (
        power_law_x ** taper_window_exponent
    ) - (window_taper_offset ** taper_window_exponent)
    scaled_power_law = unscaled_power_law / (
        taper_window_exponent * (window_taper_offset ** (taper_window_exponent - 1))
    )
    desired_num_rows = int(scaled_power_law * expand_window_per_row + min_rows)

    desired_num_rows = max(desired_num_rows, min_rows)
    if max_rows is not None:
        desired_num_rows = min(desired_num_rows, max_rows)
    return desired_num_rows


class ReplayBuffer:
    def __init__(
        self,
        min_rows=150000,
        taper_window_exponent=0.8,
        expand_window_per_row=0.3,
        keep_target_rows=10000000,
    ):
        self.min_rows = max(int(min_rows), 1)
        self.taper_window_exponent = float(taper_window_exponent)
        self.expand_window_per_row = float(expand_window_per_row)
        self.keep_target_rows = max(int(keep_target_rows), 1)
        self.buffer = []
        self.total_samples_added = 0

    def __len__(self):
        return len(self.buffer)

    def window_size(self):
        return compute_desired_num_rows(
            num_usable_rows=self.total_samples_added,
            min_rows=self.min_rows,
            add_to_data_rows=0.0,
            taper_window_exponent=self.taper_window_exponent,
            expand_window_per_row=self.expand_window_per_row,
            taper_window_scale=self.min_rows,
            max_rows=None,
        )

    def capacity(self):
        return min(self.window_size(), self.keep_target_rows)

    def add_game(self, game_memory):
        self.buffer.extend(game_memory)
        self.total_samples_added += len(game_memory)
        overflow = len(self.buffer) - self.capacity()
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
