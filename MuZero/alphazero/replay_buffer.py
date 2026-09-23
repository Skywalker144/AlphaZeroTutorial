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
        self.num_rows = 0

    def __len__(self):
        return self.num_rows

    def window_size(self):
        e = self.taper_window_exponent
        unscaled = (self.total_samples_added ** e) - (self.min_rows ** e)
        scaled = unscaled / (e * self.min_rows ** (e - 1))
        desired = int(scaled * self.expand_window_per_row + self.min_rows)
        desired = max(desired, self.min_rows)
        if self.max_rows is not None:
            desired = min(desired, self.max_rows)
        return desired

    def add_game(self, game):
        # 一局存成一个列表，窗口按累计步数裁剪整局
        if len(game) == 0:
            return
        self.buffer.append(game)
        self.total_samples_added += len(game)
        self.num_rows += len(game)
        window = self.window_size()
        while self.buffer and self.num_rows - len(self.buffer[0]) >= window:
            self.num_rows -= len(self.buffer.pop(0))

    def sample(self, batch_size, unroll_steps, action_size):
        if not self.is_ready() or self.num_rows < batch_size:
            return []
        lengths = np.array([len(game) for game in self.buffer])
        probabilities = lengths / lengths.sum()
        game_indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = []
        for game_index in game_indices:
            game = self.buffer[game_index]
            start = int(np.random.randint(len(game)))
            samples.append(self._build_sample(game, start, unroll_steps, action_size))
        return samples

    def _build_sample(self, game, start, unroll_steps, action_size):
        length = len(game)
        actions = np.zeros(unroll_steps, dtype=np.int64)
        policy_targets = np.zeros((unroll_steps + 1, action_size), dtype=np.float32)
        value_targets = np.zeros(unroll_steps + 1, dtype=np.float32)
        policy_mask = np.zeros(unroll_steps + 1, dtype=np.float32)
        to_plays = np.zeros(unroll_steps + 1, dtype=np.float32)
        base_value = game[start]["value_target"]
        start_player = game[start]["player"]
        for i in range(unroll_steps + 1):
            to_plays[i] = start_player * ((-1) ** i)
            if start + i < length:
                step = game[start + i]
                policy_targets[i] = step["mcts_policy"]
                value_targets[i] = step["value_target"]
                policy_mask[i] = 1.0
            else:
                # 终局之后不再有真实 MCTS 策略，屏蔽 policy loss；value 沿用最终胜负
                value_targets[i] = base_value * ((-1) ** i)
            if i < unroll_steps and start + i < length:
                actions[i] = game[start + i]["action"]
        return {
            "observation": game[start]["observation"],
            "actions": actions,
            "policy_targets": policy_targets,
            "value_targets": value_targets,
            "policy_mask": policy_mask,
            "to_plays": to_plays,
        }

    def is_ready(self):
        return self.total_samples_added >= self.min_rows

    def get_state(self):
        return {
            "buffer": list(self.buffer),
            "total_samples_added": self.total_samples_added,
            "num_rows": self.num_rows,
        }

    def load_state(self, state):
        self.buffer = list(state.get("buffer", []))
        self.total_samples_added = state.get("total_samples_added", 0)
        self.num_rows = state.get("num_rows", sum(len(game) for game in self.buffer))
