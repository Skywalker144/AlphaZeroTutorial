import math
from collections import deque

BACKFILL_FACTOR = 1.05


class SelfPlayScheduler:
    def __init__(
        self,
        bootstrap_games,
        min_rows,
        rows_needed_per_iteration,
        fallback_rows_per_game,
        history_length=20,
    ):
        self.bootstrap_games = int(bootstrap_games)
        self.min_rows = int(min_rows)
        self.rows_needed_per_iteration = float(rows_needed_per_iteration)
        self.fallback_rows_per_game = float(fallback_rows_per_game)
        self._iteration_history = deque(maxlen=history_length)
        self._cumulative_target_rows = 0.0
        self._target_initialized = False

    @property
    def is_bootstrapped(self):
        return bool(self._iteration_history)

    def games_to_order(self, total_rows_produced):
        # Iteration 0: no history yet, play the fixed bootstrap batch.
        if not self._iteration_history:
            return self.bootstrap_games
        # Iteration 1: backfill to min_rows using the average rows/game learned
        # from the bootstrap, with a safety factor so min_rows is actually met.
        if not self._target_initialized:
            self._cumulative_target_rows = float(self.min_rows)
            self._target_initialized = True
            deficit_rows = self._cumulative_target_rows - total_rows_produced
            if deficit_rows <= 0:
                return 0
            games = deficit_rows / self._rows_per_game()
            return math.ceil(games * BACKFILL_FACTOR)
        # Iteration 2+: grow the cumulative target by the per-iteration need.
        self._cumulative_target_rows += self.rows_needed_per_iteration
        deficit_rows = self._cumulative_target_rows - total_rows_produced
        if deficit_rows <= 0:
            return 0
        return math.ceil(deficit_rows / self._rows_per_game())

    def record_iteration(self, games_played, rows_produced):
        self._iteration_history.append((games_played, rows_produced))

    def _rows_per_game(self):
        for games_played, rows_produced in reversed(self._iteration_history):
            if games_played > 0 and rows_produced > 0:
                return rows_produced / games_played
        return self.fallback_rows_per_game

    def state(self):
        return {
            "cumulative_target_rows": self._cumulative_target_rows,
            "target_initialized": self._target_initialized,
            "iteration_history": list(self._iteration_history),
        }

    def load_state(self, state):
        self._cumulative_target_rows = float(state["cumulative_target_rows"])
        self._target_initialized = bool(state["target_initialized"])
        self._iteration_history = deque(
            state["iteration_history"], maxlen=self._iteration_history.maxlen
        )
