import numpy as np


class Gomoku:
    """Freestyle Gomoku: five or more stones in a row wins."""

    def __init__(self, board_size=9):
        self.board_size = board_size
        self.num_planes = 3

    def get_initial_state(self):
        return np.zeros((self.board_size, self.board_size), dtype=np.int8)

    @staticmethod
    def get_is_legal_actions(state, to_play):
        return state.flatten() == 0

    def get_next_state(self, state, action, to_play):
        state = state.copy()
        row, col = divmod(action, self.board_size)
        state[row, col] = to_play
        return state

    def get_winner(self, state):
        size = self.board_size
        dirs = ((1, 0), (0, 1), (1, 1), (1, -1))
        for r in range(size):
            for c in range(size):
                stone = state[r, c]
                if stone == 0:
                    continue
                for dr, dc in dirs:
                    pr, pc = r - dr, c - dc
                    if 0 <= pr < size and 0 <= pc < size and state[pr, pc] == stone:
                        continue
                    length = 1
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < size and 0 <= nc < size and state[nr, nc] == stone:
                        length += 1
                        nr += dr
                        nc += dc
                    if length >= 5:
                        return stone
        if np.all(state != 0):
            return 0
        return None

    def is_terminal(self, state):
        return self.get_winner(state) is not None

    def encode_state(self, state, to_play):
        size = self.board_size
        encoded = np.zeros((self.num_planes, size, size), dtype=np.int8)
        encoded[0] = (state == to_play)
        encoded[1] = (state == -to_play)
        encoded[2] = (to_play > 0) * np.ones((size, size), dtype=np.int8)
        return encoded
