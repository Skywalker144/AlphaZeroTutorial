import numpy as np


class TicTacToe:
    def __init__(self):
        self.board_size = 3
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
        lines = [list(range(i * size, (i + 1) * size)) for i in range(size)] + [
            list(range(i, size * size, size)) for i in range(size)
        ] + [[i * size + i for i in range(size)]] + [
            [i * size + (size - 1 - i) for i in range(size)]
        ]
        for line in lines:
            stones = [state.flatten()[i] for i in line]
            if all(s == 1 for s in stones):
                return 1
            if all(s == -1 for s in stones):
                return -1
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
