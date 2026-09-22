import numpy as np


class TicTacToe:
    def __init__(self):
        self.board_size = 3
        self.num_planes = 3

    def get_initial_state(self):
        return np.zeros((self.board_size, self.board_size), dtype=np.int8)

    @staticmethod
    def get_legal_action_mask(state, to_play):
        return state.flatten() == 0

    def mask_illegal_actions(self, state, to_play, policy_logits):
        """Return logits with illegal actions set to -inf before softmax."""
        return np.where(self.get_legal_action_mask(state, to_play), policy_logits, -np.inf)

    def get_next_state(self, state, action, to_play):
        state = state.copy()
        row, col = divmod(action, self.board_size)
        state[row, col] = to_play
        return state

    def get_winner(self, state, to_play):
        """Return the absolute winner: 1 (X) / -1 (O) / 0 (draw) / None (ongoing).

        ``to_play`` does not affect the result for this game (the winner is a
        pure function of the board state); it is kept for interface
        uniformity and forward compatibility with games whose outcome
        depends on the side to move (e.g. Go with pass rules).
        """
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

    def is_terminal(self, state, to_play):
        return self.get_winner(state, to_play) is not None

    def encode_state(self, state, to_play):
        size = self.board_size
        encoded = np.zeros((self.num_planes, size, size), dtype=np.int8)
        encoded[0] = (state == to_play)
        encoded[1] = (state == -to_play)
        encoded[2] = (to_play > 0) * np.ones((size, size), dtype=np.int8)
        return encoded
