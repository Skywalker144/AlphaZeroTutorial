import numpy as np
import pytest

from envs.gomoku import Gomoku
from envs.tictactoe import TicTacToe


def horizontal_five():
    state = np.zeros((9, 9), dtype=np.int8)
    state[4, 1:6] = 1
    return state


def overline_six():
    state = np.zeros((9, 9), dtype=np.int8)
    state[4, 1:7] = 1
    return state


def diagonal_five():
    state = np.zeros((9, 9), dtype=np.int8)
    for i in range(5):
        state[i, i] = 1
    return state


def full_draw():
    # Random full 9x9 board with no five in a row in any direction.
    return np.array([
        [1, -1, 1, -1, -1, -1, 1, -1, 1],
        [-1, 1, 1, -1, 1, 1, 1, -1, -1],
        [-1, 1, -1, -1, 1, -1, 1, 1, 1],
        [-1, -1, 1, -1, 1, 1, -1, 1, 1],
        [1, 1, -1, 1, -1, 1, -1, 1, -1],
        [-1, -1, -1, 1, 1, 1, 1, -1, -1],
        [1, 1, -1, 1, -1, -1, -1, 1, 1],
        [1, 1, -1, 1, -1, 1, 1, 1, -1],
        [-1, -1, 1, -1, 1, 1, 1, -1, 1],
    ], dtype=np.int8)


class TestGomoku:
    @pytest.fixture(autouse=True)
    def game(self):
        self.game = Gomoku(board_size=9)

    def test_initial_state(self):
        state = self.game.get_initial_state()
        assert state.shape == (9, 9)
        assert np.all(state == 0)

    def test_five_in_a_row_wins(self):
        assert self.game.get_winner(horizontal_five()) == 1
        assert self.game.is_terminal(horizontal_five())

    def test_overline_wins_freestyle(self):
        assert self.game.get_winner(overline_six()) == 1

    def test_diagonal_five_wins(self):
        assert self.game.get_winner(diagonal_five()) == 1

    def test_white_wins(self):
        state = horizontal_five()
        state[4, 1:6] = -1
        assert self.game.get_winner(state) == -1

    def test_empty_not_terminal(self):
        assert self.game.get_winner(self.game.get_initial_state()) is None
        assert not self.game.is_terminal(self.game.get_initial_state())

    def test_full_board_draw(self):
        assert self.game.get_winner(full_draw()) == 0
        assert self.game.is_terminal(full_draw())

    def test_next_state_places_stone(self):
        state = self.game.get_initial_state()
        next_state = self.game.get_next_state(state, 5, 1)
        assert next_state[0, 5] == 1
        assert np.all(state == 0)

    def test_legal_actions_are_empty_cells(self):
        state = self.game.get_initial_state()
        legal = self.game.get_is_legal_actions(state, 1)
        assert np.sum(legal) == 81
        state[0, 0] = 1
        legal = self.game.get_is_legal_actions(state, -1)
        assert not legal[0]
        assert np.sum(legal) == 80

    def test_encode_state_channels(self):
        state = self.game.get_initial_state()
        encoded = self.game.encode_state(state, 1)
        assert encoded.shape == (3, 9, 9)
        assert encoded[0].sum() == 0
        assert encoded[2].sum() == 81

    def test_nonterminal_three_in_row(self):
        state = np.zeros((9, 9), dtype=np.int8)
        state[4, 1:4] = 1
        assert self.game.get_winner(state) is None


class TestTicTacToe:
    @pytest.fixture(autouse=True)
    def game(self):
        self.game = TicTacToe()

    def test_row_win(self):
        state = np.zeros((3, 3), dtype=np.int8)
        state[0, :] = 1
        assert self.game.get_winner(state) == 1

    def test_col_win(self):
        state = np.zeros((3, 3), dtype=np.int8)
        state[:, 1] = -1
        assert self.game.get_winner(state) == -1

    def test_diag_win(self):
        state = np.zeros((3, 3), dtype=np.int8)
        np.fill_diagonal(state, 1)
        assert self.game.get_winner(state) == 1

    def test_draw(self):
        state = np.array([[1, -1, 1], [1, -1, -1], [-1, 1, 1]], dtype=np.int8)
        assert self.game.get_winner(state) == 0

    def test_incomplete(self):
        state = np.zeros((3, 3), dtype=np.int8)
        state[0, 0] = 1
        assert self.game.get_winner(state) is None

    def test_encode_state(self):
        state = np.zeros((3, 3), dtype=np.int8)
        state[0, 0] = 1
        encoded = self.game.encode_state(state, 1)
        assert encoded.shape == (3, 3, 3)
        assert encoded[0][0, 0] == 1
        assert encoded[1][0, 0] == 0
