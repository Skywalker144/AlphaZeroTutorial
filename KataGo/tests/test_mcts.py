import numpy as np
import pytest
import torch

from alphazero import MCTS, ResNet, auto_device
from alphazero.mcts import Node
from alphazero.utils import add_dirichlet_noise, softmax
from envs.gomoku import Gomoku


def make_mcts(num_simulations=20):
    game = Gomoku(board_size=9)
    model = ResNet(game.board_size, game.num_planes, num_blocks=1, num_channels=8)
    args = {"num_simulations": num_simulations, "c_puct": 1.5, "dirichlet_total_concentration": 0.03 * 9 ** 2, "dirichlet_noise_weight": 0.25}
    device = auto_device()
    return game, MCTS(game, args, model, device)


class TestSoftmax:
    def test_sums_to_one(self):
        p = softmax(np.array([1.0, 2.0, 3.0]))
        assert np.isclose(p.sum(), 1.0)

    def test_high_logit_wins(self):
        p = softmax(np.array([0.0, 0.0, 5.0]))
        assert np.argmax(p) == 2
        assert p[2] > 0.9


class TestDirichlet:
    def test_total_concentration_is_split_across_legal_actions(self, monkeypatch):
        captured = {}

        def fake_dirichlet(concentrations):
            captured["concentrations"] = np.asarray(concentrations)
            return np.full(len(concentrations), 1.0 / len(concentrations))

        monkeypatch.setattr(np.random, "dirichlet", fake_dirichlet)
        add_dirichlet_noise(
            np.array([0.5, 0.25, 0.25, 0.0]),
            total_concentration=1.2,
            noise_weight=0.25,
            legal_actions_mask=np.array([True, True, True, False]),
        )

        assert np.isclose(captured["concentrations"].sum(), 1.2)
        assert np.allclose(captured["concentrations"], 0.4)

    def test_preserves_support(self):
        policy = np.array([0.5, 0.5, 0.0, 0.0])
        noisy = add_dirichlet_noise(
            policy, total_concentration=0.06, noise_weight=0.25,
            legal_actions_mask=np.array([True, True, False, False]),
        )
        assert noisy[2] == 0.0 and noisy[3] == 0.0
        assert np.isclose(noisy.sum(), 1.0)

    def test_single_support_unchanged(self):
        policy = np.array([1.0, 0.0, 0.0])
        assert np.array_equal(add_dirichlet_noise(
            policy, total_concentration=0.03,
            legal_actions_mask=np.array([True, False, False]),
        ), policy)

    def test_noise_includes_tiny_and_underflowed_legal_probabilities(self, monkeypatch):
        monkeypatch.setattr(
            np.random, "dirichlet",
            lambda concentrations: np.full(len(concentrations), 1.0 / len(concentrations)),
        )
        policy = np.array([1.0 - 1e-9, 1e-9, 0.0, 0.0])
        noisy = add_dirichlet_noise(
            policy, total_concentration=0.06, noise_weight=0.25,
            legal_actions_mask=np.array([True, True, True, False]),
        )
        assert noisy[1] > 0.08
        assert noisy[2] > 0.08
        assert noisy[3] == 0.0
        assert np.isclose(noisy.sum(), 1.0)


class TestMCTS:
    def test_expand_uses_legality_instead_of_probability(self):
        game, mcts = make_mcts()
        state = game.get_initial_state()
        state[0, 0] = 1
        node = Node(state, -1)
        policy = np.zeros(game.board_size ** 2)
        policy[0] = 0.5  # Even a positive prior cannot make an occupied cell legal.
        policy[1] = 0.5 - 1e-9
        policy[2] = 1e-9
        mcts.expand(node, policy)
        assert {child.action_taken for child in node.children} == set(range(1, 81))

    def test_policy_sums_to_one(self):
        game, mcts = make_mcts()
        state = game.get_initial_state()
        policy, _ = mcts.search(state, 1, 20)
        assert np.isclose(policy.sum(), 1.0)

    def test_policy_only_on_legal(self):
        game, mcts = make_mcts()
        state = game.get_initial_state()
        state[0, 0] = 1
        policy, _ = mcts.search(state, -1, 20)
        assert policy[0] == 0.0

    def test_terminal_value_correct(self):
        import torch

        torch.manual_seed(0)
        np.random.seed(0)
        game = Gomoku(board_size=9)
        model = ResNet(9, 3, num_blocks=1, num_channels=8)
        mcts = MCTS(game, {"c_puct": 1.5}, model, "cpu")
        # Four in a row pinned against the left edge: (4, 4) is the only win.
        state = np.zeros((9, 9), dtype=np.int8)
        state[4, 0:4] = 1
        policy, _ = mcts.search(state, 1, 120)
        assert np.argmax(policy) == 4 * 9 + 4
        assert policy[4 * 9 + 4] > 0.5

    def test_deterministic_under_no_gpu_noise(self):
        # With epsilon=0 the search is deterministic for a fixed model.
        game = Gomoku(board_size=9)
        model = ResNet(9, 3, num_blocks=1, num_channels=8)
        model.eval()
        args = {"c_puct": 1.5, "dirichlet_total_concentration": 0.03 * 9 ** 2, "dirichlet_noise_weight": 0.0}
        state = game.get_initial_state()
        p1, _ = MCTS(game, args, model, "cpu").search(state, 1, 15)
        p2, _ = MCTS(game, args, model, "cpu").search(state, 1, 15)
        assert np.array_equal(p1, p2)
