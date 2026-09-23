import numpy as np
import pytest
import torch

from alphazero import MCTS, MuZeroNet, auto_device
from alphazero.alphazero_parallel import _select
from alphazero.mcts import Node
from alphazero.utils import add_dirichlet_noise, softmax
from envs.gomoku import Gomoku


def make_mcts(num_simulations=20):
    game = Gomoku(board_size=9)
    model = MuZeroNet(game.board_size, game.num_planes, num_blocks=1, num_channels=8)
    args = {"num_simulations": num_simulations, "pb_c_init": 1.25, "pb_c_base": 19652, "dirichlet_total_concentration": 0.03 * 9 ** 2, "dirichlet_noise_weight": 0.25}
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


class TestMCTS:
    @pytest.mark.parametrize(
        "args, expected_action",
        [({}, 1), ({"pb_c_base": 1e9}, 0), ({"pb_c_init": 0.0}, 0)],
    )
    def test_visit_dependent_exploration_changes_selected_action(self, args, expected_action):
        game = Gomoku(board_size=9)
        model = MuZeroNet(9, 3, num_blocks=1, num_channels=8)
        mcts = MCTS(game, args, model, "cpu")
        root = Node(1)
        root.visits = 19651
        for action, prior, value in [(0, 0.1, -0.04), (1, 0.9, 0.0)]:
            child = Node(-1, prior=prior, parent=root, action_taken=action)
            child.visits = 9825
            child.value_sum = value * child.visits
            root.children.append(child)

        assert mcts.select(root).action_taken == expected_action
        assert _select(
            root, args.get("pb_c_base", 19652), args.get("pb_c_init", 1.25)
        ).action_taken == expected_action

    @pytest.mark.parametrize(
        "child_value, expected_action", [(-1.0, 0), (0.0, 0), (0.1, 0), (1.0, 1)]
    )
    def test_selection_scales_visited_values_but_keeps_unvisited_value_zero(
        self, child_value, expected_action
    ):
        game = Gomoku(board_size=9)
        model = MuZeroNet(9, 3, num_blocks=1, num_channels=8)
        mcts = MCTS(game, {}, model, "cpu")
        root = Node(1)
        root.visits = 2
        visited = Node(-1, prior=0.6, parent=root, action_taken=0)
        visited.update(child_value)
        unvisited = Node(-1, prior=0.4, parent=root, action_taken=1)
        root.children = [visited, unvisited]

        assert mcts.select(root).action_taken == expected_action
        assert _select(root, 19652, 1.25).action_taken == expected_action
        assert visited.q_value() == child_value
        assert visited.value_sum == child_value
        assert visited.visits == 1
        assert unvisited.value_sum == 0.0
        assert unvisited.visits == 0

    def test_policy_sums_to_one(self):
        game, mcts = make_mcts()
        state = game.get_initial_state()
        policy, _ = mcts.search(state, 1, 20)
        assert np.isclose(policy.sum(), 1.0)

    def test_root_policy_only_on_legal(self):
        game, mcts = make_mcts()
        state = game.get_initial_state()
        state[0, 0] = 1
        policy, _ = mcts.search(state, -1, 20)
        assert policy[0] == 0.0

    def test_tree_expand_uses_full_action_space(self):
        # 树内节点按完整动作空间展开，不查询真实棋盘的合法动作
        game, mcts = make_mcts()
        node = Node(-1)
        mcts.expand(node, np.full(81, 1.0 / 81))
        assert {child.action_taken for child in node.children} == set(range(81))

    def test_eval_zero_simulations_returns_masked_policy(self):
        game, mcts = make_mcts()
        mcts.args["mode"] = "eval"
        state = game.get_initial_state()
        state[0, 0] = 1
        policy, _ = mcts.search(state, -1, 0)
        assert policy[0] == 0.0
        assert np.isclose(policy.sum(), 1.0)

    def test_deterministic_under_no_gpu_noise(self):
        # With epsilon=0 the search is deterministic for a fixed model.
        game = Gomoku(board_size=9)
        model = MuZeroNet(9, 3, num_blocks=1, num_channels=8)
        model.eval()
        args = {"pb_c_init": 1.25, "pb_c_base": 19652, "dirichlet_total_concentration": 0.03 * 9 ** 2, "dirichlet_noise_weight": 0.0}
        state = game.get_initial_state()
        p1, _ = MCTS(game, args, model, "cpu").search(state, 1, 15)
        p2, _ = MCTS(game, args, model, "cpu").search(state, 1, 15)
        assert np.array_equal(p1, p2)
