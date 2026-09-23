import numpy as np
import pytest
import torch

from alphazero import MCTS, ResNet, auto_device
from alphazero.mcts import Node
from alphazero.utils import (
    add_dirichlet_noise,
    apply_temperature,
    chosen_move_temperature,
    interpolate_early,
    root_policy_temperature,
    search_visit_counts,
    softmax,
    value_target,
)
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
            board_size=3,
        )

        concentrations = captured["concentrations"]
        assert np.isclose(concentrations.sum(), 1.2)
        assert concentrations[0] > concentrations[1]
        assert np.isclose(concentrations[1], concentrations[2])

    def test_preserves_support(self):
        policy = np.array([0.5, 0.5, 0.0, 0.0])
        noisy = add_dirichlet_noise(
            policy, total_concentration=0.06, noise_weight=0.25,
            legal_actions_mask=np.array([True, True, False, False]),
            board_size=3,
        )
        assert noisy[2] == 0.0 and noisy[3] == 0.0
        assert np.isclose(noisy.sum(), 1.0)

    def test_single_support_unchanged(self):
        policy = np.array([1.0, 0.0, 0.0])
        assert np.array_equal(add_dirichlet_noise(
            policy, total_concentration=0.03,
            legal_actions_mask=np.array([True, False, False]),
            board_size=3,
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
            board_size=3,
        )
        assert noisy[1] > 0.08
        assert noisy[2] > 0.08
        assert noisy[3] == 0.0
        assert np.isclose(noisy.sum(), 1.0)


class TestTemperature:
    def test_interpolate_early_endpoints(self):
        assert np.isclose(interpolate_early(0, 19, 0.75, 0.15, 9), 0.75)
        assert interpolate_early(10_000, 19, 0.75, 0.15, 9) < 0.15 + 1e-9

    def test_interpolate_early_board_scaling(self):
        assert np.isclose(interpolate_early(9, 19, 0.75, 0.15, 9), 0.45)
        assert np.isclose(interpolate_early(19, 19, 0.75, 0.15, 19), 0.45)

    def test_apply_temperature_identity(self):
        probs = np.array([0.5, 0.25, 0.25])
        assert np.allclose(apply_temperature(probs, 1.0), probs)

    def test_apply_temperature_flattens_and_sharpens(self):
        probs = np.array([0.5, 0.25, 0.25])
        assert apply_temperature(probs, 2.0)[0] < probs[0]
        assert apply_temperature(probs, 0.5)[0] > probs[0]

    def test_apply_temperature_near_zero_is_argmax(self):
        probs = np.array([0.5, 0.25, 0.25])
        assert np.array_equal(apply_temperature(probs, 0.0), np.array([1.0, 0.0, 0.0]))

    def test_apply_temperature_keeps_zeros(self):
        probs = np.array([0.9, 0.1, 0.0])
        assert apply_temperature(probs, 1.5)[2] == 0.0

    def test_temperature_schedules_use_configured_values(self):
        args = {
            "chosen_move_temperature_halflife": 9,
            "root_policy_temperature_early": 1.3,
            "root_policy_temperature": 1.1,
            "chosen_move_temperature_early": 0.75,
            "chosen_move_temperature": 0.15,
        }
        assert np.isclose(root_policy_temperature(args, 0, 9), 1.3)
        assert np.isclose(chosen_move_temperature(args, 0, 9), 0.75)


class TestCheapSearch:
    def test_cheap_search_skips_root_temperature_and_noise(self, monkeypatch):
        import alphazero.mcts as mcts_module

        calls = {"temperature": 0, "noise": 0}
        original_temperature = mcts_module.apply_temperature
        original_noise = mcts_module.add_dirichlet_noise

        def spy_temperature(*args, **kwargs):
            calls["temperature"] += 1
            return original_temperature(*args, **kwargs)

        def spy_noise(*args, **kwargs):
            calls["noise"] += 1
            return original_noise(*args, **kwargs)

        monkeypatch.setattr(mcts_module, "apply_temperature", spy_temperature)
        monkeypatch.setattr(mcts_module, "add_dirichlet_noise", spy_noise)

        game, mcts = make_mcts()
        state = game.get_initial_state()
        mcts.search(state, 1, 5, 0, True)
        assert calls == {"temperature": 0, "noise": 0}
        mcts.search(state, 1, 5, 0, False)
        assert calls == {"temperature": 1, "noise": 1}

    def test_evaluated_child_keeps_network_outputs_for_reuse(self):
        game, mcts = make_mcts()
        _, _, root = mcts.search(game.get_initial_state(), 1, 5, 0, False)
        evaluated = [child for child in root.children if child.children]
        assert evaluated
        for child in evaluated:
            assert child.prior_policy is not None
            assert child.nn_wdl is not None


class TestFpu:
    def test_unvisited_value_blends_parent_and_network(self):
        game, mcts = make_mcts()
        mcts.args = {"c_puct": 0.0, "fpu_reduction_max": 0.2}
        root = Node(game.get_initial_state(), 1)
        root.visits = 10
        root.wdl_sum = np.array([8.0, 0.0, 2.0])
        root.nn_wdl = np.array([0.1, 0.0, 0.9])
        visited = Node(None, -1, prior=0.2, parent=root)
        visited.visits = 1
        visited.wdl_sum = np.array([0.9, 0.0, 0.1])
        unvisited = Node(None, -1, prior=0.8, parent=root)
        root.children = [visited, unvisited]

        assert mcts.select(root, False, False) is visited


class TestValueTarget:
    def test_win_draw_loss_encoding(self):
        assert np.array_equal(value_target(1, 1), np.array([1, 0, 0], dtype=np.float32))
        assert np.array_equal(value_target(0, 1), np.array([0, 1, 0], dtype=np.float32))
        assert np.array_equal(value_target(-1, 1), np.array([0, 0, 1], dtype=np.float32))
        assert np.array_equal(value_target(-1, -1), np.array([1, 0, 0], dtype=np.float32))


class TestVisitFloors:
    def test_floors_apply_on_small_board(self):
        assert search_visit_counts({}, 3) == (50, 20)

    def test_formula_wins_on_large_board(self):
        assert search_visit_counts({}, 9) == (round(1.66 * 81), round(0.28 * 81))

    def test_explicit_values_with_zero_floors(self):
        args = {
            "num_simulations": 12,
            "cheap_search_visits": 3,
            "full_search_visits_floor": 0,
            "cheap_search_visits_floor": 0,
        }
        assert search_visit_counts(args, 3) == (12, 3)

    def test_cheap_never_exceeds_full(self):
        args = {
            "num_simulations": 10,
            "cheap_search_visits": 40,
            "full_search_visits_floor": 0,
            "cheap_search_visits_floor": 0,
        }
        assert search_visit_counts(args, 3) == (10, 10)


class TestTreeReuse:
    def test_advance_promotes_matching_child(self):
        game, mcts = make_mcts()
        state = game.get_initial_state()
        policy, _, root = mcts.search(state, 1, 10, 0, False)
        action = int(np.argmax(policy))
        child = mcts.advance(root, action)
        assert child is not None
        assert child.parent is None
        assert child.action_taken == action

    def test_advance_missing_action_returns_none(self):
        game, mcts = make_mcts()
        state = game.get_initial_state()
        state[0, 0] = 1
        _, _, root = mcts.search(state, -1, 5, 0, False)
        assert mcts.advance(root, 0) is None

    def test_reuse_continues_until_target_visits(self):
        game, mcts = make_mcts()
        state = game.get_initial_state()
        policy, _, root = mcts.search(state, 1, 10, 0, False)
        action = int(np.argmax(policy))
        reused = mcts.advance(root, action)
        inherited = reused.visits
        _, _, root2 = mcts.search(state, -1, 30, 1, True, root=reused)
        assert root2 is reused
        assert root2.visits == max(inherited, 31)


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
        policy, _, _ = mcts.search(state, 1, 20, 0, False)
        assert np.isclose(policy.sum(), 1.0)

    def test_policy_only_on_legal(self):
        game, mcts = make_mcts()
        state = game.get_initial_state()
        state[0, 0] = 1
        policy, _, _ = mcts.search(state, -1, 20, 1, False)
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
        policy, _, _ = mcts.search(state, 1, 120, 4, False)
        assert np.argmax(policy) == 4 * 9 + 4
        assert policy[4 * 9 + 4] > 0.5

    def test_deterministic_under_no_gpu_noise(self):
        # With epsilon=0 the search is deterministic for a fixed model.
        game = Gomoku(board_size=9)
        model = ResNet(9, 3, num_blocks=1, num_channels=8)
        model.eval()
        args = {"c_puct": 1.5, "dirichlet_total_concentration": 0.03 * 9 ** 2, "dirichlet_noise_weight": 0.0}
        state = game.get_initial_state()
        p1, _, _ = MCTS(game, args, model, "cpu").search(state, 1, 15, 0, False)
        p2, _, _ = MCTS(game, args, model, "cpu").search(state, 1, 15, 0, False)
        assert np.array_equal(p1, p2)
