import numpy as np
import pytest
import torch

from alphazero import AlphaZero, ParallelSelfPlayer, ResNet
from alphazero.mcts import MCTS, Node
from envs.gomoku import Gomoku
from envs.tictactoe import TicTacToe


def make_args(**overrides):
    args = {
        "num_simulations": 8,
        "c_puct": 1.5,
        "num_blocks": 1,
        "num_channels": 8,
        "dirichlet_total_concentration": 0.03 * 3 ** 2,
        "cheap_search_prob": 0.0,
        "num_parallel_games": 4,
    }
    args.update(overrides)
    return args


def make_player(game=None, args=None):
    game = game or TicTacToe()
    model = ResNet(game.board_size, game.num_planes, num_blocks=1, num_channels=8).eval()
    return game, ParallelSelfPlayer(game, args or make_args(), model, "cpu")


def assert_valid_game(samples, winner, game_len, game):
    assert game_len == len(samples) > 0
    assert winner in (-1, 0, 1)
    for sample in samples:
        assert sample["encoded_state"].shape == (game.num_planes, game.board_size, game.board_size)
        assert sample["policy_target"].shape == (game.board_size ** 2,)
        assert np.isclose(sample["policy_target"].sum(), 1.0)
        assert sample["value_target"] in (-1.0, 0.0, 1.0)
        to_play = 1 if sample["encoded_state"][2].all() else -1
        assert sample["value_target"] == float(winner) * to_play


class TestBatchedInference:
    def test_matches_sequential_inference(self):
        game = Gomoku(board_size=9)
        args = make_args(dirichlet_total_concentration=0.03 * 9 ** 2)
        model = ResNet(9, 3, num_blocks=1, num_channels=8).eval()
        mcts = MCTS(game, args, model, "cpu")
        player = ParallelSelfPlayer(game, args, model, "cpu")

        # 摆几个不同的局面
        states = [game.get_initial_state()]
        state = game.get_initial_state()
        for action, to_play in [(40, 1), (30, -1), (41, 1), (31, -1)]:
            state = game.get_next_state(state, action, to_play)
            states.append(state)

        nodes = [Node(s, 1) for s in states]
        batch = player._batch_inference(nodes)

        assert len(batch) == len(nodes)
        for node, (policy, value) in zip(nodes, batch):
            seq_policy, seq_value = mcts.nn_inference(node.state, node.to_play)
            assert np.allclose(policy, seq_policy, atol=1e-5)
            assert abs(value - seq_value) < 1e-5


class TestGames:
    def test_collects_requested_number_of_valid_games(self):
        game, player = make_player()
        games = list(player.run(6))
        assert len(games) == 6
        for samples, winner, game_len in games:
            assert_valid_game(samples, winner, game_len, game)

    def test_pool_shrinks_when_fewer_games_than_workers(self):
        game, player = make_player(args=make_args(num_parallel_games=8))
        games = list(player.run(2))
        assert len(games) == 2

    def test_batches_across_games(self):
        game, player = make_player(args=make_args(num_parallel_games=4))
        sizes = []
        original = player._batch_inference

        def recording(nodes):
            sizes.append(len(nodes))
            return original(nodes)

        player._batch_inference = recording
        list(player.run(8))
        assert max(sizes) > 1

    def test_simulation_budget_per_move(self):
        # 回归测试：每步棋的 NN 评估次数不能超过 1 次根评估 + num_simulations。
        # 早期 bug 下 simulations 不增长，树会无限膨胀（评估次数爆炸）。
        sims = 6
        game, player = make_player(args=make_args(num_simulations=sims, num_parallel_games=3))
        evals = [0]
        original = player._batch_inference

        def counting(nodes):
            evals[0] += len(nodes)
            return original(nodes)

        player._batch_inference = counting
        games = list(player.run(5))
        total_moves = sum(game_len for _, _, game_len in games)
        assert total_moves > 0
        assert evals[0] <= (sims + 1) * total_moves
        assert evals[0] >= total_moves

    def test_deterministic_with_fixed_seed(self):
        game, player = make_player()
        np.random.seed(123)
        first = list(player.run(4))
        np.random.seed(123)
        second = list(player.run(4))

        assert len(first) == len(second)
        for (s1, w1, l1), (s2, w2, l2) in zip(first, second):
            assert (w1, l1) == (w2, l2)
            for a, b in zip(s1, s2):
                assert np.array_equal(a["encoded_state"], b["encoded_state"])
                assert np.array_equal(a["policy_target"], b["policy_target"])
                assert a["value_target"] == b["value_target"]


class TestEquivalence:
    def test_single_game_matches_sequential_backend(self):
        # num_parallel_games=1 时，并行后端每轮 batch=1，推理结果与串行版
        # 逐位一致，且 RNG 消耗顺序也相同，因此应当产生完全相同的对局。
        game = TicTacToe()
        base_args = {
            "num_simulations": 12,
            "c_puct": 1.5,
            "num_blocks": 1,
            "num_channels": 8,
            "dirichlet_total_concentration": 0.03 * 3 ** 2,
        }
        seq = AlphaZero(game, {**base_args, "parallel": False})
        par = AlphaZero(game, {**base_args, "parallel": True, "num_parallel_games": 1})
        par.model.load_state_dict(seq.model.state_dict())

        np.random.seed(42)
        seq_samples, seq_winner, seq_len = seq.selfplay()
        np.random.seed(42)
        par_samples, par_winner, par_len = next(par.parallel_player.run(1))

        assert (seq_winner, seq_len) == (par_winner, par_len)
        for a, b in zip(seq_samples, par_samples):
            assert np.array_equal(a["encoded_state"], b["encoded_state"])
            assert np.array_equal(a["policy_target"], b["policy_target"])
            assert a["value_target"] == b["value_target"]


class TestTrainerIntegration:
    @pytest.fixture
    def tiny_args(self, tmp_path):
        return {
            "num_simulations": 8,
            "c_puct": 1.5,
            "num_blocks": 1,
            "num_channels": 8,
            "dirichlet_total_concentration": 0.03 * 3 ** 2,
            "cheap_search_prob": 0.0,
            "parallel": True,
            "num_parallel_games": 3,
            "num_iterations": 1,
            "train_steps": 2,
            "batch_size": 16,
            "replay_ratio": 8,
            "bootstrap_games": 2,
            "min_rows": 16,
            "taper_window_exponent": 1.0,
            "expand_window_per_row": 0.3,
            "save_interval": 1,
            "data_dir": str(tmp_path),
        }

    def test_run_iteration_collects_games(self, tiny_args):
        az = AlphaZero(TicTacToe(), tiny_args)
        assert az.parallel_player is not None
        az._run_iteration(0)
        assert az.game_count > 0
        assert len(az.replay_buffer) > 0

    def test_learn_end_to_end(self, tiny_args):
        args = {**tiny_args, "num_iterations": 2, "min_rows": 8}
        az = AlphaZero(TicTacToe(), args)
        az.learn()
        assert az.iteration == 2
        assert az.game_count > 0
        assert len(az.metrics.game_records) == az.game_count

    def test_sequential_path_unused_when_parallel(self, tiny_args, monkeypatch):
        az = AlphaZero(TicTacToe(), tiny_args)

        def fail(*args, **kwargs):
            raise AssertionError("serial selfplay should not be called when parallel=True")

        monkeypatch.setattr(az, "selfplay", fail)
        az._run_iteration(0)
        assert az.game_count > 0
