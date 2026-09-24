import os

import numpy as np
import pytest
import torch

from alphazero import AlphaZero
from alphazero.metrics import MetricsTracker
from alphazero.utils import (
    finish_game_samples,
    random_augment_batch,
    soft_policy_target,
    value_surprise,
)
from envs.tictactoe import TicTacToe


@pytest.fixture
def tiny_args(tmp_path):
    return {
        "num_simulations": 8,
        "c_puct": 1.5,
        "num_blocks": 1,
        "num_channels": 8,
        "dirichlet_total_concentration": 0.03 * 3 ** 2,
        "dirichlet_noise_weight": 0.25,
        "cheap_search_prob": 0.0,
        "cheap_search_visits_floor": 0,
        "full_search_visits_floor": 0,
        "num_iterations": 2,
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


class TestSelfplay:
    def test_generates_terminal_game(self, tiny_args):
        az = AlphaZero(TicTacToe(), tiny_args)
        game_data, winner, game_len = az.selfplay()
        assert game_len > 0
        assert winner in (-1, 0, 1)
        for sample in game_data:
            assert sample["encoded_state"].shape == (3, 3, 3)
            assert sample["policy_target"].shape == (9,)
            assert np.isclose(sample["policy_target"].sum(), 1.0)
            assert sample["value_target"].shape == (3,)
            assert np.isclose(sample["value_target"].sum(), 1.0)

    def test_training_returns_shared_model_to_eval_mode(self, tiny_args):
        args = {**tiny_args, "batch_size": 1, "min_rows": 1}
        az = AlphaZero(TicTacToe(), args)
        az.replay_buffer.add_game(az.selfplay()[0])

        assert az.train_step() is not None
        assert not az.model.training
        before = [p.detach().clone() for p in az.model.parameters()]
        az.selfplay()
        assert not az.model.training
        after = [p.detach() for p in az.model.parameters()]
        assert all(torch.equal(a, b) for a, b in zip(before, after))


class TestPlayoutCapRandomization:
    def test_all_full_records_every_move(self, tiny_args):
        args = {
            **tiny_args,
            "cheap_search_prob": 0.0,
            "reduce_visits": False,
            "policy_surprise_data_weight": 0.0,
            "value_surprise_data_weight": 0.0,
        }
        az = AlphaZero(TicTacToe(), args)
        samples, winner, game_len = az.selfplay()
        assert len(samples) == game_len > 0

    def test_all_cheap_records_nothing(self, tiny_args):
        args = {**tiny_args, "cheap_search_prob": 1.0}
        az = AlphaZero(TicTacToe(), args)
        samples, winner, game_len = az.selfplay()
        assert samples == []
        assert game_len > 0


class TestSurpriseRows:
    def test_cheap_position_can_gain_training_weight(self, monkeypatch):
        game = TicTacToe()
        initial = game.get_initial_state()
        next_state = game.get_next_state(initial, 0, 1)
        wdl = np.array([0.5, 0.0, 0.5])
        memory = [
            {
                "state": initial,
                "to_play": 1,
                "mcts_policy": np.full(9, 1.0 / 9),
                "weight": 1.0,
                "policy_surprise": 1.0,
                "search_wdl": wdl,
                "nn_wdl": wdl,
            },
            {
                "state": next_state,
                "to_play": -1,
                "mcts_policy": np.full(9, 1.0 / 9),
                "weight": 0.0,
                "policy_surprise": 3.0,
                "search_wdl": wdl,
                "nn_wdl": wdl,
            },
        ]
        draws = iter([0.8, 0.2])
        monkeypatch.setattr(np.random, "random", lambda: next(draws))

        rows = finish_game_samples(
            memory, 1, game,
            {"policy_surprise_data_weight": 0.5, "value_surprise_data_weight": 0.0},
        )

        assert np.allclose([sample["weight"] for sample in memory], [0.7, 0.3])
        assert len(rows) == 1
        assert np.array_equal(rows[0]["encoded_state"], game.encode_state(next_state, -1))
        assert "weight" not in rows[0]

    def test_value_surprise_uses_forward_smoothed_outcome(self):
        game = TicTacToe()
        initial = game.get_initial_state()
        next_state = game.get_next_state(initial, 0, 1)
        first_search = np.array([0.6, 0.1, 0.3])
        second_search = np.array([0.7, 0.1, 0.2])
        first_nn = np.array([0.4, 0.1, 0.5])
        second_nn = np.array([0.4, 0.1, 0.5])
        memory = [
            {
                "state": initial,
                "to_play": 1,
                "mcts_policy": np.full(9, 1.0 / 9),
                "weight": 1.0,
                "policy_surprise": 0.0,
                "search_wdl": first_search,
                "nn_wdl": first_nn,
            },
            {
                "state": next_state,
                "to_play": -1,
                "mcts_policy": np.full(9, 1.0 / 9),
                "weight": 1.0,
                "policy_surprise": 0.0,
                "search_wdl": second_search,
                "nn_wdl": second_nn,
            },
        ]

        finish_game_samples(
            memory, 1, game,
            {"policy_surprise_data_weight": 0.0, "value_surprise_data_weight": 0.0},
        )

        factor = 1.0 / (1.0 + 9 * 0.016)
        second_smoothed = np.array([1.0, 0.0, 0.0]) + factor * (
            second_search[::-1] - np.array([1.0, 0.0, 0.0])
        )
        first_smoothed = second_smoothed + factor * (first_search - second_smoothed)
        assert memory[1]["value_surprise"] == pytest.approx(
            value_surprise(second_smoothed, second_nn[::-1])
        )
        assert memory[0]["value_surprise"] == pytest.approx(
            value_surprise(first_smoothed, first_nn)
        )

    def test_weight_above_one_writes_duplicate_rows(self, monkeypatch):
        game = TicTacToe()
        state = game.get_initial_state()
        wdl = np.array([0.5, 0.0, 0.5])
        memory = [
            {
                "state": state,
                "to_play": 1,
                "mcts_policy": np.full(9, 1.0 / 9),
                "weight": 1.0,
                "policy_surprise": surprise,
                "search_wdl": wdl,
                "nn_wdl": wdl,
            }
            for surprise in (1.0, 3.0)
        ]
        draws = iter([0.9, 0.1])
        monkeypatch.setattr(np.random, "random", lambda: next(draws))

        rows = finish_game_samples(
            memory, 1, game,
            {"policy_surprise_data_weight": 1.0, "value_surprise_data_weight": 0.0},
        )

        assert np.allclose([sample["weight"] for sample in memory], [0.5, 1.5])
        assert len(rows) == 2
        assert rows[0] is not rows[1]
        assert np.array_equal(rows[0]["encoded_state"], rows[1]["encoded_state"])


class TestValueTargets:
    def test_value_target_from_player_view(self, tiny_args):
        az = AlphaZero(TicTacToe(), tiny_args)
        game_data, winner, _ = az.selfplay()
        for sample in game_data:
            to_play = 1 if sample["encoded_state"][2].all() else -1
            outcome = winner * to_play
            expected = np.array([outcome > 0, outcome == 0, outcome < 0], dtype=np.float32)
            assert np.array_equal(sample["value_target"], expected)


class TestCheckpoint:
    def test_roundtrip(self, tiny_args):
        az = AlphaZero(TicTacToe(), tiny_args)

        for _ in range(3):
            az.replay_buffer.add_game(az.selfplay()[0])
            az.train_step()
        az.metrics.record_game(1, 1, 5, 0)
        az.metrics.record_losses(
            1.0,
            0.6,
            0.4,
            {
                "policy_player": 0.5,
                "policy_opponent": 0.05,
                "policy_soft": 0.4,
                "policy_opponent_soft": 0.04,
            },
        )
        az.scheduler.record_iteration(games_played=2, rows_produced=12)
        az.scheduler.games_to_order(total_rows_produced=12)

        az.iteration = 7
        az.save_checkpoint()
        path = os.path.join(tiny_args["data_dir"], "checkpoints", "checkpoint.pth")
        assert os.path.exists(path)

        az2 = AlphaZero(TicTacToe(), tiny_args)
        assert az2.load_checkpoint(path)
        for p1, p2 in zip(az.model.parameters(), az2.model.parameters()):
            assert torch.equal(p1.data, p2.data)
        assert az2.iteration == 7
        assert az2.game_count == az.game_count
        assert az2.scheduler.state() == az.scheduler.state()
        assert az2.metrics.game_records == az.metrics.game_records
        assert az2.metrics.losses == az.metrics.losses
        assert az2.replay_buffer.total_samples_added == az.replay_buffer.total_samples_added

    def test_load_default_path(self, tiny_args):
        az = AlphaZero(TicTacToe(), tiny_args)
        az.save_checkpoint()
        assert az.load_checkpoint() is True


class TestMetricsTracker:
    def _tracker_with_games(self, n, winner_pattern=(1, -1, 0), start=1):
        from alphazero.metrics import MetricsTracker

        tracker = MetricsTracker()
        for i in range(n):
            tracker.record_game(
                start + i, winner_pattern[i % len(winner_pattern)], 10 + i, i
            )
        return tracker

    def test_winrate_summary(self):
        assert MetricsTracker.winrate_summary([]) == (0.0, 0.0, 0.0)
        b, d, w = MetricsTracker.winrate_summary([1, 1, -1, 0])
        assert b == 0.5 and w == 0.25 and abs(d - 0.25) < 1e-12

    def test_iteration_means_use_each_game_and_iteration_end(self):
        from alphazero.plots import selfplay_iteration_history

        # Unequal game lengths must not weight the outcome rates. Iteration
        # gaps and an incomplete latest iteration must preserve sample positions.
        records = [(1, 1, 5, 0), (2, 0, 9, 0), (3, -1, 6, 2)]
        history = selfplay_iteration_history(records)
        assert history == [(14, 0.5, 0.5, 0.0, 7.0), (20, 0.0, 0.0, 1.0, 6.0)]
        assert selfplay_iteration_history(records[:2]) == history[:1]
        assert selfplay_iteration_history([]) == []

    def test_state_roundtrip(self):
        tracker = self._tracker_with_games(3)
        tracker.record_losses(1.0, 0.6, 0.4)
        fresh = MetricsTracker()
        fresh.load_state(tracker.state())
        assert fresh.game_records == tracker.game_records
        assert fresh.losses == tracker.losses



class TestReplayBuffer:
    def test_not_ready_below_min(self, tiny_args):
        az = AlphaZero(TicTacToe(), tiny_args)
        az.replay_buffer.add_game(az.selfplay()[0])
        assert not az.replay_buffer.is_ready()
        assert az.train_step() is None

    def test_window_grows_with_total_added(self):
        from alphazero.replay_buffer import ReplayBuffer

        buf = ReplayBuffer(
            min_rows=100, taper_window_exponent=1.0, expand_window_per_row=0.3
        )
        buf.total_samples_added = 100
        assert buf.window_size() == 100
        buf.total_samples_added = 1000
        assert buf.window_size() == 370

    def test_capacity_capped_by_max_rows(self):
        from alphazero.replay_buffer import ReplayBuffer

        buf = ReplayBuffer(
            min_rows=100,
            taper_window_exponent=1.0,
            expand_window_per_row=0.3,
            max_rows=200,
        )
        buf.total_samples_added = 1000
        # 窗口自然增长到 370, 但被 max_rows 封顶在 200
        assert buf.window_size() == 200

    def test_evicts_oldest_beyond_capacity(self):
        from alphazero.replay_buffer import ReplayBuffer

        buf = ReplayBuffer(
            min_rows=4, taper_window_exponent=1.0, expand_window_per_row=0.0
        )
        buf.add_game([{"i": i} for i in range(4)])
        buf.add_game([{"i": 4}])
        assert len(buf) == 4
        assert buf.buffer[0]["i"] == 1
        assert buf.buffer[-1]["i"] == 4

    def test_sample_uniform_from_window(self):
        from alphazero.replay_buffer import ReplayBuffer

        np.random.seed(0)
        buf = ReplayBuffer(
            min_rows=10, taper_window_exponent=1.0, expand_window_per_row=0.0
        )
        buf.add_game([{"i": i} for i in range(10)])
        # 窗口内所有样本(包括老样本)都可被采样到
        seen = set()
        for _ in range(50):
            batch = buf.sample(4)
            seen.update(row["i"] for row in batch)
        assert seen == set(range(10))


class TestLearnLoop:
    def _make_az(self, args):
        return AlphaZero(TicTacToe(), args)

    def test_learn_runs_adaptive(self, tiny_args):
        args = {**tiny_args, "num_iterations": 2, "min_rows": 10, "batch_size": 8}
        az = self._make_az(args)
        az.learn()
        assert az.game_count > 0
        assert len(az.metrics.losses["total"]) >= 1
        assert len(az.replay_buffer) > 0
        assert len(az.metrics.game_records) == az.game_count

    def test_learn_interrupted_saves_final(self, tiny_args, monkeypatch):
        args = dict(tiny_args)
        args.pop("num_iterations")
        az = self._make_az(args)

        def interrupt(total_rows_produced):
            raise KeyboardInterrupt()

        monkeypatch.setattr(az.scheduler, "games_to_order", interrupt)
        with pytest.raises(KeyboardInterrupt):
            az.learn()
        assert os.path.exists(
            os.path.join(args["data_dir"], "checkpoints", "checkpoint.pth")
        )

    def test_learn_saves_model_each_interval(self, tiny_args):
        args = {**tiny_args, "num_iterations": 3, "save_interval": 2}
        az = self._make_az(args)
        az.learn()
        models_dir = os.path.join(args["data_dir"], "models")
        assert os.path.exists(os.path.join(models_dir, "model_0.pth"))
        assert not os.path.exists(os.path.join(models_dir, "model_1.pth"))
        assert os.path.exists(os.path.join(models_dir, "model_2.pth"))

    def test_plots_written_under_data_dir(self, tiny_args):
        args = {**tiny_args, "num_iterations": 2, "min_rows": 10, "batch_size": 8}
        az = self._make_az(args)
        az.learn()
        assert os.path.exists(os.path.join(args["data_dir"], "training.png"))
        assert os.path.exists(os.path.join(args["data_dir"], "losses.csv"))
        assert os.path.exists(os.path.join(args["data_dir"], "games.csv"))
        assert not os.path.exists(os.path.join(args["data_dir"], "logs"))

    def test_learn_resumes_iteration(self, tiny_args):
        args = {**tiny_args, "num_iterations": 2}
        az = self._make_az(args)
        az.learn()
        assert az.iteration == 2

        resumed = self._make_az(args)
        resumed.learn()
        assert resumed.iteration == 2
        assert resumed.game_count == az.game_count


class TestAuxiliaryPolicyTargets:
    def _memory(self, game, policies):
        state = game.get_initial_state()
        memory = []
        to_play = 1
        for policy in policies:
            memory.append(
                {
                    "state": state,
                    "to_play": to_play,
                    "mcts_policy": policy,
                    "weight": 1.0,
                    "policy_surprise": 0.0,
                    "search_wdl": np.array([0.4, 0.2, 0.4]),
                    "nn_wdl": np.array([0.4, 0.2, 0.4]),
                }
            )
            to_play = -to_play
        return memory

    def test_soft_policy_target_sums_and_flattens(self):
        policy = np.array([0.6, 0.3, 0.1, 0.0])
        soft = soft_policy_target(policy, 4.0)
        assert np.isclose(soft.sum(), 1.0)
        assert soft[3] == 0.0
        assert soft[0] < policy[0]
        assert soft[0] / soft[2] < policy[0] / policy[2]

    def test_soft_policy_target_uniform_is_unchanged(self):
        policy = np.full(4, 0.25)
        assert np.allclose(soft_policy_target(policy, 4.0), policy)

    def test_opponent_target_is_next_step_policy(self):
        game = TicTacToe()
        policies = [np.eye(9)[i] for i in range(3)]
        memory = self._memory(game, policies)
        rows = finish_game_samples(
            memory,
            1,
            game,
            {
                "policy_surprise_data_weight": 0.0,
                "value_surprise_data_weight": 0.0,
                "soft_policy_temperature": 4.0,
            },
        )
        assert len(rows) == 3
        for index in range(2):
            assert np.array_equal(rows[index]["opponent_policy"], policies[index + 1])
            assert rows[index]["opponent_weight"] == 1.0
        assert rows[2]["opponent_weight"] == 0.0
        assert np.array_equal(rows[2]["opponent_policy"], np.zeros(9))

    def test_soft_targets_present_and_normalized(self):
        game = TicTacToe()
        policies = [np.eye(9)[0], np.full(9, 1.0 / 9)]
        memory = self._memory(game, policies)
        rows = finish_game_samples(
            memory,
            1,
            game,
            {"policy_surprise_data_weight": 0.0, "value_surprise_data_weight": 0.0},
        )
        for row in rows:
            assert row["policy_target_soft"].shape == (9,)
            assert row["opponent_policy_soft"].shape == (9,)
            assert np.isclose(row["policy_target_soft"].sum(), 1.0)

    def test_augment_keeps_four_planes_consistent(self):
        batch = [
            {
                "encoded_state": np.zeros((3, 3, 3), dtype=np.int8),
                "policy_target": np.eye(9)[0].astype(np.float32),
                "opponent_policy": np.eye(9)[1].astype(np.float32),
                "policy_target_soft": np.full(9, 1.0 / 9, dtype=np.float32),
                "opponent_policy_soft": np.full(9, 1.0 / 9, dtype=np.float32),
                "opponent_weight": 1.0,
                "value_target": np.array([1.0, 0.0, 0.0]),
            }
        ]
        np.random.seed(0)
        augmented = random_augment_batch(batch, 3)[0]
        for key in (
            "policy_target",
            "opponent_policy",
            "policy_target_soft",
            "opponent_policy_soft",
        ):
            assert augmented[key].shape == (9,)
            assert np.isclose(augmented[key].sum(), 1.0)

    def test_train_step_returns_loss_components(self, tiny_args):
        args = {**tiny_args, "batch_size": 1, "min_rows": 1}
        az = AlphaZero(TicTacToe(), args)
        az.replay_buffer.add_game(az.selfplay()[0])
        result = az.train_step()
        for key in (
            "total",
            "policy",
            "value",
            "policy_player",
            "policy_opponent",
            "policy_soft",
            "policy_opponent_soft",
        ):
            assert key in result
        components = (
            result["policy_player"]
            + result["policy_opponent"]
            + result["policy_soft"]
            + result["policy_opponent_soft"]
        )
        assert np.isclose(result["policy"], components)


class TestPolicyHeadOutputs:
    def test_network_returns_four_policy_planes(self):
        from alphazero.network import ResNet

        model = ResNet(board_size=3, num_planes=3, num_blocks=1, num_channels=8)
        policy_logits, value_logits = model(torch.zeros(2, 3, 3, 3))
        assert policy_logits.shape == (2, 4, 9)
        assert value_logits.shape == (2, 3)
