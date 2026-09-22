import os

import numpy as np
import pytest
import torch

from alphazero import AlphaZero
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
        game_data = az.selfplay()
        assert len(game_data) > 0
        for sample in game_data:
            assert sample["encoded_state"].shape == (3, 3, 3)
            assert sample["policy_target"].shape == (9,)
            assert np.isclose(sample["policy_target"].sum(), 1.0)
            assert np.isscalar(sample["value_target"])
            assert sample["value_target"] in (-1.0, 0.0, 1.0)

    def test_training_returns_shared_model_to_eval_mode(self, tiny_args):
        args = {**tiny_args, "batch_size": 1, "min_rows": 1}
        az = AlphaZero(TicTacToe(), args)
        az.replay_buffer.add_game(az.selfplay())

        assert az.train_step() is not None
        assert not az.model.training
        before = az.model.start_layer[1].running_mean.detach().clone()
        az.selfplay()
        after = az.model.start_layer[1].running_mean.detach()
        assert torch.equal(after, before)


class TestValueTargets:
    def test_value_target_from_player_view(self, tiny_args):
        az = AlphaZero(TicTacToe(), tiny_args)
        game_data = az.selfplay()
        winner = az.last_game_result[0]
        for sample in game_data:
            to_play = 1 if sample["encoded_state"][2].all() else -1
            assert sample["value_target"] == float(winner) * to_play


class TestCheckpoint:
    def test_roundtrip(self, tiny_args):
        az = AlphaZero(TicTacToe(), tiny_args)

        for _ in range(3):
            az.replay_buffer.add_game(az.selfplay())
            az.train_step()
        az._target_cum = 123.5
        az._rpg_history.append((2, 12))

        az.save_checkpoint("roundtrip.pth")
        path = os.path.join(tiny_args["data_dir"], "checkpoints", "roundtrip.pth")
        assert os.path.exists(path)

        az2 = AlphaZero(TicTacToe(), tiny_args)
        assert az2.load_checkpoint(path)
        for p1, p2 in zip(az.model.parameters(), az2.model.parameters()):
            assert torch.equal(p1.data, p2.data)
        assert az2.game_count == az.game_count
        assert az2._target_cum == az._target_cum
        assert list(az2._rpg_history) == list(az._rpg_history)
        assert az2.replay_buffer.total_samples_added == az.replay_buffer.total_samples_added

    def test_load_latest_when_no_filename(self, tiny_args):
        az = AlphaZero(TicTacToe(), tiny_args)
        az.save_checkpoint("a.pth")
        assert az.load_checkpoint() is True


class TestReplayBuffer:
    def test_not_ready_below_min(self, tiny_args):
        az = AlphaZero(TicTacToe(), tiny_args)
        az.replay_buffer.add_game(az.selfplay())
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


class TestAdaptiveGames:
    def _make_az(self, args):
        return AlphaZero(TicTacToe(), args)

    def test_target_cum_seeds_with_min_rows(self, tiny_args):
        az = self._make_az(tiny_args)
        needed = tiny_args["train_steps"] * tiny_args["batch_size"] / tiny_args["replay_ratio"]
        az._rpg_history.append((2, 8))
        assert az._next_target_cum() == max(tiny_args["min_rows"], needed)
        assert az._next_target_cum() == max(tiny_args["min_rows"], needed) + needed

    def test_rows_per_game_fallback_is_board_area(self, tiny_args):
        az = self._make_az(tiny_args)
        assert az._rows_per_game() == float(TicTacToe().board_size ** 2)

    def test_cold_start_orders_bootstrap_games(self, tiny_args):
        az = self._make_az(tiny_args)
        assert az._games_to_order() == tiny_args["bootstrap_games"]

    def test_rows_per_game_uses_last_iter(self, tiny_args):
        az = self._make_az(tiny_args)
        az._rpg_history.append((100, 500))
        az._rpg_history.append((5, 10))
        assert az._rows_per_game() == 2.0

    def test_rows_per_game_skips_invalid_last(self, tiny_args):
        az = self._make_az(tiny_args)
        az._rpg_history.append((0, 0))
        az._rpg_history.append((100, 500))
        assert az._rows_per_game() == 5.0

    def test_iter1_fills_min_rows(self, tiny_args):
        import math

        az = self._make_az(tiny_args)
        az._games_to_order()
        rpg0 = 4.0
        az._rpg_history.append((2, 8))
        az.game_count = 2
        az.replay_buffer.total_samples_added = 8
        needed = tiny_args["train_steps"] * tiny_args["batch_size"] / tiny_args["replay_ratio"]
        target = max(tiny_args["min_rows"], needed)
        assert az._games_to_order() == math.ceil((target - 8) / rpg0)

    def test_orders_zero_when_production_ahead(self, tiny_args):
        az = self._make_az(tiny_args)
        az._rpg_history.append((100, 500))
        az.replay_buffer.total_samples_added = 10 ** 9
        assert az._games_to_order() == 0

    def test_learn_runs_adaptive(self, tiny_args):
        args = {**tiny_args, "num_iterations": 2, "min_rows": 10, "batch_size": 8}
        az = self._make_az(args)
        az.learn()
        assert az.game_count > 0
        assert len(az.losses["total"]) >= 1
        assert len(az.replay_buffer) > 0

    def test_learn_interrupted_saves_final(self, tiny_args, monkeypatch):
        args = dict(tiny_args)
        args.pop("num_iterations")
        az = self._make_az(args)

        def interrupt():
            raise KeyboardInterrupt()

        monkeypatch.setattr(az, "_games_to_order", interrupt)
        with pytest.raises(KeyboardInterrupt):
            az.learn()
        assert os.path.exists(
            os.path.join(args["data_dir"], "checkpoints", "checkpoint_final.pth")
        )
