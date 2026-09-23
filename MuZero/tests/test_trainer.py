import os

import numpy as np
import pytest
import torch

from alphazero import MuZero
from alphazero.metrics import MetricsTracker
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
        "parallel": False,
        "unroll_steps": 3,
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
        mz = MuZero(TicTacToe(), tiny_args)
        game_data, winner, game_len = mz.selfplay()
        assert len(game_data) > 0
        assert game_len == len(game_data)
        assert winner in (-1, 0, 1)
        for sample in game_data:
            assert sample["observation"].shape == (3, 3, 3)
            assert sample["player"] in (-1, 1)
            assert 0 <= sample["action"] < 9
            assert sample["mcts_policy"].shape == (9,)
            assert np.isclose(sample["mcts_policy"].sum(), 1.0)
            assert sample["value_target"] in (-1.0, 0.0, 1.0)

    def test_training_returns_shared_model_to_eval_mode(self, tiny_args):
        args = {**tiny_args, "batch_size": 1, "min_rows": 1}
        mz = MuZero(TicTacToe(), args)
        mz.replay_buffer.add_game(mz.selfplay()[0])

        result = mz.train_step()
        assert result is not None
        assert all(np.isfinite(norm) and norm > 0 for norm in result["grad_norms"].values())
        assert not mz.model.training
        before = [p.detach().clone() for p in mz.model.parameters()]
        mz.selfplay()
        assert not mz.model.training
        after = [p.detach() for p in mz.model.parameters()]
        assert all(torch.equal(a, b) for a, b in zip(before, after))


class TestValueTargets:
    def test_value_target_from_player_view(self, tiny_args):
        mz = MuZero(TicTacToe(), tiny_args)
        game_data, winner, _ = mz.selfplay()
        for sample in game_data:
            assert sample["value_target"] == float(winner) * sample["player"]


class TestGradientScaling:
    @pytest.mark.parametrize(
        "unroll_steps, expected_h, expected_g, expected_f",
        [(0, 2.0, 0.0, 2.0), (1, 4.0, 2.0, 4.0), (3, 19 / 6, 17 / 6, 4.0)],
    )
    def test_train_step_matches_analytic_gradients(
        self, tiny_args, monkeypatch, unroll_steps, expected_h, expected_g, expected_f
    ):
        # h=w_h, g(s)=w_g*s, v(s)=w_f*s，所有权重初值为 1，value 目标为 0。
        # K=3 时，h 的梯度为 2*[1+(1+1/2+1/4)/3]；
        # g 的共享参数梯度为 2*[1+(1+1/2)+(1+1/2+1/4)]/3。
        class Representation(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(()))

            def forward(self, observation):
                return observation.mean(dim=(1, 2, 3)) * self.weight

        class Dynamics(Representation):
            def forward(self, hidden, action):
                return hidden * self.weight

        class Prediction(Representation):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.zeros(9))

            def forward(self, hidden):
                return self.logits.expand(hidden.shape[0], -1), hidden * self.weight

        mz = MuZero(TicTacToe(), {
            **tiny_args, "batch_size": 1, "min_rows": 1, "unroll_steps": unroll_steps,
        })
        mz.model = torch.nn.ModuleDict({
            "representation": Representation(),
            "dynamics": Dynamics(),
            "prediction": Prediction(),
        }).to(mz.device)
        mz.optimizer = torch.optim.SGD(mz.model.parameters(), lr=0.01)
        game = [{
            "observation": np.ones((3, 3, 3), dtype=np.float32),
            "player": (-1) ** k,
            "action": 0,
            "mcts_policy": np.eye(9, dtype=np.float32)[0],
            "value_target": 0.0,
        } for k in range(unroll_steps + 1)]
        sample = mz.replay_buffer._build_sample(game, 0, unroll_steps, 9)
        monkeypatch.setattr(mz.replay_buffer, "sample", lambda *args: [sample])
        monkeypatch.setattr("alphazero.trainer.random_augment_batch", lambda batch, size: batch)

        result = mz.train_step()

        assert mz.model.representation.weight.grad.item() == pytest.approx(expected_h)
        grad_g = mz.model.dynamics.weight.grad
        assert (0.0 if grad_g is None else grad_g.item()) == pytest.approx(expected_g)
        assert mz.model.prediction.weight.grad.item() == pytest.approx(expected_f)
        policy_gradient = torch.full((9,), 1 / 9, device=mz.device)
        policy_gradient[0] -= 1
        if unroll_steps:
            policy_gradient *= 2
        torch.testing.assert_close(mz.model.prediction.logits.grad, policy_gradient)
        # scale_gradient 只改变反向传播，日志记录各预测步损失之和的 batch 均值。
        assert result["value"] == pytest.approx(unroll_steps + 1)
        assert result["policy"] == pytest.approx((unroll_steps + 1) * np.log(9))


class TestCheckpoint:
    def test_roundtrip(self, tiny_args):
        mz = MuZero(TicTacToe(), tiny_args)

        for _ in range(3):
            mz.replay_buffer.add_game(mz.selfplay()[0])
            mz.train_step()
        mz.metrics.record_game(1, 1, 5, 0)
        mz.metrics.record_losses(
            1.0, 0.6, 0.4, [0.5, 0.4, 0.3, 0.2],
            {"representation": 1.0, "dynamics": 2.0, "prediction": 3.0},
        )
        mz.scheduler.record_iteration(games_played=2, rows_produced=12)
        mz.scheduler.games_to_order(total_rows_produced=12)

        mz.iteration = 7
        mz.save_checkpoint()
        path = os.path.join(tiny_args["data_dir"], "checkpoints", "checkpoint.pth")
        assert os.path.exists(path)

        mz2 = MuZero(TicTacToe(), tiny_args)
        assert mz2.load_checkpoint(path)
        for p1, p2 in zip(mz.model.parameters(), mz2.model.parameters()):
            assert torch.equal(p1.data, p2.data)
        assert mz2.iteration == 7
        assert mz2.game_count == mz.game_count
        assert mz2.scheduler.state() == mz.scheduler.state()
        assert mz2.metrics.game_records == mz.metrics.game_records
        assert mz2.metrics.losses == mz.metrics.losses
        assert mz2.replay_buffer.total_samples_added == mz.replay_buffer.total_samples_added

    def test_load_default_path(self, tiny_args):
        mz = MuZero(TicTacToe(), tiny_args)
        mz.save_checkpoint()
        assert mz.load_checkpoint() is True


class TestMetricsTracker:
    def _tracker_with_games(self, n, winner_pattern=(1, -1, 0), start=1):
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
    @staticmethod
    def _game(marker, length=2):
        return [
            {
                "observation": np.zeros((3, 3, 3), dtype=np.float32),
                "player": 1,
                "action": marker,
                "mcts_policy": np.full(9, 1.0 / 9),
                "value_target": 1.0,
            }
            for _ in range(length)
        ]

    def test_not_ready_below_min(self, tiny_args):
        mz = MuZero(TicTacToe(), tiny_args)
        mz.replay_buffer.add_game(mz.selfplay()[0])
        assert not mz.replay_buffer.is_ready()
        assert mz.train_step() is None

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
        assert buf.window_size() == 200

    def test_evicts_oldest_beyond_capacity(self):
        from alphazero.replay_buffer import ReplayBuffer

        buf = ReplayBuffer(
            min_rows=4, taper_window_exponent=1.0, expand_window_per_row=0.0
        )
        for marker in range(4):
            buf.add_game(self._game(marker, length=1))
        buf.add_game(self._game(4, length=1))
        assert len(buf) == 4
        assert buf.buffer[0][0]["action"] == 1
        assert buf.buffer[-1][0]["action"] == 4

    def test_sample_builds_unrolled_segments(self):
        from alphazero.replay_buffer import ReplayBuffer

        np.random.seed(0)
        buf = ReplayBuffer(
            min_rows=8, taper_window_exponent=1.0, expand_window_per_row=0.0
        )
        for marker in range(8):
            buf.add_game(self._game(marker, length=2))
        batch = buf.sample(4, unroll_steps=3, action_size=9)
        assert len(batch) == 4
        for sample in batch:
            assert sample["observation"].shape == (3, 3, 3)
            assert sample["actions"].shape == (3,)
            assert sample["policy_targets"].shape == (4, 9)
            assert sample["value_targets"].shape == (4,)
            assert sample["policy_mask"].shape == (4,)

    def test_terminal_steps_mask_policy_and_absorb_value(self):
        from alphazero.replay_buffer import ReplayBuffer

        buf = ReplayBuffer(
            min_rows=1, taper_window_exponent=1.0, expand_window_per_row=0.0
        )
        game = [
            {"observation": np.zeros((3, 3, 3)), "player": 1, "action": 0,
             "mcts_policy": np.full(9, 1.0 / 9), "value_target": 1.0},
            {"observation": np.zeros((3, 3, 3)), "player": -1, "action": 1,
             "mcts_policy": np.full(9, 1.0 / 9), "value_target": -1.0},
        ]
        buf.add_game(game)
        sample = buf._build_sample(game, start=0, unroll_steps=3, action_size=9)
        assert list(sample["policy_mask"]) == [1.0, 1.0, 0.0, 0.0]
        # 终局之后 value 目标按交替视角沿用最终胜负
        assert list(sample["value_targets"]) == [1.0, -1.0, 1.0, -1.0]


class TestLearnLoop:
    def _make_mz(self, args):
        return MuZero(TicTacToe(), args)

    def test_learn_runs_adaptive(self, tiny_args):
        args = {**tiny_args, "num_iterations": 2, "min_rows": 10, "batch_size": 8}
        mz = self._make_mz(args)
        mz.learn()
        assert mz.game_count > 0
        assert len(mz.metrics.losses["total"]) >= 1
        assert len(mz.replay_buffer) > 0
        assert len(mz.metrics.game_records) == mz.game_count

    def test_learn_interrupted_saves_final(self, tiny_args, monkeypatch):
        args = dict(tiny_args)
        args.pop("num_iterations")
        mz = self._make_mz(args)

        def interrupt(total_rows_produced):
            raise KeyboardInterrupt()

        monkeypatch.setattr(mz.scheduler, "games_to_order", interrupt)
        with pytest.raises(KeyboardInterrupt):
            mz.learn()
        assert os.path.exists(
            os.path.join(args["data_dir"], "checkpoints", "checkpoint.pth")
        )

    def test_learn_saves_model_each_interval(self, tiny_args):
        args = {**tiny_args, "num_iterations": 3, "save_interval": 2}
        mz = self._make_mz(args)
        mz.learn()
        models_dir = os.path.join(args["data_dir"], "models")
        assert os.path.exists(os.path.join(models_dir, "model_0.pth"))
        assert not os.path.exists(os.path.join(models_dir, "model_1.pth"))
        assert os.path.exists(os.path.join(models_dir, "model_2.pth"))

    def test_plots_written_under_data_dir(self, tiny_args):
        args = {**tiny_args, "num_iterations": 2, "min_rows": 10, "batch_size": 8}
        mz = self._make_mz(args)
        mz.learn()
        assert os.path.exists(os.path.join(args["data_dir"], "training.png"))
        assert os.path.exists(os.path.join(args["data_dir"], "losses.csv"))
        assert os.path.exists(os.path.join(args["data_dir"], "games.csv"))
        assert not os.path.exists(os.path.join(args["data_dir"], "logs"))

    def test_learn_resumes_iteration(self, tiny_args):
        args = {**tiny_args, "num_iterations": 2}
        mz = self._make_mz(args)
        mz.learn()
        assert mz.iteration == 2

        resumed = self._make_mz(args)
        resumed.learn()
        assert resumed.iteration == 2
        assert resumed.game_count == mz.game_count
