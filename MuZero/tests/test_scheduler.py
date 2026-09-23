import math

import pytest

from alphazero.scheduler import SelfPlayScheduler

# Mirrors tiny_args in test_trainer.py:
#   rows_needed_per_iteration = train_steps(2) * batch_size(16) / replay_ratio(8) = 4
#   fallback_rows_per_game    = board_size(3) ** 2 = 9
BOOTSTRAP_GAMES = 2
MIN_ROWS = 16
ROWS_NEEDED = 4.0
FALLBACK_ROWS_PER_GAME = 9.0


@pytest.fixture
def scheduler():
    return SelfPlayScheduler(
        bootstrap_games=BOOTSTRAP_GAMES,
        min_rows=MIN_ROWS,
        rows_needed_per_iteration=ROWS_NEEDED,
        fallback_rows_per_game=FALLBACK_ROWS_PER_GAME,
    )


class TestColdStart:
    def test_cold_start_orders_bootstrap_games(self, scheduler):
        assert scheduler.games_to_order(total_rows_produced=0) == BOOTSTRAP_GAMES

    def test_not_bootstrapped_until_first_iteration_recorded(self, scheduler):
        assert not scheduler.is_bootstrapped
        scheduler.record_iteration(games_played=2, rows_produced=8)
        assert scheduler.is_bootstrapped


class TestGamesToOrder:
    def test_iteration_1_backfills_to_min_rows_with_factor(self, scheduler):
        # Bootstrap produced 8 rows; target is min_rows = 16, so 8 rows are
        # missing at 4 rows/game -> 2 games, scaled by BACKFILL_FACTOR 1.05.
        scheduler.record_iteration(games_played=2, rows_produced=8)
        expected = math.ceil((8 / 4) * 1.05)
        assert expected == 3
        assert scheduler.games_to_order(total_rows_produced=8) == expected

    def test_steady_state_orders_rows_needed_per_iteration(self, scheduler):
        scheduler.record_iteration(games_played=100, rows_produced=500)  # 5 rows/game
        # First call is the backfill: target 16, produced 16 -> no deficit.
        assert scheduler.games_to_order(total_rows_produced=16) == 0
        # Target 20, produced 16 -> deficit 4 -> ceil(4/5) = 1 game.
        assert scheduler.games_to_order(total_rows_produced=16) == 1
        # Target 24, produced 17 -> deficit 7 -> ceil(7/5) = 2 games.
        assert scheduler.games_to_order(total_rows_produced=17) == 2

    def test_orders_zero_when_production_ahead(self, scheduler):
        scheduler.record_iteration(games_played=100, rows_produced=500)
        assert scheduler.games_to_order(total_rows_produced=10 ** 9) == 0


class TestRowsPerGameEstimate:
    def test_uses_most_recent_iteration(self, scheduler):
        scheduler.record_iteration(games_played=100, rows_produced=500)  # 5.0
        scheduler.record_iteration(games_played=5, rows_produced=10)  # 2.0
        # Target 16, produced 0 -> deficit 16 at 2 rows/game -> 8 games,
        # scaled by the 1.05 backfill factor -> ceil(8.4) = 9.
        assert scheduler.games_to_order(total_rows_produced=0) == 9

    def test_falls_back_when_history_has_no_valid_entry(self, scheduler):
        # E.g. an iteration ordered 0 games and produced 0 rows.
        scheduler.record_iteration(games_played=0, rows_produced=0)
        # Target 16, produced 10 -> deficit 6 at fallback 9 rows/game -> 1 game.
        assert scheduler.games_to_order(total_rows_produced=10) == 1


class TestState:
    def test_state_roundtrip_preserves_behavior(self, scheduler):
        scheduler.record_iteration(games_played=2, rows_produced=8)
        scheduler.games_to_order(total_rows_produced=8)  # advance the ledger

        fresh = SelfPlayScheduler(
            bootstrap_games=BOOTSTRAP_GAMES,
            min_rows=MIN_ROWS,
            rows_needed_per_iteration=ROWS_NEEDED,
            fallback_rows_per_game=FALLBACK_ROWS_PER_GAME,
        )
        fresh.load_state(scheduler.state())

        assert fresh.state() == scheduler.state()
        # Both order identically afterwards: target 20, produced 10 ->
        # deficit 10 at 4 rows/game -> 3 games. A lost state would instead
        # return the bootstrap game count.
        assert (
            fresh.games_to_order(total_rows_produced=10)
            == scheduler.games_to_order(total_rows_produced=10)
            == 3
        )
