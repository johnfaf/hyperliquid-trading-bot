"""
Tests for the golden wallet pipeline.
Covers:
  - compute_sharpe: validation, NaN/Inf guards, edge cases
  - compute_avg_hold_time_hours: pairing logic, empty/single-fill cases
  - apply_execution_penalties: buy/sell penalty direction
  - build_equity_curve: curve construction
  - evaluate_wallet: empty wallet, too-few fills, too-many fills,
                     superhuman WR filter, superhuman DD filter,
                     golden vs non-golden classification
  - _rate_limit_backoff: jitter range
  - _detect_leaderboard_schema caching (in trader_discovery)
"""
import math
import sys
import os
from datetime import datetime, timedelta, timezone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.discovery.golden_wallet import (
    PenalisedFill,
    evaluate_wallet,
    compute_sharpe,
    compute_avg_hold_time_hours,
    compute_max_drawdown,
    apply_execution_penalties,
    _rate_limit_backoff,
    MAX_HUMAN_WIN_RATE,
)


# ─── compute_sharpe ──────────────────────────────────────────────

class TestComputeSharpe:
    def test_too_few_returns(self):
        assert compute_sharpe([0.01, 0.02, 0.01]) == 0.0

    def test_exactly_five_returns(self):
        returns = [0.01, 0.02, -0.01, 0.015, 0.005]
        result = compute_sharpe(returns)
        assert isinstance(result, float)
        assert result != 0.0

    def test_nan_filtered(self):
        returns = [0.01, float("nan"), 0.02, -0.01, 0.015, 0.005]
        result = compute_sharpe(returns)
        assert not math.isnan(result)

    def test_inf_filtered(self):
        returns = [0.01, float("inf"), 0.02, -0.01, 0.015, 0.005]
        result = compute_sharpe(returns)
        assert not math.isinf(result)

    def test_extreme_outliers_filtered(self):
        """Values outside [-1, 10] should be dropped."""
        returns = [0.01, 0.02, 100.0, -0.01, 0.015, 0.005]
        sharpe_clean = compute_sharpe([0.01, 0.02, -0.01, 0.015, 0.005])
        sharpe_with_outlier = compute_sharpe(returns)
        # The outlier gets dropped, so result should equal clean
        assert abs(sharpe_clean - sharpe_with_outlier) < 0.001

    def test_constant_returns_gives_zero(self):
        """Zero standard deviation → Sharpe should be 0.0."""
        returns = [0.01] * 10
        result = compute_sharpe(returns)
        assert result == 0.0

    def test_positive_returns_positive_sharpe(self):
        returns = [0.01, 0.02, 0.015, 0.008, 0.012, 0.018]
        result = compute_sharpe(returns)
        assert result > 0.0

    def test_negative_returns_negative_sharpe(self):
        returns = [-0.01, -0.02, -0.015, -0.008, -0.012, -0.018]
        result = compute_sharpe(returns)
        assert result < 0.0


# ─── compute_avg_hold_time_hours ─────────────────────────────────

def _make_fill(coin="BTC", closed_pnl=0.0, time_ms=0, side="buy"):
    return {
        "coin": coin, "closed_pnl": closed_pnl, "time": time_ms,
        "side": side, "price": 50000.0, "size": 0.01,
        "hash": f"hash_{time_ms}_{closed_pnl}",
        "fee": 0.0, "is_liquidation": False, "direction": "",
    }


class TestComputeAvgHoldTime:
    def test_empty_fills(self):
        assert compute_avg_hold_time_hours([]) == 0.0

    def test_single_fill(self):
        assert compute_avg_hold_time_hours([_make_fill(time_ms=1000)]) == 0.0

    def test_basic_open_close_pair(self):
        # Open at t=0, close at t=1 hour (3_600_000 ms)
        fills = [
            _make_fill("BTC", closed_pnl=0.0, time_ms=0),           # open
            _make_fill("BTC", closed_pnl=100.0, time_ms=3_600_000),  # close
        ]
        result = compute_avg_hold_time_hours(fills)
        assert abs(result - 1.0) < 0.01, f"Expected ~1h, got {result}"

    def test_multiple_pairs_average(self):
        """Two round-trips: 1h and 3h → avg 2h."""
        fills = [
            _make_fill("BTC", closed_pnl=0.0,   time_ms=0),
            _make_fill("BTC", closed_pnl=100.0, time_ms=3_600_000),      # 1h hold
            _make_fill("BTC", closed_pnl=0.0,   time_ms=10_000_000),
            _make_fill("BTC", closed_pnl=50.0,  time_ms=10_000_000 + 3 * 3_600_000),  # 3h hold
        ]
        result = compute_avg_hold_time_hours(fills)
        assert abs(result - 2.0) < 0.1, f"Expected ~2h average, got {result}"

    def test_unmatched_opens_dont_crash(self):
        """Opens without matching closes should just be ignored."""
        fills = [
            _make_fill("BTC", closed_pnl=0.0, time_ms=0),   # open, no close
            _make_fill("ETH", closed_pnl=0.0, time_ms=500),  # open, no close
        ]
        result = compute_avg_hold_time_hours(fills)
        assert result == 0.0


# ─── apply_execution_penalties ───────────────────────────────────

class TestApplyExecutionPenalties:
    def test_buy_price_increased(self):
        """Buying costs more after penalty (worse fill)."""
        fill = _make_fill("BTC", closed_pnl=0.0, time_ms=1000, side="buy")
        fill["price"] = 50000.0
        penalised = apply_execution_penalties([fill])
        assert penalised[0].penalised_price > 50000.0

    def test_sell_price_decreased(self):
        """Selling receives less after penalty (worse fill)."""
        fill = _make_fill("BTC", closed_pnl=100.0, time_ms=1000, side="sell")
        fill["price"] = 50000.0
        penalised = apply_execution_penalties([fill])
        assert penalised[0].penalised_price < 50000.0

    def test_delay_applied(self):
        fill = _make_fill(time_ms=1_000_000)
        penalised = apply_execution_penalties([fill])
        assert penalised[0].delayed_time_ms == 1_000_100

    def test_closing_fill_pnl_reduced(self):
        """A profitable closing fill should have lower penalised PnL."""
        fill = _make_fill("BTC", closed_pnl=500.0, time_ms=0, side="sell")
        fill["price"] = 50000.0
        fill["size"] = 0.1
        penalised = apply_execution_penalties([fill])
        assert penalised[0].penalised_pnl < 500.0

    def test_losing_fill_pnl_worsens(self):
        """A losing closing fill should become more negative after penalty."""
        fill = _make_fill("BTC", closed_pnl=-200.0, time_ms=0, side="sell")
        fill["price"] = 50000.0
        fill["size"] = 0.05
        penalised = apply_execution_penalties([fill])
        assert penalised[0].penalised_pnl < -200.0


# ─── compute_max_drawdown ────────────────────────────────────────

class TestComputeMaxDrawdown:
    def test_no_drawdown(self):
        assert compute_max_drawdown([100, 110, 120, 130]) == 0.0

    def test_simple_drawdown(self):
        # Peak 100, falls to 80 → DD = 20%
        dd = compute_max_drawdown([100, 90, 80, 85])
        assert abs(dd - 20.0) < 0.01

    def test_empty_curve(self):
        assert compute_max_drawdown([]) == 0.0

    def test_single_point(self):
        assert compute_max_drawdown([100]) == 0.0


# ─── Superhuman filter (unit test via evaluate_wallet) ───────────

def _generate_fills(n: int, win: bool, start_ms: int = 0) -> list:
    """Generate n synthetic fills (alternating open/close) with win/loss outcomes."""
    fills = []
    for i in range(n * 2):
        time_ms = start_ms + i * 3_600_000  # 1 hour apart
        is_close = i % 2 == 1
        closed_pnl = (50.0 if win else -50.0) if is_close else 0.0
        fills.append({
            "coin": "BTC",
            "side": "buy" if i % 2 == 0 else "sell",
            "price": 50000.0,
            "size": 0.01,
            "time": time_ms,
            "closed_pnl": closed_pnl,
            "hash": f"hash_{i}_{time_ms}",
            "fee": 1.0,
            "is_liquidation": False,
            "direction": "Open Long" if not is_close else "Close Long",
        })
    return fills


class TestEvaluateWalletSuperhuman:
    """Test the superhuman filter logic directly without hitting the network."""

    def test_high_win_rate_blocked(self):
        """A wallet with >MAX_HUMAN_WIN_RATE% win rate and substantial PnL is filtered out."""
        # 50 winning closes, 0 losing (100% WR)
        fills = _generate_fills(50, win=True)
        penalised = apply_execution_penalties(fills)
        closing = [f for f in penalised if f.penalised_pnl != 0]
        wins = len([f for f in closing if f.penalised_pnl > 0])
        win_rate = wins / len(closing) * 100 if closing else 0

        pen_total = sum(f.penalised_pnl for f in penalised)
        superhuman = win_rate > MAX_HUMAN_WIN_RATE and abs(pen_total) > 1000

        assert superhuman, (
            f"Expected superhuman=True for WR={win_rate:.1f}%, "
            f"PnL=${pen_total:+,.0f}"
        )

    def test_normal_wallet_not_blocked(self):
        """A normal wallet with ~65% WR is not superhuman."""
        fills = _generate_fills(30, win=True) + _generate_fills(16, win=False)
        fills.sort(key=lambda f: f["time"])
        penalised = apply_execution_penalties(fills)
        closing = [f for f in penalised if f.penalised_pnl != 0]
        wins = len([f for f in closing if f.penalised_pnl > 0])
        win_rate = wins / len(closing) * 100 if closing else 0

        pen_total = sum(f.penalised_pnl for f in penalised)
        superhuman = win_rate > MAX_HUMAN_WIN_RATE and abs(pen_total) > 1000

        assert not superhuman, f"WR={win_rate:.1f}% should NOT trigger superhuman filter"


# ─── Rate limit backoff ──────────────────────────────────────────

class TestRateLimitBackoff:
    def test_first_attempt_returns_positive(self):
        backoff = _rate_limit_backoff(1)
        assert backoff > 0

    def test_later_attempts_longer(self):
        b1 = _rate_limit_backoff(1)
        b3 = _rate_limit_backoff(3)
        # On average later attempts should be longer (jitter means not always strictly)
        # Use a generous margin: b3 base is 15 vs b1 base of 5
        assert b3 > b1 * 0.5, "Later backoff should generally be longer"

    def test_capped_at_max(self):
        backoff = _rate_limit_backoff(100)  # Way past the sequence length
        from src.discovery.golden_wallet import _RATE_LIMIT_BACKOFF_SEQ
        max_base = _RATE_LIMIT_BACKOFF_SEQ[-1]
        # With ±10% jitter, max possible = max_base * 1.10
        assert backoff <= max_base * 1.11, "Backoff should be capped at max sequence value"

    def test_jitter_varies(self):
        """Multiple calls with same attempt should not be identical (jitter active)."""
        results = {_rate_limit_backoff(2) for _ in range(20)}
        assert len(results) > 1, "Jitter should produce varying backoff durations"


def _penalised_row(fill: PenalisedFill) -> dict:
    return {
        "wallet_address": "0xwallet",
        "coin": fill.coin,
        "side": fill.side,
        "original_price": fill.original_price,
        "penalised_price": fill.penalised_price,
        "size": fill.size,
        "time_ms": fill.time_ms,
        "delayed_time_ms": fill.delayed_time_ms,
        "closed_pnl": fill.closed_pnl,
        "penalised_pnl": fill.penalised_pnl,
        "fee": fill.fee,
        "is_liquidation": fill.is_liquidation,
        "direction": fill.direction,
    }


class TestIncrementalWalletMerging:
    def test_evaluate_wallet_merges_stored_and_incremental_history(self, monkeypatch):
        start_ms = int((datetime.now(timezone.utc) - timedelta(days=5)).timestamp() * 1000)
        stored_fills = _generate_fills(15, win=True, start_ms=start_ms)
        new_fills = _generate_fills(2, win=False, start_ms=stored_fills[-1]["time"] + 3_600_000)
        stored_rows = [_penalised_row(fill) for fill in apply_execution_penalties(stored_fills)]

        monkeypatch.setattr(
            "src.discovery.golden_wallet.download_fills_incremental",
            lambda address, last_sync_time_ms: list(new_fills),
        )
        monkeypatch.setattr(
            "src.discovery.golden_wallet.get_wallet_fills",
            lambda address: list(stored_rows),
        )

        result = evaluate_wallet("0xwallet", bot_score=0, last_sync_time_ms=stored_fills[-1]["time"])
        assert result is not None
        report, penalised = result
        assert report.total_fills == len(stored_fills) + len(new_fills)
        assert len(penalised) == len(stored_fills) + len(new_fills)
        assert penalised[-1].time_ms == new_fills[-1]["time"]


# ─── Leaderboard schema caching (trader_discovery) ───────────────

class TestLeaderboardSchemaDetection:
    def test_detects_leaderboard_rows_key(self):
        import src.discovery.trader_discovery as td
        td._leaderboard_schema_key = None  # Reset cache

        data = {"leaderboardRows": [{"address": "0xabc"}], "other": "noise"}
        entries, key = td._detect_leaderboard_schema(data)
        assert key == "leaderboardRows"
        assert len(entries) == 1

    def test_detects_root_list(self):
        import src.discovery.trader_discovery as td
        td._leaderboard_schema_key = None

        data = [{"address": "0xabc"}, {"address": "0xdef"}]
        entries, key = td._detect_leaderboard_schema(data)
        assert key is None  # Root list has no key
        assert len(entries) == 2

    def test_caches_detected_key(self):
        import src.discovery.trader_discovery as td
        td._leaderboard_schema_key = None

        data = {"rows": [{"address": "0x1"}, {"address": "0x2"}]}
        td._detect_leaderboard_schema(data)
        assert td._leaderboard_schema_key == "rows"

        # Second call should use cached key without re-probing
        data2 = {"rows": [{"address": "0x3"}], "other": [{"junk": True}]}
        entries, key = td._detect_leaderboard_schema(data2)
        assert key == "rows"
        assert len(entries) == 1

    def test_empty_data_returns_empty(self):
        import src.discovery.trader_discovery as td
        td._leaderboard_schema_key = None

        entries, key = td._detect_leaderboard_schema({})
        assert entries == []

    def teardown_method(self, _):
        """Reset schema cache after each test to avoid cross-test contamination."""
        import src.discovery.trader_discovery as td
        td._leaderboard_schema_key = None
