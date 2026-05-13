"""Research helpers for cross-coin candle sweeps and walk-forward tuning.

These functions are intentionally offline-only. They do not alter live bot
state; they convert cached candles into JSON-ready reports that answer:

* Which strategies survive across the 10-coin universe?
* Do RSI / mean-reversion parameters stay stable in walk-forward OOS tests?
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from src.backtest.candle_backtester import (
    CandleBacktestConfig,
    CandleBacktestResult,
    CandleBacktester,
)

DEFAULT_10_COIN_UNIVERSE = (
    "BTC",
    "ETH",
    "SOL",
    "HYPE",
    "XRP",
    "DOGE",
    "BNB",
    "ADA",
    "AVAX",
    "LINK",
)

DEFAULT_12_STRATEGIES = (
    "momentum",
    "ma_crossover",
    "mean_reversion",
    "breakout",
    "rsi",
    "macd",
    "macd_histogram",
    "vwap_reversion",
    "stochastic",
    "adx_trend",
    "supertrend",
    "ema_rsi_combo",
)

DEFAULT_WALK_FORWARD_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "rsi": {
        "rsi_period": [7, 14, 21],
        "rsi_oversold": [20.0, 30.0, 35.0],
        "rsi_overbought": [65.0, 70.0, 80.0],
    },
    "mean_reversion": {
        "bb_period": [14, 20, 30],
        "bb_std": [1.5, 2.0, 2.5],
    },
}


@dataclass
class StrategySweepReport:
    universe: List[str]
    strategies: List[str]
    per_strategy: Dict[str, Dict[str, Any]]
    per_coin: Dict[str, Dict[str, Any]]
    skipped: Dict[str, str]


@dataclass
class WalkForwardReport:
    strategy: str
    coin: str
    folds: List[Dict[str, Any]]
    selected_param_counts: Dict[str, int]
    aggregate: Dict[str, Any]


def _clone_config(base: CandleBacktestConfig | None, **overrides: Any) -> CandleBacktestConfig:
    cfg = CandleBacktestConfig(**((base or CandleBacktestConfig()).to_dict()))
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def _metric_summary(result: CandleBacktestResult) -> Dict[str, Any]:
    return {
        "coin": result.coin,
        "timeframe": result.timeframe,
        "candles": result.candle_count,
        "trades": result.total_trades,
        "win_rate": result.win_rate,
        "total_pnl": result.total_pnl,
        "total_pnl_pct": result.total_pnl_pct,
        "max_drawdown_pct": result.max_drawdown_pct,
        "sharpe": result.sharpe_ratio,
        "profit_factor": result.profit_factor,
    }


def _score_result(result: CandleBacktestResult) -> float:
    """Risk-aware score for ranking parameter candidates."""
    if result.total_trades <= 0:
        return -1_000_000.0
    profit_factor = min(float(result.profit_factor or 0.0), 10.0)
    return (
        float(result.total_pnl)
        + profit_factor * 25.0
        + float(result.sharpe_ratio) * 50.0
        - float(result.max_drawdown_pct) * 20.0
    )


def _param_candidates(grid: Mapping[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    if not keys:
        return [{}]
    return [dict(zip(keys, values)) for values in product(*(grid[k] for k in keys))]


def _timestamp_ms(candle: Any) -> int:
    if isinstance(candle, dict):
        return int(candle["timestamp_ms"])
    return int(getattr(candle, "timestamp_ms"))


def slice_candles(candles: Sequence[Any], start_ms: int, end_ms: int) -> List[Any]:
    return [c for c in candles if start_ms <= _timestamp_ms(c) < end_ms]


def aggregate_strategy_results(results: Mapping[str, CandleBacktestResult]) -> Dict[str, Any]:
    total_trades = sum(r.total_trades for r in results.values())
    total_wins = sum(r.winning_trades for r in results.values())
    total_pnl = sum(r.total_pnl for r in results.values())
    coins_with_trades = sum(1 for r in results.values() if r.total_trades > 0)
    avg_sharpe = (
        sum(r.sharpe_ratio for r in results.values()) / len(results)
        if results else 0.0
    )
    avg_drawdown = (
        sum(r.max_drawdown_pct for r in results.values()) / len(results)
        if results else 0.0
    )
    return {
        "coins_tested": len(results),
        "coins_with_trades": coins_with_trades,
        "total_trades": total_trades,
        "win_rate": round((total_wins / total_trades * 100.0) if total_trades else 0.0, 3),
        "total_pnl": round(total_pnl, 2),
        "avg_sharpe": round(avg_sharpe, 4),
        "avg_max_drawdown_pct": round(avg_drawdown, 3),
        "survives_cross_coin": (
            len(results) > 0
            and coins_with_trades >= max(1, len(results) // 2)
            and total_trades >= len(results)
            and total_pnl > 0
        ),
        "coins": {coin: _metric_summary(result) for coin, result in results.items()},
    }


def run_cross_coin_strategy_sweep(
    candle_sets: Mapping[str, Sequence[Any]],
    *,
    strategies: Iterable[str] = DEFAULT_12_STRATEGIES,
    base_config: CandleBacktestConfig | None = None,
    min_candles: int = 50,
) -> StrategySweepReport:
    per_strategy_raw: Dict[str, Dict[str, CandleBacktestResult]] = {}
    per_coin: Dict[str, Dict[str, Any]] = {}
    skipped: Dict[str, str] = {}

    for coin, candles in candle_sets.items():
        if len(candles) < min_candles:
            skipped[coin] = f"insufficient_candles:{len(candles)}<{min_candles}"
            continue
        per_coin.setdefault(coin, {})
        for strategy in strategies:
            cfg = _clone_config(base_config, strategy=strategy)
            result = CandleBacktester(cfg).run(candles, strategy=strategy, coin=coin)
            per_strategy_raw.setdefault(strategy, {})[coin] = result
            per_coin[coin][strategy] = _metric_summary(result)

    per_strategy = {
        strategy: aggregate_strategy_results(results)
        for strategy, results in per_strategy_raw.items()
    }
    return StrategySweepReport(
        universe=list(candle_sets.keys()),
        strategies=list(strategies),
        per_strategy=per_strategy,
        per_coin=per_coin,
        skipped=skipped,
    )


def run_walk_forward_parameter_optimization(
    candles: Sequence[Any],
    *,
    coin: str,
    strategy: str,
    param_grid: Mapping[str, Sequence[Any]] | None = None,
    base_config: CandleBacktestConfig | None = None,
    train_days: int = 30,
    test_days: int = 30,
    step_days: int = 30,
    min_train_candles: int = 50,
    min_test_candles: int = 20,
) -> WalkForwardReport:
    if not candles:
        return WalkForwardReport(strategy, coin, [], {}, {"fold_count": 0})

    ordered = sorted(candles, key=_timestamp_ms)
    grid = param_grid or DEFAULT_WALK_FORWARD_GRIDS.get(strategy, {})
    candidates = _param_candidates(grid)
    day_ms = 86_400_000
    train_ms = train_days * day_ms
    test_ms = test_days * day_ms
    step_ms = step_days * day_ms
    start_ms = _timestamp_ms(ordered[0])
    final_ms = _timestamp_ms(ordered[-1]) + 1

    folds: List[Dict[str, Any]] = []
    selected_counts: Dict[str, int] = {}
    fold_start = start_ms
    while fold_start + train_ms + test_ms <= final_ms:
        train_start = fold_start
        train_end = fold_start + train_ms
        test_end = train_end + test_ms
        train_candles = slice_candles(ordered, train_start, train_end)
        test_candles = slice_candles(ordered, train_end, test_end)
        if len(train_candles) < min_train_candles or len(test_candles) < min_test_candles:
            fold_start += step_ms
            continue

        ranked: List[tuple[float, Dict[str, Any], CandleBacktestResult]] = []
        for params in candidates:
            cfg = _clone_config(base_config, strategy=strategy, **params)
            result = CandleBacktester(cfg).run(train_candles, strategy=strategy, coin=coin)
            ranked.append((_score_result(result), params, result))
        ranked.sort(key=lambda item: item[0], reverse=True)
        best_score, best_params, train_result = ranked[0]
        test_cfg = _clone_config(base_config, strategy=strategy, **best_params)
        test_result = CandleBacktester(test_cfg).run(test_candles, strategy=strategy, coin=coin)
        params_key = ",".join(f"{k}={v}" for k, v in sorted(best_params.items())) or "default"
        selected_counts[params_key] = selected_counts.get(params_key, 0) + 1
        folds.append({
            "train_start_ms": train_start,
            "train_end_ms": train_end,
            "test_end_ms": test_end,
            "selected_params": best_params,
            "train_score": round(best_score, 4),
            "train": _metric_summary(train_result),
            "test": _metric_summary(test_result),
        })
        fold_start += step_ms

    test_trades = sum(int(f["test"]["trades"]) for f in folds)
    test_pnl = sum(float(f["test"]["total_pnl"]) for f in folds)
    weighted_wins = sum(float(f["test"]["win_rate"]) * int(f["test"]["trades"]) for f in folds)
    aggregate = {
        "fold_count": len(folds),
        "test_trades": test_trades,
        "test_total_pnl": round(test_pnl, 2),
        "test_win_rate": round((weighted_wins / test_trades) if test_trades else 0.0, 3),
        "stable_params": len(selected_counts) <= max(1, len(folds) // 2) if folds else False,
    }
    return WalkForwardReport(strategy, coin, folds, selected_counts, aggregate)


def report_to_dict(report: StrategySweepReport | WalkForwardReport) -> Dict[str, Any]:
    return asdict(report)
