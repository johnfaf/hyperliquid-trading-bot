"""
Canonical Sharpe Helpers
========================

★ H25 FIX: Sharpe was being recomputed in 6 places across the codebase
with incompatible semantics — some annualized, some not; some used
sample stdev (ddof=1), some population (ddof=0); some operated on
trade returns, some on daily returns.  Numbers labelled "sharpe_ratio"
in dashboards / scoring / reports were not directly comparable.

This module is the single canonical implementation.  All callers should
go through one of these helpers:

  sharpe_per_trade(returns)
      Per-trade Sharpe.  No annualization.  ddof=1 (sample stdev).
      Used by replay backtester, agent scoring, alpha arena reports
      where the "trade" is the unit of observation.

  sharpe_daily(daily_returns, periods_per_year=365)
      Annualized Sharpe from daily returns.  ddof=1 (sample stdev).
      Used by golden_wallet evaluation, UI report exporter, and any
      place that aggregates per-day P&L.

Conventions:
  * Returns 0.0 on insufficient data (n < 2) or zero stdev.
  * Always uses sample stdev (ddof=1) — matches scipy / numpy default.
  * Filters NaN / Inf inputs silently (caller is responsible for
    validating data quality if it cares about the dropped count).
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def _clean_returns(values: Iterable[float]) -> list[float]:
    out: list[float] = []
    for v in values or []:
        if isinstance(v, (int, float)) and not (math.isnan(v) or math.isinf(v)):
            out.append(float(v))
    return out


def _sample_stdev(values: Sequence[float], mean_value: float) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    var = sum((v - mean_value) ** 2 for v in values) / (n - 1)
    return math.sqrt(max(var, 0.0))


def sharpe_per_trade(returns: Iterable[float]) -> float:
    """Sharpe ratio computed over per-trade returns, NOT annualized.

    For trade-level units of observation.  Returns 0.0 if fewer than 2
    finite values are present or if stdev is zero.
    """
    cleaned = _clean_returns(returns)
    if len(cleaned) < 2:
        return 0.0
    mean_value = sum(cleaned) / len(cleaned)
    sd = _sample_stdev(cleaned, mean_value)
    if sd <= 1e-12:
        return 0.0
    return mean_value / sd


def sharpe_daily(
    daily_returns: Iterable[float],
    periods_per_year: float = 365.0,
    min_samples: int = 5,
) -> float:
    """Annualized Sharpe ratio from a series of daily returns.

    Uses sample stdev (ddof=1) and √periods_per_year scaling.  Returns
    0.0 if fewer than ``min_samples`` finite values are present.
    """
    cleaned = _clean_returns(daily_returns)
    if len(cleaned) < min_samples:
        return 0.0
    mean_value = sum(cleaned) / len(cleaned)
    sd = _sample_stdev(cleaned, mean_value)
    if sd <= 1e-12:
        return 0.0
    return (mean_value / sd) * math.sqrt(max(periods_per_year, 1.0))
