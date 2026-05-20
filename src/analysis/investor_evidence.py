"""Investor evidence, benchmarks, and audit-pack helpers.

This module is intentionally boring: it reads existing bot tables, computes
plain investor-facing metrics, writes immutable artifacts, and avoids importing
the trading loop.  The command-line scripts in ``scripts/`` are thin wrappers
around these pure-ish helpers.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


UTC = timezone.utc
MISSING = "unavailable"


@dataclass(frozen=True)
class Artifact:
    path: str
    format: str
    rows: int
    sha256: str
    note: str = ""


def utc_now() -> datetime:
    # Route through the canonical clock provider so replay swaps the
    # source consistently across the four audited trees.
    from src.core import clock_provider
    return clock_provider.utc_now()


def utc_now_slug() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def iso_utc(dt: datetime | None = None) -> str:
    value = dt or utc_now()
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def parse_window(value: str | int | None, *, default_days: int = 90) -> int:
    if value is None:
        return int(default_days)
    if isinstance(value, int):
        return max(1, value)
    raw = str(value).strip().lower()
    if not raw:
        return int(default_days)
    if raw.endswith("d"):
        return max(1, int(float(raw[:-1])))
    if raw.endswith("w"):
        return max(1, int(float(raw[:-1]) * 7))
    return max(1, int(float(raw)))


def parse_timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, (int, float)):
        raw = float(value)
        if raw > 10_000_000_000:
            raw /= 1000.0
        try:
            return datetime.fromtimestamp(raw, tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    for candidate in (text, text.replace(" ", "T")):
        try:
            parsed = datetime.fromisoformat(candidate)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except ValueError:
            pass
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=UTC)
        except ValueError:
            pass
    return None


def epoch_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.astimezone(UTC).timestamp() * 1000)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def row_to_dict(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, dict):
        return dict(row)
    try:
        return dict(row)
    except (TypeError, ValueError):
        return {}


def fetch_all(conn: Any, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
    rows = conn.execute(sql, tuple(params)).fetchall()
    return [row_to_dict(row) for row in rows]


def fetch_one(conn: Any, sql: str, params: Sequence[Any] = ()) -> dict[str, Any]:
    row = conn.execute(sql, tuple(params)).fetchone()
    return row_to_dict(row)


def table_exists(conn: Any, table: str) -> bool:
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        return row is not None
    except Exception:
        return False


def table_columns(conn: Any, table: str) -> set[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except Exception:
        return set()
    out: set[str] = set()
    for row in rows:
        d = row_to_dict(row)
        name = d.get("name") or d.get("column_name")
        if name:
            out.add(str(name))
    return out


def parse_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return sha256_file(target)


def write_jsonl(path: str | Path, rows: Sequence[dict[str, Any]]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str))
            handle.write("\n")
    return sha256_file(target)


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return sha256_file(target)


def write_table_artifact(
    out_dir: str | Path,
    stem: str,
    rows: Sequence[dict[str, Any]],
    *,
    require_parquet: bool = False,
) -> Artifact:
    """Write rows as Parquet when an engine is installed, otherwise JSONL.

    The runtime lock now supports optional Parquet through pandas/pyarrow when
    present.  For local development machines without a Parquet engine, the
    manifest records the JSONL fallback rather than pretending the artifact is
    Parquet.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    note = ""
    if rows:
        try:
            import pandas as pd  # type: ignore

            path = out / f"{stem}.parquet"
            pd.DataFrame(list(rows)).to_parquet(path, index=False)
            return Artifact(
                path=str(path),
                format="parquet",
                rows=len(rows),
                sha256=sha256_file(path),
            )
        except Exception as exc:
            if require_parquet:
                raise RuntimeError(f"Parquet artifact requested for {stem}, but write failed: {exc}") from exc
            note = f"parquet_unavailable:{type(exc).__name__}"
    elif require_parquet:
        try:
            import pandas as pd  # type: ignore

            path = out / f"{stem}.parquet"
            pd.DataFrame([]).to_parquet(path, index=False)
            return Artifact(path=str(path), format="parquet", rows=0, sha256=sha256_file(path))
        except Exception as exc:
            raise RuntimeError(f"Parquet artifact requested for empty {stem}, but write failed: {exc}") from exc

    path = out / f"{stem}.jsonl"
    digest = write_jsonl(path, rows)
    return Artifact(path=str(path), format="jsonl", rows=len(rows), sha256=digest, note=note)


def _filter_by_dt(
    rows: Iterable[dict[str, Any]],
    field_names: Sequence[str],
    start: datetime | None,
    end: datetime | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        dt = None
        for name in field_names:
            dt = parse_timestamp(row.get(name))
            if dt is not None:
                break
        if start and dt and dt < start:
            continue
        if end and dt and dt > end:
            continue
        out.append(row)
    return out


def load_closed_paper_trades(
    conn: Any,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if not table_exists(conn, "paper_trades"):
        return []
    sql = "SELECT * FROM paper_trades WHERE LOWER(COALESCE(status, '')) = 'closed'"
    sql += " ORDER BY closed_at ASC, id ASC"
    if limit:
        sql += " LIMIT ?"
        rows = fetch_all(conn, sql, (int(limit),))
    else:
        rows = fetch_all(conn, sql)
    return _filter_by_dt(rows, ("closed_at", "opened_at"), start, end)


def load_wallet_fills(
    conn: Any,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[dict[str, Any]]:
    if not table_exists(conn, "wallet_fills"):
        return []
    rows = fetch_all(conn, "SELECT * FROM wallet_fills ORDER BY time_ms ASC, id ASC")
    start_ms = epoch_ms(start) if start else None
    end_ms = epoch_ms(end) if end else None
    out = []
    for row in rows:
        ts = safe_int(row.get("delayed_time_ms") or row.get("time_ms"), 0)
        if start_ms is not None and ts and ts < start_ms:
            continue
        if end_ms is not None and ts and ts > end_ms:
            continue
        out.append(row)
    return out


def load_candles(
    conn: Any,
    coin: str,
    *,
    timeframe: str = "1h",
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[dict[str, Any]]:
    if not table_exists(conn, "candles"):
        return []
    cols = table_columns(conn, "candles")
    required = {"coin", "timeframe", "timestamp_ms", "close"}
    if not required.issubset(cols):
        return []
    params: list[Any] = [coin.upper(), timeframe]
    sql = "SELECT * FROM candles WHERE UPPER(coin)=? AND timeframe=?"
    if start:
        sql += " AND timestamp_ms >= ?"
        params.append(epoch_ms(start))
    if end:
        sql += " AND timestamp_ms <= ?"
        params.append(epoch_ms(end))
    sql += " ORDER BY timestamp_ms ASC"
    return fetch_all(conn, sql, params)


def period_from_trades(trades: Sequence[dict[str, Any]], *, fallback_days: int = 90) -> tuple[datetime, datetime]:
    dates = [
        dt for trade in trades
        for dt in (parse_timestamp(trade.get("opened_at")), parse_timestamp(trade.get("closed_at")))
        if dt is not None
    ]
    if dates:
        return min(dates), max(dates)
    end = utc_now()
    return end - timedelta(days=fallback_days), end


def trade_notional(trade: dict[str, Any]) -> float:
    meta = parse_metadata(trade.get("metadata"))
    for key in ("notional", "entry_notional", "submitted_notional", "live_notional"):
        value = safe_float(meta.get(key), 0.0)
        if value > 0:
            return value
    return abs(safe_float(trade.get("size"), 0.0) * safe_float(trade.get("entry_price"), 0.0))


def _sum_keys(value: Any, names: set[str]) -> float:
    if isinstance(value, dict):
        total = 0.0
        for key, item in value.items():
            lowered = str(key).strip().lower()
            if lowered in names:
                total += safe_float(item, 0.0)
            elif isinstance(item, (dict, list, tuple)):
                total += _sum_keys(item, names)
        return total
    if isinstance(value, (list, tuple)):
        return sum(_sum_keys(item, names) for item in value)
    return 0.0


def trade_fee(trade: dict[str, Any]) -> float:
    meta = parse_metadata(trade.get("metadata"))
    return abs(_sum_keys(
        meta,
        {
            "fee", "fees", "entry_fee", "exit_fee", "total_fee", "total_fees",
            "trading_fee", "live_fee", "estimated_fee",
        },
    ))


def trade_funding(trade: dict[str, Any]) -> float:
    meta = parse_metadata(trade.get("metadata"))
    return _sum_keys(
        meta,
        {"funding", "funding_fee", "funding_payment", "funding_pnl", "funding_drag"},
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))


def _skew(values: Sequence[float]) -> float:
    if len(values) < 3:
        return 0.0
    mean = _mean(values)
    sigma = _std(values)
    if sigma <= 0:
        return 0.0
    return sum((x - mean) ** 3 for x in values) / (len(values) * sigma**3)


def _kurtosis(values: Sequence[float]) -> float:
    if len(values) < 4:
        return 3.0
    mean = _mean(values)
    sigma = _std(values)
    if sigma <= 0:
        return 3.0
    return sum((x - mean) ** 4 for x in values) / (len(values) * sigma**4)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def sharpe(values: Sequence[float]) -> float:
    sigma = _std(values)
    if sigma <= 0:
        return 0.0
    return _mean(values) / sigma


def probabilistic_sharpe_ratio(returns: Sequence[float], benchmark_sr: float = 0.0) -> float:
    """Bailey-style PSR probability that Sharpe exceeds a benchmark Sharpe."""
    rs = [float(x) for x in returns if math.isfinite(float(x))]
    if len(rs) <= 1:
        return 0.0
    sr = sharpe(rs)
    skew = _skew(rs)
    kurt = _kurtosis(rs)
    denominator = math.sqrt(max(1e-12, 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2))
    z_score = (sr - benchmark_sr) * math.sqrt(len(rs) - 1) / denominator
    return _normal_cdf(z_score)


def bootstrap_ci(
    values: Sequence[float],
    metric: Callable[[Sequence[float]], float],
    *,
    iterations: int = 250,
    seed: int = 7,
) -> tuple[float, float]:
    if len(values) < 3:
        point = metric(values)
        return point, point
    rng = random.Random(seed)
    draws: list[float] = []
    vals = list(values)
    for _ in range(iterations):
        sample = [rng.choice(vals) for _ in vals]
        draws.append(metric(sample))
    draws.sort()
    low_idx = max(0, int(0.025 * (len(draws) - 1)))
    high_idx = min(len(draws) - 1, int(0.975 * (len(draws) - 1)))
    return draws[low_idx], draws[high_idx]


def metrics_from_trades(
    trades: Sequence[dict[str, Any]],
    *,
    starting_capital: float = 10_000.0,
    num_trials: int = 1,
) -> dict[str, Any]:
    ordered = sorted(
        list(trades),
        key=lambda row: (
            parse_timestamp(row.get("closed_at")) or parse_timestamp(row.get("opened_at")) or datetime.min.replace(tzinfo=UTC),
            safe_int(row.get("id"), 0),
        ),
    )
    pnls = [safe_float(row.get("pnl"), 0.0) for row in ordered]
    returns = [pnl / starting_capital for pnl in pnls if starting_capital > 0]
    cumulative = 0.0
    peak = float(starting_capital)
    max_dd = 0.0
    equity_curve: list[dict[str, Any]] = []
    for row, pnl in zip(ordered, pnls):
        cumulative += pnl
        equity = float(starting_capital) + cumulative
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
        dt = parse_timestamp(row.get("closed_at")) or parse_timestamp(row.get("opened_at"))
        equity_curve.append({"timestamp": iso_utc(dt) if dt else None, "equity": equity, "pnl": pnl})

    wins = sum(1 for pnl in pnls if pnl > 0)
    losses = sum(1 for pnl in pnls if pnl < 0)
    total_pnl = sum(pnls)
    total_notional = sum(trade_notional(row) for row in ordered)
    fees = sum(trade_fee(row) for row in ordered)
    funding = sum(trade_funding(row) for row in ordered)
    sr = sharpe(returns)
    try:
        from src.learning.promotion_stats import deflated_sharpe

        dsr_result = deflated_sharpe(returns, num_trials=max(1, int(num_trials)))
        dsr = dsr_result.deflated_sharpe
        dsr_p = dsr_result.p_value
    except Exception:
        dsr = 0.0
        dsr_p = 1.0
    psr = probabilistic_sharpe_ratio(returns)
    dsr_ci = bootstrap_ci(
        returns,
        lambda sample: metrics_sharpe_dsr(sample, num_trials=max(1, int(num_trials))),
    )
    psr_ci = bootstrap_ci(returns, probabilistic_sharpe_ratio)
    start, end = period_from_trades(ordered)
    days = max((end - start).total_seconds() / 86_400.0, 1 / 24)
    return {
        "trades": len(ordered),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(ordered) if ordered else 0.0,
        "total_pnl": total_pnl,
        "return_pct": total_pnl / starting_capital if starting_capital else 0.0,
        "avg_pnl": _mean(pnls),
        "max_drawdown": max_dd,
        "max_drawdown_pct": abs(max_dd) / starting_capital if starting_capital else 0.0,
        "sharpe": sr,
        "deflated_sharpe": dsr,
        "deflated_sharpe_p_value": dsr_p,
        "deflated_sharpe_ci": {"low": dsr_ci[0], "high": dsr_ci[1]},
        "probabilistic_sharpe": psr,
        "probabilistic_sharpe_ci": {"low": psr_ci[0], "high": psr_ci[1]},
        "total_notional": total_notional,
        "avg_notional": total_notional / len(ordered) if ordered else 0.0,
        "turnover_per_day": total_notional / days,
        "fees": fees,
        "funding": funding,
        "period_start": iso_utc(start),
        "period_end": iso_utc(end),
        "period_days": days,
        "equity_curve": equity_curve,
    }


def metrics_sharpe_dsr(returns: Sequence[float], *, num_trials: int = 1) -> float:
    try:
        from src.learning.promotion_stats import deflated_sharpe

        return deflated_sharpe(returns, num_trials=max(1, int(num_trials))).deflated_sharpe
    except Exception:
        return 0.0


def _price_return(candles: Sequence[dict[str, Any]], capital: float) -> tuple[float | None, str]:
    if len(candles) < 2:
        return None, "missing candle coverage"
    start_px = safe_float(candles[0].get("close"), 0.0)
    end_px = safe_float(candles[-1].get("close"), 0.0)
    if start_px <= 0 or end_px <= 0:
        return None, "invalid candle prices"
    return capital * ((end_px / start_px) - 1.0), f"{start_px:.4f}->{end_px:.4f}"


def _benchmark_row(
    name: str,
    pnl: float | None,
    capital: float,
    *,
    trades: int = 0,
    notes: str = "",
) -> dict[str, Any]:
    available = pnl is not None
    pnl_value = float(pnl or 0.0)
    return {
        "benchmark": name,
        "available": available,
        "pnl": pnl_value if available else None,
        "return_pct": (pnl_value / capital) if available and capital else None,
        "trades": trades,
        "notes": notes if available else notes or MISSING,
    }


def build_baselines(
    conn: Any,
    *,
    candle_conn: Any | None = None,
    starting_capital: float = 10_000.0,
    window_days: int = 90,
    random_wallets: int = 5,
    seed: int = 42,
    fee_bps: float = 4.5,
    funding_hourly_rate: float = 0.0000125,
) -> dict[str, Any]:
    end = utc_now()
    start = end - timedelta(days=window_days)
    trades = load_closed_paper_trades(conn, start=start, end=end)
    if trades:
        start, end = period_from_trades(trades, fallback_days=window_days)

    bot_metrics = metrics_from_trades(trades, starting_capital=starting_capital)
    rows: list[dict[str, Any]] = [
        _benchmark_row(
            "Bot closed paper trades",
            bot_metrics["total_pnl"],
            starting_capital,
            trades=bot_metrics["trades"],
            notes="actual closed paper_trades rows",
        )
    ]

    price_conn = candle_conn or conn
    for coin in ("BTC", "ETH"):
        candles = load_candles(price_conn, coin, timeframe="1h", start=start, end=end)
        pnl, note = _price_return(candles, starting_capital)
        rows.append(_benchmark_row(f"{coin} buy-and-hold", pnl, starting_capital, notes=note))

    fills = load_wallet_fills(conn, start=start, end=end)
    by_wallet: dict[str, list[dict[str, Any]]] = {}
    for fill in fills:
        wallet = str(fill.get("wallet_address") or "").strip()
        if wallet:
            by_wallet.setdefault(wallet, []).append(fill)
    if by_wallet:
        wallets = sorted(by_wallet)
        rng = random.Random(seed)
        sample = rng.sample(wallets, min(random_wallets, len(wallets)))
        sample_pnls = [
            sum(safe_float(row.get("penalised_pnl"), safe_float(row.get("closed_pnl"), 0.0)) for row in by_wallet[wallet])
            for wallet in sample
        ]
        rows.append(_benchmark_row(
            f"Random copy-wallet baseline ({len(sample)} wallets)",
            _mean(sample_pnls),
            starting_capital,
            trades=sum(len(by_wallet[wallet]) for wallet in sample),
            notes="mean penalised wallet_fills PnL",
        ))
        best_wallet = max(
            wallets,
            key=lambda wallet: sum(
                safe_float(row.get("penalised_pnl"), safe_float(row.get("closed_pnl"), 0.0))
                for row in by_wallet[wallet]
            ),
        )
        best_pnl = sum(
            safe_float(row.get("penalised_pnl"), safe_float(row.get("closed_pnl"), 0.0))
            for row in by_wallet[best_wallet]
        )
        rows.append(_benchmark_row(
            "Top-wallet naive mirror",
            best_pnl,
            starting_capital,
            trades=len(by_wallet[best_wallet]),
            notes=f"best sampled wallet_fills source {best_wallet[:10]}...",
        ))
    else:
        rows.append(_benchmark_row("Random copy-wallet baseline (5 wallets)", None, starting_capital))
        rows.append(_benchmark_row("Top-wallet naive mirror", None, starting_capital))

    avg_notional = safe_float(bot_metrics.get("avg_notional"), 0.0)
    n_trades = safe_int(bot_metrics.get("trades"), 0)
    fee_drag = -1.0 * n_trades * avg_notional * (fee_bps / 10_000.0) * 2.0
    rows.append(_benchmark_row(
        "Fee-only drag",
        fee_drag,
        starting_capital,
        trades=n_trades,
        notes=f"{n_trades} round trips, {fee_bps:.2f} bps per leg",
    ))

    hours = max((end - start).total_seconds() / 3600.0, 0.0)
    funding_drag = -1.0 * avg_notional * funding_hourly_rate * hours
    rows.append(_benchmark_row(
        "Funding drag",
        funding_drag,
        starting_capital,
        notes=f"static delta ${avg_notional:,.2f}, hourly rate {funding_hourly_rate:.8f}",
    ))
    rows.append(_benchmark_row("No-trade benchmark", 0.0, starting_capital, notes="cash account sits idle"))

    return {
        "generated_at": iso_utc(),
        "window_days": window_days,
        "period_start": iso_utc(start),
        "period_end": iso_utc(end),
        "starting_capital": starting_capital,
        "bot_metrics": {k: v for k, v in bot_metrics.items() if k != "equity_curve"},
        "benchmarks": rows,
    }


def _fmt_money(value: Any) -> str:
    if value is None:
        return MISSING
    return f"${safe_float(value):+,.2f}"


def _fmt_pct(value: Any) -> str:
    if value is None:
        return MISSING
    return f"{safe_float(value) * 100:+.2f}%"


def _fmt_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return MISSING
    return f"{safe_float(value):.{digits}f}"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def render_baseline_markdown(report: dict[str, Any]) -> str:
    bot = report.get("bot_metrics", {})
    rows = []
    bot_pnl = safe_float(bot.get("total_pnl"), 0.0)
    for bench in report.get("benchmarks", []):
        pnl = bench.get("pnl")
        delta = None if pnl is None else bot_pnl - safe_float(pnl)
        rows.append([
            bench.get("benchmark"),
            "yes" if bench.get("available") else "no",
            _fmt_money(pnl),
            _fmt_pct(bench.get("return_pct")),
            bench.get("trades", 0),
            _fmt_money(delta),
            bench.get("notes", ""),
        ])
    return "\n\n".join([
        "# Baseline Benchmark Report",
        f"Generated: {report.get('generated_at')}",
        f"Window: {report.get('period_start')} -> {report.get('period_end')}",
        f"Starting capital: ${safe_float(report.get('starting_capital')):,.2f}",
        "## Bot Summary",
        markdown_table(
            ["Trades", "Win Rate", "PnL", "Return", "Max DD", "Sharpe", "DSR", "PSR"],
            [[
                bot.get("trades", 0),
                _fmt_pct(bot.get("win_rate")),
                _fmt_money(bot.get("total_pnl")),
                _fmt_pct(bot.get("return_pct")),
                _fmt_pct(bot.get("max_drawdown_pct")),
                _fmt_float(bot.get("sharpe")),
                _fmt_float(bot.get("deflated_sharpe")),
                _fmt_pct(bot.get("probabilistic_sharpe")),
            ]],
        ),
        "## Benchmarks",
        markdown_table(
            ["Benchmark", "Available", "PnL", "Return", "Trades", "Bot Delta", "Notes"],
            rows,
        ),
    ]) + "\n"


def snapshot_dataset(
    conn: Any,
    out_dir: str | Path,
    *,
    window_days: int = 90,
    require_parquet: bool = False,
) -> dict[str, Any]:
    end = utc_now()
    start = end - timedelta(days=window_days)
    paper_trades = load_closed_paper_trades(conn, start=start, end=end)
    if paper_trades:
        start, end = period_from_trades(paper_trades, fallback_days=window_days)
    tables: dict[str, list[dict[str, Any]]] = {"paper_trades": paper_trades}
    for table in ("audit_trail", "wallet_fills"):
        if not table_exists(conn, table):
            tables[table] = []
            continue
        rows = fetch_all(conn, f"SELECT * FROM {table}")
        if table == "wallet_fills":
            rows = load_wallet_fills(conn, start=start, end=end)
        else:
            rows = _filter_by_dt(rows, ("timestamp", "created_at", "time"), start, end)
        tables[table] = rows

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifacts = {
        name: write_table_artifact(out, name, rows, require_parquet=require_parquet).__dict__
        for name, rows in tables.items()
    }
    dataset_payload = {
        "period_start": iso_utc(start),
        "period_end": iso_utc(end),
        "tables": tables,
    }
    manifest = {
        "generated_at": iso_utc(),
        "window_days": window_days,
        "period_start": iso_utc(start),
        "period_end": iso_utc(end),
        "row_counts": {name: len(rows) for name, rows in tables.items()},
        "artifacts": artifacts,
        "dataset_sha256": sha256_json(dataset_payload),
        "dataset_format_note": (
            "Parquet is used when pandas has a parquet engine available; "
            "otherwise JSONL fallback artifacts are recorded explicitly."
        ),
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    write_json(out / "manifest.json", manifest)
    return manifest


def split_chronological(
    trades: Sequence[dict[str, Any]],
    *,
    train_pct: float = 0.60,
    val_pct: float = 0.20,
) -> dict[str, list[dict[str, Any]]]:
    ordered = sorted(
        list(trades),
        key=lambda row: (
            parse_timestamp(row.get("closed_at")) or parse_timestamp(row.get("opened_at")) or datetime.min.replace(tzinfo=UTC),
            safe_int(row.get("id"), 0),
        ),
    )
    n = len(ordered)
    train_end = int(n * train_pct)
    val_end = train_end + int(n * val_pct)
    if n and train_end == 0:
        train_end = 1
    if n > 2 and val_end <= train_end:
        val_end = train_end + 1
    return {
        "train": ordered[:train_end],
        "validation": ordered[train_end:val_end],
        "test": ordered[val_end:],
    }


def build_walk_forward_report(
    conn: Any,
    *,
    window_days: int = 90,
    starting_capital: float = 10_000.0,
    num_trials: int = 1,
) -> dict[str, Any]:
    end = utc_now()
    start = end - timedelta(days=window_days)
    trades = load_closed_paper_trades(conn, start=start, end=end)
    if trades:
        start, end = period_from_trades(trades, fallback_days=window_days)
    splits = split_chronological(trades)
    windows = {
        name: {k: v for k, v in metrics_from_trades(
            rows,
            starting_capital=starting_capital,
            num_trials=num_trials,
        ).items() if k != "equity_curve"}
        for name, rows in splits.items()
    }
    return {
        "generated_at": iso_utc(),
        "period_start": iso_utc(start),
        "period_end": iso_utc(end),
        "window_days": window_days,
        "starting_capital": starting_capital,
        "split": "chronological 60/20/20",
        "windows": windows,
        "row_counts": {name: len(rows) for name, rows in splits.items()},
        "dataset_sha256": sha256_json({"paper_trades": trades}),
        "note": (
            "This metrics runner uses immutable paper_trades rows. "
            "scripts/run_walkforward.py can also invoke scripts/run_replay.py "
            "per chronological window when cached candles are available."
        ),
    }


def render_walk_forward_markdown(report: dict[str, Any]) -> str:
    rows = []
    for name, metrics in report.get("windows", {}).items():
        rows.append([
            name,
            report.get("row_counts", {}).get(name, 0),
            _fmt_money(metrics.get("total_pnl")),
            _fmt_pct(metrics.get("return_pct")),
            _fmt_pct(metrics.get("max_drawdown_pct")),
            _fmt_float(metrics.get("sharpe")),
            _fmt_float(metrics.get("deflated_sharpe")),
            _fmt_pct(metrics.get("probabilistic_sharpe")),
        ])
    return "\n\n".join([
        "# Walk-Forward Evidence Report",
        f"Generated: {report.get('generated_at')}",
        f"Window: {report.get('period_start')} -> {report.get('period_end')}",
        f"Split: {report.get('split')}",
        f"Dataset SHA256: `{report.get('dataset_sha256')}`",
        markdown_table(
            ["Window", "Trades", "PnL", "Return", "Max DD", "Sharpe", "DSR", "PSR"],
            rows,
        ),
    ]) + "\n"


def sign_hash_with_agent_key(hex_digest: str) -> dict[str, Any]:
    key = os.environ.get("HL_AGENT_PRIVATE_KEY", "").strip()
    if not key:
        try:
            from src.core.secret_manager import load_agent_private_key

            key = load_agent_private_key("none") or ""
        except Exception:
            key = ""
    if not key:
        return {"signed": False, "reason": "HL_AGENT_PRIVATE_KEY missing"}
    if not key.startswith("0x"):
        key = "0x" + key
    try:
        from eth_account import Account
        from eth_account.messages import encode_defunct

        account = Account.from_key(key)
        message_text = f"hyperliquid-trading-bot evidence sha256:{hex_digest}"
        signed = account.sign_message(encode_defunct(text=message_text))
        signature = signed.signature.hex()
        if not signature.startswith("0x"):
            signature = "0x" + signature
        return {
            "signed": True,
            "scheme": "eth_account.personal_sign",
            "address": account.address,
            "message": message_text,
            "signature": signature,
        }
    except Exception as exc:
        return {"signed": False, "reason": f"{type(exc).__name__}: {exc}"}


def build_live_evidence_pack(
    conn: Any,
    out_dir: str | Path,
    *,
    window_days: int = 90,
    starting_capital: float = 10_000.0,
    num_trials: int = 1,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    end = utc_now()
    start = end - timedelta(days=window_days)
    trades = load_closed_paper_trades(conn, start=start, end=end)
    if trades:
        start, end = period_from_trades(trades, fallback_days=window_days)
    metrics = metrics_from_trades(trades, starting_capital=starting_capital, num_trials=num_trials)
    rejected = 0
    if table_exists(conn, "audit_trail"):
        audit_rows = _filter_by_dt(fetch_all(conn, "SELECT * FROM audit_trail"), ("timestamp",), start, end)
        rejected = sum(1 for row in audit_rows if "reject" in str(row.get("action", "")).lower())
    else:
        audit_rows = []
    source_sha = sha256_json({"paper_trades": trades, "audit_trail": audit_rows})
    trade_csv = out / f"trades_{window_days}d.csv"
    trade_csv_sha = write_csv(trade_csv, trades)
    markdown = render_live_evidence_markdown(
        window_days=window_days,
        metrics=metrics,
        source_sha=source_sha,
        trade_csv=trade_csv,
        trade_csv_sha=trade_csv_sha,
        rejected_signals=rejected,
    )
    report_path = out / f"live_evidence_T{window_days}.md"
    report_path.write_text(markdown, encoding="utf-8")
    report_sha = sha256_file(report_path)
    signature = sign_hash_with_agent_key(sha256_json({"source_sha": source_sha, "report_sha": report_sha}))
    sig_path = out / f"live_evidence_T{window_days}.sig.json"
    write_json(sig_path, {
        "generated_at": iso_utc(),
        "source_sha256": source_sha,
        "report_sha256": report_sha,
        "trade_csv_sha256": trade_csv_sha,
        "signature": signature,
    })
    return {
        "generated_at": iso_utc(),
        "window_days": window_days,
        "report_path": str(report_path),
        "report_sha256": report_sha,
        "trade_csv_path": str(trade_csv),
        "trade_csv_sha256": trade_csv_sha,
        "source_sha256": source_sha,
        "signature_path": str(sig_path),
        "signature": signature,
        "metrics": {k: v for k, v in metrics.items() if k != "equity_curve"},
    }


def render_live_evidence_markdown(
    *,
    window_days: int,
    metrics: dict[str, Any],
    source_sha: str,
    trade_csv: Path,
    trade_csv_sha: str,
    rejected_signals: int,
) -> str:
    return "\n\n".join([
        f"# Live Evidence Pack T+{window_days}",
        f"Generated: {iso_utc()}",
        f"Source rows SHA256: `{source_sha}`",
        f"Trade CSV: `{trade_csv}` (`{trade_csv_sha}`)",
        "## Realized Performance",
        markdown_table(
            ["Trades", "PnL", "Win Rate", "Max DD", "Turnover/Day", "Fees", "Funding", "Rejected Signals"],
            [[
                metrics.get("trades", 0),
                _fmt_money(metrics.get("total_pnl")),
                _fmt_pct(metrics.get("win_rate")),
                _fmt_pct(metrics.get("max_drawdown_pct")),
                _fmt_money(metrics.get("turnover_per_day")),
                _fmt_money(metrics.get("fees")),
                _fmt_money(metrics.get("funding")),
                rejected_signals,
            ]],
        ),
        "## Statistical Evidence",
        markdown_table(
            ["Sharpe", "DSR", "DSR 95% CI", "PSR", "PSR 95% CI"],
            [[
                _fmt_float(metrics.get("sharpe")),
                _fmt_float(metrics.get("deflated_sharpe")),
                f"{_fmt_float(metrics.get('deflated_sharpe_ci', {}).get('low'))} to "
                f"{_fmt_float(metrics.get('deflated_sharpe_ci', {}).get('high'))}",
                _fmt_pct(metrics.get("probabilistic_sharpe")),
                f"{_fmt_pct(metrics.get('probabilistic_sharpe_ci', {}).get('low'))} to "
                f"{_fmt_pct(metrics.get('probabilistic_sharpe_ci', {}).get('high'))}",
            ]],
        ),
    ]) + "\n"


def hash_file_if_exists(path: str | Path | None) -> str | None:
    if not path:
        return None
    target = Path(path)
    if not target.exists():
        return None
    return sha256_file(target)


def build_investor_report(
    *,
    baseline_report: dict[str, Any],
    walk_forward_report: dict[str, Any],
    dataset_manifest: dict[str, Any],
    evidence_pack: dict[str, Any],
    config_path: str | Path = "config.py",
) -> str:
    bot = baseline_report.get("bot_metrics", {})
    wf_test = (walk_forward_report.get("windows") or {}).get("test", {})
    manifest_sha = dataset_manifest.get("manifest_sha256")
    dataset_sha = dataset_manifest.get("dataset_sha256")
    signature = evidence_pack.get("signature") or {}
    baseline_md = render_baseline_markdown(baseline_report)
    wf_md = render_walk_forward_markdown(walk_forward_report)
    return "\n\n".join([
        "# Investor Evidence Report",
        f"Generated: {iso_utc()}",
        "## Audit Anchors",
        markdown_table(
            ["Artifact", "SHA256 / Value"],
            [
                ["Dataset", f"`{dataset_sha}`"],
                ["Dataset manifest", f"`{manifest_sha}`"],
                ["Config", f"`{hash_file_if_exists(config_path) or MISSING}`"],
                ["Evidence report", f"`{evidence_pack.get('report_sha256')}`"],
                ["Trade CSV", f"`{evidence_pack.get('trade_csv_sha256')}`"],
                ["Signature", "signed" if signature.get("signed") else signature.get("reason", "unsigned")],
            ],
        ),
        "## Executive Metrics",
        markdown_table(
            ["Metric", "Full Window", "Walk-Forward Test"],
            [
                ["Trades", bot.get("trades", 0), wf_test.get("trades", 0)],
                ["PnL", _fmt_money(bot.get("total_pnl")), _fmt_money(wf_test.get("total_pnl"))],
                ["Max DD", _fmt_pct(bot.get("max_drawdown_pct")), _fmt_pct(wf_test.get("max_drawdown_pct"))],
                ["Sharpe", _fmt_float(bot.get("sharpe")), _fmt_float(wf_test.get("sharpe"))],
                ["DSR", _fmt_float(bot.get("deflated_sharpe")), _fmt_float(wf_test.get("deflated_sharpe"))],
                ["PSR", _fmt_pct(bot.get("probabilistic_sharpe")), _fmt_pct(wf_test.get("probabilistic_sharpe"))],
                ["Turnover/Day", _fmt_money(bot.get("turnover_per_day")), _fmt_money(wf_test.get("turnover_per_day"))],
                ["Fees + Funding", _fmt_money(safe_float(bot.get("fees")) + safe_float(bot.get("funding"))),
                 _fmt_money(safe_float(wf_test.get("fees")) + safe_float(wf_test.get("funding")))],
            ],
        ),
        "## Baseline Comparison",
        baseline_md.split("## Benchmarks", 1)[-1].strip(),
        "## Walk-Forward",
        wf_md.split("Dataset SHA256:", 1)[-1].strip(),
        "## Next Quant Steps",
        "\n".join([
            "1. Require the immutable dataset manifest in every model-promotion review.",
            "2. Gate live sizing on walk-forward test PSR/DSR and per-source drawdown stability.",
            "3. Keep benchmark deltas in CI or a scheduled report so alpha decay is visible quickly.",
            "4. Reconcile live fills into the dataset before scaling capital beyond canary size.",
        ]),
    ]) + "\n"
