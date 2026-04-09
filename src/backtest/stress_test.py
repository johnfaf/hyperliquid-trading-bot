"""
Stress Test Engine
===================
Runs the backtester against multiple extreme-market scenarios and
produces a survival report.

For each scenario:
  1. Load baseline fills (from golden wallets or sample data)
  2. Inject the scenario's mutations into the fill stream
  3. Run the backtester on the mutated fills
  4. Compare results to the unmodified baseline

Outputs a structured report with:
  - Per-scenario survival metrics (max DD, PnL, liquidations, Sharpe)
  - Baseline vs. stressed comparison
  - Kill zones: which scenarios would have blown the account
  - Composite stress score (0-100, higher = more resilient)

Usage:
    python -m src.backtest.stress_test                  # all scenarios
    python -m src.backtest.stress_test --scenario flash_crash
    python -m src.backtest.stress_test --report html     # generate HTML report
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional

# ─── Path setup ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest.backtester import (
    BacktestConfig, BacktestEngine, BacktestResult, BacktestFill,
)
from src.backtest.stress_scenarios import (
    StressScenarioSuite, apply_scenario,
)

logger = logging.getLogger("stress_test")


# ─── Result structures ─────────────────────────────────────────

@dataclass
class ScenarioResult:
    """Result from a single stress scenario."""
    scenario_name: str
    scenario_key: str
    scenario_meta: Dict

    # Backtest metrics under stress
    total_trades: int = 0
    winning_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    calmar_ratio: float = 0.0

    # Stress-specific
    liquidations: int = 0
    stop_losses_hit: int = 0
    time_limit_closes: int = 0
    peak_loss_usd: float = 0.0
    final_balance: float = 0.0
    balance_change_pct: float = 0.0

    # Comparison to baseline
    pnl_delta: float = 0.0           # stressed PnL - baseline PnL
    dd_delta: float = 0.0            # stressed DD - baseline DD
    sharpe_delta: float = 0.0        # stressed Sharpe - baseline Sharpe

    # Survival verdict
    survived: bool = True            # account stayed above 50%
    blown: bool = False              # account went to 0 or below
    severity_score: float = 0.0      # 0-100 (100 = maximum damage)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class StressTestReport:
    """Full stress test report across all scenarios."""
    timestamp: str
    baseline_pnl: float
    baseline_dd: float
    baseline_sharpe: float
    baseline_trades: int

    scenarios: List[ScenarioResult] = field(default_factory=list)

    # Aggregate
    composite_stress_score: float = 0.0     # 0-100 resilience score
    worst_scenario: str = ""
    worst_drawdown: float = 0.0
    worst_pnl: float = 0.0
    scenarios_survived: int = 0
    scenarios_blown: int = 0

    duration_seconds: float = 0.0

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["scenarios"] = [s.to_dict() for s in self.scenarios]
        return d


# ─── Engine ────────────────────────────────────────────────────

class StressTestEngine:
    """
    Orchestrates stress testing across multiple scenarios.
    """

    def __init__(self, backtest_config: Optional[BacktestConfig] = None):
        self.bt_cfg = backtest_config or BacktestConfig(
            initial_balance=10_000,
            max_position_pct=0.08,
            max_positions=5,
            max_per_coin=2,
            max_leverage=5.0,
            max_aggregate_exposure_pct=0.50,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            trailing_stop=True,
            trailing_pct=0.025,
            copy_delay_ms=2000,
            slippage_bps=4.5,
            seed=42,
        )

    def run(self, fills: List[Dict],
            suite: Optional[StressScenarioSuite] = None,
            scenarios: Optional[List[str]] = None) -> StressTestReport:
        """
        Run all enabled scenarios against the fill data.

        Args:
            fills: Raw fill dicts (from DB or seed script)
            suite: Scenario configuration (defaults to full suite)
            scenarios: Optional list of scenario keys to run (None = all enabled)

        Returns:
            StressTestReport with all results
        """
        t0 = time.time()
        suite = suite or StressScenarioSuite()

        coins = list({f["coin"] for f in fills})
        logger.info("Stress test: %d fills, %d coins: %s",
                     len(fills), len(coins), ", ".join(coins))

        # ─── 1. Run baseline (unmodified fills) ───────────────
        logger.info("Running baseline backtest...")
        baseline_bt_fills = self._convert_fills(fills)
        baseline_engine = BacktestEngine(self.bt_cfg)
        baseline = baseline_engine.run(baseline_bt_fills, experiment_id="stress_baseline")

        logger.info("Baseline: %d trades, PnL=$%+.2f, DD=%.1f%%, Sharpe=%.3f",
                     baseline.total_trades, baseline.total_pnl,
                     baseline.max_drawdown_pct, baseline.sharpe_ratio)

        report = StressTestReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            baseline_pnl=baseline.total_pnl,
            baseline_dd=baseline.max_drawdown_pct,
            baseline_sharpe=baseline.sharpe_ratio,
            baseline_trades=baseline.total_trades,
        )

        # ─── 2. Run each scenario ─────────────────────────────
        enabled = suite.enabled_scenarios()
        if scenarios:
            enabled = [(n, k, c) for n, k, c in enabled if k in scenarios]

        for name, key, cfg in enabled:
            logger.info("Running scenario: %s", name)

            try:
                # Inject scenario mutations
                mutated_fills, meta = apply_scenario(fills, key, cfg, coins)
                logger.info("  Injected: %s", json.dumps(meta, default=str)[:200])

                # Run backtester on mutated fills
                bt_fills = self._convert_fills(mutated_fills)
                engine = BacktestEngine(self.bt_cfg)
                result = engine.run(bt_fills,
                                     experiment_id=f"stress_{key}")

                # Build scenario result
                sr = self._build_scenario_result(
                    name, key, meta, result, baseline, engine
                )
                report.scenarios.append(sr)

                verdict = "SURVIVED" if sr.survived else "BLOWN"
                logger.info("  %s: PnL=$%+.2f, DD=%.1f%%, score=%.0f/100 [%s]",
                            name, sr.total_pnl, sr.max_drawdown_pct,
                            sr.severity_score, verdict)

            except Exception as e:
                logger.error("  Scenario %s failed: %s", name, e)
                sr = ScenarioResult(
                    scenario_name=name, scenario_key=key,
                    scenario_meta={"error": str(e)},
                    survived=False, blown=True, severity_score=100,
                )
                report.scenarios.append(sr)

        # ─── 3. Compute aggregate metrics ──────────────────────
        report.duration_seconds = round(time.time() - t0, 2)
        self._compute_aggregates(report)

        return report

    def _convert_fills(self, fills: List[Dict]) -> List[BacktestFill]:
        """Convert raw fill dicts to BacktestFill objects."""
        bt_fills = []
        for f in fills:
            try:
                bf = BacktestFill(
                    wallet_address=f["wallet_address"],
                    coin=f["coin"],
                    side=f["side"],
                    price=f["penalised_price"],
                    original_price=f["original_price"],
                    size=f["size"],
                    time_ms=f["time_ms"],
                    closed_pnl=f.get("penalised_pnl", f.get("closed_pnl", 0)),
                    direction=f.get("direction", ""),
                    is_liquidation=bool(f.get("is_liquidation", 0)),
                )
                bt_fills.append(bf)
            except (KeyError, TypeError):
                continue
        bt_fills.sort(key=lambda f: f.time_ms)
        return bt_fills

    def _build_scenario_result(self, name: str, key: str, meta: Dict,
                                result: BacktestResult,
                                baseline: BacktestResult,
                                engine: BacktestEngine) -> ScenarioResult:
        """Build a ScenarioResult from backtest output."""
        # Count exit reasons
        liquidations = sum(1 for t in result.trades if t.exit_reason == "liquidation")
        stop_losses = sum(1 for t in result.trades if t.exit_reason == "stop_loss")
        time_limits = sum(1 for t in result.trades if t.exit_reason == "time_limit")

        final_balance = self.bt_cfg.initial_balance + result.total_pnl
        balance_change = (result.total_pnl / self.bt_cfg.initial_balance) * 100

        survived = final_balance > self.bt_cfg.initial_balance * 0.50
        blown = final_balance <= 0

        # Severity score: 0 = no impact, 100 = total destruction
        severity = 0.0
        # DD component (0-40 points)
        severity += min(40, result.max_drawdown_pct * 0.8)
        # PnL loss component (0-30 points)
        pnl_loss_pct = max(0, -balance_change)
        severity += min(30, pnl_loss_pct * 0.6)
        # Liquidation component (0-20 points)
        severity += min(20, liquidations * 4)
        # Sharpe degradation (0-10 points)
        sharpe_loss = max(0, baseline.sharpe_ratio - result.sharpe_ratio)
        severity += min(10, sharpe_loss * 2)

        if blown:
            severity = 100

        return ScenarioResult(
            scenario_name=name,
            scenario_key=key,
            scenario_meta=meta,
            total_trades=result.total_trades,
            winning_trades=result.winning_trades,
            win_rate=result.win_rate,
            total_pnl=result.total_pnl,
            max_drawdown_pct=result.max_drawdown_pct,
            sharpe_ratio=result.sharpe_ratio,
            sortino_ratio=result.sortino_ratio,
            profit_factor=result.profit_factor,
            calmar_ratio=result.calmar_ratio,
            liquidations=liquidations,
            stop_losses_hit=stop_losses,
            time_limit_closes=time_limits,
            peak_loss_usd=round(self.bt_cfg.initial_balance * result.max_drawdown_pct / 100, 2),
            final_balance=round(final_balance, 2),
            balance_change_pct=round(balance_change, 2),
            pnl_delta=round(result.total_pnl - baseline.total_pnl, 2),
            dd_delta=round(result.max_drawdown_pct - baseline.max_drawdown_pct, 2),
            sharpe_delta=round(result.sharpe_ratio - baseline.sharpe_ratio, 4),
            survived=survived,
            blown=blown,
            severity_score=round(severity, 1),
        )

    def _compute_aggregates(self, report: StressTestReport):
        """Compute aggregate stress metrics across all scenarios."""
        if not report.scenarios:
            return

        report.scenarios_survived = sum(1 for s in report.scenarios if s.survived)
        report.scenarios_blown = sum(1 for s in report.scenarios if s.blown)

        worst = max(report.scenarios, key=lambda s: s.severity_score)
        report.worst_scenario = worst.scenario_name
        report.worst_drawdown = worst.max_drawdown_pct
        report.worst_pnl = worst.total_pnl

        # Composite resilience score: 100 - avg severity
        avg_severity = sum(s.severity_score for s in report.scenarios) / len(report.scenarios)
        report.composite_stress_score = round(max(0, 100 - avg_severity), 1)


# ─── Report generation ─────────────────────────────────────────

def generate_html_report(report: StressTestReport, output_path: str):
    """Generate an interactive HTML stress test report."""
    scenarios_html = ""
    for s in report.scenarios:
        status_text = "SURVIVED" if s.survived else ("BLOWN" if s.blown else "DAMAGED")
        status_emoji = "&#9989;" if s.survived else "&#10060;"

        bar_width = min(100, s.severity_score)
        bar_color = (
            "#22c55e" if s.severity_score < 30 else
            "#eab308" if s.severity_score < 60 else
            "#ef4444"
        )

        meta_items = ""
        for k, v in s.scenario_meta.items():
            if k == "components":
                continue
            meta_items += f'<span style="background:#f1f5f9;padding:2px 8px;border-radius:4px;margin:2px;font-size:13px">{k}: <b>{v}</b></span> '

        scenarios_html += f"""
        <div style="border:1px solid #e2e8f0;border-radius:12px;padding:20px;margin-bottom:16px;
                     background:{'#fef2f2' if s.blown else '#f8fafc'}">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
                <h3 style="margin:0;font-size:18px">{status_emoji} {s.scenario_name}</h3>
                <span style="font-size:14px;font-weight:bold;color:{'#dc2626' if not s.survived else '#16a34a'}">{status_text}</span>
            </div>

            <div style="background:#e2e8f0;border-radius:6px;height:8px;margin-bottom:16px">
                <div style="background:{bar_color};width:{bar_width}%;height:100%;border-radius:6px"></div>
            </div>
            <div style="text-align:right;font-size:12px;color:#64748b;margin-top:-12px;margin-bottom:12px">
                Severity: {s.severity_score}/100
            </div>

            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px">
                <div style="text-align:center;padding:8px;background:#fff;border-radius:8px;border:1px solid #e2e8f0">
                    <div style="font-size:12px;color:#64748b">PnL</div>
                    <div style="font-size:18px;font-weight:bold;color:{'#dc2626' if s.total_pnl < 0 else '#16a34a'}">${s.total_pnl:+,.0f}</div>
                    <div style="font-size:11px;color:#94a3b8">{s.pnl_delta:+,.0f} vs base</div>
                </div>
                <div style="text-align:center;padding:8px;background:#fff;border-radius:8px;border:1px solid #e2e8f0">
                    <div style="font-size:12px;color:#64748b">Max Drawdown</div>
                    <div style="font-size:18px;font-weight:bold;color:#dc2626">{s.max_drawdown_pct:.1f}%</div>
                    <div style="font-size:11px;color:#94a3b8">{s.dd_delta:+.1f}% vs base</div>
                </div>
                <div style="text-align:center;padding:8px;background:#fff;border-radius:8px;border:1px solid #e2e8f0">
                    <div style="font-size:12px;color:#64748b">Win Rate</div>
                    <div style="font-size:18px;font-weight:bold">{s.win_rate:.1f}%</div>
                    <div style="font-size:11px;color:#94a3b8">{s.total_trades} trades</div>
                </div>
                <div style="text-align:center;padding:8px;background:#fff;border-radius:8px;border:1px solid #e2e8f0">
                    <div style="font-size:12px;color:#64748b">Liquidations</div>
                    <div style="font-size:18px;font-weight:bold;color:{'#dc2626' if s.liquidations > 0 else '#64748b'}">{s.liquidations}</div>
                    <div style="font-size:11px;color:#94a3b8">{s.stop_losses_hit} SL hits</div>
                </div>
            </div>

            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:12px">
                <div style="text-align:center;padding:6px;background:#fff;border-radius:6px;border:1px solid #e2e8f0">
                    <div style="font-size:11px;color:#64748b">Sharpe</div>
                    <div style="font-weight:bold">{s.sharpe_ratio:.3f}</div>
                </div>
                <div style="text-align:center;padding:6px;background:#fff;border-radius:6px;border:1px solid #e2e8f0">
                    <div style="font-size:11px;color:#64748b">Final Balance</div>
                    <div style="font-weight:bold">${s.final_balance:,.0f}</div>
                </div>
                <div style="text-align:center;padding:6px;background:#fff;border-radius:6px;border:1px solid #e2e8f0">
                    <div style="font-size:11px;color:#64748b">Profit Factor</div>
                    <div style="font-weight:bold">{s.profit_factor:.2f}</div>
                </div>
            </div>

            <div style="margin-top:8px">{meta_items}</div>
        </div>
        """

    score = report.composite_stress_score
    score_color = "#22c55e" if score >= 70 else "#eab308" if score >= 40 else "#ef4444"
    score_label = "RESILIENT" if score >= 70 else "MODERATE" if score >= 40 else "FRAGILE"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stress Test Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f8fafc; color: #1e293b; padding: 24px; max-width: 960px; margin: 0 auto; }}
  h1 {{ font-size: 28px; margin-bottom: 4px; }}
  h2 {{ font-size: 20px; margin: 24px 0 12px; color: #334155; }}
  .subtitle {{ color: #64748b; margin-bottom: 24px; font-size: 14px; }}
</style>
</head>
<body>

<h1>Stress Test Report</h1>
<p class="subtitle">Generated {report.timestamp[:19]} &mdash; {len(report.scenarios)} scenarios, {report.duration_seconds:.1f}s runtime</p>

<!-- Composite Score -->
<div style="text-align:center;padding:32px;background:linear-gradient(135deg,#0f172a,#1e293b);
            border-radius:16px;margin-bottom:24px;color:white">
    <div style="font-size:14px;opacity:0.7;margin-bottom:8px">COMPOSITE RESILIENCE SCORE</div>
    <div style="font-size:64px;font-weight:800;color:{score_color}">{score:.0f}</div>
    <div style="font-size:18px;font-weight:600;color:{score_color};margin-top:4px">{score_label}</div>
    <div style="font-size:13px;opacity:0.6;margin-top:8px">
        {report.scenarios_survived}/{len(report.scenarios)} scenarios survived &bull;
        Worst: {report.worst_scenario}
    </div>
</div>

<!-- Baseline -->
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px">
    <div style="text-align:center;padding:16px;background:white;border-radius:12px;border:1px solid #e2e8f0">
        <div style="font-size:12px;color:#64748b">Baseline PnL</div>
        <div style="font-size:22px;font-weight:bold;color:#16a34a">${report.baseline_pnl:+,.0f}</div>
    </div>
    <div style="text-align:center;padding:16px;background:white;border-radius:12px;border:1px solid #e2e8f0">
        <div style="font-size:12px;color:#64748b">Baseline DD</div>
        <div style="font-size:22px;font-weight:bold">{report.baseline_dd:.1f}%</div>
    </div>
    <div style="text-align:center;padding:16px;background:white;border-radius:12px;border:1px solid #e2e8f0">
        <div style="font-size:12px;color:#64748b">Baseline Sharpe</div>
        <div style="font-size:22px;font-weight:bold">{report.baseline_sharpe:.3f}</div>
    </div>
    <div style="text-align:center;padding:16px;background:white;border-radius:12px;border:1px solid #e2e8f0">
        <div style="font-size:12px;color:#64748b">Baseline Trades</div>
        <div style="font-size:22px;font-weight:bold">{report.baseline_trades}</div>
    </div>
</div>

<h2>Scenario Results</h2>
{scenarios_html}

<div style="text-align:center;padding:16px;color:#94a3b8;font-size:12px;margin-top:32px">
    Hyperliquid Trading Bot &mdash; Stress Test Platform
</div>

</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    logger.info("HTML report written to %s", output_path)


# ─── CLI ───────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Run stress tests against backtester with extreme-market scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scenarios:
  flash_crash           25% vertical drop in 12 min with panic slippage
  funding_squeeze       1.5%/8h funding rate for 72h (547% annualized)
  liquidity_drain       150 bps slippage + 60% partial fills for 48h
  cascade_liquidation   4-wave liquidation waterfall with 1.4x amplification
  black_swan            All of the above simultaneously (amplified)

Examples:
  python -m src.backtest.stress_test
  python -m src.backtest.stress_test --scenario flash_crash
  python -m src.backtest.stress_test --scenario flash_crash --scenario funding_squeeze
  python -m src.backtest.stress_test --report html
  python -m src.backtest.stress_test --seed   # use sample data (no DB needed)
        """,
    )
    parser.add_argument("--scenario", action="append", dest="scenarios",
                        help="Run specific scenario(s) (can repeat)")
    parser.add_argument("--report", choices=["json", "html", "both"], default="both",
                        help="Report format (default: both)")
    parser.add_argument("--seed", action="store_true",
                        help="Use sample data from fixtures/ instead of DB")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for reports")
    args = parser.parse_args()

    print()
    print("=" * 65)
    print("  Hyperliquid Trading Bot — Stress Test Platform")
    print("=" * 65)

    # Load fills
    if args.seed:
        logger.info("Loading sample data and generating fills...")
        # Import the seed script's fill generator
        sys.path.insert(0, str(ROOT / "scripts"))
        from seed_and_replay import seed_database
        seed_database()
        fills_raw = _load_fills_from_db()
    else:
        fills_raw = _load_fills_from_db()

    if not fills_raw:
        print("\nNo fills found. Either:")
        print("  1. Run: python scripts/seed_and_replay.py --seed-only")
        print("  2. Run: python -m src.backtest.stress_test --seed")
        print("  3. Run the bot to discover golden wallets first")
        return

    n_wallets = len({f["wallet_address"] for f in fills_raw})
    n_coins = len({f["coin"] for f in fills_raw})
    logger.info("Loaded %d fills from %d wallets across %d coins",
                 len(fills_raw), n_wallets, n_coins)

    # Run stress tests
    engine = StressTestEngine()
    report = engine.run(fills_raw, scenarios=args.scenarios)

    # Print summary
    _print_summary(report)

    # Save reports
    output_dir = args.output_dir or str(ROOT / "reports")
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if args.report in ("json", "both"):
        json_path = os.path.join(output_dir, f"stress_test_{ts}.json")
        with open(json_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        logger.info("JSON report: %s", json_path)

    if args.report in ("html", "both"):
        html_path = os.path.join(output_dir, f"stress_test_{ts}.html")
        generate_html_report(report, html_path)
        print(f"\nHTML report: {html_path}")

    print()


def _load_fills_from_db() -> List[Dict]:
    """Load raw fill dicts from DB."""
    from src.data import database as db
    try:
        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT wf.wallet_address, wf.coin, wf.side, wf.original_price,
                       wf.penalised_price, wf.size, wf.time_ms, wf.delayed_time_ms,
                       wf.closed_pnl, wf.penalised_pnl, wf.fee,
                       wf.is_liquidation, wf.direction
                FROM wallet_fills wf
                INNER JOIN golden_wallets gw ON wf.wallet_address = gw.address
                WHERE gw.is_golden = 1
                ORDER BY wf.time_ms ASC
            """)
            rows = cur.fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("Could not load fills from DB: %s", e)
        return []


def _print_summary(report: StressTestReport):
    """Print console summary."""
    score = report.composite_stress_score
    grade = "RESILIENT" if score >= 70 else "MODERATE" if score >= 40 else "FRAGILE"

    print(f"\n{'='*65}")
    print(f"  STRESS TEST RESULTS — Score: {score:.0f}/100 ({grade})")
    print(f"{'='*65}")
    print(f"  Baseline: PnL=${report.baseline_pnl:+,.2f}  "
          f"DD={report.baseline_dd:.1f}%  Sharpe={report.baseline_sharpe:.3f}")
    print(f"  Survived: {report.scenarios_survived}/{len(report.scenarios)} scenarios")

    print(f"\n  {'Scenario':<26} {'PnL':>10} {'MaxDD':>8} {'Liq':>5} "
          f"{'Score':>7} {'Verdict':>10}")
    print(f"  {'─'*70}")

    for s in report.scenarios:
        verdict = "SURVIVED" if s.survived else ("BLOWN" if s.blown else "DAMAGED")
        print(f"  {s.scenario_name:<26} ${s.total_pnl:>+8,.0f} "
              f"{s.max_drawdown_pct:>7.1f}% {s.liquidations:>5} "
              f"{s.severity_score:>6.0f}/100 {verdict:>10}")

    print(f"\n  Worst scenario: {report.worst_scenario}")
    print(f"  Runtime: {report.duration_seconds:.1f}s")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
