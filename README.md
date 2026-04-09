# Hyperliquid Auto-Research Trading Bot

An autonomous research bot that discovers profitable traders on Hyperliquid, classifies their strategies, scores them over time, and paper-trades the best ones. Ships with a reproducible sample dataset so you can run a full backtest from a fresh checkout.

## Quick Start

```bash
# Clone and install
git clone <repo-url> && cd "hyperliquid trading bot"
pip install -r requirements.txt

# Seed sample data and run a backtest (no API keys needed)
python scripts/seed_and_replay.py

# Run the live bot loop (requires websocket-client, eth_account)
python main.py
```

The seed-and-replay script populates the database with 3 golden wallets, 5 strategies, and ~600 deterministic fills, then runs the backtester against them. Everything is seeded with `seed=42` so results are identical across machines.

## How It Works

The bot runs a 3-tier scheduling loop:

**Tier 1 — Fast (60s):** Position management, stop-loss/take-profit checks, copy-trade scanning.

**Tier 2 — Trading (5 min):** Market regime detection, strategy scoring, signal processing through a 6-layer pipeline (scorer, regime filter, signal processor, decision engine, firewall, paper trader).

**Tier 3 — Discovery (24h):** Leaderboard scanning of top 2000 traders, bot detection, strategy identification, golden wallet evaluation.

The signal pipeline is deliberately aggressive: of the ~900 strategies generated per cycle, only 1-3 make it through to execution. Each signal passes through schema validation, confidence checks, exposure limits, regime alignment, conflict detection, cooldown enforcement, funding rate risk, and predictive de-risking before it can trade.

## Project Structure

```
.
├── main.py                          # Orchestrator & CLI entry point
├── config.py                        # All configuration (env-overridable)
├── cli.py                           # CLI command definitions
├── requirements.txt                 # Production dependencies
├── Dockerfile                       # Container image (Railway/Docker)
├── Procfile                         # Process definition
├── fixtures/
│   └── sample_data.json             # Reproducible sample dataset
├── scripts/
│   ├── seed_and_replay.py           # Seed DB + run backtest (one command)
│   ├── diagnose_rejections.py       # Debug why signals are rejected
│   └── run_crash_monte_carlo.py     # Stress testing
├── src/
│   ├── core/
│   │   ├── boot.py                  # Logging, DB init, dependency check
│   │   ├── api_manager.py           # Rate limiter (token bucket + circuit breaker)
│   │   ├── subsystem_registry.py    # Build and wire subsystems
│   │   ├── health_registry.py       # Per-subsystem health tracking
│   │   ├── health_reporter.py       # Health endpoint
│   │   ├── dependency_validator.py  # Boot-time package check
│   │   ├── task_runner.py           # Background thread manager
│   │   └── cycles/
│   │       ├── fast_cycle.py        # Tier 1: SL/TP, copy scanning
│   │       ├── trading_cycle.py     # Tier 2: regime, scoring, trading
│   │       ├── research_cycle.py    # Tier 3: leaderboard, bot detection
│   │       └── reporting_cycle.py   # Daily report generation
│   ├── data/
│   │   ├── database.py              # SQLite schema, persistence, backup/restore
│   │   ├── hyperliquid_client.py    # Hyperliquid API wrapper (V3)
│   │   ├── cryptocom_client.py      # Crypto.com adapter
│   │   ├── options_flow.py          # Deribit/Binance options flow
│   │   ├── polymarket_scanner.py    # Polymarket prediction markets
│   │   └── exchange_aggregator.py   # Multi-exchange data
│   ├── discovery/
│   │   ├── trader_discovery.py      # Leaderboard scan, top trader tracking
│   │   ├── adaptive_bot_detector.py # Signal-based bot detection
│   │   ├── golden_wallet.py         # 90d fill download + penalised equity
│   │   └── golden_bridge.py         # Golden wallets → live copy trading
│   ├── analysis/
│   │   ├── strategy_identifier.py   # Classify trader strategies
│   │   ├── strategy_scorer.py       # 5-dimension scoring + time decay
│   │   ├── regime_detector.py       # Market regime (trending/ranging/volatile)
│   │   ├── regime_strategy_filter.py # Filter strategies by regime
│   │   ├── features.py              # Feature engineering
│   │   ├── sharpe_calculator.py     # Sharpe ratio computation
│   │   ├── liquidation_strategy.py  # Liquidation cascade detection
│   │   └── shadow_tracker.py        # Top trader shadowing
│   ├── signals/
│   │   ├── signal_schema.py         # TradeSignal dataclass (universal contract)
│   │   ├── signal_processor.py      # Cull → dedup → conflict → compress
│   │   ├── decision_engine.py       # Composite ranking + execution priority
│   │   ├── decision_firewall.py     # 11-point validation gate
│   │   ├── kelly_sizing.py          # Kelly criterion position sizing
│   │   ├── agent_scoring.py         # Source reliability tracking
│   │   ├── calibration.py           # Prediction calibration
│   │   ├── predictive_regime_forecaster.py  # 5-input composite forecaster
│   │   ├── xgboost_regime_forecaster.py     # ML regime prediction
│   │   ├── llm_filter.py            # LLM-based signal filtering
│   │   └── alpha_arena.py           # Strategy tournament
│   ├── trading/
│   │   ├── paper_trader.py          # Paper trading (firewall + slippage)
│   │   ├── copy_trader.py           # Copy trading engine
│   │   ├── live_trader.py           # Live execution (agent-wallet signing)
│   │   ├── cross_venue_hedger.py    # Multi-exchange hedging
│   │   └── trade_memory.py          # Trade outcome memory
│   ├── exchanges/
│   │   ├── base_adapter.py          # Exchange interface
│   │   ├── hyperliquid_adapter.py   # Hyperliquid adapter
│   │   ├── lighter_adapter.py       # Lighter exchange adapter
│   │   ├── scanner.py               # Multi-exchange scanner
│   │   ├── whale_scanner.py         # Whale activity detection
│   │   └── cross_venue.py           # Cross-venue arbitrage
│   ├── backtest/
│   │   ├── backtester.py            # Event-driven backtester
│   │   ├── backtest_engine.py       # Core backtest engine
│   │   ├── candle_backtester.py     # Candle-based replay
│   │   ├── data_fetcher.py          # Historical data fetcher
│   │   └── monte_carlo.py           # Monte Carlo stress testing
│   ├── notifications/
│   │   ├── telegram_alerts.py       # Telegram alerts
│   │   ├── telegram_bot.py          # Telegram command handler
│   │   └── ws_position_monitor.py   # WebSocket position monitor
│   └── ui/
│       ├── dashboard.py             # Web dashboard
│       ├── backtest_dashboard.py    # Backtest results UI
│       ├── options_dashboard.py     # Options flow UI
│       ├── reporter.py              # Report generation
│       └── report_exporter.py       # Report export
├── tests/                           # pytest test suite
├── data/                            # SQLite database (gitignored)
├── logs/                            # Daily logs (gitignored)
├── reports/                         # Generated reports (gitignored)
└── models/                          # ML models (XGBoost regime)
```

## CLI Reference

```bash
python main.py                    # Run the full 3-tier bot loop
python main.py --once             # Single discovery + trading cycle, then exit
python main.py --core-only        # Minimal profile (no optional scanners)
python main.py --report           # Generate a report and exit
python main.py --status           # Print current status
python main.py --bootstrap        # Cold-start DB seeding from Hyperliquid API
python main.py --reset-paper      # Wipe paper trades, reset to $10k balance

# Backtester
python -m src.backtest.backtester --all-golden                  # Backtest all golden wallets
python -m src.backtest.backtester --wallet 0xabc123...          # Backtest a specific wallet
python -m src.backtest.backtester --all-golden --sweep           # Parameter sweep
python -m src.backtest.backtester --all-golden --stop-loss 0.03 # Custom SL
python -m src.backtest.backtester --list-experiments             # List past experiments
python -m src.backtest.backtester --show-experiment <id>         # View past run

# Diagnostic tools
python scripts/seed_and_replay.py              # Seed sample data + backtest
python scripts/seed_and_replay.py --seed-only  # Just populate DB
python scripts/seed_and_replay.py --sweep      # Seed + parameter sweep
python scripts/diagnose_rejections.py          # Debug signal rejections
python scripts/replay_decision_cycle.py        # Replay recent approve/reject outcomes
python scripts/run_crash_monte_carlo.py        # Crash stress test
python scripts/run_rotation_shadow_mode.py     # 7-day rotation shadow mode
```


## Live Safety Hardening

Recent live-readiness hardening adds explicit guardrails and observability:

- **Canary controls:** optional caps on order size and daily signal count before full rollout.
- **Per-source/day throttles:** firewall approvals and live entries can be capped by source.
- **External kill switch:** live entries can be blocked instantly by env flag or a watched file.
- **Runtime health snapshot:** dashboard/API exposes subsystem health, stale heartbeats, firewall rejection summary, and live kill-switch state.
- **Rotation telemetry depth:** replacement decisions now capture candidate/incumbent scores, reasons, and post-close outcomes.
- **Decision replay harness:** `scripts/replay_decision_cycle.py` summarizes approval/rejection reasons and execution attribution by source/regime.
## Signal Pipeline

Signals flow through 6 layers before execution. Each layer can reject:

```
Strategy Scorer (914 strategies/cycle)
  │  5-dimension scoring: PnL (30%), win rate (25%), Sharpe (20%),
  │  consistency (15%), risk-adjusted return (10%)
  │  + sample-size penalty + time decay
  ▼
Regime Strategy Filter
  │  5 regimes: trending_up, trending_down, ranging, volatile, low_liquidity
  │  Compatibility matrix adjusts scores ±50% based on regime fit
  ▼
Signal Processor
  │  Step 1: Cull below min_score_threshold (0.30)
  │  Step 2: Dedup by (coin, direction) — merge agreeing strategies
  │  Step 3: Resolve conflicts — regime-aligned side wins
  │  Step 4: Compress to top 8 candidates
  ▼
Decision Engine
  │  Composite ranking: score (35%) + regime (25%) + diversity (20%)
  │                     + freshness (10%) + consensus (10%)
  │  Min composite threshold: 0.20
  │  Max 3 trades per cycle
  ▼
Decision Firewall (11-point validation)
  │  1. Schema valid         7. Cooldown (60s/coin)
  │  2. Confidence ≥ 15%     8. Regime alignment
  │  3. Leverage ≤ 5x        9. Source accuracy
  │  4. Position count ≤ 8   10. Daily drawdown < 3%
  │  5. Per-coin ≤ 3         11. Funding rate risk
  │  6. Exposure ≤ 150%
  ▼
Paper Trader / Live Execution
   Feature enrichment, arena consensus, Kelly sizing, trade memory
```

## Configuration

All settings live in `config.py` and most are overridable via environment variables.

### Key Settings

```python
# Trader discovery
MAX_TRACKED_TRADERS = 2000
MIN_TRADES_FOR_STRATEGY = 10

# Scoring
SCORE_DECAY_RATE = 0.95              # per day
MAX_STRATEGIES_PER_CYCLE = 15

# Paper trading
PAPER_TRADING_INITIAL_BALANCE = 10_000
PAPER_TRADING_MAX_LEVERAGE = 5
PAPER_TRADING_STOP_LOSS_PCT = 0.05
PAPER_TRADING_TAKE_PROFIT_PCT = 0.10

# 3-tier scheduling
FAST_CYCLE_INTERVAL = 60             # 1 min
TRADING_CYCLE_INTERVAL = 300         # 5 min (env: TRADING_CYCLE_INTERVAL)
DISCOVERY_CYCLE_INTERVAL = 86400     # 24h  (env: DISCOVERY_CYCLE_INTERVAL)
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HL_BOT_DB` | `./data/bot.db` | Database path (Railway: `/data/bot.db`) |
| `TRADING_CYCLE_INTERVAL` | `300` | Trading cycle interval (seconds) |
| `DISCOVERY_CYCLE_INTERVAL` | `86400` | Discovery cycle interval (seconds) |
| `MAX_ACTIVE_STRATEGIES` | `200` | Max strategies in DB |
| `MAX_STRATEGIES_PER_CYCLE` | `15` | Max strategies per trading cycle |
| `BOT_HARD_CUTOFF_TRADES` | `100` | Trades/day for instant bot rejection |
| `BOT_THRESHOLD` | `3` | Signal score for bot classification |
| `KELLY_MULTIPLIER` | `0.25` | Fraction of full Kelly (0.25 = quarter) |
| `FUNDING_RISK_ENABLED` | `true` | Block trades against expensive funding |
| `POLYMARKET_ENABLED` | `true` | Enable Polymarket sentiment scanner |
| `OPTIONS_FLOW_ENABLED` | `true` | Enable options flow scanner |
| `LIGHTER_ENABLED` | `true` | Enable Lighter exchange adapter |
| `ENABLE_PREDICTIVE_FORECASTER` | `true` | Enable regime forecaster |
| `ENABLE_XGBOOST_FORECASTER` | `true` | Enable ML regime model |
| `LIVE_TRADING_ENABLED` | `false` | Explicitly enable real order submission; otherwise the live trader stays disabled/dry-run |
| `LIVE_CANARY_MODE` | `false` | Enable live canary rollout guardrails |
| `LIVE_CANARY_MAX_ORDER_USD` | `25` | Max order notional when canary mode is enabled |
| `LIVE_CANARY_MAX_SIGNALS_PER_DAY` | `25` | Daily live entry cap when canary mode is enabled |
| `LIVE_MAX_ORDERS_PER_SOURCE_PER_DAY` | `0` | Per-source daily live entry cap (`0` disables cap) |
| `LIVE_EXTERNAL_KILL_SWITCH_FILE` | _(none)_ | Optional file path; truthy/non-empty file activates kill switch |
| `HL_WALLET_MODE` | `agent_only` | Wallet mode; only agent-wallet signing is permitted |
| `HL_PUBLIC_ADDRESS` | _(none)_ | Trading account address (master/vault) |
| `HL_AGENT_PRIVATE_KEY` | _(none)_ | Agent wallet private key (only for `SECRET_MANAGER_PROVIDER=none`) |
| `HL_AGENT_WALLET_ADDRESS` | _(none)_ | Optional explicit agent wallet address (must match signer) |
| `SECRET_MANAGER_PROVIDER` | `none` | `none`, `aws_kms`, or `hashicorp` |
| `AWS_KMS_REGION` | _(none)_ | AWS region for KMS decrypt (when provider is `aws_kms`) |
| `AWS_KMS_CIPHERTEXT_B64` | _(none)_ | Base64-encrypted private key payload for KMS decrypt |
| `VAULT_ADDR` | _(none)_ | Vault address (when provider is `hashicorp`) |
| `VAULT_TOKEN` | _(none)_ | Vault token (when provider is `hashicorp`) |
| `VAULT_SECRET_PATH` | _(none)_ | Vault secret path (when provider is `hashicorp`) |
| `ROTATION_ENGINE_ENABLED` | `false` | Enable rotation decision engine |
| `ROTATION_DRY_RUN_TELEMETRY` | `true` | Shadow mode: simulate replacements and log telemetry only |
| `ROTATION_SHADOW_MODE_DAYS` | `7` | Planned shadow window length for operations logging |
| `FIREWALL_MAX_SIGNALS_PER_SOURCE_PER_DAY` | `0` | Per-source daily firewall pass cap (`0` disables cap) |
| `FIREWALL_CANARY_MODE` | `false` | Enable firewall canary constraints |
| `FIREWALL_CANARY_MAX_POSITIONS` | `2` | Max open positions enforced when firewall canary mode is on |
| `POLYMARKET_MAX_MARKETS_PER_SCAN` | `100` | Hard cap of ranked markets processed each scan |
| `ARKHAM_API_KEY` | _(none)_ | Optional: Arkham Intelligence API key |
| `LOG_FORMAT` | `json` | Log format: `json` (production) or `text` (local) |

### Live Mode Notes

- `LIVE_TRADING_ENABLED=true` is required before any live orders can be submitted.
- When live trading is deployable, exchange positions become the source of truth for exposure, decisioning, and health reporting.
- The paper ledger remains as a shadow book for reporting and is reconciled back to exchange truth instead of driving live risk.
- Paper-only paths such as standalone options-flow, liquidation-reversal, and arena champion execution are skipped while live trading is active so capital does not drift from the tracked book.

## Strategy Types

| Type | Description |
|------|-------------|
| `momentum_long` | Rides upward trends with leveraged longs |
| `momentum_short` | Rides downward trends with leveraged shorts |
| `mean_reversion` | Fades extended moves, expecting reversion |
| `scalping` | High-frequency small profits with tight stops |
| `swing_trading` | Multi-day directional bets |
| `funding_arb` | Captures funding rate differentials |
| `delta_neutral` | Hedged positions capturing yield |
| `trend_following` | Follows trends across timeframes |
| `breakout` | Enters on breakouts from consolidation |
| `concentrated_bet` | Large single-asset high-conviction positions |

## Database Schema

The bot uses SQLite with 13 tables. Key tables:

**traders** — Top 2000 tracked traders (address, PnL, win rate, bot status).
**strategies** — Detected strategies with 5-dimension scores.
**paper_trades** — Simulated trades with full SL/TP/trailing stop tracking.
**paper_account** — Single-row account state (balance, total PnL, trade count).
**golden_wallets** — Evaluated wallets with penalised equity curves.
**wallet_fills** — Historical fills for golden wallets (penalised prices, fees).
**audit_trail** — Immutable INSERT-only trade journal.
**experiments** — Backtest results with full config and metrics.

The database auto-backs up to JSON on each cycle and on SIGTERM for Railway persistence.

## Deployment

See [RAILWAY_DEPLOYMENT.md](RAILWAY_DEPLOYMENT.md) for full Railway setup with persistent volumes.

```bash
# Local Docker
docker build -t hl-bot .
docker run -p 8080:8080 -v $(pwd)/data:/data hl-bot

# Railway auto-detects the Dockerfile. Set env vars in the dashboard.
```

## Troubleshooting

**"Passed: 0 | Rejected: N signals"** — Signal confidence is below the firewall threshold. Run `python scripts/diagnose_rejections.py` to see which check is blocking. During bootstrap, this is expected until the strategy DB accumulates enough history.

**"No fills found"** — The golden wallet pipeline hasn't run yet. Run `python main.py --bootstrap` or seed with `python scripts/seed_and_replay.py`.

**"MISSING websocket, eth_account"** — Install production deps: `pip install websocket-client eth-account`. These are only needed for the live bot loop, not for backtesting.

## Notes

- No API key is needed for research/monitoring (uses Hyperliquid's public info endpoint)
- Paper trading is fully simulated with realistic slippage, fees, and partial fills
- The bot is designed to run 24/7 and improve strategy selection over time
- All logs are structured JSON with automatic secret scrubbing
- Database backs up to JSON on each cycle for recovery after redeploys
