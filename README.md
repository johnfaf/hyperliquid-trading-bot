# Hyperliquid Auto-Research Trading Bot

An autonomous research bot that continuously discovers the most profitable traders on Hyperliquid, identifies their strategies, scores them over time, and paper trades the best ones — improving its strategy selection day by day.

## How It Works

The bot runs a continuous loop with 5 phases plus supporting infrastructure:

**Phase 1 — Trader Discovery**: Scans the top 2000 traders from the Hyperliquid leaderboard, vaults, and whale activity. Bot detection filters out automated accounts (market makers, arb bots, funding farmers) using configurable signal-based scoring. Known bots are persisted in the database and skipped on future scans.

**Phase 2 — Strategy Identification**: Analyzes each tracked trader's positions, leverage, fills, and trading frequency to classify what strategy they're running (momentum, mean-reversion, scalping, funding arb, etc.).

**Phase 3 — Strategy Scoring**: Scores every identified strategy across 5 dimensions (PnL, win rate, Sharpe, consistency, risk-adjusted returns). Applies time-decay so old results fade, and deactivates underperformers automatically.

**Phase 4 — Paper Trading & Copy Trading**: Takes the top-scoring strategies and simulates trades with a $10,000 paper account. Copy trader mirrors top traders' live position changes through the DecisionFirewall. Full risk management with stop-losses, take-profits, trailing stops, position limits, aggregate exposure caps, and Kelly-based sizing.

**Phase 5 — Reporting & Monitoring**: Generates reports with trader rankings, strategy leaderboards, paper trading P&L, and self-improvement metrics. Structured JSON logging for Railway/ELK with secret scrubbing.

**Golden Wallet Pipeline**: Downloads 90 days of fills for human-like traders, applies realistic execution penalties (taker fees + slippage on both legs), and tags wallets whose penalised equity curve is still rising as "golden." Golden wallets feed into both copy trading and backtesting.

**Backtester**: Event-driven backtester that replays golden wallet fills through copy-trading simulation with configurable parameters. Includes parameter sweep, experiment persistence, and comprehensive metrics (Sharpe, Sortino, Calmar, profit factor, max drawdown, expectancy).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Bootstrap (first deploy — seeds DB with top trader data)
python main.py --bootstrap

# Run a single research cycle
python main.py --once

# Run the bot continuously
python main.py

# Check current status
python main.py --status

# Generate a report
python main.py --report
```

## Backtester

```bash
# Backtest all golden wallets with default config
python -m src.backtester --all-golden

# Backtest a specific wallet
python -m src.backtester --wallet 0xabc123...

# Custom parameters
python -m src.backtester --all-golden --stop-loss 0.03 --take-profit 0.15

# Parameter sweep (tests all SL/TP/sizing combinations, ranks by Sharpe)
python -m src.backtester --all-golden --sweep

# View past experiments
python -m src.backtester --list-experiments
python -m src.backtester --show-experiment bt_20260324_120000_42
```

## Project Structure

```
├── main.py                       # Main orchestrator & CLI
├── config.py                     # All configuration in one place
├── requirements.txt              # Pinned dependencies
├── Dockerfile                    # Production container (Railway)
├── src/
│   ├── hyperliquid_client.py     # Hyperliquid API wrapper (V3 — via APIManager)
│   ├── api_manager.py            # Centralized rate limiter (token bucket + circuit breaker)
│   ├── trader_discovery.py       # Finds & monitors top traders, bot detection
│   ├── strategy_identifier.py    # Classifies trading strategies
│   ├── strategy_scorer.py        # Scores & ranks strategies (self-improvement)
│   ├── paper_trader.py           # Paper trading simulator (V2 — firewall + slippage)
│   ├── copy_trader.py            # Copy trading engine (V2 — mirrors top traders)
│   ├── golden_wallet.py          # Golden wallet pipeline (90d fills + penalties)
│   ├── golden_bridge.py          # Connects golden wallets to live copy trading
│   ├── backtester.py             # Vectorized backtester + experiment tracker
│   ├── decision_firewall.py      # Signal validation, exposure limits, audit trail
│   ├── signal_schema.py          # Universal TradeSignal dataclass
│   ├── agent_scoring.py          # Source reliability tracking
│   ├── kelly_sizing.py           # Kelly criterion position sizing
│   ├── regime_detector.py        # Market regime classification
│   ├── features.py               # Feature engine for signal enrichment
│   ├── options_flow.py           # Deribit + Binance options flow scanner
│   ├── exchange_aggregator.py    # Multi-exchange data aggregation
│   ├── calibration.py            # Prediction calibration tracking
│   ├── trade_memory.py           # Trade outcome memory
│   ├── alpha_arena.py            # Strategy tournament system
│   ├── signal_processor.py       # Signal processing pipeline
│   ├── decision_engine.py        # Decision orchestrator
│   ├── reporter.py               # Report generation
│   ├── dashboard.py              # Web dashboard (FastAPI)
│   ├── database.py               # SQLite persistence + audit trail
│   └── exchanges/                # Multi-exchange adapters
├── data/                         # SQLite database + backups
├── logs/                         # Daily log files (JSON + text)
└── reports/                      # Generated reports
```

## Configuration

All settings are in `config.py`. Key ones to customize:

```python
MAX_TRACKED_TRADERS = 2000        # Scan top 2000 traders

# Bot detection (tunable via env vars)
BOT_HARD_CUTOFF_TRADES = 100      # >N trades/day = instant bot
BOT_THRESHOLD = 3                  # signal score >= N = bot
BOT_ELEVATED_FREQ = 50            # trades/day for elevated freq signal

# Paper trading
PAPER_TRADING_INITIAL_BALANCE = 10_000
PAPER_TRADING_MAX_LEVERAGE = 5
PAPER_TRADING_STOP_LOSS_PCT = 0.05
PAPER_TRADING_TAKE_PROFIT_PCT = 0.10

# Timing
MAIN_LOOP_INTERVAL = 300          # Check every 5 min
RESEARCH_CYCLE_INTERVAL = 3600    # Full research every hour
SCORING_INTERVAL = 86400          # Re-score daily
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HL_BOT_DB` | `data/bot.db` | Database file path |
| `LOG_FORMAT` | `json` | Log format: `json` (production) or `text` (local dev) |
| `BOT_HARD_CUTOFF_TRADES` | `100` | Trades/day threshold for instant bot rejection |
| `BOT_THRESHOLD` | `3` | Signal score threshold for bot classification |
| `BOT_ELEVATED_FREQ` | `50` | Trades/day for elevated frequency bot signal |
| `BOT_MM_PNL_THRESHOLD` | `0.0` | Median PnL below this = spread/MM bot |
| `LIGHTER_ENABLED` | `true` | Enable Lighter exchange adapter |

## Strategy Types Detected

| Strategy | Description |
|----------|-------------|
| momentum_long | Rides upward trends with leveraged long positions |
| momentum_short | Rides downward trends with leveraged short positions |
| mean_reversion | Buys dips and sells rips, expecting price to revert |
| scalping | High-frequency small-profit trades with tight stops |
| swing_trading | Medium-term directional bets held for days |
| funding_arb | Captures funding rate differentials |
| delta_neutral | Hedged positions capturing yield |
| trend_following | Follows established trends across timeframes |
| breakout | Enters on price breakouts from consolidation |
| concentrated_bet | Large single-asset positions with high conviction |

## Risk Management

The DecisionFirewall validates every signal before execution:

- Max 5 concurrent positions (configurable)
- Max 2 positions per coin
- 30% aggregate portfolio exposure cap
- Conflict detection (no opposing positions on same coin)
- 5-minute cooldown per coin after trade
- Regime-aware size modifiers
- Immutable audit trail (INSERT-only) logging every approval/rejection

## Deployment (Railway)

```bash
# Build and run locally
docker build -t hl-bot .
docker run -p 8080:8080 hl-bot

# Railway auto-detects the Dockerfile
# Set env vars in Railway dashboard for any config overrides
```

## Regime Forecaster V2 — Polymarket + Options Flow Integration

The predictive regime forecaster uses a 5-input composite signal for forward-looking market regime detection:

| Input | Weight | Source | Description |
|-------|--------|--------|-------------|
| Funding Rate Slope | 0.30 | Hyperliquid API | Slope of recent funding rates (bearish pressure detection) |
| Orderbook Imbalance | 0.25 | Hyperliquid L2 | Bid/ask depth ratio from top 10 levels |
| Polymarket Sentiment | 0.20 | Polymarket CLOB | Aggregated crypto prediction-market odds movements |
| Options Flow Conviction | 0.15 | Deribit API | Net directional flow from unusual options prints |
| Arkham Smart-Money Flow | 0.10 | Arkham API (optional) | On-chain whale/fund flow scoring |

### How It Works

1. **Background scanners** refresh Polymarket (every 3 min) and Deribit options flow (every 2 min) on daemon threads
2. Each trading cycle, the main loop **injects** fresh sentiment + conviction data into the forecaster
3. The forecaster computes a **composite signal** from all active inputs (weights auto-redistribute when sources are unavailable)
4. Signal classification: `< -0.15` → crash, `> 0.15` → bullish, else neutral
5. The **DecisionFirewall** uses the regime prediction for dynamic de-risking (80% size reduction + 25% exposure cap during crash regimes)

### Example Regime Output

```json
{
  "signal": -0.2341,
  "regime": "crash",
  "confidence": 0.4214,
  "components": {
    "funding_slope": -0.35,
    "imbalance": -0.18,
    "arkham_flow": 0.0,
    "polymarket": -0.45,
    "options_flow": -0.72
  },
  "active_inputs": 4
}
```

### Configuration (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `POLYMARKET_ENABLED` | `true` | Enable/disable Polymarket scanner |
| `POLYMARKET_SCAN_INTERVAL` | `180` | Background scan interval (seconds) |
| `OPTIONS_FLOW_ENABLED` | `true` | Enable/disable options flow scanner |
| `OPTIONS_FLOW_SCAN_INTERVAL` | `120` | Background scan interval (seconds) |
| `FORECASTER_EXTERNAL_DATA_TTL` | `600` | Max staleness for external data (seconds) |
| `ENABLE_PREDICTIVE_FORECASTER` | `true` | Master switch for regime forecaster |

## Notes

- No API key needed for the research/monitoring features (uses public info endpoint)
- Paper trading is simulated — no real funds at risk
- The bot is designed to run 24/7 and improve its strategy selection over time
- Structured JSON logs for production (Railway, ELK, Datadog compatible)
- Secrets are automatically scrubbed from log output
- Database backs up to JSON on each cycle for recovery after redeploys
