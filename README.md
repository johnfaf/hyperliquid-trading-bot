# Hyperliquid Auto-Research Trading Bot

An autonomous research bot that continuously discovers the most profitable traders on Hyperliquid, identifies their strategies, scores them over time, and paper trades the best ones — improving its strategy selection day by day.

## How It Works

The bot runs a continuous loop with 5 phases:

**Phase 1 — Trader Discovery**: Scans the Hyperliquid leaderboard, vaults, and whale activity to find top-performing traders. Tracks their addresses and monitors their positions over time.

**Phase 2 — Strategy Identification**: Analyzes each tracked trader's positions, leverage, fills, and trading frequency to classify what strategy they're running (momentum, mean-reversion, scalping, funding arb, etc.).

**Phase 3 — Strategy Scoring**: Scores every identified strategy across 5 dimensions (PnL, win rate, Sharpe, consistency, risk-adjusted returns). Applies time-decay so old results fade, and deactivates underperformers automatically.

**Phase 4 — Paper Trading**: Takes the top-scoring strategies and simulates trades with a $10,000 paper account. Full risk management with stop-losses, take-profits, position limits, and exposure caps.

**Phase 5 — Reporting**: Generates markdown reports with trader rankings, strategy leaderboards, paper trading P&L, and self-improvement metrics.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run a single research cycle
python main.py --once

# Run the bot continuously
python main.py

# Check current status
python main.py --status

# Generate a report
python main.py --report
```

## Project Structure

```
├── main.py                    # Main orchestrator & CLI
├── config.py                  # All configuration in one place
├── requirements.txt
├── src/
│   ├── hyperliquid_client.py  # Hyperliquid API wrapper
│   ├── trader_discovery.py    # Finds & monitors top traders
│   ├── strategy_identifier.py # Classifies trading strategies
│   ├── strategy_scorer.py     # Scores & ranks strategies (self-improvement)
│   ├── paper_trader.py        # Paper trading simulator
│   ├── reporter.py            # Report generation
│   └── database.py            # SQLite persistence layer
├── data/                      # SQLite database
├── logs/                      # Daily log files
└── reports/                   # Generated markdown reports
```

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

## Self-Improvement System

The scoring system uses weighted metrics with time-decay:

- **PnL (30%)**: Total profitability normalized via sigmoid
- **Win Rate (25%)**: Percentage of winning trades (penalized if few trades)
- **Sharpe Ratio (20%)**: Risk-adjusted returns estimate
- **Consistency (15%)**: Low variance in scores over time = higher consistency
- **Risk-Adjusted Return (10%)**: PnL relative to drawdown potential

Strategies that stop performing get their scores decayed by 5% per day. Below a score of 0.1, they're deactivated entirely. Strategies that consistently outperform rise to the top and get more paper trading allocation.

## Configuration

All settings are in `config.py`. Key ones to customize:

```python
# How many top traders to track
MAX_TRACKED_TRADERS = 50

# Paper trading settings
PAPER_TRADING_INITIAL_BALANCE = 10_000
PAPER_TRADING_MAX_LEVERAGE = 5
PAPER_TRADING_STOP_LOSS_PCT = 0.05      # 5%
PAPER_TRADING_TAKE_PROFIT_PCT = 0.10    # 10%

# Timing
MAIN_LOOP_INTERVAL = 300      # Check every 5 min
RESEARCH_CYCLE_INTERVAL = 3600 # Full research every hour
SCORING_INTERVAL = 86400       # Re-score daily
```

## Database Override

If you encounter permissions issues with the default `data/bot.db` path, set:

```bash
export HL_BOT_DB=/path/to/your/bot.db
```

## Notes

- No API key needed for the research/monitoring features (uses public info endpoint)
- Paper trading is simulated — no real funds at risk
- The bot is designed to run 24/7 and improve its strategy selection over time
- Logs are written to `logs/` with daily rotation
- Reports are saved to `reports/` as markdown files
