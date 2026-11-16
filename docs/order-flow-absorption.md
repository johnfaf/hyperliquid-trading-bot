# BTC/ETH order-flow absorption research

This is a separately deployed observer, not a strategy in the trading bot. It opens only the public Hyperliquid fast `l2Book` and `trades` streams for BTC and ETH. It stores five-second top-of-book and top-five depth frames, signed trade notionals and counts, plus venue/receipt timestamps. It does not keep raw wallet addresses or an individual-fill archive. Capture and daily score storage are under `/var/lib/hyperliquid-absorption`, with 60 days of retention and no access to the bot's database or credentials.

The `flow_absorption_v1` rule is frozen. Every UTC minute, compare the preceding minute's notional with the prior ten-minute median of one-minute notionals. Require 2x volume, at least three prints, absolute signed notional imbalance of 0.60 or more, and a current spread no wider than 3 bps. A candidate needs no more than 2 bps of price progress in the aggressive direction, at least a 1 bp rebound from that extreme, and 60% of top-five depth on the resisting side. Fade only then. Both directions use the same thresholds. That is a hypothesis about public flow, not proof of who absorbed an order or whether visible liquidity would persist.

The feed freezes observations before sampling cutoffs, never revises a decision, and resets continuity on reconnect or missing/late data. Eleven complete minutes of frames are needed to emit a ready decision. Trade B/A direction is sanity-checked against observed prior quotes; future readiness requires sufficient checks and majority agreement, not a claim that every trade can be classified independently from L2.

Five minutes is the primary horizon; 15 minutes is diagnostic. The scorecard requires the first observed book after the signal, continuous ready frames through the exit, enough quoted top-level depth for a hypothetical $100 order, and no funding-hour crossing. Model both legs at bid/ask with 4.5 bps taker fees and 1/4 bps per-side additional slippage assumptions. Earlier controls are matched before scoring on asset, flow direction, UTC session, volume shock and imbalance. Missing outcomes are excluded without rematching. The report only reaches research-review eligibility after 50 matched primary pairs on 14 days, healthy coverage and positive stressed scores/uncertainty bounds. It never enables an order.

```shell
python -m scripts.capture_hyperliquid_absorption --db /private/flow.sqlite3 --seconds 30
python -m scripts.score_hyperliquid_absorption --db /private/flow.sqlite3
```

On the VPS, `bash deploy/install-vps-absorption.sh FULL_GIT_REVISION` pins a separate restartable container, using the existing dependency image without the bot's environment or mounts. The `hyperliquid-absorption-report.timer` refreshes `/var/lib/hyperliquid-absorption/reports/latest.md` at 00:20 UTC each day. Do not change a frozen rule mid-study; a changed rule needs a new ID and a new unseen evaluation window.
