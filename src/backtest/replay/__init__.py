"""Historical replay harness for the production decision pipeline.

The candle backtester in `src/backtest/candle_backtester.py` runs textbook
strategies (RSI, MACD, momentum, ...) on raw candles. It is NOT what the
production bot does -- the bot has a multi-stage pipeline (signals ->
scorer -> regime detector -> firewall -> Kelly sizing) that this module
replays t-step by t-step against historical data.

The point of this package is reproducibility and no-lookahead. Every read
that the live bot performs at decision time has to be either:
  - replayed from a frozen historical artifact (price, funding, model),
  - intercepted by a shim that returns only data <= t, or
  - explicitly stubbed to a neutral value (with STUBBED audit metadata).

What lives here:
  - clock.ReplayClock: controllable wall clock, injected via SubsystemContainer
  - candle_oracle.CandleOracle: read-only candle DB with t-strict invariants
  - api_manager_shim.ReplayAPIManager: drop-in for src.core.api_manager
  - stub_subsystems: null-objects for polymarket / options / macro / events
  - strategy_seed: frozen strategy pool snapshot loader
  - harness.ReplayHarness: top-level orchestrator
"""
