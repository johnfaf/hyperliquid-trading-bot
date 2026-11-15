# Prospective BTC paper exit evaluation

Frozen before the prospective window. This is a research screen and cannot
authorize live trading or alter the deployed paper strategy.

## Holdout and policy

- Entries: **2026-11-16 00:00 UTC through 2027-01-11 00:00 UTC**, end exclusive.
  Finish observation at **2027-01-12 01:05 UTC**. The initial November 15
  recording is a plumbing check and is excluded from the decision.
- Use executed entries from the existing BTC paper pilot. Keep their entry
  fills, sizes, hard stop, target, and maximum holding rule fixed. Run the
  [three exit variants](PAPER_EXIT_EXPERIMENT.md) on identical quotes. Do not
  change trail thresholds, add a sentiment trigger, or select dates based on
  results. Each comparison is per entry; it does not recreate a portfolio's
  later entry opportunities or aggregate investment risk.
- Base: observed spread, 4.5 bps fees per leg, 5 bps additional exit impact,
  no artificial fill delay. Adverse: twice the observed spread (center fixed),
  6 bps fees per leg, 10 bps additional exit impact, and at least a five-second
  delay from trigger to adequate executable bid. Base and adverse share the
  original paper entry fill. These are sensitivity assumptions, not live fills.

## Decision rule

Require at least 30 fully matched exits across 14 distinct UTC entry days and
six distinct holdout weeks, usable data for at least 99% of scheduled slots,
and a paired result in both scenarios for at least 95% of all holdout entries.
A missing funding estimate or a material data gap in any policy/scenario
excludes the complete entry from comparisons; overlapping flags are reported.
Require at least five matched entries in each starting-volatility group:
below 5 bps and at least 5 bps of sampled 14-minute average true range divided
by the latest available mark. Missing warmup is unknown, not low volatility.
These are minimum review gates, not a claim of statistical significance.

Against **five-second fixed exits**, trailing must earn more after base costs,
not earn less under adverse execution, retain a positive base difference when
each entry week is removed in turn, reduce average modeled giveback, and not
increase worst observed single-trade drawdown. Show the separate fixed-5s vs.
fixed-60s comparison as the speed effect. The five-minute rebound diagnostic
does not change acceptance. Passing means a candidate for more paper testing,
not a recommendation to fund; weak coverage or sparse trades mean not enough
evidence. Keep the same criteria through the end of this window.

## Operations and historical checks

The daily scorecard uses the previous seven calendar days as of UTC midnight.
It is descriptive and distinguishes zero matched exits from zero P&L. On the
VPS, `hyperliquid-exit-scorecard.timer` runs at 00:20 UTC and writes private
Markdown reports. At the holdout deadline, run the script with `--holdout`
to evaluate the frozen gate; it refuses an early verdict.

Use `--from` and `--until` with explicit timezone for a historical check of up
to seven days of existing recordings. Start with captured data without buying
additional history. [Hyperliquid's archive](https://hyperliquid.gitbook.io/hyperliquid-docs/historical-data)
has time-stamped books and asset contexts, can be incomplete, and charges
the requester for transfer. Do not reconstruct five-second fills from
minute candles. Synthetic regression scenarios are separate from performance
evidence.
