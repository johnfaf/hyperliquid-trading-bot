# SPY paired exit retry study

**Research only.** The fallback recovered the known wide-spread IEX exit in
two scenarios, but there were no natural exit failures to recover in the
fresh sample. The results do not justify enabling this in the trading workers.

## Fixed method

- Use the existing rising-trend SPY pullback signal and entry rules without
  modifications. Score January 2 through August 29, 2025 (165 sessions);
  daily warm-up starts September 3, 2024. The new scored dates do not
  overlap the prior stock holdouts.
- Compare two policies from the **same modeled entry** in each episode. Both
  try the original 9:35 a.m. New York quote window first. Evaluate episodes
  independently even if they overlap. Counts are not a portfolio return,
  and a recovered exit does not establish a profitable entry.
- If the primary exit fails, try again at 9:36, 9:38 and 9:40, taking the
  first eligible bid in time order. Each window uses seconds 2 to 30
  (20 to 30 under combined stress). Require a two-sided quote, usable displayed
  depth, and at most 35 basis points of spread after the scenario widening.
- Require net proceeds per share no worse than 65 basis points below a
  fixed reference. The reference is the first usable 9:35 midpoint (at least
  one displayed unit on both sides, raw spread at most 100 bp), or the last
  completed trigger close if no midpoint is available. Each attempt includes
  modeled fees, slippage and the scenario's additional exit-price haircut.
  The reference is never reset as bids fall. The cap applies to retries only;
  the original exit keeps its original fill rules.
- If no attempt succeeds, record one `exit_exposure_still_open` alert at
  9:40:30 with the symbol, quantity and exit reason. Do not invent a fill or
  send an actual notification. This is not a cap on trading losses: a gap,
  prolonged lack of liquidity or an outage can leave the position open.
- Use raw IEX daily bars after verifying they match split-adjusted bars.
  Replay quotes separately from [Alpaca's IEX and consolidated SIP feeds](https://docs.alpaca.markets/us/docs/historical-stock-data-1).
  The study pinned source, dates and policy before capture, and captured all
  40 requested windows for each feed. Neither feed proves actual routed
  broker fills.

## Fresh Data

Eight signal dates qualified for an entry attempt.

| Feed and scenario | Paired entries | Primary exits | Retry exits | Still open / alerts |
|---|---:|---:|---:|---:|
| IEX base | 6 | 6 | 0 | 0 |
| IEX cost stress | 5 | 5 | 0 | 0 |
| IEX combined stress | 0 | 0 | 0 | 0 |
| IEX forced exit outage | 5 | 0 | 0 | 5 |
| SIP base | 6 | 6 | 0 | 0 |
| SIP cost stress | 5 | 5 | 0 | 0 |
| SIP combined stress | 5 | 5 | 0 | 0 |
| SIP forced exit outage | 5 | 0 | 0 | 5 |

Both feeds produced one alert per forced outage episode. No qualifying IEX
entries under combined stress means there was no IEX exit to test in that
scenario. With no natural primary miss among the paired entries, this
holdout cannot estimate a recovery rate or show that retries improve P&L.

## Known Incident

Separately replayed the February 25, 2026 time-limit exit from the prior
study, reusing its original 4,196 IEX quotes at 9:35. This date was picked
because it failed previously and is not part of the fresh holdout.

| Scenario | IEX replay | Consolidated SIP replay |
|---|---|---|
| Base | Recovered at 9:36:02, 22.50 bp all-in vs reference | Eligible primary bid at 9:35:02, 0.14 bp spread |
| Cost stress | Recovered at 9:36:02, 33.48 bp all-in vs reference | Eligible primary bid at 9:35:02, 0.14 bp spread |
| Combined stress | No eligible bid in any retry; alert at 9:40:30 | Eligible primary bid at 9:35:20, 0.43 bp modeled spread |

The old IEX-only failure was not evidence that the broader market lacked an
eligible bid. The fallback helped under limited conditions, but neither feed
selection nor retries guarantee a sale. Before enabling either in a worker,
collect more missed-exit cases with feed provenance and test order
acknowledgment, partial fills, and unresolved-exposure alerts in paper. No
orders, trading configuration, automatic schedules, or approvals changed.
