# Funding evidence and forward shadow operations

## Paper settlement ledger

The VPS paper overlay enables the settlement model for newly opened
positions. Legacy open positions keep their previous accounting and
closed records are not rewritten. UTC settlement evidence is stored in
the paper volume's separate bot.funding.sqlite3, one record per asset and
hour. Before an hour changes the worker keeps the last observed public
oracle price up to two minutes before that boundary. It reads each
published historical rate after settlement. Received times prevent
future records from entering an earlier close.

Both strategy and copy closes freeze the same signed-quantity, rate,
price and time receipts in the existing atomic close-and-account-credit
transaction. Duplicate observations and close retries cannot add another
debit. Conflicting rates and absent hours are unverified. A paper exit is
not delayed for missing funding. Its ledger P&L reflects known costs only
and the trade is excluded from strategy evidence. This release does not
retroactively credit late settlements to closed positions. Modeled paper
funding is not the venue's account-level cash movement.

Strategy reports group new observations by a digest of trading code,
strategy parameters and effective risk/cost controls. The build revision
is stored separately. A dashboard-only deployment can preserve a group.
Changes to shared trading code conservatively create new cohorts. Old
release-bound cohorts retain their original identity.

## Frozen forward study

The separate BTC comparison opens one long at the next modeled 15-minute
open when a completed bar closes above the highest high of the previous
96 completed bars. The $1,000 book uses 5% unlevered allocation, a 2% price
stop, 4% target and an 18-hour maximum hold. Base and stress costs match
the earlier paper research assumptions. Cash and a passive BTC sleeve use
the same window. The research simulator models funding with the hourly
open as its price proxy, not the production paper oracle ledger.

The first run freezes the source digest and a start at the next UTC hour.
Only closed observations after the fixed start advance outcomes. Warmup
candles are not trades. The next 28 days remain fixed across restarts,
with no weekly re-selection. In-progress open trades are marked, not
force-closed by every report run. The one final window close is disclosed
separately from normal exits. Missing inputs produce a data-gap state.

The worker has no order client, credentials or access to the production
ledger. Systemd stores its inputs and trace in a separate private state
directory. The dashboard publishes only aggregate scores. The prior
BTC reversal pilot, exit experiment and absorption study are unchanged.

Offline screening produced six closes and a small loss versus cash in
the earlier window, and no trades in the later window. The fixed forward
comparison is research, not deployment of a candidate into the trading bot.

## Account audit

Use the public-address capture and the offline reconciler. Raw data and
results are private and must stay outside Git. The account manifest binds
the returned fills, funding and venue order history to the requested
public account. Durable journal links are preferred. A venue order can
also supply an exact link if its client ID matches the deterministic
decision ID and symbol, side, time and quantity all agree. Missing and
conflicting identifiers remain separately labeled.

The CSV comparison checks whether the supplied export already subtracts
the fee. Unique numerical coincidences are used only for that accounting
cross-check and never to link a bot decision. Captured funding receipts
are actual venue cash movements, stored apart from simulated funding.
Do not treat the fill-plus-funding subtotal as full account performance:
it does not establish opening/closing unrealized P&L, funds movements,
fees in other assets, or coverage beyond the available venue records.

    python -m scripts.capture_hyperliquid_account_history --account PUBLIC_ADDRESS --start UTC_START --end UTC_END --output PRIVATE_DIR
    python -m scripts.reconcile_hyperliquid_execution --capture PRIVATE_DIR --account PUBLIC_ADDRESS --decisions BOT_DB --csv EXPORT.csv --out PRIVATE_REPORT.json

## Operations

The rollout backs up the funding sidecar if it exists and pins the active
WAL database at a consistent read snapshot while the backup copies.
Restore checks use scratch files only. For archived .db.gz backups,
decompress into new scratch storage and verify integrity before any use.
Do not restore over the trading databases automatically.
