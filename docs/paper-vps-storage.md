# Paper VPS storage

The dashboard publisher measures filesystem space and the paper SQLite
ledger size without walking the database. It keeps bounded hourly samples
to estimate primary database growth and alerts on low space, or on a
trend that would reach the operating reserve within three days. The trend
is an estimate, not guaranteed runway. Alerts are deduplicated for one day
per severity. It never deletes, compacts, or resets a ledger.

The rollout checks capacity before the Docker build, and again before
the SQLite backup. It still reserves two copies of the ledgers plus a
2 GiB operating buffer. Existing WAL bytes are included. Correct a
capacity shortfall before a full rollout; do not override the check.
Keep the newest raw rollback and the independently verified encrypted
off-host copy. Only archive selected inactive older rollback files,
using the existing round-trip hash verifier. Archive savings do not
change the size of the running database.

New Polymarket market snapshots compress the opaque raw JSON payload
losslessly when the versioned JSON envelope is smaller. The replay
reader supports both formats and declines corrupt compressed records
instead of substituting current market data. This does not alter scan
cadence, timestamps, numeric price/liquidity columns, market metadata,
trade history, or historical database rows. Older application versions
should not be used to replay snapshots created with the new codec;
they cannot decode the new evidence format. No migration or automatic
VACUUM is part of this release.
