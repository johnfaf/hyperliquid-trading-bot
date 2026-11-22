# Operational follow-through

## Funding and Arena observations

Research compares 24-hour cross-hour closes by exit reason. It reports
unpublished or missing receipts separately from invalid receipts and
arithmetic mismatches. Verification recomputes the signed funding and
checks the recorded fees, gross amount, and net realized amount. No exit
is forced or delayed for this check, and no late cost is posted to a
previously closed position. An empty close sample is not a pass.

The historical BTC Arena error occurred after decision linkage. Its
short candidate faced two existing BTC shorts, equal to the configured
same-side limit. That is consistent with an admission rejection, but the
original exception message was not retained. The direct paper routes now
journal typed portfolio and planned-stop denials as expected blocks with
specific codes. Unexpected exceptions remain errors with a stage. Neither
case lowers limits or authorizes an order.

## Off-host backups

The VPS exporter reads complete immutable rollback snapshots, never active
SQLite files. It streams the ledger set and per-file hashes through age
public-key encryption. The Windows PC receives ciphertext only. The
identity stays on Windows under the account's DPAPI and private ACLs.
Preserve a separately protected copy of the recovery identity before
replacing the Windows profile; the VPS cannot recover this identity.

`deploy/recovery/Pull-HyperliquidRecovery.ps1` retrieves the latest complete
pre-rollout snapshot, authenticates the encrypted archive, extracts only
allowlisted members in scratch, and verifies every file hash, critical
table list and SQLite quick check. It uploads a small restore receipt
only. No signers, broker credentials, or environment files are archived.
After a successful drill, scratch databases are removed and the ciphertext
is retained. The SQLite copy is authoritative in dual-write mode; this
bundle is not an independent Postgres dump or a copy of frozen research
experiments.

Use the installed per-user backup pull or run it after a verified rollout
while the PC is online. It captures new complete rollout snapshots; it
does not generate daily snapshots of the active book. Snapshot age, not
download age, determines the dashboard freshness warning. Off-host RPO
equals the age of the last successful copied rollout snapshot.

## Release review

Research shows exact crypto cohort eligibility, frozen study status,
funding close checks, verified off-host recovery and stock drill evidence.
Completed BTC research still needs at least 30 natural closes and positive
results versus cash after base and stressed costs, also after removing its
largest winner. CI runs restart, lost acknowledgement and partial-fill
tests with simulated venues. Those are not a verified broker outage drill.
The release display cannot change any trading mode. A venue-connected
drill and a separately approved release are still required.
