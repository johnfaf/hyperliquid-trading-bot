# Runtime and execution audit, 22 November 2026

This pass inventories the repository and reviews the operating paths with the
largest accounting and deployment risk. The inventory was approximately
172,000 lines of tracked Python, SQL and shell/PowerShell. This is not a
manual line-by-line sign-off of every file or a certification for live orders.

| Area | Verification | Finding or next action |
| --- | --- | --- |
| Paper ledger and Postgres mirror | Read-only live comparison of paper-account totals, trade status counts and decision-row counts. SQLite is authoritative in the VPS dual-write setup. | **High:** After a Postgres error, the old adapter rolled back its earlier statements but resumed mirroring later statements in the same operation. A balance credit could commit without its close. The adapter now abandons the whole mirror operation on an error or affected-row mismatch. Connection failures are counted too. A separate CI job exercises real migrations, funded close, retry, and an injected mirror failure on disposable Postgres. |
| Funding and fees | Hour-boundary oracle sample and published rate observed on the VPS; one open position had a matching receipt. SQLite regressions cover known-zero, late/missing/conflicting inputs, and atomic close/credit. | The dashboard separates recent collector success, open settlement coverage and verified closes. A close before receipt publication is still excluded from evidence, with no retroactive credits. Observe real cross-hour closes and compare verified/incomplete rates by exit reason before interpreting cohorts. |
| Order and risk controls | Reviewed crypto entry holds, stock paper endpoint/mode, journal reservation and restart tests. VPS reports paper-only, no signer and stock shadow. | Keep live disabled. Historical ownership is still unproven without an exact archived decision/order ID link. Verify missed-order, partial-fill and outage cases on a disposable broker account before any later pilot. |
| Replay and research | Reviewed completed-bar gating, fold separation, fixed forward protocol and independent status reporting. | Leave the 28-day BTC pilot and previous experiments frozen. A green worker and a no-trade period are not strategy evidence. No performance promotion is included. |
| Dashboard and network | Checked authenticated routes on the VPS and the Windows restricted SSH profile. The bot HTTP listener is bound to loopback, and Postgres has no host listener. | Treat a healthy local tunnel and a healthy app separately. The Research page carries the funding evidence counters. |
| Backups and rollout | Reviewed snapshot, SQLite integrity check, rollback, and restore paths. | Same-disk copies are not disaster recovery. Keep the latest local rollback uncompressed. For capacity, archive specific older immutable files with round-trip SHA-256 verification; never overwrite or alter the live DB. Next: encrypted off-host copy and a timed isolated restore drill. |
| Supply chain | Repository lint and dependency audit run in CI. | The current dependency scan has explicit exceptions. Review their reachability and retirement dates separately; a successful scan does not clear the exceptions. |

## Operational checks

- Run the full CI suite and the dedicated Postgres integration job. Default
  local runs skip the latter unless a disposable database is configured.
- Verify current paper-only readiness, missing signer, stock shadow, active
  research timers and authenticated Research/Trade review pages after rollout.
- For an older sidecar-free backup, use
  `sudo python3 scripts/archive_sqlite_rollback.py bot-pre-FULL_SHA.db`.
  The command only replaces an explicitly named pre-rollout file after
  verifying the decompressed byte hash. Leave the newest backup as a raw DB.
- Before any capital deployment, require representative fresh samples,
  complete decision/fee/funding evidence, positive net under stressed costs,
  a tested operator halt, and a separately approved release.
