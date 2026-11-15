# Recover and monitor the frozen paper study

The backup timer creates a daily OpenPGP-encrypted archive at 00:40 UTC.
Only a public recovery key is installed on the VPS. The study ledger and
market recording use separate SQLite online backups. A shared replay
cutoff is taken from the earlier end of the two snapshots; it is not a
claim that the databases were committed simultaneously. The archive
contains both snapshots, their hashes, the commit ID, and an exact
reference scorecard over the common window. Original databases are
never used as restore targets.

The Windows task uses a separate SSH key restricted to `status` and
`fetch <encrypted-archive-name>`. It verifies a downloaded archive's size
and SHA-256 before publishing it locally. Two consecutive failed checks
request a Windows notification and record a line in `watch.log`.
It monitors unavailable or stale data, missed scorecards, backup delivery
and disk space. The Windows task can detect a VPS outage, but it cannot
run while the PC sleeps or is signed out. Focus Assist can suppress a
visible notification. The log is the durable record.

On the configured Windows PC, run `Verify-PaperStudyBackup.ps1` from
the installed release to verify the latest downloaded archive. For
manual recovery on another host:

1. Work with a previously downloaded encrypted archive and the
   recovery key. Decrypt it into a private scratch directory.
2. Run `python scripts/paper_study_recovery.py verify PATH_TO_DECRYPTED_TAR_GZ`
   using the verifier revision recorded for the backup. The verifier
   extracts into separate scratch storage, checks both databases, and
   requires the regenerated scorecard to match the archived digest.
3. Remove decrypted scratch copies after the check. Keep the encrypted
   original. Maintain a separate recovery-key copy under your control if
   recovery must survive loss of the Windows account or device.

No archive automatically activates a bot or changes its strategy. Storage
is not silently pruned; agree a retention policy before space gets tight.
