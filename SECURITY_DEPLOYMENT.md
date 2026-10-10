# Secure deployment checklist (Render)

- Deploy only a paid always-on service with **one** instance and a persistent disk mounted at `/data`.
- Set `HL_BOT_DB=/data/bot.db`, `DASHBOARD_AUTH_TOKEN` to a long random secret, `LIVE_TRADING_ENABLED=false`, and `DASHBOARD_V2_ENABLED=false` unless you explicitly need it.
- Keep `DASHBOARD_PUBLIC_READ=false`. Do not expose auxiliary ports or metrics unauthenticated.
- Treat `HL_AGENT_PRIVATE_KEY`, Vault tokens and Telegram tokens as secrets; use environment variables or a dedicated secret manager, never tracked files or CI logs.
- If using `SECRET_MANAGER_PROVIDER=vault`, the Vault URL must be HTTPS.
- The Docker image defaults to a non-root `bot` user. **Before deployment, verify that the user can write to the mounted `/data` disk.** Dockerfile-time chown is insufficient if a separately mounted disk overrides /data ownership. Arrange correct ownership via volume setup or your provider's mount options. Do not silently switch the trading service to root if permissions fail.
- Persist and back up the database, kill-switch state and relevant deduplication state. Monitor disk usage; do not assume 10 GB remains sufficient. Restore tests are essential.
- Keep a single live bot instance and verify the exchange's existing orders/positions after any crash or redeploy.
- Before ever enabling live mode, run paper trading, inspect recent execution/kill-switch logs and validate risk controls using a low-risk exchange-approved agent wallet, not a primary wallet.
- Review CI's pip-audit findings for each `requirements.lock` revision; do not suppress advisories without justification.
- Validate the `/api/health` route, v1/v2 login protection, error responses and all critical trading actions before merging.
