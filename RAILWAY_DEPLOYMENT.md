# Railway Deployment Guide

## Prerequisites

- A Railway account with a project created
- Git repo pushed to GitHub (Railway auto-deploys from pushes)
- The bot runs from the `Dockerfile` at the repo root

## 1. Push Code

```bash
cd "hyperliquid trading bot"
git push origin main
```

Railway auto-detects the Dockerfile and triggers a build.

## 2. Add a Persistent Volume

Without a volume, the SQLite database and all discovered trader/strategy data get wiped on every redeploy.

In the Railway dashboard:

1. Go to your service, then Settings, then Volumes.
2. Click Add Volume.
3. Set the mount path to `/data`.

```
Volume name: data
Mount path:  /data
```

The bot auto-detects `/data` at startup. When the volume is writable, `DB_PATH` resolves to `/data/bot.db` and backups go to `/data/bot_backup.json`.

## 3. Set Environment Variables

In Settings, then Variables:

```
# Required for persistence (auto-detected if /data is writable, but explicit is safer)
HL_BOT_DB=/data/bot.db

# Optional overrides
TRADING_CYCLE_INTERVAL=300         # 5 min (lower for more frequent trading)
DISCOVERY_CYCLE_INTERVAL=86400     # 24h
LOG_FORMAT=json                     # structured logs for Railway log viewer
```

No API keys are required for the research/paper-trading features. Hyperliquid's public info endpoint is unauthenticated.

Optional keys for enhanced features:

```
ARKHAM_API_KEY=...          # Arkham Intelligence smart-money flow
TELEGRAM_BOT_TOKEN=...     # Telegram alerts
TELEGRAM_CHAT_ID=...       # Telegram chat for notifications
```

## 4. How Persistence Works

```
Container starts
  │
  ▼
boot.py → init_db() → restore_from_json(/data/bot_backup.json)
  │   If backup exists and DB is empty: restores traders, golden wallets,
  │   fills, strategies, experiments.
  │   If no backup: starts fresh.
  │
  ▼
Bot runs normally
  │   Every cycle: backup_to_json() saves state to /data/bot_backup.json
  │
  ▼
Redeploy triggered (git push or manual restart)
  │   SIGTERM → signal handler calls backup_to_json() before exit
  │   Container killed
  │
  ▼
New container starts
  │   Volume still mounted at /data with bot_backup.json intact
  │   restore_from_json() recovers all data
  │
  ▼
Bot resumes with full history
```

The backup includes: traders, golden wallets, wallet fills, strategies, strategy scores, experiments, and the paper account.

## 5. Health Check

The Dockerfile configures a health check that pings `http://localhost:8080/api/health` every 30 seconds. Railway marks the service unhealthy after 3 consecutive failures.

Check the Logs tab for:

```
GET /api/health 200 OK
```

## 6. Verifying a Deploy

After pushing, watch the logs for these milestones:

```
[boot] Dependency Boot Report
  ✓ core       READY
  ✓ backtester READY
  ...

[database] Restored DB from backup: N traders, M golden wallets, K fills

[trading_cycle] Phase 1: Regime Detection
[trading_cycle] Phase 2: Strategy Scoring (N strategies)
[trading_cycle] Phase 3: Signal Processing
[trading_cycle] EXECUTING X trade(s) this cycle
```

If you see `EXECUTING 0 trade(s)` on every cycle after the first few hours, run `python scripts/diagnose_rejections.py` locally to identify which firewall check is blocking signals.

## 7. Monitoring

### Key Metrics to Watch

- **Trades per cycle**: Should be 0-3. Sustained 0 after bootstrap means the pipeline is too restrictive.
- **Firewall pass rate**: Visible in the dashboard at `/`. Healthy range: 5-20% of signals.
- **Paper PnL**: Track via `python main.py --status` or the dashboard.
- **Discovery cycle**: Should log new traders every 24h. If stuck at 0 traders, check API rate limits.

### Log Queries (Railway Log Viewer)

Search for:

- `EXECUTING` — trades taken this cycle
- `signal_rejected` — firewall rejections (audit trail)
- `golden_wallet_connected` — new golden wallets found
- `bot_detected` — traders flagged as bots
- `CRASH REGIME` — predictive de-risking activated

## 8. Troubleshooting

### "Backup failed: Permission denied"

Volume not mounted. Go to Settings, then Volumes, and verify `/data` exists.

### "No backup snapshot found, starting fresh"

Normal on first deploy. Subsequent redeploys restore from the backup.

### "MISSING websocket, eth_account"

The Dockerfile installs from `requirements.txt` which includes these. If you see this locally, run `pip install -r requirements.txt`.

### Data lost after redeploy

The volume wasn't mounted before the first cycle ran. Add the volume, then either:

- Wait for the next discovery cycle to rebuild, or
- Seed with sample data: `python scripts/seed_and_replay.py --seed-only`

### Bot stuck at "0 strategies"

The discovery cycle runs every 24h by default. For faster bootstrap, set `DISCOVERY_CYCLE_INTERVAL=3600` (1h) temporarily, or run `python main.py --bootstrap` locally and push the seeded DB.

## 9. File Paths on Railway

| Path | Contents | Persistent? |
|------|----------|-------------|
| `/app/` | Application code (from Dockerfile COPY) | No (rebuilt each deploy) |
| `/data/bot.db` | SQLite database | Yes (volume) |
| `/data/bot_backup.json` | JSON backup for recovery | Yes (volume) |
| `/app/logs/` | Log files | No (use Railway log viewer) |
| `/app/reports/` | Generated reports | No (view via dashboard) |
| `/app/models/` | ML models (XGBoost) | No (retrained from DB data) |
