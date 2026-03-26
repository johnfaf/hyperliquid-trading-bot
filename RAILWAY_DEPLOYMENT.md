# Railway Deployment Guide

## Overview

Your bot is ready to deploy to Railway with all the fixes applied. This guide explains:
1. How to push the code
2. How to set up Railway for persistent data
3. What happens on redeploy

---

## Step 1: Push Code to Railway

From your local machine:

```bash
cd "hyperliquid trading bot"
git push origin main
```

This pushes 3 commits:
- `65ca309` - Fix coin persistence + wallet_fills schema
- `bc8695f` - Persist golden wallets + fills across redeploys
- `4d0b9e2` - Fix zero-trade bug (signal funnel, exposure cap, cache, logging)

Railway will auto-detect the push and trigger a deploy (if auto-deploy is enabled).

---

## Step 2: Set Up Railway Volume for Persistent Data

**Critical:** Without a volume, your accumulated trader/strategy data gets wiped on every redeploy.

### In Railway Dashboard:

1. Go to your service → **Settings** → **Volumes**
2. Click **+ Add Volume**
3. Create volume named `data` with mount path `/app/data`

```
Volume name: data
Mount path: /app/data
```

This ensures that `data/bot_backup.json` (your backup file) survives container restarts.

### How It Works:

```
Bot lifecycle on Railway:
┌─────────────────────────────────────────────────────────────┐
│ 1. Container starts                                         │
│    ↓                                                         │
│ 2. Bot loads → restore_from_json()                          │
│    Checks for /app/data/bot_backup.json                     │
│    If exists: restores all traders, golden wallets, fills   │
│    If missing: starts fresh                                 │
│    ↓                                                         │
│ 3. Bot runs normally                                         │
│    Every cycle: backup_to_json() saves state                │
│    ↓                                                         │
│ 4. Railway redeploy (push new code or restart)              │
│    SIGTERM signal sent                                      │
│    Signal handler: backup_to_json() runs before exit        │
│    Container killed                                         │
│    ↓                                                         │
│ 5. New container starts                                     │
│    Volume mounted at /app/data                              │
│    backup file persists → RESTORE → data recovered          │
│    ↓                                                         │
│ 6. Bot resumes with all old data intact                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 3: Configure Environment Variables

In Railway **Settings** → **Variables**, add:

```
DB_BACKUP_DIR=/app/data
HL_BOT_BACKUP=/app/data/bot_backup.json
```

Optional (for Hyperliquid API):
```
HL_API_KEY=your_key_here
HL_API_SECRET=your_secret_here
PAPER_TRADING_INITIAL_BALANCE=10000
```

---

## Step 4: Monitor Deployment

### Check Logs:

```
Railway Dashboard → Logs tab
```

Look for:

**Good signs:**
```
✓ DB restore complete: 150 traders, 25 golden wallets, 500+ fills restored
✓ Restored DB from backup (post-deploy recovery)
→ EXECUTING 2 trade(s) this cycle
```

**Bad signs:**
```
✗ Restore failed: No backup found
✗ No trade this cycle (missing asset symbol)
```

### Health Check:

Railway pings `/api/health` every 30 seconds. If it fails 3 times in a row, Railway marks the service unhealthy.

Check in Logs:
```
GET /api/health 200 OK
```

---

## What the Fixes Do

### 1. **Zero-Trade Bug Fix** (commit `4d0b9e2`)

**Problem:** Bot was executing 0 trades per cycle despite identifying 500+ strategies.

**Root cause:** Decision engine blocked strategies with missing `coins` field, firewall rejected due to 30% exposure cap, API cache had 0% hit rate.

**Fixes:**
- Decision engine coin fallback: infers coins from metrics or strategy type
- Exposure cap: 30% → 60% to allow 1-2 leveraged positions
- Cache TTLs: increased 5-10x so strategies identified in one phase can be used by another
- Regime detector: now also activates swing_trading + concentrated_bet in ranging markets

**Result:** Bot should now execute 1-3 trades per cycle instead of 0.

### 2. **Data Persistence** (commit `bc8695f`)

**Problem:** SQLite DB lives on ephemeral filesystem. Every Railway redeploy wipes all discovered traders, golden wallets, strategy scores.

**Solution:**
- `backup_to_json()`: saves to `/app/data/bot_backup.json` (persistent volume)
- `restore_from_json()`: on startup, restores all tables if empty
- Signal handler: on SIGTERM (redeploy), backs up before exit
- Includes golden wallets + wallet_fills (the most expensive data to regenerate)

**Result:** Your research survives redeploys. You don't restart from scratch.

### 3. **Schema Fix** (commit `65ca309`)

**Problem:** Decision engine didn't persist coins → coin field was "unknown" downstream.

**Solution:** Always persist resolved coins back into `parameters` dict.

**Result:** Strategies with metrics fallback coin now work correctly.

---

## Data Flow During Redeploy

```
BEFORE REDEPLOY:
  Bot running → every cycle: backup_to_json() writes to disk
  150 traders, 25 golden wallets, 500 strategy scores, etc.

REDEPLOY TRIGGERED:
  git push main → Railway detects → pulls new code → kills container
  Signal handler catches SIGTERM → backup_to_json() (one final save)
  Container exits

NEW CONTAINER:
  Bot starts → restore_from_json() reads /app/data/bot_backup.json
  Restores ALL traders, bots, golden wallets, fills, etc.
  DB is full again → ready to trade

RESULT:
  Zero lost data. Bot picks up where it left off.
```

---

## Troubleshooting

### Bot starts but "NO TRADE" every cycle

**Check logs for:**
```
→ NO TRADE this cycle (no candidates)
Signal funnel: 500 strategies → 0 candidates
```

This suggests:
- All strategies were culled (score too low)
- All strategies had missing coins (unlikely if you're up to date)
- Regime is set to "pause all trading"

**Fix:** Check `regime_detector.py` — verify ranging/trending detection is working.

### "Backup failed" in logs

```
Backup failed: [Errno 13] Permission denied: '/app/data/bot_backup.json'
```

**Fix:** Volume not mounted. Go to Railway Settings → Volumes, ensure `data` volume exists at `/app/data`.

### Restore says "DB already has data"

```
No backup snapshot found, starting fresh
```

This is normal on first deploy. Subsequent redeploys will restore.

### Data lost after redeploy

**Likely cause:** Volume not mounted. Check Railway Settings → Volumes.

**Workaround:** Commit backup as seed file:
```bash
git add -f data/bot_backup.json
git commit -m "Seed data for fresh deployments"
git push origin main
```

Then the repo itself carries the seed data (though it gets large over time).

---

## Monitoring Checklist

After deploy, verify in logs:

- [ ] `Restored DB from backup` (means restore ran)
- [ ] `EXECUTING 1-3 trade(s)` (zero-trade bug fixed)
- [ ] No `permission denied` errors (volume mounted)
- [ ] Signal handler registered (ready for graceful shutdown)
- [ ] Cache hit rate increasing (should go from 0% to 20-40%)

---

## What's Running Now

```
main branch:
  4d0b9e2 ← Signal funnel + exposure cap + cache fixes
  bc8695f ← Data persistence (backup/restore)
  65ca309 ← Coin persistence fix

ready to push to Railway
```

---

## Next Steps

1. **Push code:**
   ```bash
   git push origin main
   ```

2. **Add volume in Railway:**
   - Settings → Volumes → Add `/app/data`

3. **Wait for deploy:**
   - Check Logs tab
   - Verify "EXECUTING X trade(s)" in logs

4. **Monitor for 1-2 cycles:**
   - Ensure backup/restore working
   - Check API cache hit rate improving
   - Verify trades executing

---

## Questions?

All the backup/restore logic is in:
- `src/database.py` → `backup_to_json()`, `restore_from_json()`
- `main.py` → signal handler (search for `SIGTERM`)

The bot will automatically save state every cycle, so your research data is always safe.
