#!/usr/bin/env bash
# Run on the VPS as the deployment user, from an unpacked Git archive.
set -Eeuo pipefail

rev=${1:?usage: rollout-vps-paper.sh FULL_GIT_SHA}
[[ "$rev" =~ ^[0-9a-f]{40}$ ]]
release=$(cd -- "$(dirname -- "$0")/.." && pwd)
previous=/home/ubuntu/hyperliquid-vps-paper
env_file=/etc/hyperliquid-bot/paper/compose.env
paper_data=/srv/hyperliquid-bot/paper
tag="hyperliquid-bot:paper-$rev"
runtime="hyperliquid-bot:vps"
compose=(sudo -n docker compose --project-directory "$previous" --env-file "$env_file"
    -f "$previous/compose.vps.yaml" -f "$previous/compose.vps.postgres.yaml")

verify_paper() {
    sudo -n docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' hyperliquid-bot |
        grep -Fx 'BOT_RUN_MODE=paper' >/dev/null || return 1
    python3 - <<'PY'
import json
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

try:
    response = urlopen("http://127.0.0.1:8080/api/live_ready", timeout=10)
except HTTPError as exc:
    if exc.code != 503:
        raise
    response = exc
except (URLError, TimeoutError):
    raise SystemExit("live-readiness endpoint still warming up")
with response as body:
    readiness = json.load(body)
checks = readiness.get("checks", {})
if checks.get("live_requested") is not False or checks.get("signer_available") is not False:
    raise SystemExit("bot is not confirmed paper-only")
PY
}

verify_paper
sudo -n grep -Fx 'BOT_RUN_MODE=paper' "$env_file" >/dev/null

# These files are local infrastructure and notification settings, not part of the stock rollout.
install -m 0644 "$previous/.dockerignore" "$release/.dockerignore"
install -D -m 0644 "$previous/deploy/vps/launch.py" "$release/deploy/vps/launch.py"
install -m 0644 "$previous/src/notifications/telegram_bot.py" "$release/src/notifications/telegram_bot.py"
sudo -n docker build --build-arg RUN_AS_USER=bot --label "org.opencontainers.image.revision=$rev" -t "$tag" "$release"
sudo -n docker run --rm --network=none --entrypoint python "$tag" -B -c \
    'from src.trading.order_journal import OrderJournal; from src.trading.live_trader import LiveTrader; from scripts.run_stock_paper import AlpacaPaper'

sudo -n install -d -m 0700 /srv/hyperliquid-bot/rollbacks
sudo -n python3 - "$paper_data/bot.db" "/srv/hyperliquid-bot/rollbacks/bot-pre-$rev.db" <<'PY'
from contextlib import closing
from pathlib import Path
import sqlite3, sys

source, dest = map(Path, sys.argv[1:])
if not dest.exists():
    temporary = dest.with_suffix(".tmp")
    if temporary.exists():
        raise SystemExit("previous incomplete backup must be inspected first")
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as src, closing(sqlite3.connect(temporary)) as out:
        src.backup(out, pages=1024, sleep=0.05)
        if out.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise SystemExit("pre-rollout SQLite backup failed quick_check")
    temporary.chmod(0o600)
    temporary.replace(dest)
PY
old_id=$(sudo -n docker inspect --format '{{.Image}}' hyperliquid-bot)
sudo -n docker tag "$old_id" "hyperliquid-bot:pre-$rev"
rollback() {
    printf '%s\n' "Rollout failed; restoring previous image (no database restore attempted)." >&2
    sudo -n docker tag "$old_id" "$runtime"
    "${compose[@]}" up -d --no-build --no-deps --force-recreate bot
}
sudo -n docker tag "$tag" "$runtime"
if ! "${compose[@]}" up -d --no-build --no-deps --force-recreate bot; then
    rollback
    exit 1
fi
ready=false
deadline=$((SECONDS + 180))
while (( SECONDS < deadline )); do
    if [[ $(sudo -n docker inspect --format '{{.State.Health.Status}}' hyperliquid-bot) == healthy ]]; then
        if verify_paper; then
            ready=true
            break
        fi
    fi
    sleep 3
done
if [[ "$ready" != true ]]; then
    rollback
    exit 1
fi

# Update the reporters, leaving the frozen pilot, collector and backup jobs running.
study=/srv/hyperliquid-audit-pilot
sudo -n install -m 0644 "$release/scripts/daily_paper_report.py" "$study/tools/daily_paper_report.py"
for script in paper_exit_experiment paper_exit_scorecard; do
    sudo -n install -m 0644 "$release/scripts/$script.py" "$study/exit-study/scorecard/scripts/$script.py"
done
sudo -n systemctl start hyperliquid-paper-pilot-report.service hyperliquid-exit-scorecard.service

# Separate immutable release and ledger. No broker credentials are provisioned here.
stock=/opt/hyperliquid-stock-paper
sudo -n install -d -m 0700 /etc/hyperliquid-stock-paper
sudo -n install -d -m 0755 "$stock/$rev"
for file in scripts/run_stock_paper.py src/trading/alpaca_paper.py src/trading/order_journal.py; do
    sudo -n install -D -m 0644 "$release/$file" "$stock/$rev/$file"
done
sudo -n ln -sfn "$stock/$rev" "$stock/current"
for unit in hyperliquid-stock-paper.service hyperliquid-stock-paper.timer; do
    sudo -n install -m 0644 "$release/deploy/stock-paper/$unit" "/etc/systemd/system/$unit"
done
sudo -n systemctl daemon-reload
sudo -n systemctl enable --now hyperliquid-stock-paper.timer
sudo -n systemctl start hyperliquid-stock-paper.service
printf '%s\n' "Verified VPS paper release $rev"
