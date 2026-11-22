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
base_compose=(sudo -n docker compose --project-directory "$previous" --env-file "$env_file"
    -f "$previous/compose.vps.yaml" -f "$previous/compose.vps.postgres.yaml")
compose=("${base_compose[@]}" -f "$release/deploy/dashboard/hyperliquid-dashboard.compose.yml")

verify_paper() {
    sudo -n docker exec -i hyperliquid-bot python - <<'PY'
import os
import config
if (os.getenv("BOT_RUN_MODE") != "paper" or config.LIVE_TRADING_ENABLED
        or config.LIGHTER_LIVE_TRADING_ENABLED or bool(os.getenv("HL_AGENT_PRIVATE_KEY"))):
    raise SystemExit("paper-only configuration or signer absence not confirmed")
PY
    if [[ "${1:-runtime}" == configuration ]]; then
        return
    fi
    python3 - <<'PY'
import json
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

try:
    response = urlopen("http://127.0.0.1:8080/api/execution-mode", timeout=5)
except HTTPError as exc:
    if exc.code != 503:
        raise
    response = exc
except (URLError, TimeoutError):
    raise SystemExit("execution-mode endpoint still warming up")
with response as body:
    readiness = json.load(body)
if readiness.get("paper_only") is not True or readiness.get("live_requested") is not False or readiness.get("signer_available") is not False:
    raise SystemExit("bot is not confirmed paper-only")
if readiness.get("live_entry_hold") != "strategy_evidence_human_approval_required":
    raise SystemExit("crypto live entry hold not confirmed")
PY
}

verify_paper configuration
sudo -n grep -Fx 'BOT_RUN_MODE=paper' "$env_file" >/dev/null
stock_env=/etc/hyperliquid-stock-paper/paper.env
if sudo -n test -e "$stock_env"; then
    stock_orders=$(sudo -n grep -E '^[[:space:]]*STOCK_PAPER_SEND_ORDERS[[:space:]]*=' "$stock_env" || true)
    if [[ -n "$stock_orders" && "$stock_orders" != STOCK_PAPER_SEND_ORDERS=false ]]; then
        printf '%s\n' "Stock shadow-only configuration was not confirmed." >&2
        exit 1
    fi
fi

# These files are local infrastructure and notification settings, not part of the stock rollout.
install -m 0644 "$previous/.dockerignore" "$release/.dockerignore"
install -D -m 0644 "$previous/deploy/vps/launch.py" "$release/deploy/vps/launch.py"
install -m 0644 "$previous/src/notifications/telegram_bot.py" "$release/src/notifications/telegram_bot.py"
sudo -n docker build --build-arg RUN_AS_USER=bot --build-arg "BOT_BUILD_REVISION=$rev" \
    --label "org.opencontainers.image.revision=$rev" -t "$tag" "$release"
sudo -n docker run --rm --network=none --entrypoint python "$tag" -B -c \
    'from src.trading.order_journal import OrderJournal; from src.trading.live_trader import LiveTrader; from scripts.run_stock_paper import AlpacaPaper; from src.ui.v2.app import create_app; create_app()'

sudo -n install -d -m 0700 /srv/hyperliquid-bot/rollbacks
sudo -n python3 - "$paper_data" "/srv/hyperliquid-bot/rollbacks" "$rev" <<'PY'
from contextlib import closing
from pathlib import Path
import shutil, sqlite3, sys

data, rollbacks = map(Path, sys.argv[1:3])
revision = sys.argv[3]
required = sum(path.stat().st_size for path in (data / "bot.db", data / "bot.decisions.sqlite3",
               data / "bot.orders.sqlite3", data / "bot.risk.sqlite3", data / "bot.funding.sqlite3",
               Path("/var/lib/hyperliquid-stock-paper/orders.sqlite3")) if path.exists())
if shutil.disk_usage(rollbacks).free < required*2 + 2*1024**3:
    raise SystemExit("not enough space for backups, isolated recovery verification, and reserve")
for kind, filename in (("bot", "bot.db"), ("decisions", "bot.decisions.sqlite3"), ("orders", "bot.orders.sqlite3"),
                       ("risk", "bot.risk.sqlite3"), ("funding", "bot.funding.sqlite3"),
                       ("stock", "/var/lib/hyperliquid-stock-paper/orders.sqlite3")):
    source = data / filename
    dest = rollbacks / f"{kind}-pre-{revision}.db"
    if dest.exists() or (kind != "bot" and not source.exists()):
        continue
    temporary = dest.with_suffix(".tmp")
    if temporary.exists():
        raise SystemExit("previous incomplete backup must be inspected first")
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as src, closing(sqlite3.connect(temporary)) as out:
        # Pin one WAL snapshot so new bot writes cannot restart the large backup.
        src.execute("BEGIN")
        src.execute("SELECT count(*) FROM sqlite_master").fetchone()
        src.backup(out, pages=1024, sleep=0.05)
        src.rollback()
        if out.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise SystemExit("pre-rollout SQLite backup failed quick_check")
    temporary.chmod(0o600)
    temporary.replace(dest)
PY
old_id=$(sudo -n docker inspect --format '{{.Image}}' hyperliquid-bot)
rollback_compose=("${base_compose[@]}")
if sudo -n docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' hyperliquid-bot |
    grep -Fx 'DASHBOARD_V2_PRIMARY=true' >/dev/null; then
    rollback_compose=("${compose[@]}")
fi
sudo -n docker tag "$old_id" "hyperliquid-bot:pre-$rev"
rollback() {
    printf '%s\n' "Rollout failed; restoring previous image (no database restore attempted)." >&2
    sudo -n docker tag "$old_id" "$runtime"
    "${rollback_compose[@]}" up -d --no-build --no-deps --force-recreate bot
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
        if verify_paper && curl -s --max-time 4 -D - -o /dev/null http://127.0.0.1:8080/decision-coverage |
            grep -qi '^location: /login'; then
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

# Install independent funding capture and a credential-free forward book.
shadow=/opt/hyperliquid-forward-shadow
for file in scripts/run_forward_shadow.py scripts/fetch_hyperliquid_replay_market.py src/backtest/candle_exit_replay.py; do
    sudo -n install -D -m 0644 "$release/$file" "$shadow/$rev/$file"
done
if ! sudo -n test -e "$shadow/current"; then
    sudo -n ln -s "$shadow/$rev" "$shadow/current"
fi
for unit in hyperliquid-paper-funding.service hyperliquid-paper-funding.timer \
    hyperliquid-forward-shadow.service hyperliquid-forward-shadow.timer; do
    sudo -n install -m 0644 "$release/deploy/research/$unit" "/etc/systemd/system/$unit"
done
sudo -n systemctl daemon-reload
sudo -n systemctl start hyperliquid-paper-funding.service hyperliquid-forward-shadow.service
sudo -n systemctl enable --now hyperliquid-paper-funding.timer hyperliquid-forward-shadow.timer

# Update the reporters, leaving the frozen pilot, collector and backup jobs running.
if sudo -n test -f /var/lib/hyperliquid-absorption/data/flow.sqlite3; then
    reporter=/opt/hyperliquid-absorption-report
    for file in scripts/score_hyperliquid_absorption.py src/backtest/absorption_study.py \
        src/backtest/candle_entry_replay.py src/backtest/candle_exit_replay.py src/signals/flow_absorption.py; do
        sudo -n install -D -m 0644 "$release/$file" "$reporter/$rev/$file"
    done
    sudo -n ln -sfn "$reporter/$rev" "$reporter/current"
    sudo -n install -m 0644 "$release/deploy/absorption/hyperliquid-absorption-report.service" /etc/systemd/system/
    sudo -n systemctl daemon-reload
    sudo -n systemctl start hyperliquid-absorption-report.service
fi
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
for file in scripts/run_stock_paper.py scripts/stock_pilot_report.py src/trading/alpaca_paper.py \
    src/trading/order_journal.py src/trading/equity_risk.py src/trading/stock_evidence.py src/core/clock_provider.py src/backtest/replay/clock.py; do
    sudo -n install -D -m 0644 "$release/$file" "$stock/$rev/$file"
done
sudo -n ln -sfn "$stock/$rev" "$stock/current"
for unit in hyperliquid-stock-paper.service hyperliquid-stock-paper.timer hyperliquid-stock-paper-drill@.service \
    hyperliquid-stock-paper-report.service hyperliquid-stock-paper-report.timer; do
    sudo -n install -m 0644 "$release/deploy/stock-paper/$unit" "/etc/systemd/system/$unit"
done
sudo -n systemctl daemon-reload
sudo -n systemctl enable --now hyperliquid-stock-paper.timer
sudo -n systemctl start hyperliquid-stock-paper.service
sudo -n systemctl enable --now hyperliquid-stock-paper-report.timer
sudo -n systemctl start hyperliquid-stock-paper-report.service

sudo -n install -D -m 0644 "$release/scripts/publish_dashboard_status.py" /opt/hyperliquid-dashboard/publish_dashboard_status.py
sudo -n install -D -m 0644 "$release/scripts/recovery_bundle.py" /opt/hyperliquid-recovery/recovery_bundle.py
for file in src/trading/alpaca_paper.py src/trading/order_journal.py src/trading/stock_evidence.py \
    src/core/clock_provider.py src/backtest/replay/clock.py src/trading/trade_decision_memory.py src/trading/equity_risk.py \
    src/trading/copy_shadow_report.py src/trading/paper_reporting.py src/trading/paper_funding.py \
    src/trading/paper_funding_health.py src/data/decision_coverage.py; do
    sudo -n install -D -m 0644 "$release/$file" "/opt/hyperliquid-dashboard/$file"
done
for unit in hyperliquid-dashboard-status.service hyperliquid-dashboard-status.timer; do
    sudo -n install -m 0644 "$release/deploy/dashboard/$unit" "/etc/systemd/system/$unit"
done
sudo -n systemctl daemon-reload
sudo -n systemctl enable --now hyperliquid-dashboard-status.timer
sudo -n systemctl start hyperliquid-dashboard-status.service
printf '%s\n' "Verified VPS paper release $rev"
