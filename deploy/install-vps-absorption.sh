#!/usr/bin/env bash
# Install a pinned public-feed research worker. Never touch the trading service or ledger.
set -euo pipefail

rev=${1:?usage: install-vps-absorption.sh FULL_GIT_REVISION}
[[ "$rev" =~ ^[0-9a-f]{40}$ ]]
release=$(cd -- "$(dirname -- "$0")/.." && pwd)
name=hyperliquid-absorption
base=/opt/hyperliquid-absorption
target="$base/$rev"
state=/var/lib/hyperliquid-absorption
if existing=$(sudo -n docker inspect "$name" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' 2>/dev/null); then
    [[ "$existing" == "$rev" ]] || { printf 'Existing research revision is frozen; no replacement performed.\n' >&2; exit 1; }
fi
sudo -n install -d -m 0755 "$base" "$target" "$target/scripts" "$target/src" \
    "$target/src/backtest" "$target/src/data" "$target/src/signals"
sudo -n install -d -m 0700 "$state" "$state/reports"
sudo -n install -d -m 0700 -o "$(id -u)" -g "$(id -g)" "$state/data"
for file in scripts/capture_hyperliquid_absorption.py scripts/score_hyperliquid_absorption.py \
    src/data/absorption_feed.py src/signals/flow_absorption.py src/backtest/absorption_study.py \
    src/backtest/candle_entry_replay.py src/backtest/candle_exit_replay.py; do
    sudo -n install -m 0644 "$release/$file" "$target/$file"
done
image=$(sudo -n docker inspect hyperliquid-bot --format '{{.Image}}')
runtime="hyperliquid-absorption-runtime:$rev"
sudo -n docker tag "$image" "$runtime"
if [[ -z "${existing:-}" ]]; then
    sudo -n docker run -d --name "$name" --restart unless-stopped --read-only --cap-drop ALL \
        --security-opt no-new-privileges --memory 128m --cpus .50 --tmpfs /tmp:rw,nosuid,nodev,mode=1777 \
        --user "$(id -u):$(id -g)" --env HOME=/tmp --mount "type=bind,src=$target,dst=/research,readonly" \
        --mount "type=bind,src=$state/data,dst=/capture" --workdir /research \
        --label "org.opencontainers.image.revision=$rev" --entrypoint python "$runtime" \
        -B -m scripts.capture_hyperliquid_absorption --db /capture/flow.sqlite3 >/dev/null
fi
sudo -n ln -sfnT "$target" "$base/current"
for unit in "$name-report.service" "$name-report.timer"; do
    sudo -n install -m 0644 "$release/deploy/absorption/$unit" "/etc/systemd/system/$unit"
done
sudo -n systemctl daemon-reload
sudo -n systemctl enable --now "$name-report.timer"
printf 'Read-only absorption research installed: %s\n' "$rev"
