#!/usr/bin/env bash
# Run on the VPS from an unpacked Git archive. No trading services are touched.
set -euo pipefail

revision=${1:?usage: install-vps-history.sh FULL_GIT_SHA}
[[ "$revision" =~ ^[0-9a-f]{40}$ ]]
release=$(cd -- "$(dirname -- "$0")/.." && pwd)
base=/opt/hyperliquid-history
destination="$base/$revision"

sudo -n install -d -m 0755 "$base" "$destination" "$destination/scripts" "$destination/src" \
    "$destination/src/backtest" "$destination/src/data" "$destination/src/trading"
for file in scripts/import_trading_history.py scripts/run_copy_walk_forward.py src/backtest/history_archive.py \
    src/data/decision_coverage.py src/trading/copy_shadow_report.py src/trading/trade_decision_memory.py; do
    sudo -n install -D -m 0644 "$release/$file" "$destination/$file"
done

cd "$destination"
/usr/bin/python3 -B -m scripts.import_trading_history --help >/dev/null
/usr/bin/python3 -B -m scripts.run_copy_walk_forward --help >/dev/null
sudo -n ln -sfnT "$destination" "$base/current"
printf '%s\n' "Installed offline history tools $revision (no trading services restarted)"
