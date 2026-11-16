#!/usr/bin/env bash
# Run from a private Git release archive, with only public market data mounted as input.
set -euo pipefail

input=$(realpath -e -- "${1:?usage: run-vps-market-replay.sh MARKET_SQLITE NEW_REPORT_DIR}")
output=$(realpath -m -- "${2:?usage: run-vps-market-replay.sh MARKET_SQLITE NEW_REPORT_DIR}")
release=$(cd -- "$(dirname -- "$0")/.." && pwd)
[[ -f "$input" ]]
for dir in "$input" "$output" "$release"; do
    [[ "$dir" != *,* && "$dir" != /srv/hyperliquid-bot && "$dir" != /srv/hyperliquid-bot/* ]]
done
mkdir -m 0700 -- "$output"
image=$(sudo -n docker inspect hyperliquid-bot --format '{{.Image}}')
sudo -n docker run --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges \
    --tmpfs /tmp:rw,nosuid,nodev,mode=1777 --user "$(id -u):$(id -g)" --env HOME=/tmp \
    --mount "type=bind,src=$release,dst=/replay,readonly" \
    --mount "type=bind,src=$input,dst=/market.sqlite3,readonly" \
    --mount "type=bind,src=$output,dst=/reports" --workdir /replay --entrypoint python "$image" \
    -B -m scripts.run_hyperliquid_exit_replay --market /market.sqlite3 \
    --report /reports/exit-test.md --trace /reports/exit-test.jsonl
