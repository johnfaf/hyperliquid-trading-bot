#!/usr/bin/env bash
# Run from a private Git release archive, with only public market data mounted as input.
set -euo pipefail

input=$(realpath -e -- "${1:?usage: run-vps-market-replay.sh MARKET_SQLITE NEW_REPORT_DIR [exit|entry|failed-breakout]}")
output=$(realpath -m -- "${2:?usage: run-vps-market-replay.sh MARKET_SQLITE NEW_REPORT_DIR [exit|entry|failed-breakout]}")
release=$(cd -- "$(dirname -- "$0")/.." && pwd)
signal_args=()
case "${3:-exit}" in
    exit) module=scripts.run_hyperliquid_exit_replay; artifact=exit-test ;;
    entry) module=scripts.run_hyperliquid_entry_replay; artifact=entry-test ;;
    failed-breakout) module=scripts.run_hyperliquid_entry_replay; artifact=failed-breakout; signal_args=(--signal failed-breakout) ;;
    *) printf 'Unknown replay type (use exit, entry, or failed-breakout).\n' >&2; exit 2 ;;
esac
shift 2
if (( $# )); then shift; fi
if (( $# )) && [[ "$artifact" == exit-test || $# != 2 || "$1" != --forward-from ]]; then
    printf 'The only optional argument for entry studies is --forward-from UTC_CUTOFF.\n' >&2
    exit 2
fi
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
    -B -m "$module" --market /market.sqlite3 \
    --report "/reports/$artifact.md" --trace "/reports/$artifact.jsonl" "${signal_args[@]}" "$@"
