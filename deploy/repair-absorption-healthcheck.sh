#!/usr/bin/env bash
# Replace only the public collector's inherited HTTP probe; preserve its pinned code and data.
set -Eeuo pipefail
rev=${1:?usage: repair-absorption-healthcheck.sh FULL_DEPLOYMENT_SHA}
[[ "$rev" =~ ^[0-9a-f]{40}$ ]]
release=$(cd -- "$(dirname -- "$0")/.." && pwd)
name=hyperliquid-absorption
old="$name-pre-health-${rev:0:12}"
state=/var/lib/hyperliquid-absorption
probe="/opt/hyperliquid-absorption-health/$rev/check.py"
check='python -B /healthcheck.py --db /capture/flow.sqlite3'
if [[ $(sudo -n docker inspect "$name" --format '{{json .Config.Healthcheck.Test}}') == "[\"CMD-SHELL\",\"$check\"]" ]]; then
    printf 'Feed freshness probe already installed.\n'
    exit 0
fi
sudo -n docker inspect "$name" | python3 -c '
import json,sys
x=json.load(sys.stdin)[0]
assert x["Config"]["Entrypoint"]==["python"]
assert x["Config"]["Cmd"]==["-B","-m","scripts.capture_hyperliquid_absorption","--db","/capture/flow.sqlite3"]
assert x["Config"]["User"]=="1000:1000" and x["Config"]["WorkingDir"]=="/research"
revision=x["Config"]["Labels"]["org.opencontainers.image.revision"]
assert len(revision)==40 and all(c in "0123456789abcdef" for c in revision)
assert {(m["Source"],m["Destination"],m["RW"]) for m in x["Mounts"]}=={
    ("/opt/hyperliquid-absorption/"+revision,"/research",False),
    ("/var/lib/hyperliquid-absorption/data","/capture",True)}
h=x["HostConfig"]
assert h["ReadonlyRootfs"] and not h["Privileged"] and not h["PortBindings"]
assert h["CapDrop"]==["ALL"] and "no-new-privileges" in h["SecurityOpt"]
assert h["Memory"]==134217728 and h["NanoCpus"]==500000000
assert not any(v.startswith("HL_AGENT_PRIVATE_KEY=") and v.split("=",1)[1] for v in x["Config"]["Env"])
'
frozen=$(sudo -n docker inspect "$name" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')
image=$(sudo -n docker inspect "$name" --format '{{.Image}}')
[[ "$image" =~ ^sha256:[0-9a-f]{64}$ ]]
if sudo -n docker inspect "$old" >/dev/null 2>&1; then
    printf 'Existing rollback container requires inspection.\n' >&2
    exit 1
fi
sudo -n install -D -m 0644 "$release/scripts/check_absorption_health.py" "$probe"
sudo -n install -d -m 0700 "$state/backups"
sudo -n python3 - "$state" "$rev" <<'PY'
from contextlib import closing
from pathlib import Path
import sqlite3,sys
root=Path(sys.argv[1])
destination=root/"backups"/("flow-pre-health-"+sys.argv[2]+".sqlite3")
if destination.exists():
    raise SystemExit("existing feed backup requires inspection")
with closing(sqlite3.connect((root/"data/flow.sqlite3").as_uri()+"?mode=ro",uri=True)) as source, closing(sqlite3.connect(destination)) as target:
    source.backup(target,pages=1024,sleep=.05)
    if target.execute("PRAGMA quick_check").fetchone() != ("ok",):
        raise SystemExit("feed backup integrity check failed")
destination.chmod(0o600)
PY
sudo -n docker rename "$name" "$old"
rollback() {
    sudo -n docker stop "$name" >/dev/null 2>&1 || true
    sudo -n docker rename "$name" "$name-failed-health-${rev:0:12}" 2>/dev/null || true
    sudo -n docker rename "$old" "$name"
    sudo -n docker start "$name" >/dev/null
}
trap rollback ERR
sudo -n docker create --name "$name" --restart unless-stopped --read-only --cap-drop ALL \
    --security-opt no-new-privileges --memory 128m --cpus .50 --tmpfs /tmp:rw,nosuid,nodev,mode=1777 \
    --user 1000:1000 --env HOME=/tmp --workdir /research \
    --mount "type=bind,src=/opt/hyperliquid-absorption/$frozen,dst=/research,readonly" \
    --mount "type=bind,src=$state/data,dst=/capture" \
    --mount "type=bind,src=$probe,dst=/healthcheck.py,readonly" \
    --health-cmd "$check" --health-interval 30s --health-timeout 5s --health-retries 3 --health-start-period 90s \
    --label "org.opencontainers.image.revision=$frozen" --entrypoint python "$image" \
    -B -m scripts.capture_hyperliquid_absorption --db /capture/flow.sqlite3 >/dev/null
sudo -n docker stop "$old" >/dev/null
sudo -n docker start "$name" >/dev/null
deadline=$((SECONDS+180))
ready=false
while (( SECONDS < deadline )); do
    if [[ $(sudo -n docker inspect "$name" --format '{{.State.Health.Status}}') == healthy ]]; then
        ready=true
        break
    fi
    sleep 3
done
[[ "$ready" == true ]]
trap - ERR
printf 'Feed healthy; frozen research revision preserved: %s. Rollback container retained.\n' "$frozen"
