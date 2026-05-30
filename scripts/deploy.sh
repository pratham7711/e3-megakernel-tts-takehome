#!/usr/bin/env bash
#
# deploy.sh -- one-shot Vast.ai start -> sync -> rebuild -> launch flow for the
# e3-megakernel-tts take-home.
#
# Usage:
#   scripts/deploy.sh             # start instance, sync code, rebuild kernel, launch ui_v2
#   scripts/deploy.sh --stop      # stop instance (preserves disk)
#   scripts/deploy.sh --destroy   # destroy instance permanently (asks confirmation)
#   scripts/deploy.sh --no-tunnel # skip the local 8080 port-forward
#
# Requires:
#   - vastai CLI logged in
#   - ~/.ssh/config has a Host block named "e3-vast" pointing at the instance
#   - ~/.ssh/id_ed25519_e3vast key uploaded to the instance
#
# Side-effects on the Mac side:
#   - rewrites the HostName and Port lines inside the `Host e3-vast` block of
#     ~/.ssh/config to match whatever Vast.ai exposes after restart
#   - opens a backgrounded `ssh -fN -L 8080:localhost:8080 e3-vast` tunnel unless
#     --no-tunnel is passed

set -euo pipefail

INSTANCE_ID="${E3_INSTANCE_ID:-38548758}"
SSH_HOST="e3-vast"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_CONFIG="${HOME}/.ssh/config"
REMOTE_PORT=8080

# Prefer the working vastai CLI: the system one at ~/Library/Python/3.9/bin
# is broken because the vastai package uses `match` (Python 3.10+ syntax)
# and Python 3.9 can't parse it. The vastvenv binary at /tmp/vastvenv/bin
# is Python 3.13 and works. Fall back to PATH if neither is present.
if [[ -x "/tmp/vastvenv/bin/vastai" ]]; then
    VASTAI="/tmp/vastvenv/bin/vastai"
elif command -v vastai >/dev/null 2>&1; then
    VASTAI="$(command -v vastai)"
else
    echo "[fatal] no working vastai CLI found" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

ACTION="deploy"
SKIP_TUNNEL=0
for arg in "$@"; do
    case "$arg" in
        --stop)       ACTION="stop" ;;
        --destroy)    ACTION="destroy" ;;
        --no-tunnel)  SKIP_TUNNEL=1 ;;
        -h|--help)
            sed -n '2,21p' "$0"
            exit 0
            ;;
        *)
            echo "unknown arg: $arg" >&2
            exit 2
            ;;
    esac
done

log()  { printf '\033[1;34m[deploy]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[fatal]\033[0m %s\n' "$*" >&2; exit 1; }

require() {
    command -v "$1" >/dev/null 2>&1 || die "missing dependency: $1"
}

# require vastai (uses $VASTAI from above)
require ssh
require rsync
require jq
require sed

# ---------------------------------------------------------------------------
# Stop / destroy paths -- short-circuit before any heavy work.
# ---------------------------------------------------------------------------

if [[ "$ACTION" == "stop" ]]; then
    log "stopping instance ${INSTANCE_ID}"
    "$VASTAI" stop instance "${INSTANCE_ID}"
    log "stop request submitted"
    exit 0
fi

if [[ "$ACTION" == "destroy" ]]; then
    warn "this will PERMANENTLY destroy instance ${INSTANCE_ID} and all of its disk."
    read -r -p "type 'destroy' to confirm: " confirm
    if [[ "$confirm" != "destroy" ]]; then
        die "aborted"
    fi
    "$VASTAI" destroy instance "${INSTANCE_ID}"
    log "destroy request submitted"
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 1: start instance and poll for actual_status=running (5 min cap).
# ---------------------------------------------------------------------------

log "starting Vast instance ${INSTANCE_ID}"
"$VASTAI" start instance "${INSTANCE_ID}" || warn "start returned non-zero (already running?)"

log "polling for actual_status=running"
deadline=$(( $(date +%s) + 300 ))
while :; do
    status_json="$("$VASTAI" show instances --raw 2>/dev/null || echo '[]')"
    status="$(echo "$status_json" | jq -r --argjson id "$INSTANCE_ID" '.[] | select(.id == $id) | .actual_status // "missing"')"
    if [[ "$status" == "running" ]]; then
        log "instance is running"
        break
    fi
    if (( $(date +%s) > deadline )); then
        die "instance did not reach running state within 5 min (last status=${status})"
    fi
    sleep 5
done

# ---------------------------------------------------------------------------
# Step 3: rewrite ~/.ssh/config with new HostName + Port for the e3-vast block.
# (Step 3 is done before step 2 because we can't SSH at all without the right
#  endpoint.)
# ---------------------------------------------------------------------------

new_host="$(echo "$status_json" | jq -r --argjson id "$INSTANCE_ID" '.[] | select(.id == $id) | .ssh_host // empty')"
new_port="$(echo "$status_json" | jq -r --argjson id "$INSTANCE_ID" '.[] | select(.id == $id) | .ssh_port // empty')"

if [[ -z "$new_host" || -z "$new_port" ]]; then
    die "could not read ssh_host / ssh_port from vastai show instances --raw"
fi

log "new SSH endpoint: ${new_host}:${new_port}"

if [[ ! -f "$SSH_CONFIG" ]]; then
    die "no ~/.ssh/config found; create the Host e3-vast block first"
fi

# Surgical in-place update of HostName / Port inside the Host e3-vast block
# only. We use awk so we don't disturb anything else in the file.
backup="${SSH_CONFIG}.bak.$(date +%s)"
cp "$SSH_CONFIG" "$backup"
awk -v host="$new_host" -v port="$new_port" '
    BEGIN { in_block = 0 }
    /^Host[ \t]+/ {
        # Entering a new Host block; mark whether it is e3-vast.
        if ($2 == "e3-vast") { in_block = 1 } else { in_block = 0 }
        print; next
    }
    in_block && /^[ \t]*HostName[ \t]+/ { print "    HostName " host; next }
    in_block && /^[ \t]*Port[ \t]+/     { print "    Port " port; next }
    { print }
' "$backup" > "$SSH_CONFIG"

log "updated ~/.ssh/config (backup at $backup)"

# ---------------------------------------------------------------------------
# Step 2: wait for SSH to actually answer.
# ---------------------------------------------------------------------------

log "waiting for SSH to come up on ${SSH_HOST}"
deadline=$(( $(date +%s) + 300 ))
until ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null "$SSH_HOST" 'echo ok' >/dev/null 2>&1; do
    if (( $(date +%s) > deadline )); then
        die "SSH did not come up within 5 min"
    fi
    sleep 3
done
log "SSH reachable"

# ---------------------------------------------------------------------------
# Step 4: rsync source trees.
# ---------------------------------------------------------------------------

RSYNC_EXCLUDES=(
    --exclude="__pycache__"
    --exclude="*.pyc"
    --exclude=".env"
    --exclude="*.bak"
    --exclude=".git"
    --exclude="*.swp"
)

log "rsync qwen_megakernel_modified/ -> /workspace/qwen_megakernel/"
rsync -az --delete "${RSYNC_EXCLUDES[@]}" \
    "$REPO_ROOT/qwen_megakernel_modified/" \
    "$SSH_HOST:/workspace/qwen_megakernel/"

log "rsync inference-server/ -> /workspace/inference-server/"
rsync -az --delete "${RSYNC_EXCLUDES[@]}" \
    "$REPO_ROOT/inference-server/" \
    "$SSH_HOST:/workspace/inference-server/"

# ---------------------------------------------------------------------------
# Step 5: blow away the JIT cache and force a rebuild.
# ---------------------------------------------------------------------------

log "rebuilding qwen_megakernel JIT extension on remote"
ssh "$SSH_HOST" \
    'rm -rf ~/.cache/torch_extensions/*/qwen_megakernel_C && cd /workspace/qwen_megakernel && python3 -c "import qwen_megakernel; print(\"qwen_megakernel imported OK\")"'

# ---------------------------------------------------------------------------
# Step 6: kill the old ui_v2.py and launch a fresh one detached.
# ---------------------------------------------------------------------------

log "restarting ui_v2.py on remote (port ${REMOTE_PORT})"
ssh "$SSH_HOST" bash <<'EOSSH'
set -e
pkill -f 'python3? .*ui_v2.py' || true
sleep 1
cd /workspace/inference-server
mkdir -p /workspace/logs
nohup env PYTHONPATH=/workspace/qwen_megakernel:/workspace/inference-server \
    python3 ui_v2.py >> /workspace/logs/ui_v2.log 2>&1 &
disown
echo "spawned ui_v2.py pid=$!"
EOSSH

# ---------------------------------------------------------------------------
# Step 7: poll for HTTP 200 on the remote.
# ---------------------------------------------------------------------------

log "polling http://127.0.0.1:${REMOTE_PORT} on remote for HTTP 200"
deadline=$(( $(date +%s) + 180 ))
until ssh "$SSH_HOST" "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:${REMOTE_PORT}" 2>/dev/null | grep -q '^200$'; do
    if (( $(date +%s) > deadline )); then
        warn "ui_v2 did not return HTTP 200 within 3 min; tail of /workspace/logs/ui_v2.log:"
        ssh "$SSH_HOST" "tail -40 /workspace/logs/ui_v2.log" || true
        die "ui_v2 startup failed"
    fi
    sleep 3
done
log "ui_v2 is serving HTTP 200"

# ---------------------------------------------------------------------------
# Step 8: open the local SSH tunnel if it's not already up.
# ---------------------------------------------------------------------------

if [[ "$SKIP_TUNNEL" -eq 0 ]]; then
    if pgrep -f "ssh -fN -L ${REMOTE_PORT}:localhost:${REMOTE_PORT} ${SSH_HOST}" >/dev/null 2>&1; then
        log "local tunnel already open"
    else
        # Kill any stale tunnel pointing at a previous instance.
        pkill -f "ssh.*-L ${REMOTE_PORT}:localhost:${REMOTE_PORT} ${SSH_HOST}" 2>/dev/null || true
        sleep 1
        ssh -fN -L "${REMOTE_PORT}:localhost:${REMOTE_PORT}" "$SSH_HOST" \
            && log "local tunnel up on http://localhost:${REMOTE_PORT}" \
            || warn "failed to open local tunnel; open manually with: ssh -fN -L ${REMOTE_PORT}:localhost:${REMOTE_PORT} ${SSH_HOST}"
    fi
fi

# ---------------------------------------------------------------------------
# Step 9: next-steps banner.
# ---------------------------------------------------------------------------

cat <<EOM

==========================================================================
deploy complete

  remote:   ${new_host}:${new_port}
  service:  ui_v2.py @ /workspace/inference-server (PYTHONPATH set)
  log:      ssh ${SSH_HOST} 'tail -f /workspace/logs/ui_v2.log'

next steps:
  1. open http://localhost:${REMOTE_PORT} in your browser
  2. click "Generate" to validate the megakernel TTS path
  3. when done:     scripts/deploy.sh --stop
     (full delete:  scripts/deploy.sh --destroy)
==========================================================================
EOM
