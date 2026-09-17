#!/usr/bin/env bash
# Multi-agent memory handover on REAL Docker Sandboxes (sbx).
#
# Two sandboxes — cognee-supervisor and cognee-worker — are created from the
# cognee-memory kit and share this demo directory as their workspace. The
# cognee state itself lives on each VM's LOCAL disk while a phase runs
# (embedded LanceDB cannot run on the shared virtiofs workspace mount) and is
# handed between sandboxes as a snapshot with `sbx cp` — a literal memory
# handover. The supervisor and worker are separate cognee USERS protected by
# cognee's ACLs, so even though the worker receives the snapshot, it can only
# read/write the datasets it was granted:
#
#   sandbox 1 (cognee-supervisor): brief + grant read/write + emit token
#   sandbox 2 (cognee-worker):     recall briefing, prove denials, report back
#   sandbox 1 (cognee-supervisor): recall the worker's report
#
# Prerequisites (one-time):
#   brew trust docker/tap && brew install docker/tap/sbx
#   sbx daemon start        (own terminal, or nohup)
#   sbx login
#   sbx policy init deny-all
#   sbx secret set-custom --host api.openai.com --env LLM_API_KEY --value "$LLM_API_KEY"
set -euo pipefail
cd "$(dirname "$0")"

KIT="$PWD/../cognee-memory"
SANDBOXES=(cognee-supervisor cognee-worker)
PY=/home/agent/.local/share/uv/tools/cognee/bin/python
STATE=cognee-state                 # canonical snapshot on the host, between phases
SB_STATE=/home/agent/cognee-state  # VM-local working copy, during a phase

# The proxy replaces this placeholder with the real key on requests to
# api.openai.com; the key itself never enters either sandbox.
PLACEHOLDER=$(sbx secret ls | awk '$3 == "LLM_API_KEY" {print $4}' | head -1)
if [ -z "$PLACEHOLDER" ]; then
  echo "No LLM_API_KEY custom secret found. Create it with:" >&2
  echo '  sbx secret set-custom --host api.openai.com --env LLM_API_KEY --value "$LLM_API_KEY"' >&2
  exit 1
fi

mkdir -p handover-out "$STATE"

for name in "${SANDBOXES[@]}"; do
  if ! sbx ls | awk '{print $1}' | grep -qx "$name"; then
    echo "=== creating sandbox: $name (kit install runs inside the VM) ==="
    sbx run shell --kit "$KIT" --name "$name" --detached .
  fi
done

run_phase() {
  echo
  echo "=== sandbox: $1 (phase: $2) ==="
  # Hand the memory snapshot in, run the phase on VM-local disk, hand it back.
  # sbx cp preserves host ownership (your host uid), so re-own it to agent.
  sbx exec "$1" -- sudo rm -rf "$SB_STATE"
  sbx cp "$STATE" "$1":/home/agent/
  sbx exec "$1" -- sudo chown -R agent:agent "$SB_STATE"
  sbx exec "$1" -- sh -lc "
    export LLM_API_KEY=$PLACEHOLDER LOG_LEVEL=ERROR ENABLE_BACKEND_ACCESS_CONTROL=true
    export DATA_ROOT_DIRECTORY=$SB_STATE/data SYSTEM_ROOT_DIRECTORY=$SB_STATE/system
    exec $PY supervisor_worker_handover.py --phase $2 --token-file handover-out/handover_token.json
  "
  rm -rf "$STATE"
  sbx cp "$1":"$SB_STATE" .
}

run_phase cognee-supervisor brief
run_phase cognee-worker work
run_phase cognee-supervisor review

echo
echo "Handover round trip passed across two real sandboxes."
echo "Token exchanged via: $PWD/handover-out/handover_token.json"
echo "Memory snapshot handed over via sbx cp; final state in: $PWD/$STATE"
echo "Inspect policy decisions with: sbx policy log"
echo "Clean up with: sbx rm -f ${SANDBOXES[*]} && rm -rf $STATE handover-out"
