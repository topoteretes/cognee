#!/usr/bin/env bash
# brainbox: run a task in a disposable Docker Sandbox with scoped, revocable
# access to a central cognee brain.
#
#   ./run.sh code-architecture --repo <path> --read <dataset> [--read <dataset>]...
#            [--question "..."] [--keep] [--purge]
#
# What happens (each step is one of the "walls" in the talk):
#   1. mint     a cognee agent identity on the brain, read-granted on exactly the
#               --read datasets and explicitly denied on every other one
#   2. sandbox  create it from ../cognee-memory-remote (deny-all + kit allowlist),
#               admit it to ONE host port with a sandbox-scoped policy rule
#   3. secret   store the agent key as a sandbox-scoped proxy-managed secret;
#               the VM only ever sees the placeholder
#   4. run      the task payload (recall -> local code graph -> push)
#   5. walls    show the placeholder, a blocked egress, a denied dataset, the log
#   6. revoke   drop the sandbox, the secret, the rule, and the agent's grants
#               (--keep leaves everything up; --purge also deletes the agent and
#               the dataset it wrote — otherwise that dataset stays for review)
#
# Prerequisites: sbx installed + `sbx login`, `sbx policy init deny-all`, a
# running cognee API (BRAIN_URL, default http://127.0.0.1:8011) and the owner's
# key at ~/.cognee-plugin/api_key.json (or BRAIN_OWNER_KEY / BRAIN_OWNER_KEY_FILE).
set -euo pipefail
cd "$(dirname "$0")"

BRAIN_URL="${BRAIN_URL:-http://127.0.0.1:8011}"
PORT="${BRAIN_URL##*:}"; PORT="${PORT%%/*}"
SANDBOX_HOST="${BRAIN_SANDBOX_HOST:-host.docker.internal}"
KIT="$PWD/../cognee-memory-remote"
PY_IN_SANDBOX='$HOME/.local/share/uv/tools/cognee/bin/python'

TASK="${1:-}"; shift || true
[ -n "$TASK" ] && [ -f "tasks/${TASK//-/_}.py" ] || { echo "usage: $0 <task> --repo <path> --read <dataset>..." >&2; ls tasks/*.py >&2; exit 2; }

REPO=""; READS=(); QUESTION=""; KEEP=0; PURGE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --read) READS+=("$2"); shift 2 ;;
    --question) QUESTION="$2"; shift 2 ;;
    --keep) KEEP=1; shift ;;
    --purge) PURGE=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$REPO" ] && [ -d "$REPO" ] || { echo "--repo must be a directory" >&2; exit 2; }
[ ${#READS[@]} -gt 0 ] || { echo "at least one --read dataset is required" >&2; exit 2; }

NAME="brainbox-$(printf '%s' "$TASK" | tr -c 'a-z0-9\n' '-')-$(openssl rand -hex 3)"
WORK="$PWD/work/$NAME"
OUT_DATASET="$NAME-out"
mkdir -p "$WORK/out" "$WORK/tasks"
rsync -a --exclude .git "$REPO"/ "$WORK/repo/"
cp tasks/*.py "$WORK/tasks/"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

AGENT_ID=""
cleanup() {
  [ "$KEEP" = 1 ] && { say "kept: sandbox $NAME, agent $AGENT_ID, dataset $OUT_DATASET"; return; }
  say "revoke"
  sbx secret rm COGNEE_API_KEY --sandbox "$NAME" -f >/dev/null 2>&1 || true
  sbx policy rm network --sandbox "$NAME" --resource "localhost:$PORT" --force >/dev/null 2>&1 || true
  sbx rm -f "$NAME" >/dev/null 2>&1 && echo "  sandbox removed" || true
  if [ -n "$AGENT_ID" ]; then
    if [ "$PURGE" = 1 ]; then
      python3 brain_admin.py --url "$BRAIN_URL" purge --agent "$AGENT_ID" | sed 's/^/  /'
    else
      python3 brain_admin.py --url "$BRAIN_URL" revoke --agent "$AGENT_ID" | sed 's/^/  /'
      echo "  output dataset '$OUT_DATASET' kept for review (owner: the revoked identity)"
    fi
  fi
}
trap cleanup EXIT

say "1/6 mint identity on the brain: $NAME  (read: ${READS[*]})"
MINT=$(python3 brain_admin.py --url "$BRAIN_URL" mint --name "$NAME" --read "${READS[@]}")
AGENT_ID=$(printf '%s' "$MINT" | python3 -c 'import json,sys;print(json.load(sys.stdin)["agent_id"])')
AGENT_KEY=$(printf '%s' "$MINT" | python3 -c 'import json,sys;print(json.load(sys.stdin)["api_key"])')
printf '%s' "$MINT" | python3 -c 'import json,sys;d=json.load(sys.stdin);d["api_key"]="<never written to disk>";json.dump(d,open(sys.argv[1],"w"),indent=2)' "$WORK/agent.json"
echo "  agent $AGENT_ID: read on ${#READS[@]} dataset(s), denied on $(printf '%s' "$MINT" | python3 -c 'import json,sys;print(json.load(sys.stdin)["denied"])') others"

say "2/6 create sandbox from kit (deny-all + kit allowlist), admit localhost:$PORT for this sandbox only"
sbx run shell --kit "$KIT" --name "$NAME" --detached "$WORK" | tail -1
sbx policy allow network --sandbox "$NAME" "localhost:$PORT" | tail -1

say "3/6 store the agent key as a proxy-managed secret (the VM sees only a placeholder)"
PLACEHOLDER=$(sbx secret set-custom --sandbox "$NAME" --host localhost --host "$SANDBOX_HOST" \
  --env COGNEE_API_KEY --value "$AGENT_KEY" | sed -n 's/.*placeholder: //p' | head -1)
unset AGENT_KEY
echo "  placeholder: $PLACEHOLDER"

SANDBOX_ENV="export PATH=\$HOME/.local/bin:\$PATH LOG_LEVEL=ERROR TELEMETRY_DISABLED=1 \
  COGNEE_SERVICE_URL=http://$SANDBOX_HOST:$PORT COGNEE_API_KEY=$PLACEHOLDER"

say "4/6 run task '$TASK' inside the sandbox"
QARG=""; [ -n "$QUESTION" ] && QARG="--question $(printf '%q' "$QUESTION")"
sbx exec "$NAME" -- sh -lc "$SANDBOX_ENV; $PY_IN_SANDBOX $WORK/tasks/${TASK//-/_}.py \
  --repo $WORK/repo --target $OUT_DATASET --out-dir $WORK/out $QARG" \
  2>&1 | sed '/already exists, skipping/d'

say "5/6 the walls"
echo "  credential wall — what the VM holds:"
sbx exec "$NAME" -- sh -lc "$SANDBOX_ENV; echo \"    COGNEE_API_KEY=\$COGNEE_API_KEY\""
echo "  compute wall — egress outside the allowlist:"
sbx exec "$NAME" -- sh -lc 'curl -s -m 8 -o /dev/null -w "    https://example.com -> HTTP %{http_code}\n" https://example.com || echo "    https://example.com -> blocked"'
DENIED_ID=$(python3 brain_admin.py --url "$BRAIN_URL" datasets | python3 -c '
import json,sys; ds=json.load(sys.stdin); reads=set(sys.argv[1:])
for d in ds:
    if d["name"] not in reads and d["id"] not in reads and "brainbox" not in d["name"]: print(d["id"], d["name"]); break' "${READS[@]}")
if [ -n "$DENIED_ID" ]; then
  echo "  memory wall — a dataset this identity was not granted (${DENIED_ID#* }):"
  sbx exec "$NAME" -- sh -lc "$SANDBOX_ENV; curl -s -m 60 -H \"X-Api-Key: \$COGNEE_API_KEY\" -H 'Content-Type: application/json' \
    -d '{\"query\":\"anything\",\"datasetIds\":[\"${DENIED_ID%% *}\"],\"searchType\":\"CHUNKS\",\"topK\":1}' \
    -o /dev/null -w '    recall -> HTTP %{http_code}\n' \$COGNEE_SERVICE_URL/api/v1/recall"
fi
echo "  audit trail — sbx policy log (this sandbox):"
sbx policy log 2>/dev/null | awk -v name="$NAME" '
  /^Blocked requests/ { verdict = "DENIED" } /^Allowed requests/ { verdict = "ALLOWED" }
  $1 == name { printf "    %-8s %-30s x%s\n", verdict, $3, $NF }' | sort -u | head -14

say "6/6 results"
echo "  outputs:        $WORK/out  (architecture.html, architecture.mmd, recall.json, report.json)"
echo "  brain dataset:  $OUT_DATASET  (owned by $NAME; review + promote from the workspace UI)"
