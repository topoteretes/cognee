#!/usr/bin/env bash
# Search Cognee's memory (session or permanent graph).
#
# Usage:
#   cognee-search.sh <query> [top_k] [--session | --graph]
#
# --session: search session cache only (default when COGNEE_SESSION_ID is set)
# --graph:   search permanent knowledge graph only
# No flag:   search session first, then graph if empty

set -euo pipefail

DATASET="${COGNEE_PLUGIN_DATASET:-claude_sessions}"
SESSION_ID="${COGNEE_SESSION_ID:-claude_code_session}"
QUERY="${1:-}"
TOP_K="${2:-5}"
MODE="auto"

# Parse flags from any position
for arg in "$@"; do
    case "$arg" in
        --session) MODE="session" ;;
        --graph)   MODE="graph" ;;
    esac
done

if [ -z "$QUERY" ]; then
    echo "Error: no query provided" >&2
    exit 1
fi

if [ "$MODE" = "graph" ]; then
    cognee-cli recall "$QUERY" -d "$DATASET" -k "$TOP_K" -f json 2>/dev/null
elif [ "$MODE" = "session" ]; then
    cognee-cli recall "$QUERY" -s "$SESSION_ID" -k "$TOP_K" -f json 2>/dev/null
else
    # Auto: try session first, fall back to graph
    RESULT=$(cognee-cli recall "$QUERY" -s "$SESSION_ID" -k "$TOP_K" -f json 2>/dev/null)
    if [ -n "$RESULT" ] && [ "$RESULT" != "[]" ]; then
        echo "$RESULT"
    else
        cognee-cli recall "$QUERY" -d "$DATASET" -k "$TOP_K" -f json 2>/dev/null
    fi
fi
