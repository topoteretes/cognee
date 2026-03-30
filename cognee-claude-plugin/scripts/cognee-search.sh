#!/usr/bin/env bash
# Search Cognee's knowledge graph.
# Argument: the search query text.
# Returns JSON results on stdout.

set -euo pipefail

DATASET="${COGNEE_PLUGIN_DATASET:-claude_sessions}"
QUERY="$1"
TOP_K="${2:-5}"

if [ -z "$QUERY" ]; then
    echo "Error: no query provided" >&2
    exit 1
fi

cognee-cli recall "$QUERY" -d "$DATASET" -k "$TOP_K" -f json 2>/dev/null
