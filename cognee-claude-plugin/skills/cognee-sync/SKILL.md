---
name: cognee-sync
description: Sync session cache entries into the permanent Cognee knowledge graph. Run this to make session memory searchable, or it runs automatically at session end.
---

# Sync Session to Permanent Graph

Bridge session cache entries into the permanent knowledge graph.

## Instructions

Run the sync script:

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/sync-session-to-graph.py
```

Or equivalently via CLI:

```bash
cognee-cli improve -d "${COGNEE_PLUGIN_DATASET:-claude_sessions}" -s "${COGNEE_SESSION_ID:-claude_code_session}"
```

## What this does

1. Applies feedback weights from session scores to graph nodes/edges
2. Persists session Q&A text into the permanent graph
3. Runs default enrichment (triplet embeddings)

After this, session entries become searchable via `/cognee-memory:cognee-search`.
