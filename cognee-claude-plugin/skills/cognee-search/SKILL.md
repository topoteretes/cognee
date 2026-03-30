---
name: cognee-search
description: Search the Cognee knowledge graph for relevant context from previous sessions, stored reasoning, and ingested data. Use when you need to recall information from prior conversations or look up domain knowledge.
---

# Cognee Knowledge Graph Search

Search the Cognee knowledge graph to retrieve relevant context.

## Instructions

1. Run the cognee recall command:

```bash
cognee-cli recall "$ARGUMENTS" -d "${COGNEE_PLUGIN_DATASET:-claude_sessions}" -k 5 -f json
```

2. Parse the JSON results.

3. Present the relevant findings to the user clearly. If no results are found, say so.

## When to use

- The user asks to recall something from a previous session
- The user asks "what do you know about X" referring to stored knowledge
- You need context from prior reasoning or tool calls
- The user explicitly asks to search cognee

## Note on memory types

Cognee has two memory paths:
- **Session memory** (lightweight): tool calls and responses stored during the session. Available immediately but not in the permanent graph until `cognee-cli improve` is run.
- **Permanent memory**: data ingested via `cognee-cli remember` (without --session-id). This builds the full knowledge graph via add + cognify.

This search queries the **permanent** knowledge graph. Session entries are only searchable after running `cognee-cli improve`.
