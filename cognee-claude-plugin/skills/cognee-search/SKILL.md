---
name: cognee-search
description: Search Cognee memory. Session memory is automatically searched on every prompt via hooks. Use this skill explicitly for permanent knowledge graph search or when you need more results than the automatic lookup provides.
---

# Cognee Memory Search

Search both session memory and the permanent knowledge graph.

## Automatic session search

Session memory is searched **automatically on every user prompt** via the `UserPromptSubmit` hook. Relevant context from tool calls and responses in this session is injected into your context window without any manual action. You do not need to run this skill to access current-session context.

## When to use this skill explicitly

Use this skill when you need to:
- Search the **permanent knowledge graph** (cross-session, ingested data)
- Get **more results** than the automatic lookup provides (auto returns top 3)
- Search with a **different query** than the user's prompt

## Instructions

### Search session memory (current session, more results)

```bash
cognee-cli recall "$ARGUMENTS" -s "${COGNEE_SESSION_ID:-claude_code_session}" -k 10 -f json
```

### Search permanent graph (cross-session, ingested data)

```bash
cognee-cli recall "$ARGUMENTS" -d "${COGNEE_PLUGIN_DATASET:-claude_sessions}" -k 5 -f json
```

### Search both (session first, fallback to graph)

```bash
${CLAUDE_PLUGIN_ROOT}/scripts/cognee-search.sh "$ARGUMENTS"
```

## Understanding results

Results include a `_source` field:
- `"session"` — from the session cache (current conversation)
- `"graph"` — from the permanent knowledge graph

## Decision table

| Signal | Action |
|--------|--------|
| Need current session context | Already automatic, no action needed |
| "from last time" / "previous session" | Search permanent graph with `-d` |
| "what do you know about X" | Try auto context first, then permanent graph |
| User explicitly says "search cognee" | Search permanent graph with `-d` |
| Auto context insufficient | Search session with `-s -k 10` for more results |
