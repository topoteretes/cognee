---
name: cognee-remember
description: Store data permanently in the Cognee knowledge graph via add + cognify. Use when the user explicitly wants to persist important information into the permanent graph (not just session cache).
---

# Cognee Permanent Memory Storage

Store data permanently in the Cognee knowledge graph.

## Instructions

Run the cognee remember command with the data to store:

```bash
cognee-cli remember "$ARGUMENTS" -d "${COGNEE_PLUGIN_DATASET:-claude_sessions}" -b
```

The `-b` flag runs cognify in the background so it does not block.

## When to use

- The user explicitly says to "remember this permanently" or "save this to the knowledge graph"
- Important findings or conclusions that should persist beyond the current session
- NOT for routine tool call logging (that uses session memory automatically)

## Note

This triggers the full add + cognify pipeline, which builds entities and relationships in the knowledge graph. It is heavier than session storage. Routine tool call and response logging is handled automatically by the plugin hooks using the lightweight session cache path.
