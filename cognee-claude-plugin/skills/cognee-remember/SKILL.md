---
name: cognee-remember
description: Store data permanently in the Cognee knowledge graph via add + cognify. Use when the user explicitly wants to persist important information into the permanent graph (not just session cache).
---

# Cognee Permanent Memory Storage

Store data permanently in the Cognee knowledge graph.

## Instructions

Run the cognee remember command with the data to store:

```bash
cognee-cli remember "$ARGUMENTS" -d "${COGNEE_PLUGIN_DATASET:-claude_sessions}"
```

The command outputs a summary after completion:

```
Data ingested and knowledge graph built successfully!
  Dataset ID: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
  Items processed: 1
  Content hash: a1b2c3d4...
  Elapsed: 4.2s
```

Use the dataset ID and content hash to track what was stored. If items_processed is 0 or the command errors, the data was not indexed and won't be searchable.

**IMPORTANT**: Do NOT use the `-b` (background) flag. Running cognify in the background can result in data not being fully indexed and therefore not searchable. Always run in the foreground to ensure the full pipeline completes before returning.

## When to use

- The user explicitly says to "remember this permanently" or "save this to the knowledge graph"
- Important findings or conclusions that should persist beyond the current session
- NOT for routine tool call logging (that uses session memory automatically)

## Note

This triggers the full add + cognify + improve pipeline, which builds entities and relationships in the knowledge graph, then enriches them with triplet embeddings. It is heavier than session storage. Routine tool call and response logging is handled automatically by the plugin hooks using the lightweight session cache path.

After remembering, the graph knowledge is automatically synced back to the active session for fast retrieval during completions.
