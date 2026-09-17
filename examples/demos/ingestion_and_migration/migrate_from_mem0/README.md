# Cognee Tutorials

Runnable tutorials that walk through a specific workflow end-to-end.

| Tutorial | What you'll learn |
|---|---|
| [`migrate_from_mem0.py`](migrate_from_mem0.py) | Import mem0 memories into Cognee using `Mem0Source` (`preserve` and `re-derive` modes) |

## Running a tutorial

```bash
uv run python examples/demos/ingestion_and_migration/migrate_from_mem0/migrate_from_mem0.py
```

Requires `LLM_API_KEY` in `.env` (for `re-derive` mode and `recall`).
