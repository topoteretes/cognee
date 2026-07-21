# Cognee Tutorials

Step-by-step tutorials that walk through common workflows end-to-end.

## Migration tutorials

| Tutorial | Description |
|---|---|
| [`migrate_from_mem0_tutorial.py`](migrate_from_mem0_tutorial.py) | Import mem0 memories into Cognee using ``Mem0Source`` — covers preserve and re-derive modes, file-based and inline payloads, and recall verification |

## Running a tutorial

```bash
# Install dev environment
uv sync --dev --all-extras --reinstall

# Configure LLM_API_KEY in .env (one-time)
cp .env.template .env
# edit .env: set LLM_API_KEY

# Run the tutorial
uv run python examples/tutorials/migrate_from_mem0_tutorial.py
```

## Contributing a new tutorial

1. Place the tutorial script in this directory.
2. If your tutorial needs sample data, place it in `data/`.
3. Follow the style of existing tutorials: numbered steps, ``asyncio.run(main())``, ``forget(everything=True)`` for reproducibility.
4. Add a row to the table above.
