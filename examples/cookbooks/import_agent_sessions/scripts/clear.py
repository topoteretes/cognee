"""Forget the dataset `imported_agent_sessions`, so the next import starts from an empty memory.

Runs before ingest_sessions.py when the cookbook is run with --clear. Your session files
in ~/.claude and ~/.codex are never touched; only cognee's copy is forgotten.

Run alone: uv run python examples/cookbooks/import_agent_sessions/scripts/clear.py
"""

import asyncio
import os

os.environ.setdefault("LOG_LEVEL", "ERROR")  # quiet cognee's logs; set before importing it

import cognee
from cognee.modules.engine.operations.setup import setup

try:
    from scripts.configure import configure_cognee
except ModuleNotFoundError:  # run alone from scripts/
    from configure import configure_cognee

DATASET = "imported_agent_sessions"  # the same in every script


async def clear(dataset: str = DATASET) -> None:
    await setup()  # a fresh install has no database yet
    if dataset not in {existing.name for existing in await cognee.datasets.list_datasets()}:
        print(f"[clear] Nothing to forget: the dataset {dataset} does not exist yet.")
        return
    await cognee.forget(dataset=dataset)
    print(f"[clear] Forgot the dataset {dataset}")


if __name__ == "__main__":
    configure_cognee()  # cognee checks the dataset's embedder before forgetting it
    asyncio.run(clear())
