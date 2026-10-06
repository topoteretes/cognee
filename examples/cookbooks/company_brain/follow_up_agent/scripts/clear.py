"""Forget the dataset `follow_up_agent`, so the next run starts from an empty memory.

Runs before the ingest scripts when the cookbook is run with --clear, which is on by default
for a sample run, so an earlier run's copies never mix with the new ones.

Run alone: uv run python examples/cookbooks/company_brain/follow_up_agent/scripts/clear.py
"""

import asyncio
import os

os.environ.setdefault("LOG_LEVEL", "ERROR")  # quiet cognee's logs; set before importing it

import cognee

DATASET = "follow_up_agent"  # the same in every script


async def clear() -> None:
    if DATASET not in {dataset.name for dataset in await cognee.datasets.list_datasets()}:
        print(f"[clear] Nothing to forget: the dataset {DATASET} does not exist yet.")
        return
    await cognee.forget(dataset=DATASET)
    print(f"[clear] Forgot the dataset {DATASET}")


if __name__ == "__main__":
    asyncio.run(clear())
