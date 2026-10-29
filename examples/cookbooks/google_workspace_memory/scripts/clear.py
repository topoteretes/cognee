"""Forget the dataset `google_workspace_memory`, so the next run starts from an empty memory.

Runs before the ingest scripts when the cookbook is run with --clear, which is on by default
for a sample run, so an earlier run's copies never mix with the new ones. With --sample, it
forgets the sample's dataset `google_workspace_memory_sample` instead.

Run alone: uv run python examples/cookbooks/google_workspace_memory/scripts/clear.py [--sample]
"""

import argparse
import asyncio
import os

os.environ.setdefault("LOG_LEVEL", "ERROR")  # quiet cognee's logs; set before importing it

import cognee
from cognee.modules.engine.operations.setup import setup

DATASET = "google_workspace_memory"  # the same in every script
SAMPLE_DATASET = "google_workspace_memory_sample"  # the sample never mixes with your data


async def clear(sample: bool = False) -> None:
    dataset = SAMPLE_DATASET if sample else DATASET
    await setup()  # a fresh install has no database yet
    if dataset not in {dataset.name for dataset in await cognee.datasets.list_datasets()}:
        print(f"[clear] Nothing to forget: the dataset {dataset} does not exist yet.")
        return
    await cognee.forget(dataset=dataset)
    print(f"[clear] Forgot the dataset {dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", action="store_true", help="forget the sample's dataset")
    asyncio.run(clear(parser.parse_args().sample))
