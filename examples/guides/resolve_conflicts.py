"""Inspect and resolve a stored conflict without another LLM review.

Run after ingesting conflicting sources and running
``await cognee.improve(dataset=..., review_conflicts=True)``.

Usage:
    uv run python examples/guides/resolve_conflicts.py DATASET_UUID
    uv run python examples/guides/resolve_conflicts.py DATASET_UUID --conflict CONFLICT_UUID --fact FACT_UUID --reason "Verified with owner"
"""

import argparse
import asyncio
from uuid import UUID

import cognee


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=UUID)
    parser.add_argument("--conflict", type=UUID)
    parser.add_argument("--fact", type=UUID)
    parser.add_argument("--reason")
    args = parser.parse_args()
    if bool(args.conflict) != bool(args.fact):
        parser.error("--conflict and --fact must be provided together")
    if args.conflict:
        result = await cognee.resolve_conflict(
            args.conflict, dataset_id=args.dataset, selected_fact_id=args.fact, reason=args.reason
        )
        print(result.model_dump_json(indent=2))
    else:
        for conflict in await cognee.list_conflicts(args.dataset):
            print(conflict.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
