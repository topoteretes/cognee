"""Prune or forget the TrustRAG dataset in Cognee.

Cleans up Cognee's vector, graph, and metadata stores for the specified dataset.
Requires explicit --yes / --force flag or interactive confirmation to avoid
unintentional data loss.

Run alone: uv run python examples/cookbooks/trustrag/scripts/clear.py [--dataset NAME] [--yes]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("LOG_LEVEL", "ERROR")

COOKBOOK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(COOKBOOK_DIR))

import cognee
from config import default_config


async def clear_dataset(dataset_name: str | None = None, force: bool = False) -> None:
    ds = dataset_name or default_config.dataset_name

    if not force:
        confirm = input(f"[clear] Are you sure you want to forget dataset '{ds}'? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            print("[clear] Aborted by user.")
            return

    try:
        from cognee.modules.engine.operations.setup import setup
        await setup()

        if hasattr(cognee, "forget"):
            await cognee.forget(dataset=ds)
            print(f"[clear] Successfully forgotten dataset '{ds}'.")
        elif hasattr(cognee, "prune") and hasattr(cognee.prune, "prune_data"):
            await cognee.prune.prune_data(datasets=[ds])
            print(f"[clear] Successfully pruned dataset '{ds}'.")
        else:
            print(f"[clear] Cognee cleanup API not found for dataset '{ds}'.", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"[clear] Error clearing dataset '{ds}': {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", help="name of dataset to clear (default: trustrag)")
    parser.add_argument("--yes", "-y", action="store_true", help="skip confirmation prompt")
    args = parser.parse_args()

    asyncio.run(clear_dataset(dataset_name=args.dataset, force=args.yes))

