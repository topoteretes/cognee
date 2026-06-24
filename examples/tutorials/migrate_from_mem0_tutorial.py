"""
Tutorial: migrate memories from mem0 to Cognee.

Import a mem0 export with Mem0Source, then query the migrated memories with
cognee.recall().

Usage:
    uv run python examples/tutorials/migrate_from_mem0_tutorial.py

This example runs in "re-derive" mode so the imported memory is immediately
queryable with recall(). Use "preserve" when you want to store the imported
COGXMemory records without re-running Cognee extraction.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import cognee
from cognee import SearchType
from cognee.migration import Mem0Source


DATASET_NAME = "mem0_migration_tutorial"
MODE = "re-derive"
SAMPLE_DUMP_PATH = Path(__file__).parent / "data" / "mem0_sample_dump.json"


def print_step(number: int, title: str) -> None:
    print(f"\n--- Step {number}: {title} ---")


def load_sample_dump() -> dict[str, Any]:
    return json.loads(SAMPLE_DUMP_PATH.read_text(encoding="utf-8"))


def explain_import_modes() -> None:
    print(
        """
Mem0Source accepts mem0 exports as lists, {"results": [...]}, {"memories": [...]},
or pre-fetched API response dictionaries.

- preserve: import source memories as COGXMemory records without re-running
  Cognee extraction. For a mem0 memory list, run cognify later before recall.
- re-derive: ingest the raw memory text and let Cognee cognify it into its own
  graph representation.
- hybrid: preserve source graph records when available and cognify raw text.
""".strip()
    )


async def main() -> None:
    print("=== Migrate from mem0 to Cognee ===")

    print_step(1, "Start from a clean Cognee workspace")
    await cognee.forget(everything=True)
    print("Cleared existing data.")

    print_step(2, "Load a short mem0 dump")
    mem0_dump = load_sample_dump()
    memories = mem0_dump["results"]
    if not memories:
        raise ValueError("Sample mem0 dump contains no memories.")

    print(f"Loaded {len(memories)} mem0 memories.")
    print(f"First memory: {memories[0]['memory']}")

    print_step(3, "Choose the migration mode")
    explain_import_modes()
    print(f"\nThis run uses mode={MODE!r}.")

    print_step(4, "Import mem0 memories with Mem0Source")
    import_result = await cognee.remember(
        Mem0Source(mem0_dump, mode=MODE),
        dataset_name=DATASET_NAME,
        self_improvement=False,
    )
    print(import_result)

    print_step(5, "Recall a migrated memory")
    query = "What coffee does Ava prefer before architecture reviews?"
    results = await cognee.recall(
        query,
        query_type=SearchType.CHUNKS,
        datasets=[DATASET_NAME],
        top_k=3,
        auto_route=False,
    )
    print(f"Query: {query}")
    for index, item in enumerate(results, start=1):
        print(f"Result {index}: {item}")
    if "Blue Bottle" not in "\n".join(str(item) for item in results):
        raise RuntimeError("Recall did not return the migrated mem0 content.")

    print_step(6, "COGX note")
    print(
        "Mem0Source translates each mem0 memory into a COGXMemory record, "
        "the memory record type in the COGX standard."
    )
    print(
        "COGX reference: "
        "https://github.com/topoteretes/cognee/blob/dev/cognee/modules/migration/cogx.py"
    )
    print("Public COGX spec tracking issue: https://github.com/topoteretes/cognee/issues/3400")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
