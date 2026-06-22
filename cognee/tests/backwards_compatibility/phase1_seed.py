"""
Backwards Compatibility Test

Phase 1 -  run with cognee v0.5.7

Seeds the database with Lorem Ipsum data: add → cognify → search.

Phase 2 - run with current Cognee branch

Verifies that the current branch can search the v0.5.7 cognified data, then adds + cognifies new Lorem Ipsum data with
the current branch and verifies search again.
"""

import asyncio
import sys

import cognee
from cognee.api.v1.search import SearchType

LOREM_IPSUM = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut
labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris
nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit
esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt
in culpa qui officia deserunt mollit anim id est laborum.

Lorem ipsum is placeholder text commonly used in the graphic, print, and publishing industries for
previewing layouts and visual mockups. It has been the industry standard dummy text since the 1500s
when an unknown printer scrambled a passage of text to make a type specimen book.
"""

SEARCH_QUERY = "What is Lorem Ipsum and where does it come from?"


async def main():
    print(f"Running Phase 1 with cognee version: {cognee.__version__}")

    print("Pruning existing data...")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    print("Adding Lorem Ipsum text...")
    await cognee.add(LOREM_IPSUM, dataset_name="lorem_ipsum")

    print("Cognifying...")
    await cognee.cognify(datasets=["lorem_ipsum"])

    print(f"Searching with query: '{SEARCH_QUERY}'")
    results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text=SEARCH_QUERY,
    )

    if not results:
        print("ERROR: Phase 1 search returned no results. Seeding failed.")
        sys.exit(1)

    print(f"Phase 1 completed successfully. Got {len(results)} result(s):")
    for result in results:
        print(f"  - {result}")


if __name__ == "__main__":
    asyncio.run(main())
