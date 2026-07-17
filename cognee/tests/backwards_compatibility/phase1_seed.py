"""
Backwards Compatibility Test

Phase 1 -  run with a legacy cognee version

Seeds the database with Lorem Ipsum data: add → cognify → search.
Also seeds a session (two entries via session-mode remember) and bridges it
into the graph via improve() — leaving a genuine pre-watermark session cache
for Phase 2 to take over. The legacy version must be >= v1.2.0: that is the
earliest release with the SQL session cache (cache.db) the current branch
reads, and the CI pin never goes below it.

Phase 2 - run with current Cognee branch

Verifies that the current branch can search the legacy cognified data, then adds + cognifies new Lorem Ipsum data with
the current branch and verifies search again. Also verifies the session
persistence watermark takeover (see phase2_verify docstring).
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

# Session seeding (only on legacy versions that already have session memory).
# The facts carry distinctive markers Phase 2 greps for, so keep them in sync
# with phase2_verify.py.
COMPAT_SESSION_ID = "compat_session"
SESSION_FACT_1 = "The beekeeper Anton Zorman kept blue-marked hives on the Levada terraces."
SESSION_FACT_2 = "The weaver Ilka Matova dyed her linen with walnut husks and iron water."


async def seed_session() -> None:
    """Seed + bridge a session into the already-seeded dataset.

    Uses the existing dataset on purpose: bridging silently no-ops when the
    target dataset does not exist yet (pre-existing behavior in every version).
    """
    print("Seeding session memory (remember x2 + improve bridge)...")
    await cognee.remember(SESSION_FACT_1, session_id=COMPAT_SESSION_ID, self_improvement=False)
    await cognee.remember(SESSION_FACT_2, session_id=COMPAT_SESSION_ID, self_improvement=False)
    await cognee.improve("lorem_ipsum", session_ids=[COMPAT_SESSION_ID])
    print(f"Seeded session '{COMPAT_SESSION_ID}' with 2 entries and bridged it into the graph.")


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

    print(f"Phase 1 search OK. Got {len(results)} result(s):")
    for result in results:
        print(f"  - {result}")

    await seed_session()

    print("Phase 1 completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
