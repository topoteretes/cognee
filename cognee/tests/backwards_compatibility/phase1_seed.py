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

# DLT seeding — legacy versions ingest a dlt source as one Data record PER ROW
# (external_metadata.source == "dlt"). Phase 2 verifies the migration moves
# that stamp to system_metadata, the tombstone rejects cognifying the legacy
# rows, and re-adding the source converts it to the manifest model. Markers
# must stay in sync with phase2_verify.py.
DLT_COMPAT_DATASET = "dlt_compat"
DLT_COMPAT_SOURCE = "compat_widgets"
DLT_COMPAT_ROWS = [
    {"id": "1", "name": "anemometer", "shelf": "north"},
    {"id": "2", "name": "barometer", "shelf": "east"},
    {"id": "3", "name": "hygrometer", "shelf": "south"},
]


async def seed_dlt() -> None:
    """Seed a dlt source the legacy way: one Data record per row.

    Runs on the legacy tag, so this exercises whatever DLT ingestion that
    version shipped (per-row records since v0.5.5). The dlt extra must be
    installed — a missing import is a hard failure, not a skip, so the CI
    workflow and this script cannot silently drift apart.
    """
    import dlt

    @dlt.resource(name=DLT_COMPAT_SOURCE, primary_key="id", write_disposition="replace")
    def compat_widgets():
        yield from DLT_COMPAT_ROWS

    print(f"Seeding legacy DLT source '{DLT_COMPAT_SOURCE}' ({len(DLT_COMPAT_ROWS)} rows)...")
    await cognee.add([compat_widgets()], dataset_name=DLT_COMPAT_DATASET)
    try:
        await cognee.cognify(datasets=[DLT_COMPAT_DATASET])
        print(f"Seeded and cognified dataset '{DLT_COMPAT_DATASET}'.")
    except Exception as error:
        # v1.2.0's legacy DLT cognify is broken on SQLite (upsert_edges binds
        # string UUIDs into a UUID column -> "'str' object has no attribute
        # 'hex'"). The tag is immutable, so tolerate it: the per-row Data
        # records with their source == "dlt" stamps ARE the compat surface
        # Phase 2 verifies (migration, tombstone, re-add conversion).
        print(
            f"KNOWN-ISSUE: legacy cognify of '{DLT_COMPAT_DATASET}' failed on this "
            f"backend ({type(error).__name__}); seeded per-row records only."
        )


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

    await seed_dlt()

    await seed_session()

    print("Phase 1 completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
