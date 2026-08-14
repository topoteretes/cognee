"""End-to-end regression test for issue #3481 (PR #3483, SDK-275).

``brute_force_triplet_search()`` used to append ``"EdgeType_relationship_name"``
in place to the ``collections`` list the caller passed in. Callers such as
``TripletSearchContextProvider`` keep a persistent ``self.collections`` and hand
the same list object to one search per entity, so the caller's configured list
was silently and permanently mutated after the first search.

This test runs the real add -> cognify pipeline (no mocks), then exercises both
entry points with caller-provided collections lists and asserts:

- the caller's list is never mutated, across repeated searches;
- a list that already contains the edge collection is left untouched;
- the edge collection is still searched (triplets are found with a list that
  omits it).
"""

import os

os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")

import asyncio
import pathlib
from types import SimpleNamespace

import cognee
from cognee.modules.retrieval.context_providers.TripletSearchContextProvider import (
    TripletSearchContextProvider,
)
from cognee.modules.retrieval.utils.brute_force_triplet_search import (
    brute_force_triplet_search,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger()

TEXT = (
    "Audrey Hepburn starred in Roman Holiday, a romantic comedy directed by "
    "William Wyler. Gregory Peck played the male lead in the same film."
)


async def main():
    base = pathlib.Path(__file__).parent
    data_dir = str((base / ".data_storage/test_collections_mutation").resolve())
    system_dir = str((base / ".cognee_system/test_collections_mutation").resolve())
    cognee.config.data_root_directory(data_dir)
    cognee.config.system_root_directory(system_dir)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    await cognee.add(TEXT)
    await cognee.cognify()

    # A caller-provided list is not mutated, across repeated real searches.
    caller_collections = ["Entity_name"]
    snapshot = list(caller_collections)

    triplets = await brute_force_triplet_search(
        query="Who directed Roman Holiday?", collections=caller_collections
    )
    await brute_force_triplet_search(
        query="Which film did Audrey Hepburn star in?", collections=caller_collections
    )

    assert caller_collections == snapshot, (
        f"caller's collections list was mutated: {caller_collections}"
    )
    # The edge collection is still searched even though the caller omitted it.
    assert len(triplets) > 0, "expected triplets from the real graph"

    # A list that already contains the edge collection is left untouched too.
    with_edge = ["Entity_name", "EdgeType_relationship_name"]
    snapshot_with_edge = list(with_edge)

    await brute_force_triplet_search(query="Who played the male lead?", collections=with_edge)

    assert with_edge == snapshot_with_edge, f"caller's collections list was mutated: {with_edge}"

    # TripletSearchContextProvider hands its persistent self.collections to one
    # search per entity; the configured list must survive a multi-entity search.
    provider = TripletSearchContextProvider(collections=["Entity_name"])
    entities = [SimpleNamespace(name="Audrey Hepburn"), SimpleNamespace(name="Gregory Peck")]

    context = await provider.get_context(entities, "How are they related?")

    assert provider.collections == ["Entity_name"], (
        f"provider's configured collections were mutated: {provider.collections}"
    )
    assert context, "expected a non-empty context from the real graph"

    logger.info("Collections mutation e2e test passed.")


if __name__ == "__main__":
    asyncio.run(main())
