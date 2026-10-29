"""Journey: dataset routing on real local stores.

Two datasets, each remembered from its own slice of the golden corpus. The
automatic improve after each remember builds the global context index and
mirrors it into the routing collection of the default vector store, and an
unscoped recall is then searched in the dataset whose documents match the
query, not in both.

Mock mode embeds a hashed bag of words, so a query that quotes a document is
closest to that document's bucket; that is enough to prove the mechanics
(rows written, scored by derived id, the search narrowed to the winner).
"""

from __future__ import annotations

import pytest

import cognee
from cognee.modules.data.methods import get_data_ids_by_dataset
from cognee.modules.search.dataset_routing import (
    DATASET_ROUTE_COLLECTION,
    dataset_route_ids,
    get_dataset_route_engine,
    get_dataset_routing_config,
    route_datasets,
)
from cognee.tests.journeys import _support

DATASET_A = "journey_routing_a"
DATASET_B = "journey_routing_b"


@pytest.mark.journey
@pytest.mark.asyncio
async def test_unscoped_recall_is_routed_to_the_dataset_holding_the_answer(
    clean_env, default_user, corpus
):
    first, second = corpus[:3], corpus[3:6]

    result_a = await cognee.remember([d.text for d in first], dataset_name=DATASET_A)
    result_b = await cognee.remember([d.text for d in second], dataset_name=DATASET_B)
    assert result_a.status == "completed" and result_b.status == "completed"
    await cognee.wait_for_background_tasks()
    dataset_a, dataset_b = (
        _support.as_uuid(result_a.dataset_id),
        _support.as_uuid(result_b.dataset_id),
    )

    # The improve after each remember mirrored every dataset's index into the
    # routing collection: its root plus one bucket per document, by derived id.
    engine = await get_dataset_route_engine()
    documents = await get_data_ids_by_dataset([dataset_a, dataset_b])
    for dataset_id, slice_ in ((dataset_a, first), (dataset_b, second)):
        assert len(documents[dataset_id]) == len(slice_)
        row_ids = dataset_route_ids(dataset_id, documents[dataset_id])
        rows = {
            str(row.id): row.payload
            for row in await engine.retrieve(DATASET_ROUTE_COLLECTION, row_ids)
        }
        assert set(rows) == set(row_ids), f"routing rows missing for {dataset_id}: {rows.keys()}"
        assert all(payload["dataset_id"] == str(dataset_id) for payload in rows.values())
        assert rows[row_ids[0]]["is_root"] is True

    # A query quoting one of B's documents routes to B alone.
    target = second[0]
    config = get_dataset_routing_config()
    decision = await route_datasets(
        target.text,
        [dataset_a, dataset_b],
        config.top_k,
        config.distance_cutoff,
        config.tie_margin,
    )
    assert decision.routed == (dataset_b,)
    assert decision.unindexed == ()
    assert decision.scores[dataset_b] < decision.scores[dataset_a]

    # And an unscoped recall is searched only there.
    results = await cognee.recall(target.text)
    assert results, "recall returned nothing"
    assert {entry.dataset_id for entry in results} == {str(dataset_b)}

    async def stored(dataset_id, document_ids):
        rows = await engine.retrieve(
            DATASET_ROUTE_COLLECTION, dataset_route_ids(dataset_id, document_ids)
        )
        return {str(row.id) for row in rows}

    # Forgetting one of B's documents removes its bucket row at once (its id can
    # no longer be derived once the Data row is gone) and B's root row, whose
    # text named the document; the other buckets stay. The next index build
    # writes the root again.
    gone, *kept = documents[dataset_b]
    await cognee.forget(data_id=gone, dataset_id=dataset_b)
    _root, *kept_buckets = dataset_route_ids(dataset_b, kept)
    assert await stored(dataset_b, documents[dataset_b]) == set(kept_buckets)

    # Dropping A's memory alone (its files stay) removes every row of A.
    await cognee.forget(dataset_id=dataset_a, memory_only=True)
    assert await stored(dataset_a, documents[dataset_a]) == set()

    # Deleting B removes the rest.
    await cognee.forget(dataset_id=dataset_b)
    assert await stored(dataset_b, documents[dataset_b]) == set()

    # And a prune drops the collection itself, so a later embedding model
    # change starts from a table of the new width.
    await cognee.prune.prune_system(metadata=False)
    assert await engine.has_collection(DATASET_ROUTE_COLLECTION) is False
