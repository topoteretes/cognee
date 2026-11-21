"""Journey: dataset routing on real local stores.

Two datasets, each remembered from its own slice of the golden corpus. The
automatic improve after each remember builds the global context index and
mirrors it into the routing collection of the model that built the dataset,
in the default vector store, and an unscoped recall is then searched in the
dataset whose documents match the query, not in both.

Mock mode embeds a hashed bag of words, so a query that quotes a document is
closest to that document's bucket; that is enough to prove the mechanics
(rows written, scored by derived id, the search narrowed to the winner).

The second half switches the embedding model and remembers two more datasets
with it: the unscoped recall is then routed among the new model's datasets
and never enters the old ones, and switching the configuration back searches
the first two again, off rows that waited in their collection -- no rebuild,
no prune.
"""

from __future__ import annotations

import pytest

import cognee
from cognee.modules.data.methods import get_data_ids_by_dataset, get_dataset_data
from cognee.modules.search.dataset_routing import (
    dataset_route_collection,
    dataset_route_ids,
    datasets_on_configured_model,
    get_dataset_route_engine,
    get_dataset_routing_config,
    route_datasets,
)
from cognee.tests.journeys import _support, mock_ai

DATASET_A = "journey_routing_a"
DATASET_B = "journey_routing_b"
DATASET_C = "journey_routing_c"
DATASET_D = "journey_routing_d"

MODEL_ONE = ("openai/text-embedding-3-small", 256)
MODEL_TWO = ("openai/text-embedding-3-large", 384)


def _use_model(monkeypatch, model_and_dimensions):
    """Point the deployment's embedding config at one of the two mock models.

    The config caches see the environment at construction, so the caches are
    dropped with the pin: the next engine embeds at the new width, and the
    routing collection resolves to the new model's.
    """
    from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_config

    model, dimensions = model_and_dimensions
    monkeypatch.setenv("EMBEDDING_MODEL", model)
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", str(dimensions))
    get_embedding_config.cache_clear()
    mock_ai._clear_engine_caches()


@pytest.mark.journey
@pytest.mark.asyncio
async def test_unscoped_recall_is_routed_to_the_dataset_holding_the_answer(
    monkeypatch, clean_env, default_user, corpus
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
    # routing collection of the model that built it: its root plus one bucket
    # per document, by derived id.
    config = get_dataset_routing_config()
    engine = await get_dataset_route_engine()
    collection = dataset_route_collection()
    documents = await get_data_ids_by_dataset([dataset_a, dataset_b])
    for dataset_id, slice_ in ((dataset_a, first), (dataset_b, second)):
        assert len(documents[dataset_id]) == len(slice_)
        row_ids = dataset_route_ids(dataset_id, documents[dataset_id])
        rows = {str(row.id): row.payload for row in await engine.retrieve(collection, row_ids)}
        assert set(rows) == set(row_ids), f"routing rows missing for {dataset_id}: {rows.keys()}"
        # Each row is the dataset's own index row, copied as stored: the root
        # names no single document, a bucket carries the document it summarizes.
        assert rows[row_ids[0]]["document_id"] is None
        assert [rows[row_id]["document_id"] for row_id in row_ids[1:]] == [
            str(document_id) for document_id in documents[dataset_id]
        ]

    # A query quoting one of B's documents routes to B alone.
    target = second[0]
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

    # --- a second model joins the deployment -----------------------------------

    _use_model(monkeypatch, MODEL_TWO)
    third, fourth = corpus[6:9], corpus[9:12]

    result_c = await cognee.remember([d.text for d in third], dataset_name=DATASET_C)
    result_d = await cognee.remember([d.text for d in fourth], dataset_name=DATASET_D)
    assert result_c.status == "completed" and result_d.status == "completed"
    await cognee.wait_for_background_tasks()
    dataset_c, dataset_d = (
        _support.as_uuid(result_c.dataset_id),
        _support.as_uuid(result_d.dataset_id),
    )

    # The model switch wrote into a different collection, on rows the first
    # two datasets never shared: their collections are untouched.
    other_collection = dataset_route_collection()
    assert other_collection != collection
    assert await (await get_dataset_route_engine()).has_collection(collection)
    candidates = await datasets_on_configured_model([dataset_a, dataset_b, dataset_c, dataset_d])
    assert candidates == {dataset_c, dataset_d}

    target_c = third[0]
    decision = await route_datasets(
        target_c.text,
        [dataset_a, dataset_b, dataset_c, dataset_d],
        config.top_k,
        config.distance_cutoff,
        config.tie_margin,
    )
    assert decision.routed == (dataset_c,), decision

    # An unscoped recall is routed among the new model's datasets and never
    # enters the old ones: entering one of them would fail on the model guard.
    results = await cognee.recall(target_c.text)
    assert results, "recall returned nothing"
    assert {entry.dataset_id for entry in results} == {str(dataset_c)}

    # --- switching back needs no rebuild ---------------------------------------

    _use_model(monkeypatch, MODEL_ONE)
    assert await datasets_on_configured_model([dataset_a, dataset_b, dataset_c, dataset_d]) == {
        dataset_a,
        dataset_b,
    }
    results = await cognee.recall(target.text)
    assert results, "recall returned nothing"
    assert {entry.dataset_id for entry in results} == {str(dataset_b)}

    async def stored(collection_name, dataset_id, document_ids):
        rows = await (await get_dataset_route_engine()).retrieve(
            collection_name, dataset_route_ids(dataset_id, document_ids)
        )
        return {str(row.id) for row in rows}

    # Forgetting one of B's documents removes its bucket row at once (its id can
    # no longer be derived once the Data row is gone); the other buckets stay.
    # B's root row, whose text named the document, is written again from the
    # documents that remain, so a broad question keeps routing to B before any
    # improve runs, and the forgotten document's name is nowhere in it.
    names = {row.id: row.name for row in await get_dataset_data(dataset_b)}
    gone, *kept = documents[dataset_b]
    await cognee.forget(data_id=gone, dataset_id=dataset_b)
    root_id, *kept_buckets = dataset_route_ids(dataset_b, kept)
    assert await stored(collection, dataset_b, documents[dataset_b]) == {root_id, *kept_buckets}
    [root_row] = await (await get_dataset_route_engine()).retrieve(collection, [root_id])
    assert root_row.payload["document_id"] is None
    assert names[gone] not in root_row.payload["text"]
    assert all(names[document_id] in root_row.payload["text"] for document_id in kept)
    # The row mirrors the root the dataset's own index now holds.
    assert root_row.payload["text"].lower() in await _support.vector_texts(
        dataset_b, default_user, "GlobalContextSummary_text"
    )
    decision = await route_datasets(
        root_row.payload["text"],
        [dataset_a, dataset_b],
        config.top_k,
        config.distance_cutoff,
        config.tie_margin,
    )
    assert decision.routed == (dataset_b,)

    # Dropping A's memory alone (its files stay) removes every row of A.
    await cognee.forget(dataset_id=dataset_a, memory_only=True)
    assert await stored(collection, dataset_a, documents[dataset_a]) == set()

    # Deleting B removes the rest.
    await cognee.forget(dataset_id=dataset_b)
    assert await stored(collection, dataset_b, documents[dataset_b]) == set()

    # And a prune drops the collections themselves.
    await cognee.prune.prune_system(metadata=False)
    assert await (await get_dataset_route_engine()).has_collection(collection) is False
    assert await (await get_dataset_route_engine()).has_collection(other_collection) is False
