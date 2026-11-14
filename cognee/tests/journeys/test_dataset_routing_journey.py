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
from cognee.modules.data.methods import get_data_ids_by_dataset, get_dataset_data
from cognee.modules.search.dataset_routing import (
    dataset_route_ids,
    get_dataset_route_engine,
    get_dataset_routing_config,
    route_datasets,
)
from cognee.modules.search.dataset_routing.store import (
    dataset_route_collection,
    deployment_embedding_record,
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
    identity = await deployment_embedding_record()
    collection = dataset_route_collection(
        identity["embedding_model"], identity["embedding_dimensions"]
    )
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
        rows = await engine.retrieve(collection, dataset_route_ids(dataset_id, document_ids))
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
    assert await stored(dataset_b, documents[dataset_b]) == {root_id, *kept_buckets}
    [root_row] = await engine.retrieve(collection, [root_id])
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
    assert await stored(dataset_a, documents[dataset_a]) == set()

    # Deleting B removes the rest.
    await cognee.forget(dataset_id=dataset_b)
    assert await stored(dataset_b, documents[dataset_b]) == set()

    # And a prune drops the collection itself, so a later embedding model
    # change starts from a table of the new width.
    await cognee.prune.prune_system(metadata=False)
    assert await engine.has_collection(collection) is False


@pytest.mark.journey
@pytest.mark.asyncio
@pytest.mark.skipif(not _support.IS_MOCK, reason="distinct synthetic models use mock embeddings")
async def test_two_model_switch_reuses_routing_and_never_enters_incompatible_dataset(
    clean_env, default_user, corpus, monkeypatch
):
    from contextlib import asynccontextmanager
    from importlib import import_module

    from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_config
    from cognee.modules.search.types import SearchType
    from cognee.tests.journeys import mock_ai

    config = get_embedding_config()
    original = (config.embedding_model, config.embedding_dimensions)
    groups = {}
    entered, scored = [], []
    search_module = import_module("cognee.modules.search.methods.search")
    context = search_module.set_database_global_context_variables
    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import LanceDBAdapter

    original_score = LanceDBAdapter.score_by_ids

    @asynccontextmanager
    async def tracked_context(dataset_id, *args, **kwargs):
        entered.append(dataset_id)
        async with context(dataset_id, *args, **kwargs):
            yield

    async def tracked_score(self, collection, ids, vector):
        scored.append((collection, list(ids), len(vector)))
        return await original_score(self, collection, ids, vector)

    def switch(model, width):
        config.embedding_model, config.embedding_dimensions = model, width
        mock_ai._clear_engine_caches()

    try:
        # Two datasets per model allow narrowing in both directions.
        for model, width, offset in [
            ("openai/text-embedding-3-small", 64, 0),
            ("openai/text-embedding-3-large", 128, 2),
        ]:
            switch(model, width)
            ids = []
            for i in range(2):
                result = await cognee.remember(
                    corpus[offset + i].text, dataset_name=f"mixed_{width}_{i}"
                )
                assert result.status == "completed"
                ids.append(_support.as_uuid(result.dataset_id))
            await cognee.wait_for_background_tasks()
            groups[model] = (width, ids, corpus[offset + 1].text)

        monkeypatch.setattr(search_module, "set_database_global_context_variables", tracked_context)
        monkeypatch.setattr(LanceDBAdapter, "score_by_ids", tracked_score)
        all_ids = [id for _, ids, _ in groups.values() for id in ids]
        for model in [
            "openai/text-embedding-3-small",
            "openai/text-embedding-3-large",
            "openai/text-embedding-3-small",
        ]:
            width, ids, query = groups[model]
            switch(model, width)
            entered.clear()
            scored.clear()
            results = await cognee.recall(query, query_type=SearchType.CHUNKS)
            assert {item.dataset_id for item in results} == {str(ids[1])}
            assert entered == [ids[1]]
            assert scored and scored[0][0] == dataset_route_collection(model, width)
            documents = await get_data_ids_by_dataset(ids)
            assert set(scored[0][1]) == {
                row for id in ids for row in dataset_route_ids(id, documents[id])
            }
            assert scored[0][2] == width

        # Every collection remains readable regardless of the active embedder.
        documents = await get_data_ids_by_dataset(all_ids)
        for model, (width, ids, _) in groups.items():
            switch(model, width)
            for id in ids:
                await cognee.forget(dataset_id=id)
            engine = await get_dataset_route_engine()
            collection = dataset_route_collection(model, width)
            assert (
                await engine.retrieve(
                    collection, [row for id in ids for row in dataset_route_ids(id, documents[id])]
                )
                == []
            )
        from uuid import uuid4

        from cognee.modules.search.dataset_routing.store import DATASET_ROUTE_COLLECTION

        historic = dataset_route_collection("historical-unregistered-model", 16)
        engine = await get_dataset_route_engine()
        point_id = uuid4()
        point = {
            "id": point_id,
            "vector": [1.0] + [0.0] * 15,
            "payload": {"id": str(point_id), "text": "historical"},
        }
        for collection in [historic, DATASET_ROUTE_COLLECTION, "Unrelated_text"]:
            await engine.upsert_raw_vectors(collection, [point], vector_size=16)
        await cognee.prune.prune_system(metadata=False)
        engine = await get_dataset_route_engine()
        assert await engine.list_collections() == ["Unrelated_text"]
    finally:
        switch(*original)
