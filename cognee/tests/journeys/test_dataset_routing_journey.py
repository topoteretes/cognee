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
from cognee.modules.search.dataset_routing.store import get_dataset_route_embedding
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
    collection = (await get_dataset_route_embedding()).collection
    documents = await get_data_ids_by_dataset([dataset_a, dataset_b])
    for dataset_id, slice_ in ((dataset_a, first), (dataset_b, second)):
        assert len(documents[dataset_id]) == len(slice_)
        row_ids = dataset_route_ids(dataset_id, documents[dataset_id])
        rows = {str(row.id): row.payload for row in await engine.retrieve(collection, row_ids)}
        assert set(rows) == set(row_ids), f"routing rows missing for {dataset_id}: {rows.keys()}"
        # Each row preserves the dataset's index payload: the root
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
@pytest.mark.skipif(
    not _support.IS_MOCK, reason="two-model deterministic routing requires mock mode"
)
async def test_mixed_dataset_models_route_in_deployment_space_and_fail_at_entry(
    clean_env, default_user, corpus, monkeypatch
):
    from cognee.context_global_variables import embedding_config as context_embedding_config
    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.infrastructure.databases.vector import get_vector_engine_async
    from cognee.infrastructure.databases.vector.embeddings.config import (
        EmbeddingConfig,
        get_embedding_config,
    )
    from cognee.infrastructure.databases.vector.exceptions.exceptions import (
        EmbeddingDimensionMismatchError,
    )
    from cognee.infrastructure.locks.dataset_lock import dataset_lock
    from cognee.modules.search.dataset_routing import dataset_route_ids
    from cognee.tests.journeys.mock_ai import hashed_embedding

    deployment = get_embedding_config()
    config_a = deployment.model_copy()
    config_b = EmbeddingConfig(
        embedding_provider="openai",
        embedding_model="openai/text-embedding-3-large",
        embedding_dimensions=128,
        embedding_api_key="mock-key",
    )
    a = await cognee.remember(corpus[0].text, dataset_name=DATASET_A, embedding_config=config_a)
    # Carry the override from the start, including migration/setup engine
    # discovery before ingestion enters its dataset context.
    token = context_embedding_config.set(config_b)
    try:
        b = await cognee.remember(
            corpus[3].text,
            dataset_name=DATASET_B,
            embedding_config=config_b,
            self_improvement=False,
        )
    finally:
        context_embedding_config.reset(token)
    await cognee.wait_for_background_tasks()
    assert a.status == b.status == "completed"
    ids = [_support.as_uuid(result.dataset_id) for result in (a, b)]
    documents = await get_data_ids_by_dataset(ids)
    # Build B's index under its valid dataset context, retaining deployment A
    # for routing. Automatic improve has no per-call embedding configuration.
    async with (
        dataset_lock(ids[1]),
        set_database_global_context_variables(ids[1], default_user, embedding_config=config_b),
    ):
        improved = await cognee.improve(dataset=ids[1], build_global_context_index=True)
        assert improved.status == "completed"
    routing = await get_dataset_route_engine()
    collection_a = (await get_dataset_route_embedding()).collection
    foreign_root = dataset_route_ids(ids[1], documents[ids[1]])[0]
    [route_row] = await routing.retrieve(collection_a, [foreign_root], include_vector=True)
    assert route_row.payload["vector"] == pytest.approx(
        hashed_embedding(route_row.payload["text"], config_a.embedding_dimensions)
    )
    async with set_database_global_context_variables(
        ids[1], default_user, embedding_config=config_b
    ):
        source = await get_vector_engine_async()
        [source_row] = await source.retrieve(
            "GlobalContextSummary_text", [foreign_root], include_vector=True
        )
        assert len(source_row.payload["vector"]) == 128
        # Query exactly the routing text so the foreign dataset wins deterministically.
        target = source_row.payload["text"]
    decision = await route_datasets(target, ids, 1, 0.8, 0.001)
    assert decision.routed == (ids[1],)
    results = await cognee.recall(corpus[0].text, query_type=cognee.SearchType.CHUNKS)
    assert results and any(entry.text for entry in results)
    assert {entry.dataset_id for entry in results} == {str(ids[0])}

    # Change deployment space without pruning or rebuilding dataset vectors.
    monkeypatch.setattr(deployment, "embedding_model", config_b.embedding_model)
    monkeypatch.setattr(deployment, "embedding_dimensions", config_b.embedding_dimensions)
    for dataset_id, config in zip(ids, (config_a, config_b)):
        async with (
            dataset_lock(dataset_id),
            set_database_global_context_variables(
                dataset_id, default_user, embedding_config=config
            ),
        ):
            result = await cognee.improve(dataset=dataset_id, build_global_context_index=True)
            assert result.status == "completed"
    collection_b = (await get_dataset_route_embedding()).collection
    assert collection_b != collection_a
    assert await routing.retrieve(collection_a, [foreign_root]) == []
    decision = await route_datasets(target, ids, 1, 0.8, 0.001)
    assert decision.routed == (ids[1],)
    results = await cognee.recall(target, query_type=cognee.SearchType.CHUNKS)
    assert results and any(entry.text for entry in results)
    assert {entry.dataset_id for entry in results} == {str(ids[1])}

    # Switch back and repopulate only routing summaries. A foreign dataset is
    # still routable, and its actual dataset-entry guard fails the recall.
    monkeypatch.setattr(deployment, "embedding_model", config_a.embedding_model)
    monkeypatch.setattr(deployment, "embedding_dimensions", config_a.embedding_dimensions)
    async with (
        dataset_lock(ids[1]),
        set_database_global_context_variables(ids[1], default_user, embedding_config=config_b),
    ):
        assert (
            await cognee.improve(dataset=ids[1], build_global_context_index=True)
        ).status == "completed"
    assert (await route_datasets(target, ids, 1, 0.8, 0.001)).routed == (ids[1],)
    with pytest.raises(EmbeddingDimensionMismatchError) as caught:
        await cognee.recall(target, query_type=cognee.SearchType.CHUNKS)
    assert str(ids[1]) in str(caught.value)
    assert caught.value.stored_dimensions == 128
    assert caught.value.configured_dimensions == config_a.embedding_dimensions


@pytest.mark.journey
@pytest.mark.asyncio
@pytest.mark.skipif(not _support.IS_MOCK, reason="deterministic routing scale fixture")
async def test_warm_routing_1010_rows_call_budget_and_concurrent_progress(
    clean_env, default_user, monkeypatch
):
    """Routing scales with summaries, not the 500 entities represented by each document."""
    import asyncio
    import importlib
    import time
    from unittest.mock import AsyncMock
    from uuid import uuid4

    from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import LanceDBAdapter
    from cognee.modules.data.models import Data, Dataset
    from cognee.modules.engine.models.Entity import Entity
    from cognee.tests.journeys.mock_ai import hashed_embedding

    route_module = importlib.import_module("cognee.modules.search.dataset_routing.route")
    embedding = await get_dataset_route_embedding()
    documents = {uuid4(): [uuid4() for _ in range(100)] for _ in range(10)}
    candidates = list(documents)
    async with get_relational_engine().get_async_session() as session:
        session.add_all(
            Dataset(id=dataset_id, name=f"scale_{dataset_id}", owner_id=default_user.id)
            for dataset_id in candidates
        )
        session.add_all(
            Data(id=document_id, dataset_id=dataset_id, owner_id=default_user.id)
            for dataset_id, document_ids in documents.items()
            for document_id in document_ids
        )
        await session.commit()
    # A real source graph of 500 nodes per document. Graph construction and
    # extraction are outside the measured warm routing operation.
    graph = await asyncio.to_thread(LadybugAdapter, db_path=str(clean_env / "scale.lbug"))

    def source_nodes():
        return [
            Entity(name=f"{document_id}:entity_{index}", description=str(document_id))
            for document_ids in documents.values()
            for document_id in document_ids
            for index in range(500)
        ]

    try:
        await graph.add_nodes(await asyncio.to_thread(source_nodes))
        assert (await graph.query("MATCH (n:Node) RETURN count(n)"))[0][0] == 500000
    finally:
        await graph.close()
    names = " ".join(f"entity_{index}" for index in range(500))
    points = []
    for dataset_id, document_ids in documents.items():
        for index, row_id in enumerate(dataset_route_ids(dataset_id, document_ids)):
            text = f"dataset {dataset_id} document {index}: {names}"
            points.append(
                {
                    "id": row_id,
                    "vector": hashed_embedding(text, embedding.vector_size),
                    "payload": {"id": row_id, "text": text},
                }
            )
    engine = await get_dataset_route_engine()
    await engine.upsert_raw_vectors(embedding.collection, points, vector_size=embedding.vector_size)
    ids_read = AsyncMock(wraps=get_data_ids_by_dataset)
    monkeypatch.setattr(route_module, "get_data_ids_by_dataset", ids_read)

    # Spy on orchestration, independently of backend-internal query batches.
    async def check_collection(name):
        return await engine.has_collection(name)

    async def score_rows(*args):
        return await engine.score_by_ids(*args)

    has_collection = AsyncMock(wraps=check_collection)
    score = AsyncMock(wraps=score_rows)
    embed = AsyncMock(wraps=embedding.engine.embed_text)
    monkeypatch.setattr(
        route_module,
        "get_dataset_route_engine",
        AsyncMock(
            return_value=type(
                "Store",
                (),
                {
                    "has_collection": staticmethod(has_collection),
                    "score_by_ids": staticmethod(score),
                },
            )()
        ),
    )
    monkeypatch.setattr(
        route_module, "get_dataset_route_embedding", AsyncMock(return_value=embedding)
    )
    monkeypatch.setattr(embedding.engine, "embed_text", embed)
    started = time.perf_counter()
    decision = await route_datasets(points[0]["payload"]["text"], candidates, 1, 0.8, 0.001)
    elapsed = time.perf_counter() - started
    assert len(decision.scores) == 10
    assert elapsed < 2.0
    ids_read.assert_awaited_once_with(candidates)
    has_collection.assert_awaited_once()
    embed.assert_awaited_once()
    score.assert_awaited_once()
    assert len(score.await_args.args[1]) == 1010
    print(
        f"Warm routing: {elapsed:.4f}s, 1010 rows; calls: IDs=1 existence=1 query_embedding=1 score=1"
    )

    # While the provider is suspended, a heartbeat and another store operation
    # must finish before routing. No deployment-wide or all-dataset lock.
    entered, release = asyncio.Event(), asyncio.Event()

    async def paused(texts):
        entered.set()
        await release.wait()
        return await embed(texts)

    monkeypatch.setattr(embedding.engine, "embed_text", paused)
    routing_task = asyncio.create_task(route_datasets("query", candidates, 1, 0.8, 0.001))
    other = LanceDBAdapter(
        url=str(clean_env / "other_dataset"), api_key=None, embedding_engine=embedding.engine
    )
    try:
        await asyncio.wait_for(entered.wait(), 0.1)
        heartbeat = time.perf_counter()
        await asyncio.wait_for(asyncio.sleep(0), 0.1)
        assert time.perf_counter() - heartbeat < 0.1
        await asyncio.wait_for(other.list_collections(), 1.0)
        assert not routing_task.done()
    finally:
        release.set()
        await routing_task
        await other.close()
