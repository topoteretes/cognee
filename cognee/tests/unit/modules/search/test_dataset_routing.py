"""Dataset routing: the routing collection's write path and the read path.

No database anywhere: the relational lookup and the vector engine are stubbed.
"""

import re
import types
from importlib import import_module
from uuid import UUID, uuid4

import pytest

from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult
from cognee.modules.search.dataset_routing import (
    DATASET_ROUTE_COLLECTION,
    dataset_route_ids,
    route_datasets,
    sync_dataset_routes,
)
from cognee.tasks.memify.global_context_index.ids import (
    create_document_bucket_id,
    create_root_summary_id,
)
from cognee.tasks.summarization.models import GlobalContextSummary

index_module = import_module("cognee.modules.search.dataset_routing.index")
route_module = import_module("cognee.modules.search.dataset_routing.route")
store_module = import_module("cognee.modules.search.dataset_routing.store")
model_utils_module = import_module(
    "cognee.infrastructure.databases.utils.ensure_embedding_model_matches"
)

TEST_MODEL, TEST_DIMENSIONS = "test-model", 2
TEST_COLLECTION = store_module.dataset_route_collection_name(TEST_MODEL, TEST_DIMENSIONS)


async def _route(query, dataset_ids, top_k=1, distance_cutoff=0.8, tie_margin=0.001):
    """route_datasets with the config defaults; every existing score here is within the cutoff."""
    return await route_datasets(query, dataset_ids, top_k, distance_cutoff, tie_margin)


class FakeEngine:
    """Records every call; ``scores`` maps row id -> cosine distance for score_by_ids.

    It has no ``search``: the routing rows of every tenant share a collection,
    so the read path must address rows by id, and a free search over the
    collection fails here.
    """

    def __init__(
        self,
        scores=None,
        existing=None,
        score_error=None,
        collection_exists=True,
        embed_error=None,
    ):
        self.scores = scores or {}
        self.existing = existing or {}
        self.score_error = score_error
        self.collection_exists = collection_exists
        self.embed_error = embed_error
        self.created, self.deleted, self.scored_ids, self.embedded = [], [], [], []
        self.vectors = {}  # row id -> the vector written by upsert_raw_vectors
        self.dropped = []
        self.embedding_engine = types.SimpleNamespace(
            embed_text=self._embed, get_vector_size=lambda: 2
        )

    async def _embed(self, texts):
        if self.embed_error is not None:
            raise self.embed_error
        self.embedded.extend(texts)
        return [[0.1, 0.2] for _ in texts]

    async def has_collection(self, collection_name):
        assert collection_name == TEST_COLLECTION
        return self.collection_exists

    async def existing_collections(self, collection_names):
        return list(collection_names)

    async def score_by_ids(self, collection_name, data_point_ids, query_vector):
        assert collection_name == TEST_COLLECTION
        self.scored_ids.extend(data_point_ids)
        if self.score_error is not None:
            raise self.score_error
        return [
            ScoredResult(id=UUID(row_id), score=self.scores[row_id], payload=None)
            for row_id in data_point_ids
            if row_id in self.scores
        ]

    async def retrieve(self, collection_name, data_point_ids):
        assert collection_name == TEST_COLLECTION
        return [
            ScoredResult(id=UUID(row_id), score=0, payload={"text": self.existing[row_id]})
            for row_id in data_point_ids
            if row_id in self.existing
        ]

    async def delete_data_points(self, collection_name, data_point_ids):
        assert collection_name.startswith("DatasetRoute")
        self.deleted.extend(data_point_ids)

    async def create_data_points(self, collection_name, data_points):
        raise AssertionError("the routing mirror must copy vectors, never embed its rows")

    async def upsert_raw_vectors(self, collection_name, points, payload_schema=None):
        assert collection_name == TEST_COLLECTION
        # No schema: the rows are index rows, copied from the dataset's store as stored.
        assert payload_schema is None
        for point in points:
            self.created.append(point["payload"])
            self.vectors[str(point["id"])] = point["vector"]

    async def delete_collection(self, collection_name):
        self.dropped.append(collection_name)


def _patch(monkeypatch, engine, documents, access_control=True):
    async def fake_data_ids(dataset_ids=None):
        if dataset_ids is None:
            return documents
        return {dataset_id: documents.get(dataset_id, []) for dataset_id in dataset_ids}

    async def fake_engine():
        return engine

    def fake_collection():
        return TEST_COLLECTION

    async def fake_registry_collections():
        return []

    for module in (index_module, route_module):
        monkeypatch.setattr(module, "get_dataset_route_engine", fake_engine)
        monkeypatch.setattr(module, "get_data_ids_by_dataset", fake_data_ids)
        monkeypatch.setattr(module, "dataset_route_collection", fake_collection)
    monkeypatch.setattr(index_module, "backend_access_control_enabled", lambda: access_control)
    monkeypatch.setattr(index_module, "_registry_route_collections", fake_registry_collections)


class FakeIndexStore:
    """The dataset's own store, holding the index rows the build embedded."""

    def __init__(self, vectors):
        self.vectors = vectors
        self.retrieved = []

    @staticmethod
    def stored_row(row_id):
        """The index row as ``index_data_points`` stored it, minus the vector."""
        return {"id": row_id, "text": f"indexed {row_id}", "belongs_to_set": []}

    async def retrieve(self, collection_name, data_point_ids, *, include_vector=False):
        assert collection_name == "GlobalContextSummary_text" and include_vector is True
        self.retrieved.extend(data_point_ids)
        return [
            ScoredResult(
                id=UUID(row_id), score=0, payload={**self.stored_row(row_id), "vector": vector}
            )
            for row_id, vector in self.vectors.items()
            if row_id in data_point_ids
        ]


def _summary(dataset_id, row_id, text, is_root=False):
    return GlobalContextSummary(
        id=row_id, text=text, dataset_id=str(dataset_id), level=1 if is_root else 0, is_root=is_root
    )


# --- ids -------------------------------------------------------------------


def test_route_ids_are_the_index_ids_derived_from_dataset_and_documents():
    dataset_id, doc = uuid4(), uuid4()

    assert dataset_route_ids(dataset_id, [doc]) == [
        str(create_root_summary_id(str(dataset_id))),
        str(create_document_bucket_id(str(dataset_id), str(doc))),
    ]


# --- read path ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_routes_to_the_dataset_whose_best_row_matches(monkeypatch):
    """B wins on one document bucket even though A's root is the closer root."""
    a, b, doc_a, doc_b = uuid4(), uuid4(), uuid4(), uuid4()
    root_a, bucket_a = dataset_route_ids(a, [doc_a])
    root_b, bucket_b = dataset_route_ids(b, [doc_b])
    engine = FakeEngine(scores={root_a: 0.4, bucket_a: 0.5, root_b: 0.6, bucket_b: 0.1})
    _patch(monkeypatch, engine, {a: [doc_a], b: [doc_b]})

    decision = await _route("q", [a, b], top_k=1)

    assert decision.routed == (b,)
    assert decision.unindexed == ()
    assert decision.dataset_ids == [b]
    assert decision.narrowed is True
    assert decision.scores == {a: 0.4, b: 0.1}
    # Only ids derived from the caller's own datasets are ever scored.
    assert sorted(engine.scored_ids) == sorted([root_a, bucket_a, root_b, bucket_b])


@pytest.mark.asyncio
async def test_a_dataset_with_no_routing_rows_is_left_out_when_another_is_routed(monkeypatch):
    """SDK-1516: an unindexed dataset waits for the search's widening pass; kept
    next to every routed one, a backlog of them meant routing never narrowed."""
    a, b, c = uuid4(), uuid4(), uuid4()
    (root_a,), (root_b,) = dataset_route_ids(a, []), dataset_route_ids(b, [])
    engine = FakeEngine(scores={root_a: 0.2, root_b: 0.3})
    _patch(monkeypatch, engine, {a: [], b: [], c: []})

    decision = await _route("q", [a, b, c], top_k=1)

    assert decision.routed == (a,)
    assert decision.unindexed == (c,)
    assert decision.dataset_ids == [a]
    assert decision.narrowed is True


@pytest.mark.asyncio
async def test_one_indexed_dataset_narrows_a_backlog_of_unindexed_ones(monkeypatch):
    """The case that kept routing from narrowing: one dataset indexed, the rest not."""
    indexed, *backlog = (uuid4() for _ in range(6))
    (root,) = dataset_route_ids(indexed, [])
    _patch(monkeypatch, FakeEngine(scores={root: 0.4}), {d: [] for d in [indexed, *backlog]})

    decision = await _route("q", [indexed, *backlog], top_k=1)

    assert decision.dataset_ids == [indexed]
    assert decision.unindexed == tuple(backlog)
    assert decision.narrowed is True


@pytest.mark.asyncio
async def test_with_no_dataset_indexed_every_dataset_is_searched(monkeypatch):
    a, b = uuid4(), uuid4()
    _patch(monkeypatch, FakeEngine(scores={}), {a: [], b: []})

    decision = await _route("q", [a, b], top_k=1)

    assert decision.routed == ()
    assert decision.unindexed == (a, b)
    assert decision.dataset_ids == [a, b]
    assert decision.narrowed is False


@pytest.mark.asyncio
async def test_top_k_widens_the_routed_set(monkeypatch):
    a, b, c = uuid4(), uuid4(), uuid4()
    (root_a,), (root_b,), (root_c,) = (dataset_route_ids(d, []) for d in (a, b, c))
    engine = FakeEngine(scores={root_a: 0.3, root_b: 0.1, root_c: 0.2})
    _patch(monkeypatch, engine, {a: [], b: [], c: []})

    decision = await _route("q", [a, b, c], top_k=2)

    assert decision.routed == (b, c)
    assert decision.narrowed is True
    assert (await _route("q", [a, b, c], top_k=3)).narrowed is False


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [CollectionNotFoundError("none"), NotImplementedError()])
async def test_no_collection_or_no_scoring_support_leaves_every_dataset_searched(
    monkeypatch, error
):
    a, b = uuid4(), uuid4()
    _patch(monkeypatch, FakeEngine(score_error=error), {a: [], b: []})

    decision = await _route("q", [a, b], top_k=1)

    assert decision.routed == ()
    assert decision.unindexed == (a, b)
    assert decision.narrowed is False


@pytest.mark.asyncio
async def test_no_routing_collection_means_no_embedding_call(monkeypatch):
    """A deployment that never built an index: a CODE or lexical search must not
    start needing an embedding provider for a routing step that cannot narrow."""
    a, b = uuid4(), uuid4()
    engine = FakeEngine(collection_exists=False)
    _patch(monkeypatch, engine, {a: [], b: []})

    decision = await _route("q", [a, b], top_k=1)

    assert decision.unindexed == (a, b) and decision.narrowed is False
    assert engine.embedded == [] and engine.scored_ids == []


@pytest.mark.asyncio
async def test_an_embedding_failure_warns_and_leaves_every_dataset_searched(monkeypatch, caplog):
    a, b = uuid4(), uuid4()
    engine = FakeEngine(embed_error=RuntimeError("provider down"))
    _patch(monkeypatch, engine, {a: [], b: []})

    with caplog.at_level("WARNING", logger="dataset_routing"):
        decision = await _route("q", [a, b], top_k=1)

    assert decision.unindexed == (a, b) and decision.narrowed is False
    assert engine.scored_ids == []
    assert "could not be scored" in caplog.text


@pytest.mark.asyncio
async def test_an_unreachable_routing_store_warns_and_leaves_every_dataset_searched(
    monkeypatch, caplog
):
    """Before routing no search touched the default store; a store that cannot be
    read (a database that does not exist on the vector host, a file another
    process holds) must not start failing every unscoped search."""
    a, b = uuid4(), uuid4()
    engine = FakeEngine()

    async def unreachable(collection_name):
        raise ConnectionError("database does not exist")

    engine.has_collection = unreachable
    _patch(monkeypatch, engine, {a: [], b: []})

    with caplog.at_level("WARNING", logger="dataset_routing"):
        decision = await _route("q", [a, b], top_k=1)

    assert decision.unindexed == (a, b) and decision.narrowed is False
    assert engine.embedded == []
    assert "could not be scored" in caplog.text


@pytest.mark.asyncio
async def test_the_configured_model_names_the_collection_the_read_path_scores(monkeypatch):
    """The rows of one model live in its own collection: the read path checks and
    scores only the collection of the configured model, so another model's rows
    are never ranked against this query."""
    a, b = uuid4(), uuid4()
    engine = FakeEngine(scores={})
    _patch(monkeypatch, engine, {a: [], b: []})
    seen = []

    async def spy_has_collection(collection_name):
        seen.append(collection_name)
        return False

    engine.has_collection = spy_has_collection

    decision = await _route("q", [a, b], top_k=1)

    assert seen == [TEST_COLLECTION]
    assert decision.unindexed == (a, b) and decision.narrowed is False
    assert engine.embedded == [] and engine.scored_ids == []


@pytest.mark.asyncio
async def test_top_k_below_one_is_refused(monkeypatch):
    _patch(monkeypatch, FakeEngine(), {})
    with pytest.raises(ValueError):
        await _route("q", [uuid4()], top_k=0)


# --- write path ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_query_no_dataset_is_close_to_is_searched_everywhere(monkeypatch):
    """Nothing within the cutoff: the decision covers every candidate, unindexed ones included."""
    a, b, c = uuid4(), uuid4(), uuid4()
    (root_a,), (root_b,) = dataset_route_ids(a, []), dataset_route_ids(b, [])
    engine = FakeEngine(scores={root_a: 0.87, root_b: 0.96})
    _patch(monkeypatch, engine, {a: [], b: [], c: []})

    decision = await _route("q", [a, b, c], distance_cutoff=0.8)

    assert decision.routed == ()
    assert decision.unindexed == (c,)
    assert decision.fanned_out is True
    assert decision.dataset_ids == [a, b, c]
    assert decision.narrowed is False
    assert decision.scores == {a: 0.87, b: 0.96}


@pytest.mark.asyncio
async def test_the_cutoff_drops_a_far_dataset_even_within_top_k(monkeypatch):
    a, b = uuid4(), uuid4()
    (root_a,), (root_b,) = dataset_route_ids(a, []), dataset_route_ids(b, [])
    engine = FakeEngine(scores={root_a: 0.5, root_b: 0.9})
    _patch(monkeypatch, engine, {a: [], b: []})

    decision = await _route("q", [a, b], top_k=2, distance_cutoff=0.8)

    assert decision.routed == (a,)
    assert decision.fanned_out is False
    assert decision.narrowed is True


@pytest.mark.asyncio
async def test_a_tie_for_the_top_spot_routes_every_tied_dataset(monkeypatch):
    """0.413 ties with 0.412 under a 0.001 margin (float noise included); 0.42 and 0.41 differ."""
    a, b, c = uuid4(), uuid4(), uuid4()
    roots = {dataset_id: dataset_route_ids(dataset_id, [])[0] for dataset_id in (a, b, c)}
    engine = FakeEngine(scores={roots[a]: 0.412, roots[b]: 0.413, roots[c]: 0.5})
    _patch(monkeypatch, engine, {a: [], b: [], c: []})

    decision = await _route("q", [a, b, c], top_k=1, tie_margin=0.001)

    assert decision.routed == (a, b)
    assert decision.narrowed is True

    engine.scores = {roots[a]: 0.41, roots[b]: 0.42, roots[c]: 0.5}
    assert (await _route("q", [a, b, c], top_k=1, tie_margin=0.001)).routed == (a,)


@pytest.mark.asyncio
async def test_ties_are_measured_against_the_boundary_and_never_chain(monkeypatch):
    a, b, c, d = (uuid4() for _ in range(4))
    roots = {dataset_id: dataset_route_ids(dataset_id, [])[0] for dataset_id in (a, b, c, d)}
    engine = FakeEngine(scores={roots[a]: 0.1, roots[b]: 0.2, roots[c]: 0.2009, roots[d]: 0.2018})
    _patch(monkeypatch, engine, {a: [], b: [], c: [], d: []})

    decision = await _route("q", [a, b, c, d], top_k=2, tie_margin=0.001)

    # c is within the margin of b, the boundary; d is within the margin of c but not of b.
    assert decision.routed == (a, b, c)


@pytest.mark.asyncio
async def test_sync_writes_new_and_changed_rows_and_deletes_the_rest(monkeypatch):
    dataset_id, kept, changed, gone = uuid4(), uuid4(), uuid4(), uuid4()
    root_id, kept_id, changed_id, gone_id = dataset_route_ids(dataset_id, [kept, changed, gone])
    engine = FakeEngine(
        existing={root_id: "root", changed_id: "old bucket text", gone_id: "orphan"}
    )
    _patch(monkeypatch, engine, {dataset_id: [kept, changed, gone]})
    desired = [
        _summary(dataset_id, UUID(root_id), "root", is_root=True),  # unchanged: not rewritten
        _summary(dataset_id, UUID(kept_id), "kept bucket"),  # missing from the collection
        _summary(dataset_id, UUID(changed_id), "new bucket text"),  # text changed
    ]

    index_store = FakeIndexStore({kept_id: [1.0, 0.0], changed_id: [0.0, 1.0], root_id: [0.5, 0.5]})

    touched = await sync_dataset_routes(str(dataset_id), desired, index_store)

    assert touched == 3
    assert engine.deleted == [UUID(gone_id)]
    # The rows are the index store's rows as stored, payload and vector alike;
    # nothing was embedded for them.
    assert sorted(engine.created, key=lambda row: row["id"]) == sorted(
        [index_store.stored_row(kept_id), index_store.stored_row(changed_id)],
        key=lambda row: row["id"],
    )
    assert engine.vectors == {kept_id: [1.0, 0.0], changed_id: [0.0, 1.0]}
    assert engine.embedded == []
    assert sorted(index_store.retrieved) == sorted([kept_id, changed_id])


@pytest.mark.asyncio
async def test_sync_ignores_rows_whose_ids_recall_cannot_derive(monkeypatch):
    """An llm-method cluster bucket has a clustering id: search could never score it."""
    dataset_id = uuid4()
    (root_id,) = dataset_route_ids(dataset_id, [])
    engine = FakeEngine()
    _patch(monkeypatch, engine, {dataset_id: []})

    await sync_dataset_routes(
        str(dataset_id),
        [
            _summary(dataset_id, UUID(root_id), "root", is_root=True),
            _summary(dataset_id, uuid4(), "cluster bucket"),
        ],
        FakeIndexStore({root_id: [0.1, 0.2]}),
    )

    assert [row["id"] for row in engine.created] == [root_id]


@pytest.mark.asyncio
async def test_sync_writes_into_the_collection_of_the_model_that_built_the_dataset(monkeypatch):
    """The strict collection asserts on the engine are the test: a build under the
    configured model mirrors into that model's collection, never another's."""
    dataset_id, doc = uuid4(), uuid4()
    engine = FakeEngine()
    _patch(monkeypatch, engine, {dataset_id: [doc]})
    (root_id,) = dataset_route_ids(dataset_id, [])
    root = _summary(dataset_id, UUID(root_id), "root", is_root=True)

    assert (
        await sync_dataset_routes(str(dataset_id), [root], FakeIndexStore({root_id: [0.1, 0.2]}))
        == 1
    )
    assert len(engine.created) == 1


@pytest.mark.asyncio
async def test_sync_is_a_no_op_without_backend_access_control(monkeypatch):
    dataset_id = uuid4()
    (root_id,) = dataset_route_ids(dataset_id, [])
    engine = FakeEngine()
    _patch(monkeypatch, engine, {dataset_id: []}, access_control=False)

    touched = await sync_dataset_routes(
        str(dataset_id),
        [_summary(dataset_id, UUID(root_id), "root", is_root=True)],
        FakeIndexStore({}),
    )

    assert touched == 0
    assert engine.created == [] and engine.deleted == []


@pytest.mark.asyncio
async def test_sync_fails_loudly_when_the_index_store_has_no_vector_for_a_row(monkeypatch):
    """The mirror copies, never embeds: an index row without a vector is a broken build."""
    dataset_id = uuid4()
    (root_id,) = dataset_route_ids(dataset_id, [])
    engine = FakeEngine()
    _patch(monkeypatch, engine, {dataset_id: []})

    with pytest.raises(LookupError, match="no vector"):
        await sync_dataset_routes(
            str(dataset_id),
            [_summary(dataset_id, UUID(root_id), "root", is_root=True)],
            FakeIndexStore({}),
        )
    assert engine.created == []


@pytest.mark.asyncio
async def test_sync_on_an_adapter_without_raw_vector_writes_warns_and_moves_on(monkeypatch, caplog):
    dataset_id = uuid4()
    (root_id,) = dataset_route_ids(dataset_id, [])

    class EngineWithoutRawWrites(FakeEngine):
        async def upsert_raw_vectors(self, collection_name, points, payload_schema=None):
            raise NotImplementedError

    engine = EngineWithoutRawWrites()
    _patch(monkeypatch, engine, {dataset_id: []})

    with caplog.at_level("WARNING"):
        touched = await sync_dataset_routes(
            str(dataset_id),
            [_summary(dataset_id, UUID(root_id), "root", is_root=True)],
            FakeIndexStore({root_id: [0.1, 0.2]}),
        )

    assert "cannot write precomputed vectors" in caplog.text
    assert touched == 0 and engine.created == []


@pytest.mark.asyncio
async def test_removing_a_dataset_deletes_its_root_and_every_bucket(monkeypatch):
    from cognee.modules.search.dataset_routing import remove_dataset_routes

    a, doc_a = uuid4(), uuid4()
    engine = FakeEngine()
    _patch(monkeypatch, engine, {a: [doc_a]})

    await remove_dataset_routes(a, [doc_a])

    # The fake reports every candidate collection as existing, so the deletion
    # ran against the configured model's and the legacy one alike.
    row_ids = {UUID(row_id) for row_id in dataset_route_ids(a, [doc_a])}
    assert set(engine.deleted) == row_ids
    assert len(engine.deleted) == 2 * len(row_ids)


@pytest.mark.asyncio
async def test_removing_a_dataset_covers_the_collections_of_every_recorded_model(monkeypatch):
    """A dataset moved across models carries rows in each model's collection; the
    deletion runs against the registry's collections too, not just the current one."""
    from cognee.modules.search.dataset_routing import remove_dataset_routes

    async def registry_collections():
        return [store_module.dataset_route_collection_name("old-model", 2)]

    a, doc_a = uuid4(), uuid4()
    engine = FakeEngine()
    _patch(monkeypatch, engine, {a: [doc_a]})
    monkeypatch.setattr(index_module, "_registry_route_collections", registry_collections)

    await remove_dataset_routes(a, [doc_a])

    row_ids = {UUID(row_id) for row_id in dataset_route_ids(a, [doc_a])}
    assert set(engine.deleted) == row_ids
    # legacy + current + the registry's old model
    assert len(engine.deleted) == 3 * len(row_ids)


@pytest.mark.asyncio
async def test_removing_a_document_deletes_its_bucket_and_the_root_not_the_others(monkeypatch):
    """Deletion-time removal: the bucket id derives from the Data row, so it is
    deleted while that row still exists; the root names every document, so it
    goes too, and the next build rewrites it. The other buckets stay."""
    from cognee.modules.search.dataset_routing import remove_dataset_routes

    a, gone, kept = uuid4(), uuid4(), uuid4()
    engine = FakeEngine()
    _patch(monkeypatch, engine, {a: [gone, kept]})

    await remove_dataset_routes(a, [gone])

    root, bucket = dataset_route_ids(a, [gone])
    assert set(engine.deleted) == {UUID(root), UUID(bucket)}
    _root, kept_bucket = dataset_route_ids(a, [kept])
    assert UUID(kept_bucket) not in engine.deleted


@pytest.mark.asyncio
async def test_prune_drops_every_routing_collection_rather_than_emptying_them(monkeypatch):
    from cognee.modules.search.dataset_routing import prune_dataset_routes

    engine = FakeEngine()
    _patch(monkeypatch, engine, {})

    await prune_dataset_routes()

    # The legacy collection an upgraded deployment still holds, plus the
    # configured model's own.
    assert set(engine.dropped) == {DATASET_ROUTE_COLLECTION, TEST_COLLECTION}
    assert engine.deleted == []


@pytest.mark.asyncio
async def test_prune_drops_the_collections_of_models_no_longer_configured(monkeypatch):
    from cognee.modules.search.dataset_routing import prune_dataset_routes

    async def registry_collections():
        return [store_module.dataset_route_collection_name("old-model", 2)]

    engine = FakeEngine()
    _patch(monkeypatch, engine, {})
    monkeypatch.setattr(index_module, "_registry_route_collections", registry_collections)

    await prune_dataset_routes()

    assert set(engine.dropped) == {
        DATASET_ROUTE_COLLECTION,
        TEST_COLLECTION,
        store_module.dataset_route_collection_name("old-model", 2),
    }


@pytest.mark.asyncio
async def test_prune_on_an_adapter_that_cannot_drop_warns_and_moves_on(monkeypatch, caplog):
    from cognee.modules.search.dataset_routing import prune_dataset_routes

    class EngineWithoutDrop(FakeEngine):
        async def delete_collection(self, collection_name):
            raise NotImplementedError

    _patch(monkeypatch, EngineWithoutDrop(), {})

    with caplog.at_level("WARNING", logger="dataset_routing"):
        await prune_dataset_routes()

    assert "cannot drop a collection" in caplog.text


@pytest.mark.asyncio
async def test_removal_is_a_no_op_without_backend_access_control(monkeypatch):
    from cognee.modules.search.dataset_routing import (
        prune_dataset_routes,
        remove_dataset_routes,
    )

    a, doc = uuid4(), uuid4()
    engine = FakeEngine()
    _patch(monkeypatch, engine, {a: [doc]}, access_control=False)

    await remove_dataset_routes(a, [doc])
    await prune_dataset_routes()

    assert engine.deleted == [] and engine.dropped == []


# --- engine -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_route_engine_embeds_with_the_deployment_settings_not_a_per_call_override(
    monkeypatch,
):
    """The engine cache keys on the store alone and bakes in the embedder of whoever
    creates the engine: the shared routing engine must be created with the per-call
    LLM and embedding overrides cleared, whatever context first touches it."""
    from cognee.context_global_variables import embedding_config, llm_config
    from cognee.modules.search.dataset_routing import get_dataset_route_engine

    async def no_store_setup():
        return None

    monkeypatch.setattr(store_module, "ensure_default_store", no_store_setup)

    handle_module = import_module("cognee.infrastructure.databases.vector.get_vector_engine")

    seen = []

    def fake_create_vector_engine(**config):
        seen.append((embedding_config.get(), llm_config.get()))
        return types.SimpleNamespace(name="routing-engine")

    monkeypatch.setattr(handle_module, "create_vector_engine", fake_create_vector_engine)
    override = types.SimpleNamespace(embedding_endpoint="https://tenant-a.example")
    embedding_token = embedding_config.set(override)
    llm_token = llm_config.set(override)
    try:
        assert (await get_dataset_route_engine()).name == "routing-engine"
        # The caller's own context keeps its override.
        assert embedding_config.get() is override and llm_config.get() is override
    finally:
        llm_config.reset(llm_token)
        embedding_config.reset(embedding_token)

    assert seen == [(None, None)]


# --- per-model collections ----------------------------------------------------


def test_collection_names_are_deterministic_and_model_specific():
    small = store_module.dataset_route_collection_name("openai/text-embedding-3-small", 1536)
    assert small == store_module.dataset_route_collection_name(
        "openai/text-embedding-3-small", 1536
    )
    # Another width, another model, a different collection each.
    assert small != store_module.dataset_route_collection_name(
        "openai/text-embedding-3-large", 3072
    )
    assert small != store_module.dataset_route_collection_name("other/text-embedding-3-small", 1536)


def test_collection_names_are_storage_safe_and_bounded():
    """Postgres caps identifiers at 63 bytes, and adapters accept the same charset."""
    name = store_module.dataset_route_collection_name("openai/text-embedding-3-small", 1536)
    assert name.startswith("DatasetRoute_text__")
    assert re.fullmatch(r"[\w-]+", name)
    assert len(name) <= 63
    sprawling = store_module.dataset_route_collection_name("org/" + "x" * 120, 384)
    assert len(sprawling) <= 63


def test_collection_names_keep_sanitized_and_truncated_models_apart():
    # "model/a" and "model_a" reduce to the same readable part; the digest of
    # the full pair is what keeps their collections apart.
    assert store_module.dataset_route_collection_name("model/a", 384) != (
        store_module.dataset_route_collection_name("model_a", 384)
    )
    assert store_module.dataset_route_collection_name("org/" + "a" * 60, 384) != (
        store_module.dataset_route_collection_name("org/" + "a" * 61, 384)
    )


def test_a_model_without_a_name_shares_no_collection():
    assert store_module.dataset_route_collection_name(None, 384) != (
        store_module.dataset_route_collection_name("unknown", 384)
    )


def test_the_route_collection_resolves_the_deployment_model_not_a_per_call_override(monkeypatch):
    """The rows are mirrored under the deployment's model: naming the collection
    after a caller's override would score it against a collection the writes
    never filled."""
    from cognee.context_global_variables import embedding_config, llm_config

    seen = []

    def fake_record():
        seen.append((embedding_config.get(), llm_config.get()))
        return {
            model_utils_module.MODEL_KEY: "deployed-model",
            model_utils_module.DIMENSIONS_KEY: 512,
        }

    monkeypatch.setattr(model_utils_module, "embedding_model_record", fake_record)
    override = types.SimpleNamespace(model="per-call-model")
    embedding_token = embedding_config.set(override)
    llm_token = llm_config.set(override)
    try:
        assert store_module.dataset_route_collection() == (
            store_module.dataset_route_collection_name("deployed-model", 512)
        )
        # The caller's own context keeps its override.
        assert embedding_config.get() is override and llm_config.get() is override
    finally:
        llm_config.reset(llm_token)
        embedding_config.reset(embedding_token)

    assert seen == [(None, None)]


def _pgvector_store(
    monkeypatch, *, access_control=True, vector_db_name="", relational_db_name="cognee_db"
):
    """A pgvector default store under access control, with a recorder in place of CREATE DATABASE."""
    created = []

    async def fake_create(db_name, host, port, username, password):
        created.append((db_name, host, port, username, password))
        return True

    monkeypatch.setattr(store_module, "_default_store_ensured", False)
    monkeypatch.setattr(
        store_module,
        "get_vectordb_config",
        lambda: types.SimpleNamespace(
            vector_db_provider="pgvector",
            vector_db_name=vector_db_name,
            vector_db_host="vectors.example",
            vector_db_port=5432,
            vector_db_username="cognee",
            vector_db_password="secret",
            to_dict=dict,
        ),
    )
    monkeypatch.setattr(
        "cognee.context_global_variables.backend_access_control_enabled", lambda: access_control
    )
    monkeypatch.setattr(
        "cognee.infrastructure.databases.postgres.admin.create_pg_database_if_not_exists",
        fake_create,
    )
    monkeypatch.setattr(
        "cognee.infrastructure.databases.relational.get_relational_config",
        lambda: types.SimpleNamespace(db_name=relational_db_name),
    )
    return created


@pytest.mark.asyncio
async def test_the_pgvector_default_store_is_created_on_first_use_with_sqlite_relational(
    monkeypatch,
):
    """No VECTOR_DB_NAME: the store is named after the relational database, which with
    SQLite does not exist on the vector host, so the routing store creates it there,
    once, with the vector credentials the dataset handler uses."""
    created = _pgvector_store(monkeypatch, relational_db_name="cognee_db")

    await store_module.ensure_default_store()
    await store_module.ensure_default_store()

    assert created == [("cognee_db", "vectors.example", 5432, "cognee", "secret")]


@pytest.mark.asyncio
async def test_an_explicit_vector_db_name_names_the_default_store(monkeypatch):
    created = _pgvector_store(monkeypatch, vector_db_name="routing_store")

    await store_module.ensure_default_store()

    assert [name for name, *_ in created] == ["routing_store"]


@pytest.mark.asyncio
async def test_no_database_is_created_without_access_control_or_outside_pgvector(monkeypatch):
    created = _pgvector_store(monkeypatch, access_control=False)
    await store_module.ensure_default_store()
    assert created == []

    created = _pgvector_store(monkeypatch)
    monkeypatch.setattr(
        store_module,
        "get_vectordb_config",
        lambda: types.SimpleNamespace(vector_db_provider="lancedb", to_dict=dict),
    )
    await store_module.ensure_default_store()
    assert created == []
