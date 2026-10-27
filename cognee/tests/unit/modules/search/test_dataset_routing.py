"""Dataset routing: the routing collection's write path and the read path.

No database anywhere: the relational lookup and the vector engine are stubbed.
"""

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


class FakeEngine:
    """Records every call; ``scores`` maps row id -> cosine distance for score_by_ids."""

    def __init__(self, scores=None, existing=None, score_error=None):
        self.scores = scores or {}
        self.existing = existing or {}
        self.score_error = score_error
        self.created, self.deleted, self.scored_ids = [], [], []
        self.embedding_engine = types.SimpleNamespace(embed_text=self._embed)

    async def _embed(self, texts):
        return [[0.1, 0.2] for _ in texts]

    async def score_by_ids(self, collection_name, data_point_ids, query_vector):
        assert collection_name == DATASET_ROUTE_COLLECTION
        self.scored_ids.extend(data_point_ids)
        if self.score_error is not None:
            raise self.score_error
        return [
            ScoredResult(id=UUID(row_id), score=self.scores[row_id], payload=None)
            for row_id in data_point_ids
            if row_id in self.scores
        ]

    async def retrieve(self, collection_name, data_point_ids):
        assert collection_name == DATASET_ROUTE_COLLECTION
        return [
            ScoredResult(id=UUID(row_id), score=0, payload={"text": self.existing[row_id]})
            for row_id in data_point_ids
            if row_id in self.existing
        ]

    async def delete_data_points(self, collection_name, data_point_ids):
        assert collection_name == DATASET_ROUTE_COLLECTION
        self.deleted.extend(data_point_ids)

    async def create_data_points(self, collection_name, data_points):
        assert collection_name == DATASET_ROUTE_COLLECTION
        self.created.extend(data_points)


def _patch(monkeypatch, engine, documents, access_control=True):
    async def fake_data_ids(dataset_ids=None):
        if dataset_ids is None:
            return documents
        return {dataset_id: documents.get(dataset_id, []) for dataset_id in dataset_ids}

    for module in (index_module, route_module):
        monkeypatch.setattr(module, "get_dataset_route_engine", lambda: engine)
        monkeypatch.setattr(module, "get_data_ids_by_dataset", fake_data_ids)
    monkeypatch.setattr(index_module, "backend_access_control_enabled", lambda: access_control)


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

    decision = await route_datasets("q", [a, b], top_k=1)

    assert decision.routed == (b,)
    assert decision.unindexed == ()
    assert decision.dataset_ids == [b]
    assert decision.narrowed is True
    assert decision.scores == {a: 0.4, b: 0.1}
    # Only ids derived from the caller's own datasets are ever scored.
    assert sorted(engine.scored_ids) == sorted([root_a, bucket_a, root_b, bucket_b])


@pytest.mark.asyncio
async def test_a_dataset_with_no_routing_rows_is_always_searched(monkeypatch):
    a, b, c = uuid4(), uuid4(), uuid4()
    (root_a,), (root_b,) = dataset_route_ids(a, []), dataset_route_ids(b, [])
    engine = FakeEngine(scores={root_a: 0.2, root_b: 0.3})
    _patch(monkeypatch, engine, {a: [], b: [], c: []})

    decision = await route_datasets("q", [a, b, c], top_k=1)

    assert decision.routed == (a,)
    assert decision.unindexed == (c,)
    assert decision.dataset_ids == [a, c]
    assert decision.narrowed is True


@pytest.mark.asyncio
async def test_top_k_widens_the_routed_set(monkeypatch):
    a, b, c = uuid4(), uuid4(), uuid4()
    (root_a,), (root_b,), (root_c,) = (dataset_route_ids(d, []) for d in (a, b, c))
    engine = FakeEngine(scores={root_a: 0.3, root_b: 0.1, root_c: 0.2})
    _patch(monkeypatch, engine, {a: [], b: [], c: []})

    decision = await route_datasets("q", [a, b, c], top_k=2)

    assert decision.routed == (b, c)
    assert decision.narrowed is True
    assert (await route_datasets("q", [a, b, c], top_k=3)).narrowed is False


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [CollectionNotFoundError("none"), NotImplementedError()])
async def test_no_collection_or_no_scoring_support_leaves_every_dataset_searched(
    monkeypatch, error
):
    a, b = uuid4(), uuid4()
    _patch(monkeypatch, FakeEngine(score_error=error), {a: [], b: []})

    decision = await route_datasets("q", [a, b], top_k=1)

    assert decision.routed == ()
    assert decision.unindexed == (a, b)
    assert decision.narrowed is False


@pytest.mark.asyncio
async def test_top_k_below_one_is_refused(monkeypatch):
    _patch(monkeypatch, FakeEngine(), {})
    with pytest.raises(ValueError):
        await route_datasets("q", [uuid4()], top_k=0)


# --- write path ---------------------------------------------------------------


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

    touched = await sync_dataset_routes(str(dataset_id), desired)

    assert touched == 3
    assert engine.deleted == [UUID(gone_id)]
    assert sorted((str(row.id), row.text) for row in engine.created) == sorted(
        [(kept_id, "kept bucket"), (changed_id, "new bucket text")]
    )
    assert all(row.dataset_id == str(dataset_id) for row in engine.created)
    assert {str(row.id): row.is_root for row in engine.created} == {
        kept_id: False,
        changed_id: False,
    }


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
    )

    assert [str(row.id) for row in engine.created] == [root_id]


@pytest.mark.asyncio
async def test_sync_is_a_no_op_without_backend_access_control(monkeypatch):
    dataset_id = uuid4()
    (root_id,) = dataset_route_ids(dataset_id, [])
    engine = FakeEngine()
    _patch(monkeypatch, engine, {dataset_id: []}, access_control=False)

    touched = await sync_dataset_routes(
        str(dataset_id), [_summary(dataset_id, UUID(root_id), "root", is_root=True)]
    )

    assert touched == 0
    assert engine.created == [] and engine.deleted == []


@pytest.mark.asyncio
async def test_remove_and_prune_delete_every_derivable_row(monkeypatch):
    from cognee.modules.search.dataset_routing import prune_dataset_routes, remove_dataset_routes

    a, b, doc_a = uuid4(), uuid4(), uuid4()
    engine = FakeEngine()
    _patch(monkeypatch, engine, {a: [doc_a], b: []})

    await remove_dataset_routes(a, [doc_a])
    assert engine.deleted == [UUID(row_id) for row_id in dataset_route_ids(a, [doc_a])]

    engine.deleted.clear()
    await prune_dataset_routes()
    assert sorted(map(str, engine.deleted)) == sorted(
        dataset_route_ids(a, [doc_a]) + dataset_route_ids(b, [])
    )
