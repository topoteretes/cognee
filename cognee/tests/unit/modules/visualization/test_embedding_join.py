"""Unit tests for the embedding join — the seam between graph nodes and vectors.

Fully offline and deterministic: a fake vector engine stands in for the real
adapter (the single mocking seam), so there are no LLM calls and no live store.
Covers collection resolution, batched retrieve, the re-embed fallback for adapters
without ``include_vector`` support, and the deterministic over-cap sample.
"""

import asyncio
from types import SimpleNamespace

import pytest

from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult
from cognee.modules.visualization.embedding_join import (
    DEFAULT_INDEX_FIELDS,
    fetch_node_embeddings,
)


class FakeEmbeddingEngine:
    """Deterministic embed_text: one call, one vector per text, length-based."""

    def __init__(self):
        self.calls = []

    async def embed_text(self, texts):
        self.calls.append(list(texts))
        return [[float(len(t)), 1.0, 2.0] for t in texts]


class FakeVectorEngine:
    """Stands in for a real adapter. ``store`` maps collection -> {id: vector}."""

    def __init__(self, store, supports_include_vector=True):
        self.store = store
        self.supports_include_vector = supports_include_vector
        self.embedding_engine = FakeEmbeddingEngine()
        self.retrieve_calls = []

    async def has_collection(self, collection_name):
        return collection_name in self.store

    async def retrieve(self, collection_name, data_point_ids, **kwargs):
        if not self.supports_include_vector:
            # Mimic an unmodified adapter whose retrieve() rejects the kwarg.
            raise TypeError("retrieve() got an unexpected keyword argument 'include_vector'")
        include_vector = kwargs.get("include_vector", False)
        self.retrieve_calls.append((collection_name, list(data_point_ids)))
        rows = self.store.get(collection_name, {})
        results = []
        for did in data_point_ids:
            if did in rows:
                payload = {"text": f"payload-{did}"}
                if include_vector:
                    payload = {**payload, "vector": rows[did]}
                results.append(ScoredResult(id=did, payload=payload, score=0))
        return results


# ScoredResult.id is a UUID, so fixtures use canonical UUID strings.
E1 = "11111111-1111-1111-1111-111111111111"
E2 = "22222222-2222-2222-2222-222222222222"
T1 = "33333333-3333-3333-3333-333333333333"
G1 = "44444444-4444-4444-4444-444444444444"
S1 = "55555555-5555-5555-5555-555555555555"
C1 = "66666666-6666-6666-6666-666666666666"


def _node(nid, ntype, **extra):
    return {"id": nid, "type": ntype, "name": f"name-{nid}", **extra}


def test_collection_resolution_one_batched_retrieve_per_type():
    # Two Entity nodes + one EntityType node -> one retrieve on Entity_name and
    # one on EntityType_name, each batched with the correct ids.
    store = {
        "Entity_name": {E1: [0.1, 0.2, 0.3], E2: [0.4, 0.5, 0.6]},
        "EntityType_name": {T1: [0.7, 0.8, 0.9]},
    }
    engine = FakeVectorEngine(store)
    nodes = [_node(E1, "Entity"), _node(E2, "Entity"), _node(T1, "EntityType")]

    result = asyncio.run(fetch_node_embeddings(nodes, vector_engine=engine))

    assert result == {
        E1: [0.1, 0.2, 0.3],
        E2: [0.4, 0.5, 0.6],
        T1: [0.7, 0.8, 0.9],
    }
    # Exactly one batched retrieve per collection, never per node.
    called = {c[0]: c[1] for c in engine.retrieve_calls}
    assert set(called) == {"Entity_name", "EntityType_name"}
    assert len(engine.retrieve_calls) == 2
    assert set(called["Entity_name"]) == {E1, E2}


def test_unknown_type_and_missing_collection_skipped():
    store = {"Entity_name": {E1: [1.0, 0.0, 0.0]}}
    engine = FakeVectorEngine(store)
    nodes = [
        _node(E1, "Entity"),
        _node(G1, "GraphNodeType"),  # not in DEFAULT_INDEX_FIELDS -> skipped
        _node(S1, "TextSummary"),  # known type but no collection -> skipped
    ]

    result = asyncio.run(fetch_node_embeddings(nodes, vector_engine=engine))

    assert result == {E1: [1.0, 0.0, 0.0]}
    assert [c[0] for c in engine.retrieve_calls] == ["Entity_name"]


def test_missing_ids_absent_not_reembedded():
    # Collection exists and returns vectors for present ids; ids absent from the
    # store are simply missing (layout handles them), NOT re-embedded.
    store = {"Entity_name": {E1: [0.1, 0.1, 0.1]}}
    engine = FakeVectorEngine(store)
    nodes = [_node(E1, "Entity"), _node(E2, "Entity")]

    result = asyncio.run(fetch_node_embeddings(nodes, vector_engine=engine))

    assert result == {E1: [0.1, 0.1, 0.1]}
    assert engine.embedding_engine.calls == []  # no re-embed


def test_reembed_fallback_when_include_vector_unsupported():
    # Adapter without include_vector support: every node is positioned via one
    # batched embed_text call.
    store = {"Entity_name": {E1: [9.9], E2: [9.9]}}
    engine = FakeVectorEngine(store, supports_include_vector=False)
    nodes = [_node(E1, "Entity", name="abc"), _node(E2, "Entity", name="abcd")]

    result = asyncio.run(fetch_node_embeddings(nodes, vector_engine=engine))

    # Fallback embeds the indexed field (name) once, in a single batch.
    assert len(engine.embedding_engine.calls) == 1
    assert engine.embedding_engine.calls[0] == ["abc", "abcd"]
    assert result == {E1: [3.0, 1.0, 2.0], E2: [4.0, 1.0, 2.0]}


def test_reembed_uses_text_field_for_text_types():
    store = {"DocumentChunk_text": {C1: [0.0]}}
    engine = FakeVectorEngine(store, supports_include_vector=False)
    nodes = [{"id": C1, "type": "DocumentChunk", "name": "ignored", "text": "hello"}]

    asyncio.run(fetch_node_embeddings(nodes, vector_engine=engine))

    assert engine.embedding_engine.calls == [["hello"]]


def test_sampling_cap_deterministic():
    nodes = [_node(f"{i:05d}", "Entity") for i in range(3000)]

    # Cap at 2000 and assert two runs agree (deterministic seeded sample).
    from cognee.modules.visualization import embedding_join

    picked1 = embedding_join._select_nodes(nodes, 2000)
    picked2 = embedding_join._select_nodes(nodes, 2000)
    assert len(picked1) == 2000
    assert [n["id"] for n in picked1] == [n["id"] for n in picked2]
    # id-sorted output
    assert [n["id"] for n in picked1] == sorted(n["id"] for n in picked1)


def test_default_index_fields_cover_core_types():
    for t in ("Entity", "EntityType", "TextSummary", "DocumentChunk", "TextDocument"):
        assert t in DEFAULT_INDEX_FIELDS
