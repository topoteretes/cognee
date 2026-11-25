"""A typed scope on a chunk search: the graph projects it, the vector store scores its ids."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.exceptions import CogneeConfigurationError
from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult
from cognee.infrastructure.databases.vector.vector_db_interface import VectorDBInterface
from cognee.modules.data.processing.document_types import TextDocument
from cognee.modules.retrieval.utils.scoped_chunk_search import (
    scoped_chunk_ids,
    search_scoped_chunks,
)

try:
    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import (
        IndexSchema,
        LanceDBAdapter,
    )

    HAS_LANCEDB = True
except ModuleNotFoundError:
    HAS_LANCEDB = False

DOC_ID, CHUNK_A, CHUNK_B, CHUNK_C = (str(uuid4()) for _ in range(4))
OUT_OF_SCOPE = str(uuid4())


class _GraphEngine:
    """get_nodeset_subgraph: the document plus its chunks, as the adapters return them."""

    def __init__(self):
        self.calls = []

    async def get_nodeset_subgraph(self, node_type, node_name, node_name_filter_operator="OR"):
        self.calls.append((node_type, node_name, node_name_filter_operator))
        if node_name != ["doc_a"]:
            return [], []
        nodes = [
            (DOC_ID, {"id": DOC_ID, "name": "doc_a", "type": "TextDocument"}),
            (CHUNK_A, {"id": CHUNK_A, "type": "DocumentChunk"}),
            (CHUNK_B, {"id": CHUNK_B, "type": "DocumentChunk"}),
            (CHUNK_C, {"id": CHUNK_C, "type": "DocumentChunk"}),
        ]
        return nodes, [(CHUNK_A, DOC_ID, "is_part_of", {})]


class _Embedder:
    async def embed_text(self, texts):
        return [[1.0, 0.0] for _ in texts]


_DISTANCES = {CHUNK_A: 0.1, CHUNK_B: 0.5, CHUNK_C: 0.3, OUT_OF_SCOPE: 0.01}
_PAYLOADS = {chunk_id: {"text": f"text {i}"} for i, chunk_id in enumerate(_DISTANCES)}


class _ScoringEngine(VectorDBInterface):
    """An adapter with score_by_ids; abstract methods stubbed as needed."""

    embedding_engine = _Embedder()

    def __init__(self):
        self.scored_ids = None
        self.searched = False

    async def score_by_ids(self, collection_name, data_point_ids, query_vector):
        self.scored_ids = list(data_point_ids)
        return [
            ScoredResult(id=chunk_id, score=_DISTANCES[chunk_id], payload=None)
            for chunk_id in data_point_ids
            if chunk_id in _DISTANCES
        ]

    async def search(self, *args, **kwargs):
        self.searched = True
        raise AssertionError("an adapter with score_by_ids must not scan")

    async def retrieve(self, collection_name, data_point_ids, *, include_vector=False):
        return [
            SimpleNamespace(id=chunk_id, payload=_PAYLOADS[chunk_id]) for chunk_id in data_point_ids
        ]

    # Unused abstract methods.
    async def embed_data(self, *a, **k): ...
    async def has_collection(self, *a, **k): ...
    async def create_collection(self, *a, **k): ...
    async def create_data_points(self, *a, **k): ...
    async def batch_search(self, *a, **k): ...
    async def delete_data_points(self, *a, **k): ...
    async def prune(self, *a, **k): ...


class _NoIdScoringEngine(_ScoringEngine):
    """An adapter without score_by_ids: inherits the interface stub."""

    score_by_ids = VectorDBInterface.score_by_ids

    async def search(self, *args, **kwargs):
        raise AssertionError("an adapter without score_by_ids must not be scanned")


@pytest.mark.asyncio
async def test_scope_is_the_chunk_nodes_of_the_projection():
    graph = _GraphEngine()
    ids = await scoped_chunk_ids(graph, TextDocument, ["doc_a"], "AND")
    assert sorted(ids) == sorted([CHUNK_A, CHUNK_B, CHUNK_C])
    assert graph.calls == [(TextDocument, ["doc_a"], "AND")]


@pytest.mark.asyncio
async def test_id_path_scores_only_the_scope_and_ranks_nearest_first():
    engine = _ScoringEngine()
    results = await search_scoped_chunks(
        engine, _GraphEngine(), "apples", TextDocument, ["doc_a"], "OR", limit=10
    )
    assert sorted(engine.scored_ids) == sorted([CHUNK_A, CHUNK_B, CHUNK_C])
    assert [str(r.id) for r in results] == [CHUNK_A, CHUNK_C, CHUNK_B]
    assert [r.score for r in results] == [0.1, 0.3, 0.5]
    assert results[0].payload == _PAYLOADS[CHUNK_A]
    assert engine.searched is False


@pytest.mark.asyncio
async def test_limit_cuts_after_ranking():
    results = await search_scoped_chunks(
        _ScoringEngine(), _GraphEngine(), "apples", TextDocument, ["doc_a"], "OR", limit=2
    )
    assert [str(r.id) for r in results] == [CHUNK_A, CHUNK_C]


@pytest.mark.asyncio
async def test_an_adapter_without_id_scoring_is_refused_not_scanned():
    engine = _NoIdScoringEngine()
    with pytest.raises(CogneeConfigurationError) as excinfo:
        await search_scoped_chunks(
            engine, _GraphEngine(), "apples", TextDocument, ["doc_a"], "OR", limit=10
        )
    assert "score_by_ids" in str(excinfo.value)
    assert "_NoIdScoringEngine" in str(excinfo.value)


@pytest.mark.asyncio
async def test_empty_projection_returns_nothing_without_touching_the_vector_store():
    engine = _ScoringEngine()
    results = await search_scoped_chunks(
        engine, _GraphEngine(), "apples", TextDocument, ["missing"], "OR", limit=10
    )
    assert results == []
    assert engine.scored_ids is None and engine.searched is False


class _KeywordEmbeddingEngine:
    def get_vector_size(self):
        return 3

    def get_batch_size(self):
        return 100

    async def embed_text(self, texts):
        return [[1.0, 0.0, 0.1] if "apple" in text.lower() else [0.0, 1.0, 0.1] for text in texts]


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb extra is not installed")
async def test_lancedb_scores_the_projected_chunks(tmp_path):
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"), api_key=None, embedding_engine=_KeywordEmbeddingEngine()
    )
    rows = {
        CHUNK_A: "Apples are red fruit",
        CHUNK_B: "Bananas are yellow fruit",
        CHUNK_C: "Apples grow on trees",
        OUT_OF_SCOPE: "Apple pie recipe from another document",
    }
    await adapter.create_collection("DocumentChunk_text", IndexSchema)
    await adapter.create_data_points(
        "DocumentChunk_text",
        [IndexSchema(id=chunk_id, text=text) for chunk_id, text in rows.items()],
    )

    results = await search_scoped_chunks(
        adapter, _GraphEngine(), "apple", TextDocument, ["doc_a"], "OR", limit=10
    )

    # The two apple chunks tie on distance and rank ahead of the banana chunk; the
    # nearer out-of-scope row never appears.
    assert {str(r.id) for r in results[:2]} == {CHUNK_A, CHUNK_C}
    assert {str(r.id) for r in results} == {CHUNK_A, CHUNK_B, CHUNK_C}
    assert all(r.payload["text"] == rows[str(r.id)] for r in results)
    assert [r.score for r in results] == sorted(r.score for r in results)
