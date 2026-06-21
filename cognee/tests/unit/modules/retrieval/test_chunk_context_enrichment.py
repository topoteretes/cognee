import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from cognee.modules.retrieval.chunk_context_enrichment import (
    enrich_chunk_results,
    extract_chunk_ids,
    fetch_neighbor_chunks,
    fetch_parent_documents,
)
from cognee.modules.retrieval.chunks_retriever import ChunksRetriever
from cognee.modules.retrieval.exceptions.exceptions import NoDataError


def test_extract_chunk_ids_validates_uuid_format():
    valid_id = str(uuid4())
    chunk = SimpleNamespace(id=valid_id, payload={"text": "hello"})
    chunk_ids, chunk_map = extract_chunk_ids([chunk])
    assert chunk_ids == [valid_id]
    assert chunk_map[valid_id] is chunk

    invalid_ids, _ = extract_chunk_ids([SimpleNamespace(id="not-a-uuid", payload={})])
    assert invalid_ids == []


@pytest.mark.asyncio
async def test_fetch_parent_documents_batched_lookup():
    graph = AsyncMock()
    chunk_id = str(uuid4())
    doc_id = str(uuid4())
    graph.query = AsyncMock(return_value=[(chunk_id, doc_id, "report.pdf", "PdfDocument")])

    parent_map = await fetch_parent_documents(graph, [chunk_id])
    assert parent_map[chunk_id]["id"] == doc_id
    assert parent_map[chunk_id]["name"] == "report.pdf"


@pytest.mark.asyncio
async def test_fetch_neighbor_chunks_filters_by_index_window():
    graph = AsyncMock()
    chunk_id = str(uuid4())
    neighbor_id = str(uuid4())
    graph.query = AsyncMock(
        return_value=[
            (chunk_id, 2, neighbor_id, 3, "neighbor text"),
            (chunk_id, 2, chunk_id, 2, "target text"),
        ]
    )

    neighbors = await fetch_neighbor_chunks(graph, [chunk_id], neighbor_count=1)
    assert len(neighbors[chunk_id]) == 1
    assert neighbors[chunk_id][0]["chunk_id"] == neighbor_id


@pytest.mark.asyncio
async def test_enrich_chunk_results_attaches_parent_and_neighbors():
    graph = AsyncMock()
    chunk_id = str(uuid4())
    doc_id = str(uuid4())
    neighbor_id = str(uuid4())

    async def _query_side_effect(query, params=None):
        if "sibling" in query:
            return [(chunk_id, 1, neighbor_id, 2, "next")]
        return [(chunk_id, doc_id, "Doc", "TextDocument")]

    graph.query = AsyncMock(side_effect=_query_side_effect)
    chunk = SimpleNamespace(id=chunk_id, payload={"id": chunk_id, "text": "hit"})

    enriched = await enrich_chunk_results(
        graph, [chunk], expand_neighbors=1, strict_enrichment=True
    )

    assert enriched[0].payload["parent_document"]["id"] == doc_id
    assert enriched[0].payload["is_part_of"]["id"] == doc_id
    assert enriched[0].payload["neighboring_chunks"][0]["chunk_id"] == neighbor_id


@pytest.mark.asyncio
async def test_chunks_retriever_enriches_results():
    chunk_id = str(uuid4())
    doc_id = str(uuid4())
    mock_result = SimpleNamespace(id=chunk_id, payload={"id": chunk_id, "text": "alpha"})
    mock_vector_engine = AsyncMock()
    mock_vector_engine.search = AsyncMock(return_value=[mock_result])

    graph = AsyncMock()
    graph.query = AsyncMock(return_value=[(chunk_id, doc_id, "Doc", "TextDocument")])

    unified = AsyncMock()
    unified.vector = mock_vector_engine
    unified.graph = graph

    retriever = ChunksRetriever(top_k=1)

    with patch(
        "cognee.modules.retrieval.chunks_retriever.get_unified_engine",
        return_value=unified,
    ):
        objects = await retriever.get_retrieved_objects("alpha")

    assert objects[0].payload["parent_document"]["id"] == doc_id


@pytest.mark.asyncio
async def test_chunks_retriever_skips_enrichment_when_graph_unavailable():
    mock_vector_engine = AsyncMock()
    mock_vector_engine.search = AsyncMock(
        return_value=[SimpleNamespace(id=str(uuid4()), payload={"text": "x"})]
    )
    unified = AsyncMock()
    unified.vector = mock_vector_engine
    type(unified).graph = property(lambda self: (_ for _ in ()).throw(RuntimeError("no graph")))

    retriever = ChunksRetriever(strict_enrichment=False)

    with patch(
        "cognee.modules.retrieval.chunks_retriever.get_unified_engine",
        return_value=unified,
    ):
        objects = await retriever.get_retrieved_objects("alpha")

    assert len(objects) == 1
