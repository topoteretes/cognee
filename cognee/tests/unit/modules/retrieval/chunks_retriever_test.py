import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch, MagicMock

from cognee.modules.retrieval.chunks_retriever import ChunksRetriever
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError


@pytest.fixture
def mock_vector_engine():
    engine = AsyncMock()
    engine.search = AsyncMock()
    return engine


@pytest.mark.asyncio
async def test_get_context_success(mock_vector_engine):
    mock_result1 = MagicMock()
    mock_result1.payload = {"text": "Steve Rodger", "chunk_index": 0}
    mock_result2 = MagicMock()
    mock_result2.payload = {"text": "Mike Broski", "chunk_index": 1}

    mock_vector_engine.search.return_value = [mock_result1, mock_result2]

    retriever = ChunksRetriever(top_k=5)

    with patch(
        "cognee.modules.retrieval.chunks_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        objects = await retriever.get_retrieved_objects("test query")

    assert len(objects) == 2
    assert objects[0].payload["text"] == "Steve Rodger"
    assert objects[1].payload["text"] == "Mike Broski"
    mock_vector_engine.search.assert_awaited_once_with(
        "DocumentChunk_text", "test query", limit=5, include_payload=True
    )


@pytest.mark.asyncio
async def test_get_context_collection_not_found_error(mock_vector_engine):
    """Test that CollectionNotFoundError is converted to NoDataError."""
    mock_vector_engine.search.side_effect = CollectionNotFoundError("Collection not found")

    retriever = ChunksRetriever()

    with patch(
        "cognee.modules.retrieval.chunks_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        with pytest.raises(NoDataError, match="No data found"):
            await retriever.get_retrieved_objects("test query")


@pytest.mark.asyncio
async def test_get_context_empty_results(mock_vector_engine):
    """Test that empty list is returned when no chunks are found."""
    mock_vector_engine.search.return_value = []

    retriever = ChunksRetriever()

    with patch(
        "cognee.modules.retrieval.chunks_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        objects = await retriever.get_retrieved_objects("test query")

    assert objects == []


@pytest.mark.asyncio
async def test_get_context_top_k_limit(mock_vector_engine):
    """Test that top_k parameter limits the number of results."""
    mock_results = [MagicMock() for _ in range(3)]
    for i, result in enumerate(mock_results):
        result.payload = {"text": f"Chunk {i}"}

    mock_vector_engine.search.return_value = mock_results

    retriever = ChunksRetriever(top_k=3)

    with patch(
        "cognee.modules.retrieval.chunks_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        objects = await retriever.get_retrieved_objects("test query")

    assert len(objects) == 3
    mock_vector_engine.search.assert_awaited_once_with(
        "DocumentChunk_text", "test query", limit=3, include_payload=True
    )


@pytest.mark.asyncio
async def test_get_context(mock_vector_engine):
    """Test get_completion returns provided context."""
    retriever = ChunksRetriever()

    retrieved_objects = [
        {"payload": {"text": "Steve Rodger"}},
        {"payload": {"text": "Mike Broski"}},
    ]
    mock_objects = [SimpleNamespace(**obj) for obj in retrieved_objects]

    context = await retriever.get_context_from_objects("test query", retrieved_objects=mock_objects)

    assert context == "Steve Rodger\nMike Broski"


@pytest.mark.asyncio
async def test_init_defaults():
    """Test ChunksRetriever initialization with defaults."""
    retriever = ChunksRetriever()

    assert retriever.top_k == 5


@pytest.mark.asyncio
async def test_init_custom_top_k():
    """Test ChunksRetriever initialization with custom top_k."""
    retriever = ChunksRetriever(top_k=10)

    assert retriever.top_k == 10


@pytest.mark.asyncio
async def test_init_none_top_k():
    """Test ChunksRetriever initialization with None top_k."""
    retriever = ChunksRetriever(top_k=None)

    assert retriever.top_k is None


@pytest.mark.asyncio
async def test_get_context_empty_payload(mock_vector_engine):
    """Test get_context handles empty payload."""
    mock_vector_engine.search.return_value = []

    retriever = ChunksRetriever()

    with patch(
        "cognee.modules.retrieval.chunks_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        retrieved_objects = await retriever.get_retrieved_objects("test query")
        context = await retriever.get_context_from_objects(
            "test query", retrieved_objects=retrieved_objects
        )

    assert context == ""


@pytest.fixture
def mock_graph_engine():
    """Create a mock graph engine for testing parent document enrichment."""
    engine = AsyncMock()
    engine.query = AsyncMock()
    return engine


@pytest.mark.asyncio
async def test_enrichment_successful_batched_query(mock_vector_engine, mock_graph_engine):
    """Test successful enrichment with batched graph query."""
    mock_chunk1 = MagicMock()
    mock_chunk1.id = "chunk-123"
    mock_chunk1.payload = {"id": "chunk-123", "text": "Python is great", "chunk_index": 0}

    mock_chunk2 = MagicMock()
    mock_chunk2.id = "chunk-456"
    mock_chunk2.payload = {"id": "chunk-456", "text": "Python was created", "chunk_index": 1}

    mock_vector_engine.search.return_value = [mock_chunk1, mock_chunk2]

    mock_graph_engine.query.return_value = [
        {
            "chunk_id": "chunk-123",
            "doc_id": "doc-789",
            "doc_name": "python_tutorial.pdf",
            "doc_type": "pdf",
        },
        {
            "chunk_id": "chunk-456",
            "doc_id": "doc-789",
            "doc_name": "python_tutorial.pdf",
            "doc_type": "pdf",
        },
    ]

    retriever = ChunksRetriever(top_k=5)

    with patch(
        "cognee.modules.retrieval.chunks_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        with patch(
            "cognee.modules.retrieval.chunks_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ):
            with patch(
                "cognee.modules.retrieval.chunks_retriever.update_node_access_timestamps",
                new_callable=AsyncMock,
            ):
                chunks = await retriever.get_retrieved_objects("test query")

    assert len(chunks) == 2
    assert "parent_document" in chunks[0].payload
    assert chunks[0].payload["parent_document"]["id"] == "doc-789"
    assert chunks[0].payload["parent_document"]["name"] == "python_tutorial.pdf"

    assert "parent_document" in chunks[1].payload
    assert chunks[1].payload["parent_document"]["id"] == "doc-789"

    mock_graph_engine.query.assert_awaited()


@pytest.mark.asyncio
async def test_enrichment_batched_query_fails_falls_back_to_individual(
    mock_vector_engine, mock_graph_engine
):
    """Test that batched query failure triggers fallback to individual queries."""
    mock_chunk = MagicMock()
    mock_chunk.id = "chunk-123"
    mock_chunk.payload = {"id": "chunk-123", "text": "Python is great"}

    mock_vector_engine.search.return_value = [mock_chunk]

    mock_graph_engine.query.side_effect = [
        Exception("Batched query failed"),
        [{"doc_id": "doc-789", "doc_name": "python_tutorial.pdf"}],
    ]

    retriever = ChunksRetriever(top_k=5)

    with patch(
        "cognee.modules.retrieval.chunks_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        with patch(
            "cognee.modules.retrieval.chunks_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ):
            with patch(
                "cognee.modules.retrieval.chunks_retriever.update_node_access_timestamps",
                new_callable=AsyncMock,
            ):
                chunks = await retriever.get_retrieved_objects("test query")

    assert "parent_document" in chunks[0].payload
    assert chunks[0].payload["parent_document"]["id"] == "doc-789"

    assert mock_graph_engine.query.call_count == 2


@pytest.mark.asyncio
async def test_enrichment_graph_engine_unavailable_graceful_degradation(mock_vector_engine):
    """Test that unavailable graph engine doesn't break search (graceful degradation)."""
    mock_chunk = MagicMock()
    mock_chunk.id = "chunk-123"
    mock_chunk.payload = {"id": "chunk-123", "text": "Python is great"}

    mock_vector_engine.search.return_value = [mock_chunk]

    retriever = ChunksRetriever(top_k=5, strict_enrichment=False)

    with patch(
        "cognee.modules.retrieval.chunks_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        with patch(
            "cognee.modules.retrieval.chunks_retriever.get_graph_engine",
            side_effect=Exception("Graph DB unavailable"),
        ):
            with patch(
                "cognee.modules.retrieval.chunks_retriever.update_node_access_timestamps",
                new_callable=AsyncMock,
            ):
                chunks = await retriever.get_retrieved_objects("test query")

    assert len(chunks) == 1
    assert "parent_document" not in chunks[0].payload


@pytest.mark.asyncio
async def test_enrichment_strict_mode_raises_on_graph_unavailable(mock_vector_engine):
    """Test that strict_enrichment mode raises error when graph DB unavailable."""
    mock_chunk = MagicMock()
    mock_chunk.id = "chunk-123"
    mock_chunk.payload = {"id": "chunk-123", "text": "Python is great"}

    mock_vector_engine.search.return_value = [mock_chunk]

    retriever = ChunksRetriever(top_k=5, strict_enrichment=True)

    with patch(
        "cognee.modules.retrieval.chunks_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        with patch(
            "cognee.modules.retrieval.chunks_retriever.get_graph_engine",
            side_effect=Exception("Graph DB unavailable"),
        ):
            with patch(
                "cognee.modules.retrieval.chunks_retriever.update_node_access_timestamps",
                new_callable=AsyncMock,
            ):
                with pytest.raises(NoDataError, match="Graph engine unavailable"):
                    await retriever.get_retrieved_objects("test query")


@pytest.mark.asyncio
async def test_enrichment_no_parent_found_graceful(mock_vector_engine, mock_graph_engine):
    """Test that missing parent document doesn't break search (non-strict mode)."""
    mock_chunk = MagicMock()
    mock_chunk.id = "chunk-123"
    mock_chunk.payload = {"id": "chunk-123", "text": "Python is great"}

    mock_vector_engine.search.return_value = [mock_chunk]

    mock_graph_engine.query.return_value = []

    retriever = ChunksRetriever(top_k=5, strict_enrichment=False)

    with patch(
        "cognee.modules.retrieval.chunks_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        with patch(
            "cognee.modules.retrieval.chunks_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ):
            with patch(
                "cognee.modules.retrieval.chunks_retriever.update_node_access_timestamps",
                new_callable=AsyncMock,
            ):
                chunks = await retriever.get_retrieved_objects("test query")

    assert len(chunks) == 1
    assert "parent_document" not in chunks[0].payload


@pytest.mark.asyncio
async def test_enrichment_strict_mode_raises_on_no_parent(mock_vector_engine, mock_graph_engine):
    """Test that strict mode raises error when no parent found."""
    mock_chunk = MagicMock()
    mock_chunk.id = "chunk-123"
    mock_chunk.payload = {"id": "chunk-123", "text": "Python is great"}

    mock_vector_engine.search.return_value = [mock_chunk]
    mock_graph_engine.query.return_value = []

    retriever = ChunksRetriever(top_k=5, strict_enrichment=True)

    with patch(
        "cognee.modules.retrieval.chunks_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        with patch(
            "cognee.modules.retrieval.chunks_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ):
            with patch(
                "cognee.modules.retrieval.chunks_retriever.update_node_access_timestamps",
                new_callable=AsyncMock,
            ):
                with pytest.raises(NoDataError, match="No parent found for chunk"):
                    await retriever.get_retrieved_objects("test query")


@pytest.mark.asyncio
async def test_enrichment_partial_success(mock_vector_engine, mock_graph_engine):
    """Test that partial enrichment works (some chunks enriched, others not)."""
    mock_chunk1 = MagicMock()
    mock_chunk1.id = "chunk-123"
    mock_chunk1.payload = {"id": "chunk-123", "text": "Python is great"}

    mock_chunk2 = MagicMock()
    mock_chunk2.id = "chunk-456"
    mock_chunk2.payload = {"id": "chunk-456", "text": "Python was created"}

    mock_vector_engine.search.return_value = [mock_chunk1, mock_chunk2]

    mock_graph_engine.query.return_value = [
        {"chunk_id": "chunk-123", "doc_id": "doc-789", "doc_name": "python_tutorial.pdf"},
    ]

    retriever = ChunksRetriever(top_k=5, strict_enrichment=False)

    with patch(
        "cognee.modules.retrieval.chunks_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        with patch(
            "cognee.modules.retrieval.chunks_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ):
            with patch(
                "cognee.modules.retrieval.chunks_retriever.update_node_access_timestamps",
                new_callable=AsyncMock,
            ):
                chunks = await retriever.get_retrieved_objects("test query")

    assert "parent_document" in chunks[0].payload
    assert "parent_document" not in chunks[1].payload


@pytest.mark.asyncio
async def test_enrichment_empty_chunks_list(mock_vector_engine, mock_graph_engine):
    """Test that empty chunks list is handled correctly."""
    mock_vector_engine.search.return_value = []

    retriever = ChunksRetriever(top_k=5)

    with patch(
        "cognee.modules.retrieval.chunks_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        with patch(
            "cognee.modules.retrieval.chunks_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ):
            chunks = await retriever.get_retrieved_objects("test query")

    assert chunks == []
    mock_graph_engine.query.assert_not_awaited()


@pytest.mark.asyncio
async def test_enrichment_chunk_without_id(mock_vector_engine, mock_graph_engine):
    """Test handling of chunks without ID field."""
    mock_chunk = MagicMock(spec=[])
    mock_chunk.payload = {"text": "Python is great"}

    mock_vector_engine.search.return_value = [mock_chunk]
    mock_graph_engine.query.return_value = []

    retriever = ChunksRetriever(top_k=5, strict_enrichment=False)

    with patch(
        "cognee.modules.retrieval.chunks_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        with patch(
            "cognee.modules.retrieval.chunks_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ):
            with patch(
                "cognee.modules.retrieval.chunks_retriever.update_node_access_timestamps",
                new_callable=AsyncMock,
            ):
                chunks = await retriever.get_retrieved_objects("test query")

    assert len(chunks) == 1


@pytest.mark.asyncio
async def test_init_strict_enrichment_default():
    """Test ChunksRetriever initialization with strict_enrichment default."""
    retriever = ChunksRetriever()

    assert retriever.strict_enrichment is False


@pytest.mark.asyncio
async def test_init_strict_enrichment_enabled():
    """Test ChunksRetriever initialization with strict_enrichment enabled."""
    retriever = ChunksRetriever(strict_enrichment=True)

    assert retriever.strict_enrichment is True


@pytest.mark.asyncio
async def test_enrichment_low_success_rate_warning(mock_vector_engine, mock_graph_engine):
    """Test warning logged when enrichment success rate is below 50%."""
    chunks = []
    for i in range(10):
        chunk = MagicMock()
        chunk.id = f"chunk-{i}"
        chunk.payload = {"id": f"chunk-{i}", "text": f"Text {i}"}
        chunks.append(chunk)

    mock_vector_engine.search.return_value = chunks
    mock_graph_engine.query.return_value = [
        {"chunk_id": "chunk-0", "doc_id": "doc-1", "doc_name": "doc1.pdf"},
        {"chunk_id": "chunk-1", "doc_id": "doc-1", "doc_name": "doc1.pdf"},
    ]

    retriever = ChunksRetriever(top_k=10, strict_enrichment=False)

    with patch(
        "cognee.modules.retrieval.chunks_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        with patch(
            "cognee.modules.retrieval.chunks_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ):
            with patch(
                "cognee.modules.retrieval.chunks_retriever.update_node_access_timestamps",
                new_callable=AsyncMock,
            ):
                with patch("cognee.modules.retrieval.chunks_retriever.logger") as mock_logger:
                    result = await retriever.get_retrieved_objects("test query")

    assert len(result) == 10
    assert sum(1 for c in result if "parent_document" in c.payload) == 2
    mock_logger.warning.assert_called()
    warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
    assert any("Low enrichment rate" in str(call) or "20.0%" in str(call) for call in warning_calls)


@pytest.mark.asyncio
async def test_enrichment_timestamp_update_failure_non_strict(
    mock_vector_engine, mock_graph_engine
):
    """Test that timestamp update failure doesn't break search in non-strict mode."""
    mock_chunk = MagicMock()
    mock_chunk.id = "chunk-123"
    mock_chunk.payload = {"id": "chunk-123", "text": "Python is great"}

    mock_vector_engine.search.return_value = [mock_chunk]
    mock_graph_engine.query.return_value = [
        {"chunk_id": "chunk-123", "doc_id": "doc-789", "doc_name": "doc.pdf"}
    ]

    retriever = ChunksRetriever(top_k=5, strict_enrichment=False)

    mock_update = AsyncMock(side_effect=Exception("Timestamp update failed"))

    with patch(
        "cognee.modules.retrieval.chunks_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        with patch(
            "cognee.modules.retrieval.chunks_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ):
            with patch(
                "cognee.modules.retrieval.chunks_retriever.update_node_access_timestamps",
                mock_update,
            ):
                chunks = await retriever.get_retrieved_objects("test query")

    assert len(chunks) == 1
    assert "parent_document" in chunks[0].payload


@pytest.mark.asyncio
async def test_enrichment_timestamp_update_failure_strict(mock_vector_engine):
    """Test that timestamp update failure raises error in strict mode."""
    mock_chunk = MagicMock()
    mock_chunk.id = "chunk-123"
    mock_chunk.payload = {"id": "chunk-123", "text": "Python is great"}

    mock_vector_engine.search.return_value = [mock_chunk]

    retriever = ChunksRetriever(top_k=5, strict_enrichment=True)

    mock_update = AsyncMock(side_effect=Exception("Timestamp update failed"))

    with patch(
        "cognee.modules.retrieval.chunks_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        with patch(
            "cognee.modules.retrieval.chunks_retriever.update_node_access_timestamps", mock_update
        ):
            with pytest.raises(NoDataError, match="Failed to update timestamps"):
                await retriever.get_retrieved_objects("test query")


@pytest.mark.asyncio
async def test_get_completion_from_context_with_objects():
    """Test get_completion_from_context returns chunk payloads."""
    retriever = ChunksRetriever()

    mock_chunk1 = MagicMock()
    mock_chunk1.payload = {"text": "Python is great", "chunk_index": 0}

    mock_chunk2 = MagicMock()
    mock_chunk2.payload = {"text": "Python was created", "chunk_index": 1}

    retrieved_objects = [mock_chunk1, mock_chunk2]

    result = await retriever.get_completion_from_context(
        "test query", retrieved_objects=retrieved_objects, context=None
    )

    assert len(result) == 2
    assert result[0] == {"text": "Python is great", "chunk_index": 0}
    assert result[1] == {"text": "Python was created", "chunk_index": 1}


@pytest.mark.asyncio
async def test_get_completion_from_context_empty():
    """Test get_completion_from_context returns empty list when no objects."""
    retriever = ChunksRetriever()

    result = await retriever.get_completion_from_context(
        "test query", retrieved_objects=[], context=None
    )

    assert result == []


@pytest.mark.asyncio
async def test_enrichment_chunk_missing_id_strict_mode(mock_vector_engine):
    """Test that chunk missing ID raises NoDataError in strict mode."""
    mock_chunk = MagicMock(spec=[])
    mock_chunk.payload = {"text": "Python is great"}

    mock_vector_engine.search.return_value = [mock_chunk]

    retriever = ChunksRetriever(top_k=5, strict_enrichment=True)

    with patch(
        "cognee.modules.retrieval.chunks_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        with patch(
            "cognee.modules.retrieval.chunks_retriever.update_node_access_timestamps",
            new_callable=AsyncMock,
        ):
            with pytest.raises(NoDataError, match="Failed to extract chunk ID"):
                await retriever.get_retrieved_objects("test query")


@pytest.mark.asyncio
async def test_enrichment_batched_row_parse_error(mock_vector_engine, mock_graph_engine):
    """Test that a malformed row in batched query results is skipped gracefully."""
    mock_chunk1 = MagicMock()
    mock_chunk1.id = "chunk-123"
    mock_chunk1.payload = {"id": "chunk-123", "text": "Python is great"}

    mock_chunk2 = MagicMock()
    mock_chunk2.id = "chunk-456"
    mock_chunk2.payload = {"id": "chunk-456", "text": "Python was created"}

    mock_vector_engine.search.return_value = [mock_chunk1, mock_chunk2]

    good_row = {"chunk_id": "chunk-123", "doc_id": "doc-789", "doc_name": "tutorial.pdf"}
    bad_row = MagicMock()
    bad_row.__getitem__ = MagicMock(side_effect=KeyError("chunk_id"))
    bad_row.__contains__ = MagicMock(return_value=False)

    mock_graph_engine.query.return_value = [good_row, bad_row]

    retriever = ChunksRetriever(top_k=5, strict_enrichment=False)

    with patch(
        "cognee.modules.retrieval.chunks_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        with patch(
            "cognee.modules.retrieval.chunks_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ):
            with patch(
                "cognee.modules.retrieval.chunks_retriever.update_node_access_timestamps",
                new_callable=AsyncMock,
            ):
                chunks = await retriever.get_retrieved_objects("test query")

    assert "parent_document" in chunks[0].payload
    assert "parent_document" not in chunks[1].payload


@pytest.mark.asyncio
async def test_enrichment_individual_query_fails_strict_mode(mock_vector_engine, mock_graph_engine):
    """Test that individual query failure raises NoDataError in strict mode during fallback."""
    mock_chunk = MagicMock()
    mock_chunk.id = "chunk-123"
    mock_chunk.payload = {"id": "chunk-123", "text": "Python is great"}

    mock_vector_engine.search.return_value = [mock_chunk]

    mock_graph_engine.query.side_effect = [
        Exception("Batched query failed"),
        Exception("Individual query also failed"),
    ]

    retriever = ChunksRetriever(top_k=5, strict_enrichment=True)

    with patch(
        "cognee.modules.retrieval.chunks_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        with patch(
            "cognee.modules.retrieval.chunks_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ):
            with patch(
                "cognee.modules.retrieval.chunks_retriever.update_node_access_timestamps",
                new_callable=AsyncMock,
            ):
                with pytest.raises(NoDataError, match="Failed to fetch parent for chunk-123"):
                    await retriever.get_retrieved_objects("test query")


@pytest.mark.asyncio
async def test_enrichment_handles_dict_chunks(mock_vector_engine, mock_graph_engine):
    """Test enrichment works when modifying chunk.payload on dict-type chunks."""
    mock_chunk = SimpleNamespace()
    mock_chunk.id = "chunk-123"
    mock_chunk.payload = {"id": "chunk-123", "text": "Python is great"}

    mock_vector_engine.search.return_value = [mock_chunk]
    mock_graph_engine.query.return_value = [
        {"chunk_id": "chunk-123", "doc_id": "doc-789", "doc_name": "doc.pdf"}
    ]

    retriever = ChunksRetriever(top_k=5)

    with patch(
        "cognee.modules.retrieval.chunks_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        with patch(
            "cognee.modules.retrieval.chunks_retriever.get_graph_engine",
            return_value=mock_graph_engine,
        ):
            with patch(
                "cognee.modules.retrieval.chunks_retriever.update_node_access_timestamps",
                new_callable=AsyncMock,
            ):
                chunks = await retriever.get_retrieved_objects("test query")

    assert len(chunks) == 1
    assert "parent_document" in chunks[0].payload
    assert chunks[0].payload["parent_document"]["id"] == "doc-789"
