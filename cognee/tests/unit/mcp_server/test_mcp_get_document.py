import json
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch, MagicMock


def _make_graph_rows(chunk_count):
    """Build fake graph query rows for a document with N chunks."""
    doc = SimpleNamespace(id="doc-1", name="test.pdf", type="pdf", metadata={"author": "test"})
    rows = []
    for i in range(chunk_count):
        rows.append(
            {
                "doc": doc,
                "chunk_id": f"chunk-{i}",
                "chunk_index": i,
                "chunk_text": f"Text of chunk {i}",
                "chunk_word_count": 4,
            }
        )
    if not rows:
        rows.append(
            {
                "doc": doc,
                "chunk_id": None,
                "chunk_index": None,
                "chunk_text": None,
                "chunk_word_count": None,
            }
        )
    return rows


@pytest.fixture
def mock_graph_engine():
    engine = AsyncMock()
    engine.query = AsyncMock()
    return engine


@pytest.fixture
def mock_cognee_client():
    client = MagicMock()
    client.use_api = False
    return client


def _patches(mock_graph_engine, mock_cognee_client):
    """Return a context manager that patches graph engine, normalize, and client."""
    from contextlib import contextmanager

    @contextmanager
    def ctx():
        with (
            patch(
                "cognee.infrastructure.databases.graph.get_graph_engine",
                new_callable=AsyncMock,
                return_value=mock_graph_engine,
            ),
            patch(
                "server.normalize_graph_result",
                side_effect=lambda r, c: r,
            ),
            patch("server.cognee_client", mock_cognee_client),
        ):
            yield

    return ctx()


@pytest.mark.asyncio
async def test_get_document_returns_all_chunks(mock_graph_engine, mock_cognee_client):
    """Test get_document returns all chunks when max_chunks=0 (default)."""
    from server import get_document

    rows = _make_graph_rows(5)
    mock_graph_engine.query.return_value = rows

    with _patches(mock_graph_engine, mock_cognee_client):
        result = await get_document(document_id="doc-1", max_chunks=0)

    data = json.loads(result[0].text)
    assert data["chunk_count"] == 5
    assert data["total_chunks"] == 5
    assert data["is_truncated"] is False
    assert len(data["chunks"]) == 5


@pytest.mark.asyncio
async def test_get_document_truncates_with_max_chunks(mock_graph_engine, mock_cognee_client):
    """Test get_document truncates when max_chunks < total chunks."""
    from server import get_document

    rows = _make_graph_rows(10)
    mock_graph_engine.query.return_value = rows

    with _patches(mock_graph_engine, mock_cognee_client):
        result = await get_document(document_id="doc-1", max_chunks=3)

    data = json.loads(result[0].text)
    assert data["chunk_count"] == 3
    assert data["total_chunks"] == 10
    assert data["is_truncated"] is True
    assert len(data["chunks"]) == 3
    assert data["chunks"][0]["chunk_index"] == 0
    assert data["chunks"][2]["chunk_index"] == 2


@pytest.mark.asyncio
async def test_get_document_no_truncation_when_max_exceeds_total(
    mock_graph_engine, mock_cognee_client
):
    """Test get_document returns all chunks when max_chunks >= total."""
    from server import get_document

    rows = _make_graph_rows(3)
    mock_graph_engine.query.return_value = rows

    with _patches(mock_graph_engine, mock_cognee_client):
        result = await get_document(document_id="doc-1", max_chunks=50)

    data = json.loads(result[0].text)
    assert data["chunk_count"] == 3
    assert data["total_chunks"] == 3
    assert data["is_truncated"] is False


@pytest.mark.asyncio
async def test_get_document_not_found(mock_graph_engine, mock_cognee_client):
    """Test get_document returns error when document not found."""
    from server import get_document

    mock_graph_engine.query.return_value = [{"doc": None}]

    with _patches(mock_graph_engine, mock_cognee_client):
        result = await get_document(document_id="nonexistent")

    data = json.loads(result[0].text)
    assert "error" in data
    assert data["document_id"] == "nonexistent"


@pytest.mark.asyncio
async def test_get_document_empty_result(mock_graph_engine, mock_cognee_client):
    """Test get_document returns error when query returns empty."""
    from server import get_document

    mock_graph_engine.query.return_value = []

    with _patches(mock_graph_engine, mock_cognee_client):
        result = await get_document(document_id="doc-1")

    data = json.loads(result[0].text)
    assert "error" in data


@pytest.mark.asyncio
async def test_get_document_includes_metadata(mock_graph_engine, mock_cognee_client):
    """Test get_document includes metadata when include_metadata=True."""
    from server import get_document

    rows = _make_graph_rows(1)
    mock_graph_engine.query.return_value = rows

    with _patches(mock_graph_engine, mock_cognee_client):
        result = await get_document(document_id="doc-1", include_metadata=True)

    data = json.loads(result[0].text)
    assert "metadata" in data
    assert data["metadata"]["author"] == "test"


@pytest.mark.asyncio
async def test_get_document_excludes_metadata(mock_graph_engine, mock_cognee_client):
    """Test get_document excludes metadata when include_metadata=False."""
    from server import get_document

    rows = _make_graph_rows(1)
    mock_graph_engine.query.return_value = rows

    with _patches(mock_graph_engine, mock_cognee_client):
        result = await get_document(document_id="doc-1", include_metadata=False)

    data = json.loads(result[0].text)
    assert "metadata" not in data


@pytest.mark.asyncio
async def test_get_document_api_mode_returns_error(mock_cognee_client):
    """Test get_document returns error in API mode."""
    from server import get_document

    mock_cognee_client.use_api = True

    with patch("server.cognee_client", mock_cognee_client):
        result = await get_document(document_id="doc-1")

    data = json.loads(result[0].text)
    assert "error" in data
    assert "not supported in API mode" in data["error"]


@pytest.mark.asyncio
async def test_get_document_graph_engine_error(mock_graph_engine, mock_cognee_client):
    """Test get_document handles graph engine exceptions gracefully."""
    from server import get_document

    mock_graph_engine.query.side_effect = Exception("Connection refused")

    with (
        patch(
            "cognee.infrastructure.databases.graph.get_graph_engine",
            new_callable=AsyncMock,
            return_value=mock_graph_engine,
        ),
        patch("server.cognee_client", mock_cognee_client),
    ):
        result = await get_document(document_id="doc-1")

    data = json.loads(result[0].text)
    assert "error" in data
    assert "Connection refused" in data["error"]


@pytest.mark.asyncio
async def test_get_document_max_chunks_1(mock_graph_engine, mock_cognee_client):
    """Test get_document with max_chunks=1 returns only the first chunk."""
    from server import get_document

    rows = _make_graph_rows(5)
    mock_graph_engine.query.return_value = rows

    with _patches(mock_graph_engine, mock_cognee_client):
        result = await get_document(document_id="doc-1", max_chunks=1)

    data = json.loads(result[0].text)
    assert data["chunk_count"] == 1
    assert data["total_chunks"] == 5
    assert data["is_truncated"] is True
    assert data["chunks"][0]["chunk_id"] == "chunk-0"


@pytest.mark.asyncio
async def test_get_document_dict_doc(mock_graph_engine, mock_cognee_client):
    """Test get_document handles dict-style doc (Neo4j) instead of SimpleNamespace."""
    from server import get_document

    doc_dict = {"id": "doc-1", "name": "test.pdf", "type": "pdf", "metadata": {"author": "test"}}
    rows = [
        {
            "doc": doc_dict,
            "chunk_id": "chunk-0",
            "chunk_index": 0,
            "chunk_text": "Hello",
            "chunk_word_count": 1,
        }
    ]
    mock_graph_engine.query.return_value = rows

    with _patches(mock_graph_engine, mock_cognee_client):
        result = await get_document(document_id="doc-1", include_metadata=True)

    data = json.loads(result[0].text)
    assert data["document_id"] == "doc-1"
    assert data["name"] == "test.pdf"
    assert data["type"] == "pdf"
    assert data["metadata"]["author"] == "test"


@pytest.mark.asyncio
async def test_get_document_no_chunks(mock_graph_engine, mock_cognee_client):
    """Test get_document with a document that has no chunks attached."""
    from server import get_document

    doc = SimpleNamespace(id="doc-1", name="empty.pdf", type="pdf")
    rows = [
        {
            "doc": doc,
            "chunk_id": None,
            "chunk_index": None,
            "chunk_text": None,
            "chunk_word_count": None,
        }
    ]
    mock_graph_engine.query.return_value = rows

    with _patches(mock_graph_engine, mock_cognee_client):
        result = await get_document(document_id="doc-1")

    data = json.loads(result[0].text)
    assert data["chunk_count"] == 0
    assert data["total_chunks"] == 0
    assert data["is_truncated"] is False
    assert data["chunks"] == []
