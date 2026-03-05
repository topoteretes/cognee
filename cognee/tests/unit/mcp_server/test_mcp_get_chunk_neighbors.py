import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


def _make_target_result(target_index=5, doc_id="doc-1", doc_name="test.pdf"):
    return [{"target_index": target_index, "doc_id": doc_id, "doc_name": doc_name}]


def _make_neighbor_rows(start_index, end_index):
    rows = []
    for i in range(start_index, end_index + 1):
        rows.append(
            {
                "chunk_id": f"chunk-{i}",
                "chunk_index": i,
                "text": f"Text of chunk {i}",
                "word_count": 4,
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
async def test_get_chunk_neighbors_both_directions(mock_graph_engine, mock_cognee_client):
    """Test default direction='both' returns neighbors on both sides."""
    from server import get_chunk_neighbors

    mock_graph_engine.query.side_effect = [
        _make_target_result(target_index=5),
        _make_neighbor_rows(3, 7),
    ]

    with _patches(mock_graph_engine, mock_cognee_client):
        result = await get_chunk_neighbors(chunk_id="chunk-5", neighbor_count=2)

    data = json.loads(result[0].text)
    assert data["direction"] == "both"
    assert data["chunks_returned"] == 5
    assert data["chunks"][0]["chunk_index"] == 3
    assert data["chunks"][-1]["chunk_index"] == 7
    target = [c for c in data["chunks"] if c["is_target"]]
    assert len(target) == 1
    assert target[0]["chunk_index"] == 5


@pytest.mark.asyncio
async def test_get_chunk_neighbors_forward(mock_graph_engine, mock_cognee_client):
    """Test direction='forward' only returns chunks after target."""
    from server import get_chunk_neighbors

    mock_graph_engine.query.side_effect = [
        _make_target_result(target_index=5),
        _make_neighbor_rows(5, 8),
    ]

    with _patches(mock_graph_engine, mock_cognee_client):
        result = await get_chunk_neighbors(
            chunk_id="chunk-5", neighbor_count=3, direction="forward"
        )

    data = json.loads(result[0].text)
    assert data["direction"] == "forward"
    assert all(c["chunk_index"] >= 5 for c in data["chunks"])


@pytest.mark.asyncio
async def test_get_chunk_neighbors_backward(mock_graph_engine, mock_cognee_client):
    """Test direction='backward' only returns chunks before target."""
    from server import get_chunk_neighbors

    mock_graph_engine.query.side_effect = [
        _make_target_result(target_index=5),
        _make_neighbor_rows(2, 5),
    ]

    with _patches(mock_graph_engine, mock_cognee_client):
        result = await get_chunk_neighbors(
            chunk_id="chunk-5", neighbor_count=3, direction="backward"
        )

    data = json.loads(result[0].text)
    assert data["direction"] == "backward"
    assert all(c["chunk_index"] <= 5 for c in data["chunks"])


@pytest.mark.asyncio
async def test_get_chunk_neighbors_exclude_target(mock_graph_engine, mock_cognee_client):
    """Test include_target=False excludes the target chunk from results."""
    from server import get_chunk_neighbors

    mock_graph_engine.query.side_effect = [
        _make_target_result(target_index=5),
        _make_neighbor_rows(3, 7),
    ]

    with _patches(mock_graph_engine, mock_cognee_client):
        result = await get_chunk_neighbors(
            chunk_id="chunk-5", neighbor_count=2, include_target=False
        )

    data = json.loads(result[0].text)
    target_chunks = [c for c in data["chunks"] if c["is_target"]]
    assert len(target_chunks) == 0
    assert data["chunks_returned"] == 4


@pytest.mark.asyncio
async def test_get_chunk_neighbors_chunk_not_found(mock_graph_engine, mock_cognee_client):
    """Test returns error when chunk is not found."""
    from server import get_chunk_neighbors

    mock_graph_engine.query.return_value = []

    with _patches(mock_graph_engine, mock_cognee_client):
        result = await get_chunk_neighbors(chunk_id="nonexistent")

    data = json.loads(result[0].text)
    assert "error" in data
    assert data["chunk_id"] == "nonexistent"


@pytest.mark.asyncio
async def test_get_chunk_neighbors_caps_neighbor_count(mock_graph_engine, mock_cognee_client):
    """Test that neighbor_count is capped at 10."""
    from server import get_chunk_neighbors

    mock_graph_engine.query.side_effect = [
        _make_target_result(target_index=50),
        _make_neighbor_rows(40, 60),
    ]

    with _patches(mock_graph_engine, mock_cognee_client):
        result = await get_chunk_neighbors(chunk_id="chunk-50", neighbor_count=100)

    data = json.loads(result[0].text)
    assert data["neighbor_count"] == 10


@pytest.mark.asyncio
async def test_get_chunk_neighbors_invalid_direction_defaults_to_both(
    mock_graph_engine, mock_cognee_client
):
    """Test that invalid direction defaults to 'both'."""
    from server import get_chunk_neighbors

    mock_graph_engine.query.side_effect = [
        _make_target_result(target_index=5),
        _make_neighbor_rows(3, 7),
    ]

    with _patches(mock_graph_engine, mock_cognee_client):
        result = await get_chunk_neighbors(
            chunk_id="chunk-5", neighbor_count=2, direction="invalid"
        )

    data = json.loads(result[0].text)
    assert data["direction"] == "both"


@pytest.mark.asyncio
async def test_get_chunk_neighbors_api_mode_returns_error(mock_cognee_client):
    """Test returns error in API mode."""
    from server import get_chunk_neighbors

    mock_cognee_client.use_api = True

    with patch("server.cognee_client", mock_cognee_client):
        result = await get_chunk_neighbors(chunk_id="chunk-1")

    data = json.loads(result[0].text)
    assert "error" in data
    assert "not supported in API mode" in data["error"]


@pytest.mark.asyncio
async def test_get_chunk_neighbors_graph_engine_error(mock_graph_engine, mock_cognee_client):
    """Test handles graph engine exceptions gracefully."""
    from server import get_chunk_neighbors

    mock_graph_engine.query.side_effect = Exception("Connection refused")

    with (
        patch(
            "cognee.infrastructure.databases.graph.get_graph_engine",
            new_callable=AsyncMock,
            return_value=mock_graph_engine,
        ),
        patch("server.cognee_client", mock_cognee_client),
    ):
        result = await get_chunk_neighbors(chunk_id="chunk-1")

    data = json.loads(result[0].text)
    assert "error" in data
    assert "Connection refused" in data["error"]


@pytest.mark.asyncio
async def test_get_chunk_neighbors_min_neighbor_count(mock_graph_engine, mock_cognee_client):
    """Test that neighbor_count below 1 is clamped to 1."""
    from server import get_chunk_neighbors

    mock_graph_engine.query.side_effect = [
        _make_target_result(target_index=5),
        _make_neighbor_rows(4, 6),
    ]

    with _patches(mock_graph_engine, mock_cognee_client):
        result = await get_chunk_neighbors(chunk_id="chunk-5", neighbor_count=0)

    data = json.loads(result[0].text)
    assert data["neighbor_count"] == 1
