import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import cognee.infrastructure.databases.vector.embeddings.config as embedding_config_module
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingResult
from cognee.infrastructure.databases.vector.embeddings.EmbeddingGateway import EmbeddingGateway
from cognee.infrastructure.engine import DataPoint
from cognee.shared import rate_limiting
from cognee.tasks.storage.index_data_points import index_data_points


class TestDataPoint(DataPoint):
    name: str
    description: str = "test description"


@pytest.fixture(autouse=True)
def embedding_cap(monkeypatch):
    """Fresh process-wide slots per test; returns a setter for EMBEDDING_MAX_CONCURRENT_REQUESTS."""
    monkeypatch.setattr(rate_limiting, "_concurrency_state", {})

    def set_cap(cap: int) -> None:
        config = SimpleNamespace(embedding_max_concurrent_requests=cap)
        monkeypatch.setattr(embedding_config_module, "get_embedding_config", lambda: config)

    set_cap(128)
    return set_cap


@pytest.mark.asyncio
async def test_duplicate_ids_across_write_batches_merge_tags_and_embed_once_per_field():
    first = TestDataPoint(name="old", description="old description", belongs_to_set=["a", "shared"])
    other = TestDataPoint(name="other")
    last = TestDataPoint(
        id=first.id, name="new", description="new description", belongs_to_set=["shared", "b"]
    )
    points = [first, other, last]
    for point in points:
        point.metadata = {"index_fields": ["name", "description"]}
    before = [point.model_dump() for point in points]
    engine = AsyncMock()
    engine.embedding_engine.get_batch_size = MagicMock(return_value=32)
    with patch.dict(index_data_points.__globals__, {"write_batch_size": lambda: 1}):
        assert await index_data_points(points, vector_engine=engine) is points

    written = [
        (call.args[1], point)
        for call in engine.index_data_points.await_args_list
        for point in call.args[2]
    ]
    assert len(written) == 4
    assert len({(field, point.id) for field, point in written}) == 4
    for field, point in written:
        assert point.metadata["index_fields"] == [field]
        if point.id == first.id:
            assert point.name == "new"
            assert point.description == "new description"
            assert point.belongs_to_set == ["a", "shared", "b"]
    assert [point.model_dump() for point in points] == before


@pytest.mark.asyncio
async def test_edge_text_identity_collisions_are_indexed_once():
    from cognee.modules.graph.models.EdgeType import EdgeType

    first = EdgeType(relationship_name="IS A", number_of_edges=1)
    last = EdgeType(relationship_name="is a", number_of_edges=2)
    assert first.id == last.id
    engine = AsyncMock()
    engine.embedding_engine.get_batch_size = MagicMock(return_value=32)
    with patch.dict(index_data_points.__globals__, {"write_batch_size": lambda: 1}):
        await index_data_points([first, last], vector_engine=engine)
    engine.index_data_points.assert_awaited_once_with("EdgeType", "relationship_name", [last])


@pytest.mark.asyncio
async def test_index_data_points_calls_vector_engine():
    """Test that index_data_points creates vector index and indexes data."""
    data_points = [TestDataPoint(name="test1")]
    data_points[0].metadata["index_fields"] = ["name"]

    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine.get_batch_size = MagicMock(return_value=100)

    async def _get_vector_engine():
        return mock_vector_engine

    with patch.dict(
        index_data_points.__globals__,
        {"get_vector_engine_async": _get_vector_engine},
    ):
        await index_data_points(data_points)

    assert mock_vector_engine.create_vector_index.await_count >= 1
    assert mock_vector_engine.index_data_points.await_count >= 1


async def _run_and_measure_peak_concurrency(data_points, batch_size):
    """Measure provider requests through the real gateway and storage planner."""
    in_flight = 0
    peak = 0

    async def _index(*args, **kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0)
        in_flight -= 1
        return EmbeddingResult([[1.0]] * len(args[0]), tokens=len(args[0]))

    mock_vector_engine = AsyncMock()
    raw = SimpleNamespace(get_batch_size=lambda: batch_size, embed_text=_index)
    mock_vector_engine.embedding_engine = EmbeddingGateway(raw)
    mock_vector_engine.embedding_engine._record_usage = AsyncMock()

    async def index(_type, _field, points):
        await mock_vector_engine.embedding_engine.embed_text([p.name for p in points])

    mock_vector_engine.index_data_points = AsyncMock(side_effect=index)

    with patch.dict(
        index_data_points.__globals__,
        {
            # One write per embedding request, so the batch count is the request count.
            "get_vectordb_config": lambda: SimpleNamespace(vector_write_batch_size=batch_size),
        },
    ):
        await index_data_points(data_points, vector_engine=mock_vector_engine)

    return peak


@pytest.mark.asyncio
async def test_concurrent_batches_bounded_by_max_concurrent_requests(embedding_cap):
    """Batches admitted at once = EMBEDDING_MAX_CONCURRENT_REQUESTS."""
    embedding_cap(3)
    data_points = [TestDataPoint(name=f"point{i}") for i in range(20)]
    for data_point in data_points:
        data_point.metadata["index_fields"] = ["name"]

    peak = await _run_and_measure_peak_concurrency(data_points, batch_size=2)

    assert peak == 3


@pytest.mark.asyncio
async def test_batch_size_above_the_cap_still_runs(embedding_cap):
    """A batch larger than EMBEDDING_MAX_CONCURRENT_REQUESTS still runs, not deadlocks."""
    embedding_cap(6)
    data_points = [TestDataPoint(name=f"point{i}") for i in range(10)]
    for data_point in data_points:
        data_point.metadata["index_fields"] = ["name"]

    peak = await _run_and_measure_peak_concurrency(data_points, batch_size=100)

    assert peak == 1


@pytest.mark.asyncio
async def test_write_batches_hold_vector_write_batch_size_rows():
    """The write batch is VECTOR_WRITE_BATCH_SIZE rows, not the embedding request size:
    300 points with 32-text requests and 128-row writes are three writes."""
    data_points = [TestDataPoint(name=f"point{i}") for i in range(300)]
    for data_point in data_points:
        data_point.metadata["index_fields"] = ["name"]
    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine.get_batch_size = MagicMock(return_value=32)

    with patch.dict(
        index_data_points.__globals__,
        {"get_vectordb_config": lambda: SimpleNamespace(vector_write_batch_size=128)},
    ):
        await index_data_points(data_points, vector_engine=mock_vector_engine)

    batch_sizes = [
        len(call.args[2]) for call in mock_vector_engine.index_data_points.await_args_list
    ]
    assert sorted(batch_sizes, reverse=True) == [128, 128, 44]


@pytest.mark.asyncio
async def test_a_point_with_one_index_field_is_handed_over_without_a_copy():
    """Adapters embed the first index field, so nothing has to change on the point; only
    a point indexed on several fields gets a copy per field, each naming its field."""

    class TwoFieldPoint(DataPoint):
        name: str
        description: str
        metadata: dict = {"index_fields": ["name", "description"]}

    single = TestDataPoint(name="single")
    single.metadata["index_fields"] = ["name"]
    double = TwoFieldPoint(name="n", description="d")
    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine.get_batch_size = MagicMock(return_value=100)

    await index_data_points([single, double], vector_engine=mock_vector_engine)

    handed_over = {
        (call.args[0], call.args[1]): call.args[2]
        for call in mock_vector_engine.index_data_points.await_args_list
    }
    assert handed_over[("TestDataPoint", "name")][0] is single
    (by_name,) = handed_over[("TwoFieldPoint", "name")]
    (by_description,) = handed_over[("TwoFieldPoint", "description")]
    assert by_name is not double and by_description is not double
    assert by_name.metadata["index_fields"] == ["name"]
    assert by_description.metadata["index_fields"] == ["description"]
    assert double.metadata["index_fields"] == ["name", "description"]


@pytest.mark.asyncio
async def test_concurrent_calls_share_the_process_wide_bound(embedding_cap):
    """Two concurrent index calls never exceed EMBEDDING_MAX_CONCURRENT_REQUESTS between them."""
    embedding_cap(2)
    in_flight = 0
    peak = 0

    async def _index(*args, **kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0)
        in_flight -= 1
        return EmbeddingResult([[1.0]] * len(args[0]), tokens=len(args[0]))

    mock_vector_engine = AsyncMock()
    raw = SimpleNamespace(get_batch_size=lambda: 1, embed_text=_index)
    mock_vector_engine.embedding_engine = EmbeddingGateway(raw)
    mock_vector_engine.embedding_engine._record_usage = AsyncMock()

    async def index(_type, _field, points):
        await mock_vector_engine.embedding_engine.embed_text([p.name for p in points])

    mock_vector_engine.index_data_points = AsyncMock(side_effect=index)
    first = [TestDataPoint(name=f"a{i}") for i in range(6)]
    second = [TestDataPoint(name=f"b{i}") for i in range(6)]
    for data_point in first + second:
        data_point.metadata["index_fields"] = ["name"]

    with patch.dict(
        index_data_points.__globals__,
        {"get_vectordb_config": lambda: SimpleNamespace(vector_write_batch_size=1)},
    ):
        await asyncio.gather(
            index_data_points(first, vector_engine=mock_vector_engine),
            index_data_points(second, vector_engine=mock_vector_engine),
        )

    assert peak == 2


@pytest.mark.asyncio
async def test_index_data_points_does_not_mutate_metadata():
    data_point = TestDataPoint(name="test")
    data_point.metadata["index_fields"] = ["name", "description"]

    original_index_fields = list(data_point.metadata["index_fields"])

    mock_vector_engine = AsyncMock()
    mock_vector_engine.embedding_engine.get_batch_size = MagicMock(return_value=100)

    async def _get_vector_engine():
        return mock_vector_engine

    with patch.dict(
        index_data_points.__globals__,
        {"get_vector_engine_async": _get_vector_engine},
    ):
        await index_data_points([data_point])

    assert data_point.metadata["index_fields"] == original_index_fields


@pytest.mark.asyncio
async def test_failed_batch_waits_for_sibling_writes():
    completed = asyncio.Event()
    started = asyncio.Event()

    async def write(_index, _property, batch):
        if batch[0].name == "fails":
            await started.wait()
            raise ValueError("write failed")
        started.set()
        await asyncio.sleep(0.01)
        completed.set()

    engine = AsyncMock()
    engine.index_data_points.side_effect = write
    points = [TestDataPoint(name=name) for name in ("fails", "finishes")]
    for point in points:
        point.metadata = {"index_fields": ["name"]}
    with (
        patch.dict(index_data_points.__globals__, {"write_batch_size": lambda: 1}),
        pytest.raises(ValueError, match="write failed"),
    ):
        await index_data_points(points, vector_engine=engine)
    assert completed.is_set(), "no sibling write may outlive the failed storage call"


@pytest.mark.asyncio
async def test_a_blocked_store_holds_back_embedding(embedding_cap):
    """Batches waiting on the store keep their slots, so embedding cannot run ahead of it."""
    total = 4096
    embedded = 0
    all_embedded = asyncio.Event()
    release_storage = asyncio.Event()

    async def embed(texts):
        nonlocal embedded
        embedded += len(texts)
        if embedded == total:
            all_embedded.set()
        await asyncio.sleep(0)
        return EmbeddingResult([[1.0]] * len(texts), tokens=0)

    engine = AsyncMock()
    engine.embedding_engine = EmbeddingGateway(
        SimpleNamespace(get_batch_size=lambda: 32, embed_text=embed)
    )
    engine.embedding_engine._record_usage = AsyncMock()

    async def write(_type, _field, points):
        await engine.embedding_engine.embed_text([point.name for point in points])
        await release_storage.wait()

    engine.index_data_points = AsyncMock(side_effect=write)
    points = [TestDataPoint(name=str(index)) for index in range(total)]
    for point in points:
        point.metadata = {"index_fields": ["name"]}

    embedding_cap(4)
    with patch.dict(
        index_data_points.__globals__,
        {"get_vectordb_config": lambda: SimpleNamespace(vector_write_batch_size=128)},
    ):
        indexing = asyncio.create_task(index_data_points(points, vector_engine=engine))
        try:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(all_embedded.wait(), timeout=0.5)
            # 4 batches admitted, 128 rows each.
            assert embedded == 4 * 128
        finally:
            release_storage.set()
            await indexing
    assert embedded == total


@pytest.mark.asyncio
@pytest.mark.parametrize("count,request_size,cap,expected", [(128, 32, 1, 4), (1024, 256, 8, 8)])
async def test_provider_requests_overlap_independently_of_write_batches(
    count, request_size, cap, expected, embedding_cap
):
    embedding_cap(cap)
    active = peak = 0
    reached = asyncio.Event()

    async def embed(texts):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if active == expected:
            reached.set()
        await asyncio.wait_for(reached.wait(), 1)
        active -= 1
        return EmbeddingResult([[float(text)] for text in texts], tokens=len(texts))

    gateway = EmbeddingGateway(
        SimpleNamespace(get_batch_size=lambda: request_size, embed_text=embed)
    )
    gateway._record_usage = AsyncMock()
    writes = []

    async def index(_type, _field, points):
        vectors = await gateway.embed_text([p.name for p in points])
        assert vectors == [[float(p.name)] for p in points]
        writes.append(len(points))

    engine = SimpleNamespace(
        embedding_engine=gateway, create_vector_index=AsyncMock(), index_data_points=index
    )
    points = [TestDataPoint(name=str(i), metadata={"index_fields": ["name"]}) for i in range(count)]
    with patch.dict(
        index_data_points.__globals__,
        {"write_batch_size": lambda: 128},
    ):
        await index_data_points(points, vector_engine=engine)
    assert peak == expected
    assert writes == [128] * (count // 128)
