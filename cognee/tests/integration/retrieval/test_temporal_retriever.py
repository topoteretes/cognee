import os
import pytest
import pathlib
import pytest_asyncio
import cognee

from cognee.low_level import setup, DataPoint
from cognee.tasks.storage import add_data_points
from cognee.modules.retrieval.temporal_retriever import TemporalRetriever
from cognee.modules.engine.models.Event import Event
from cognee.modules.engine.models.Timestamp import Timestamp
from cognee.modules.engine.models.Interval import Interval


@pytest_asyncio.fixture
async def setup_test_environment_with_events():
    """Set up a clean test environment with temporal events."""
    base_dir = pathlib.Path(__file__).parent.parent.parent.parent
    system_directory_path = str(base_dir / ".cognee_system/test_temporal_retriever_with_events")
    data_directory_path = str(base_dir / ".data_storage/test_temporal_retriever_with_events")

    cognee.config.system_root_directory(system_directory_path)
    cognee.config.data_root_directory(data_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    # Create timestamps for events
    timestamp1 = Timestamp(
        time_at=1609459200,  # 2021-01-01 00:00:00
        year=2021,
        month=1,
        day=1,
        hour=0,
        minute=0,
        second=0,
        timestamp_str="2021-01-01T00:00:00",
    )

    timestamp2 = Timestamp(
        time_at=1612137600,  # 2021-02-01 00:00:00
        year=2021,
        month=2,
        day=1,
        hour=0,
        minute=0,
        second=0,
        timestamp_str="2021-02-01T00:00:00",
    )

    timestamp3 = Timestamp(
        time_at=1614556800,  # 2021-03-01 00:00:00
        year=2021,
        month=3,
        day=1,
        hour=0,
        minute=0,
        second=0,
        timestamp_str="2021-03-01T00:00:00",
    )

    timestamp4 = Timestamp(
        time_at=1625097600,  # 2021-07-01 00:00:00
        year=2021,
        month=7,
        day=1,
        hour=0,
        minute=0,
        second=0,
        timestamp_str="2021-07-01T00:00:00",
    )

    timestamp5 = Timestamp(
        time_at=1633046400,  # 2021-10-01 00:00:00
        year=2021,
        month=10,
        day=1,
        hour=0,
        minute=0,
        second=0,
        timestamp_str="2021-10-01T00:00:00",
    )

    # Create interval for event spanning multiple timestamps
    interval1 = Interval(time_from=timestamp2, time_to=timestamp3)

    # Create events with timestamps
    event1 = Event(
        name="Project Alpha Launch",
        description="Launched Project Alpha at the beginning of 2021",
        at=timestamp1,
        location="San Francisco",
    )

    event2 = Event(
        name="Team Meeting",
        description="Monthly team meeting discussing Q1 goals",
        during=interval1,
        location="New York",
    )

    event3 = Event(
        name="Product Release",
        description="Released new product features in July",
        at=timestamp4,
        location="Remote",
    )

    event4 = Event(
        name="Company Retreat",
        description="Annual company retreat in October",
        at=timestamp5,
        location="Lake Tahoe",
    )

    entities = [event1, event2, event3, event4]

    await add_data_points(entities)

    yield

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        pass


@pytest_asyncio.fixture
async def setup_test_environment_with_graph_data():
    """Set up a clean test environment with graph data (for fallback to triplets)."""
    base_dir = pathlib.Path(__file__).parent.parent.parent.parent
    system_directory_path = str(base_dir / ".cognee_system/test_temporal_retriever_with_graph")
    data_directory_path = str(base_dir / ".data_storage/test_temporal_retriever_with_graph")

    cognee.config.system_root_directory(system_directory_path)
    cognee.config.data_root_directory(data_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    class Company(DataPoint):
        name: str
        description: str

    class Person(DataPoint):
        name: str
        description: str
        works_for: Company

    company1 = Company(name="Figma", description="Figma is a company")
    person1 = Person(
        name="Steve Rodger",
        description="This is description about Steve Rodger",
        works_for=company1,
    )

    entities = [company1, person1]

    await add_data_points(entities)

    yield

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        pass


@pytest_asyncio.fixture
async def setup_test_environment_empty():
    """Set up a clean test environment without data."""
    base_dir = pathlib.Path(__file__).parent.parent.parent.parent
    system_directory_path = str(base_dir / ".cognee_system/test_temporal_retriever_empty")
    data_directory_path = str(base_dir / ".data_storage/test_temporal_retriever_empty")

    cognee.config.system_root_directory(system_directory_path)
    cognee.config.data_root_directory(data_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    yield

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        pass


@pytest.mark.asyncio
async def test_temporal_retriever_context_with_time_range(setup_test_environment_with_events):
    """Integration test: verify TemporalRetriever can retrieve events within time range."""
    retriever = TemporalRetriever(top_k=5)
    query = "What happened in January 2021?"

    triplets = await retriever.get_retrieved_objects(query)

    context = await retriever.get_context_from_objects(query=query, retrieved_objects=triplets)

    assert isinstance(context, str), "Context should be a string"
    assert len(context) > 0, "Context should not be empty"
    assert "Project Alpha" in context or "Launch" in context, (
        "Should retrieve Project Alpha Launch event from January 2021"
    )


@pytest.mark.asyncio
async def test_temporal_retriever_context_with_single_time(setup_test_environment_with_events):
    """Integration test: verify TemporalRetriever can retrieve events at specific time."""
    retriever = TemporalRetriever(top_k=5)
    query = "What happened in July 2021?"

    triplets = await retriever.get_retrieved_objects(query)

    context = await retriever.get_context_from_objects(query=query, retrieved_objects=triplets)

    assert isinstance(context, str), "Context should be a string"
    assert len(context) > 0, "Context should not be empty"
    assert "Product Release" in context or "July" in context, (
        "Should retrieve Product Release event from July 2021"
    )


@pytest.mark.asyncio
async def test_temporal_retriever_context_fallback_to_triplets(
    setup_test_environment_with_graph_data,
):
    """Integration test: verify TemporalRetriever falls back to triplets when no time extracted."""
    retriever = TemporalRetriever(top_k=5)
    query = "Who works at Figma?"

    triplets = await retriever.get_retrieved_objects(query)

    context = await retriever.get_context_from_objects(query=query, retrieved_objects=triplets)

    assert isinstance(context, str), "Context should be a string"
    assert len(context) > 0, "Context should not be empty"
    assert "Steve" in context or "Figma" in context, (
        "Should retrieve graph data via triplet search fallback"
    )


@pytest.mark.asyncio
async def test_temporal_retriever_context_empty_graph(setup_test_environment_empty):
    """Integration test: verify TemporalRetriever handles empty graph correctly."""
    retriever = TemporalRetriever()
    query = "What happened?"

    triplets = await retriever.get_retrieved_objects(query)

    context = await retriever.get_context_from_objects(query=query, retrieved_objects=triplets)

    assert isinstance(context, str), "Context should be a string"
    assert len(context) >= 0, "Context should be a string (possibly empty)"


@pytest.mark.asyncio
async def test_temporal_retriever_get_completion(setup_test_environment_with_events):
    """Integration test: verify TemporalRetriever can generate completions."""
    retriever = TemporalRetriever()
    query = "What happened in January 2021?"

    triplets = await retriever.get_retrieved_objects(query)
    context = await retriever.get_context_from_objects(query=query, retrieved_objects=triplets)

    completion = await retriever.get_completion_from_context(
        query=query, retrieved_objects=triplets, context=context
    )

    assert isinstance(completion, list), "Completion should be a list"
    assert len(completion) > 0, "Completion should not be empty"
    assert all(isinstance(item, str) and item.strip() for item in completion), (
        "Completion items should be non-empty strings"
    )


@pytest.mark.asyncio
async def test_temporal_retriever_get_completion_fallback(setup_test_environment_with_graph_data):
    """Integration test: verify TemporalRetriever get_completion works with triplet fallback."""
    retriever = TemporalRetriever()
    query = "Who works at Figma?"

    triplets = await retriever.get_retrieved_objects(query)
    context = await retriever.get_context_from_objects(query=query, retrieved_objects=triplets)

    completion = await retriever.get_completion_from_context(
        query=query, retrieved_objects=triplets, context=context
    )

    assert isinstance(completion, list), "Completion should be a list"
    assert len(completion) > 0, "Completion should not be empty"
    assert all(isinstance(item, str) and item.strip() for item in completion), (
        "Completion items should be non-empty strings"
    )


@pytest.mark.asyncio
async def test_temporal_retriever_top_k_limit(setup_test_environment_with_events):
    """Integration test: verify TemporalRetriever respects top_k parameter."""
    retriever = TemporalRetriever(top_k=2)
    query = "What happened in 2021?"

    triplets = await retriever.get_retrieved_objects(query)

    context = await retriever.get_context_from_objects(query=query, retrieved_objects=triplets)

    assert isinstance(context, str), "Context should be a string"
    separator_count = context.count("#####################")
    assert separator_count <= 1, "Should respect top_k limit of 2 events"


@pytest.mark.asyncio
async def test_temporal_retriever_multiple_events(setup_test_environment_with_events):
    """Integration test: verify TemporalRetriever can retrieve multiple events."""
    retriever = TemporalRetriever(top_k=10)
    query = "What events occurred in 2021?"

    triplets = await retriever.get_retrieved_objects(query)

    context = await retriever.get_context_from_objects(query=query, retrieved_objects=triplets)

    assert isinstance(context, str), "Context should be a string"
    assert len(context) > 0, "Context should not be empty"

    assert (
        "Project Alpha" in context
        or "Team Meeting" in context
        or "Product Release" in context
        or "Company Retreat" in context
    ), "Should retrieve at least one event from 2021"
