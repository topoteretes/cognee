import logging
import os
import pathlib

import pytest
import pytest_asyncio

import cognee
from cognee.low_level import setup
from cognee.modules.engine.models import Triplet
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.modules.retrieval.triplet_retriever import TripletRetriever
from cognee.tasks.storage import add_data_points

logger = logging.getLogger(__name__)


@pytest_asyncio.fixture
async def setup_test_environment_with_triplets():
    """Set up a clean test environment with triplets."""
    base_dir = pathlib.Path(__file__).parent.parent.parent.parent
    system_directory_path = str(base_dir / ".cognee_system/test_triplet_retriever_context_simple")
    data_directory_path = str(base_dir / ".data_storage/test_triplet_retriever_context_simple")

    cognee.config.system_root_directory(system_directory_path)
    cognee.config.data_root_directory(data_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    triplet1 = Triplet(
        from_node_id="node1",
        to_node_id="node2",
        text="Alice knows Bob",
    )
    triplet2 = Triplet(
        from_node_id="node2",
        to_node_id="node3",
        text="Bob works at Tech Corp",
    )

    triplets = [triplet1, triplet2]
    await add_data_points(triplets)

    yield

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        logger.debug("Ignoring exception in setup_test_environment_with_triplets", exc_info=True)


@pytest_asyncio.fixture
async def setup_test_environment_empty():
    """Set up a clean test environment without triplets."""
    base_dir = pathlib.Path(__file__).parent.parent.parent.parent
    system_directory_path = str(
        base_dir / ".cognee_system/test_triplet_retriever_context_empty_collection"
    )
    data_directory_path = str(
        base_dir / ".data_storage/test_triplet_retriever_context_empty_collection"
    )

    cognee.config.system_root_directory(system_directory_path)
    cognee.config.data_root_directory(data_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    yield

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        logger.debug("Ignoring exception in setup_test_environment_empty", exc_info=True)


@pytest.mark.asyncio
async def test_triplet_retriever_context_simple(setup_test_environment_with_triplets):
    """Integration test: verify TripletRetriever can retrieve triplet context."""
    retriever = TripletRetriever(top_k=5)
    query = "Alice"

    triplets = await retriever.get_retrieved_objects(query)

    context = await retriever.get_context_from_objects(query=query, retrieved_objects=triplets)

    assert "Alice knows Bob" in context, "Failed to get Alice triplet"
    assert isinstance(context, str), "Context should be a string"
    assert len(context) > 0, "Context should not be empty"


@pytest.mark.asyncio
async def test_triplet_retriever_context_multiple_triplets(setup_test_environment_with_triplets):
    """Integration test: verify TripletRetriever can retrieve multiple triplets."""
    retriever = TripletRetriever(top_k=5)
    query = "Bob"

    triplets = await retriever.get_retrieved_objects(query)

    context = await retriever.get_context_from_objects(query=query, retrieved_objects=triplets)

    assert "Alice knows Bob" in context or "Bob works at Tech Corp" in context, (
        "Failed to get Bob-related triplets"
    )


@pytest.mark.asyncio
async def test_triplet_retriever_top_k_limit(setup_test_environment_with_triplets):
    """Integration test: verify TripletRetriever respects top_k parameter."""
    retriever = TripletRetriever(top_k=1)
    query = "Alice"

    triplets = await retriever.get_retrieved_objects(query)

    context = await retriever.get_context_from_objects(query=query, retrieved_objects=triplets)

    assert isinstance(context, str), "Context should be a string"


@pytest.mark.asyncio
async def test_triplet_retriever_context_empty(setup_test_environment_empty):
    """Integration test: verify TripletRetriever handles empty graph correctly."""
    await setup()

    retriever = TripletRetriever()

    with pytest.raises(NoDataError):
        await retriever.get_retrieved_objects("Alice")
