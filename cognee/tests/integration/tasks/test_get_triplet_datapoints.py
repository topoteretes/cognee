import os
import pathlib
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch

import cognee
from cognee.tasks.memify.get_triplet_datapoints import get_triplet_datapoints
from cognee.modules.engine.models import Triplet


@pytest_asyncio.fixture
async def setup_test_environment():
    """Set up a clean test environment with a simple graph."""
    base_dir = pathlib.Path(__file__).parent.parent.parent.parent
    data_directory_path = str(base_dir / ".data_storage/test_get_triplet_datapoints_integration")
    cognee_directory_path = str(base_dir / ".cognee_system/test_get_triplet_datapoints_integration")

    cognee.config.data_root_directory(data_directory_path)
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    dataset_name = "test_triplets"

    text = "Volkswagen is a german car manufacturer from Wolfsburg. They produce different models such as Golf, Polo and Touareg."
    await cognee.add(text, dataset_name)
    await cognee.cognify([dataset_name])

    yield dataset_name

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


@pytest.mark.asyncio
async def test_get_triplet_datapoints_integration(setup_test_environment):
    """Integration test: verify get_triplet_datapoints works with real graph data."""

    from cognee.infrastructure.databases.graph import get_graph_engine

    graph_engine = await get_graph_engine()

    if not hasattr(graph_engine, "get_triplets_batch"):
        pytest.skip("Graph engine does not support get_triplets_batch")

    triplets = []
    with patch(
        "cognee.tasks.memify.get_triplet_datapoints.index_data_points", new_callable=AsyncMock
    ):
        async for triplet in get_triplet_datapoints([{}], triplets_batch_size=10):
            triplets.append(triplet)

    nodes, edges = await graph_engine.get_graph_data()

    if len(edges) > 0 and len(triplets) == 0:
        test_triplets = await graph_engine.get_triplets_batch(offset=0, limit=10)
        if len(test_triplets) == 0:
            pytest.fail(
                f"Edges exist in graph ({len(edges)} edges) but get_triplets_batch found none. "
                f"This indicates the query pattern may not match the graph structure."
            )

    for triplet in triplets:
        assert isinstance(triplet, Triplet), "Each item should be a Triplet instance"
        assert triplet.from_node_id, "Triplet should have from_node_id"
        assert triplet.to_node_id, "Triplet should have to_node_id"
        assert triplet.text, "Triplet should have embeddable text"
