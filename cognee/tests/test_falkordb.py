"""Integration test for FalkorDB graph and vector adapters.

This test verifies FalkorDB as both graph and vector database provider.
Requires a running FalkorDB instance on localhost:6379.

To run this test:
    1. Start FalkorDB: docker run -d -p 6379:6379 falkordb/falkordb:latest
    2. Run: pytest cognee/tests/test_falkordb.py -v
"""

import os
import pathlib
import pytest
import cognee
from cognee.infrastructure.files.storage import get_storage_config
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.search.operations import get_history
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger
from cognee.modules.search.types import SearchType
from cognee.modules.engine.models import NodeSet

logger = get_logger()


# Skip if FalkorDB is not available
def falkordb_available():
    """Check if FalkorDB is available on localhost:6379."""
    try:
        from falkordb import FalkorDB

        client = FalkorDB(host="localhost", port=6379)
        graph = client.select_graph("test_connection")
        graph.query("RETURN 1")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not falkordb_available(), reason="FalkorDB not available")
async def test_falkordb():
    """Test FalkorDB as graph and vector database provider."""
    # Configure FalkorDB as the database provider
    cognee.config.set_graph_database_provider("falkordb")
    cognee.config.set_vector_db_provider("falkordb")

    # Set connection details from environment or defaults
    os.environ.setdefault("GRAPH_DATABASE_URL", "localhost")
    os.environ.setdefault("GRAPH_DATABASE_PORT", "6379")
    os.environ.setdefault("VECTOR_DATABASE_URL", "localhost")
    os.environ.setdefault("VECTOR_DATABASE_PORT", "6379")

    # Set up test directories
    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_falkordb")
        ).resolve()
    )
    cognee.config.data_root_directory(data_directory_path)

    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_falkordb")
        ).resolve()
    )
    cognee.config.system_root_directory(cognee_directory_path)

    # Clean up any existing data
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    dataset_name = "falkordb_test"

    # Get the graph engine and verify it's empty
    from cognee.infrastructure.databases.graph import get_graph_engine

    graph_engine = await get_graph_engine()
    is_empty = await graph_engine.is_empty()
    assert is_empty, "Graph should be empty after prune"

    # Add test data
    test_file_path = os.path.join(
        pathlib.Path(__file__).parent, "test_data/Natural_language_processing.txt"
    )
    await cognee.add([test_file_path], dataset_name)

    # Graph should still be empty before cognify
    is_empty = await graph_engine.is_empty()
    assert is_empty, "Graph should be empty before cognify"

    # Run cognify to process the data
    await cognee.cognify([dataset_name])

    # Verify graph is no longer empty
    is_empty = await graph_engine.is_empty()
    assert not is_empty, "Graph should not be empty after cognify"

    # Test vector search
    from cognee.infrastructure.databases.vector import get_vector_engine

    vector_engine = get_vector_engine()

    # Search for entities
    search_results = await cognee.search(
        query_type=SearchType.CHUNKS, query_text="natural language"
    )
    assert len(search_results) > 0, "Search should return results"

    logger.info("FalkorDB test passed: %d results found", len(search_results))

    # Test graph metrics
    metrics = await graph_engine.get_graph_metrics()
    assert metrics["nodes"] > 0, "Graph should have nodes"
    assert metrics["edges"] > 0, "Graph should have edges"

    logger.info(
        "Graph metrics: %d nodes, %d edges", metrics["nodes"], metrics["edges"]
    )

    # Clean up
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


@pytest.mark.skipif(not falkordb_available(), reason="FalkorDB not available")
async def test_falkordb_multi_agent_isolation():
    """Test multi-agent graph isolation in FalkorDB."""
    from cognee.context_global_variables import agent_graph_name_ctx
    from cognee.infrastructure.databases.graph import get_graph_engine

    # Configure FalkorDB
    cognee.config.set_graph_database_provider("falkordb")
    os.environ.setdefault("GRAPH_DATABASE_URL", "localhost")
    os.environ.setdefault("GRAPH_DATABASE_PORT", "6379")

    # Test with agent context
    agent_graph_name_ctx.set("TestAgentGraph")

    graph_engine = await get_graph_engine()

    # Verify the graph name is set correctly
    graph_name = graph_engine._get_graph_name_from_ctx()
    assert graph_name == "TestAgentGraph", f"Expected 'TestAgentGraph', got '{graph_name}'"

    # Clean up context
    agent_graph_name_ctx.set(None)

    # Verify default is used when no agent context
    graph_name = graph_engine._get_graph_name_from_ctx()
    assert "Cognee" in graph_name, f"Expected default graph name, got '{graph_name}'"

    logger.info("Multi-agent isolation test passed")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_falkordb())
    asyncio.run(test_falkordb_multi_agent_isolation())
