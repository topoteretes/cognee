import os
import pathlib
import sys
from typing import Optional, List, Union
from uuid import NAMESPACE_OID, uuid5
import asyncio

# Ensure local cognee is used
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import cognee
from cognee.infrastructure.files.storage import get_storage_config
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.search.operations import get_history
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger
from cognee.modules.search.types import SearchType
from cognee.modules.engine.models import NodeSet
from cognee.context_global_variables import agent_graph_name_ctx
from cognee.infrastructure.engine.models.Edge import Edge
from cognee.modules.engine.models import Entity
from cognee.tasks.temporal_graph.models import Event
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.tasks.chunks import chunk_by_paragraph
from cognee.modules.chunking.Chunker import Chunker

logger = get_logger()

# Test graph names - isolated from production data
TEST_GRAPH_COGNEE = "test_falkordb_cognee"
TEST_GRAPH_RED = "test_agentRed"
TEST_GRAPH_BLUE = "test_agentBlue"
TEST_GRAPHS = [TEST_GRAPH_COGNEE, TEST_GRAPH_RED, TEST_GRAPH_BLUE]


def delete_test_graphs_sync():
    """Delete all test graphs directly from FalkorDB (synchronous)."""
    try:
        import falkordb

        client = falkordb.FalkorDB(host="localhost", port=6379)
        # Get list of existing graphs
        existing_graphs = client.list_graphs()
        for graph_name in TEST_GRAPHS:
            if graph_name in existing_graphs:
                try:
                    graph = client.select_graph(graph_name)
                    graph.delete()
                    print(f"Deleted graph: {graph_name}")
                except Exception as e:
                    print(f"Could not delete {graph_name}: {e}")
    except Exception as e:
        print(f"FalkorDB cleanup error: {e}")


async def delete_test_graphs():
    """Delete all test graphs directly from FalkorDB."""
    delete_test_graphs_sync()


async def main():
    # 1. Config for FalkorDB
    # Disable internal multi-user check to allow our agent-based isolation
    os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"

    cognee.config.set_graph_database_provider("falkordb")
    cognee.config.set_vector_db_provider("falkordb")

    # Ensuring explicit connection details
    cognee.config.set_graph_db_config(
        {
            "graph_database_url": "localhost",
            "graph_database_port": 6379,
        }
    )
    cognee.config.set_vector_db_config(
        {
            "vector_db_url": "localhost",
            "vector_db_port": 6379,
        }
    )

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

    test_results = []

    async def run_step(name, coro):
        try:
            await coro
            test_results.append((name, "PASS", ""))
        except Exception as e:
            test_results.append((name, "FAIL", str(e)))
            print(f"Step {name} failed: {e}")

    # Initial cleanup - delete test graphs if they exist
    await run_step("Initial test graph cleanup", delete_test_graphs())

    # Initialize Cognee system state (creates internal tables, etc.)
    await run_step(
        "Initialize Cognee system",
        cognee.prune.prune_system(metadata=True),
    )

    # Set default context to test graph
    agent_graph_name_ctx.set(TEST_GRAPH_COGNEE)

    # Graph check
    from cognee.infrastructure.databases.graph import get_graph_engine

    graph_engine = await get_graph_engine()

    async def check_empty():
        is_empty = await graph_engine.is_empty()
        assert is_empty, "Test graph should be empty after cleanup"

    await run_step("Verify empty test graph", check_empty())

    dataset_name = "cs_explanations"

    # NodeSet Integration Test
    async def nodeset_test():
        nodeset_text = "FalkorDB is a graph database that supports cypher."
        await cognee.add([nodeset_text], dataset_name, node_set=["first"])
        await cognee.cognify([dataset_name])

        context_nonempty = await GraphCompletionRetriever(
            node_type=NodeSet,
            node_name=["first"],
        ).get_context("What is in the context?")

        assert isinstance(context_nonempty, list) and context_nonempty != [], (
            f"Expected non-empty context for 'first', got {context_nonempty}"
        )

        context_empty = await GraphCompletionRetriever(
            node_type=NodeSet,
            node_name=["nonexistent"],
        ).get_context("What is in the context?")

        assert context_empty == [], f"Expected empty context for 'nonexistent', got {context_empty}"

    await run_step("NodeSet Integration & Search", nodeset_test())

    # Multi-Agent Isolation Test - Red Agent
    async def multi_agent_red():
        print(f"Switching to {TEST_GRAPH_RED}...")
        agent_graph_name_ctx.set(TEST_GRAPH_RED)
        await cognee.add(["Apple is a red fruit."], "red_dataset")
        await cognee.cognify(["red_dataset"])
        red_results = await cognee.search("Apple", SearchType.CHUNKS)
        assert len(red_results) > 0, "AgentRed should find 'Apple'"

    await run_step("Multi-Agent Isolation: AgentRed", multi_agent_red())

    # Multi-Agent Isolation Test - Blue Agent
    async def multi_agent_blue():
        print(f"Switching to {TEST_GRAPH_BLUE}...")
        agent_graph_name_ctx.set(TEST_GRAPH_BLUE)
        blue_leak_results = await cognee.search("Apple", SearchType.CHUNKS)

        leak_found = False
        for res in blue_leak_results:
            if "red fruit" in str(res).lower():
                leak_found = True

        assert not leak_found, "Multi-agent isolation failed: AgentBlue saw AgentRed's data!"

    await run_step("Multi-Agent Isolation: AgentBlue (No Leakage)", multi_agent_blue())

    # Final Cleanup - delete test graphs
    await run_step("Final test graph cleanup", delete_test_graphs())

    # Print Summary Table
    print("\n" + "=" * 80)
    print(f"{'TEST NAME':<50} | {'STATUS':<7} | {'ERROR'}")
    print("-" * 80)
    for name, status, error in test_results:
        err_msg = (error[:25] + "...") if len(error) > 25 else error
        print(f"{name:<50} | {status:<7} | {err_msg}")
    print("=" * 80 + "\n")

    # Exit with code 1 if any test failed
    if any(status == "FAIL" for _, status, _ in test_results):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
