import os
import shutil
import asyncio
import pathlib
import cognee
from cognee.infrastructure.files.storage import get_storage_config
from cognee.modules.engine.models import NodeSet
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.shared.logging_utils import get_logger
from cognee.modules.search.types import SearchType
from cognee.modules.search.operations import get_history
from cognee.modules.users.methods import get_default_user
from cognee.context_global_variables import agent_graph_name_ctx

logger = get_logger()

async def main():
    # Clean up test directories before starting
    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_falkordb")
        ).resolve()
    )
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_falkordb")
        ).resolve()
    )

    try:
        # Set FalkorDB as the graph and vector database provider
        cognee.config.set_graph_database_provider("falkordb")
        os.environ["VECTOR_DB_PROVIDER"] = "falkordb"
        
        # Use localhost for tests (standard falkordb setup)
        os.environ["GRAPH_DATABASE_URL"] = "localhost"
        os.environ["GRAPH_DATABASE_PORT"] = "6379"
        
        cognee.config.data_root_directory(data_directory_path)
        cognee.config.system_root_directory(cognee_directory_path)

        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

        dataset_name = "falkordb_test"

        # 1. Basic Functionality Test
        test_text = "FalkorDB is a low-latency graph database."
        await cognee.add([test_text], dataset_name)
        
        from cognee.infrastructure.databases.graph import get_graph_engine
        graph_engine = await get_graph_engine()
        
        # Should be empty before cognify
        assert await graph_engine.is_empty(), "Graph should be empty before cognify"
        
        await cognee.cognify([dataset_name])
        
        # Should not be empty after cognify
        assert not await graph_engine.is_empty(), "Graph should NOT be empty after cognify"

        # 2. Search Integration Test
        search_results = await cognee.search(
            query_type = SearchType.CHUNKS,
            query_text = "low-latency"
        )
        assert len(search_results) > 0, "Vector search should return results"

        # 3. Multi-Agent Isolation Test
        print("\nTesting Multi-Agent Isolation...")
        
        # Clear data root for fresh isolation test
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        
        # Agent Blue Context
        agent_graph_name_ctx.set("AgentBlue")
        await cognee.add(["Azure is a cloud provider."], "blue_data")
        await cognee.cognify(["blue_data"])
        
        # Agent Red Context
        agent_graph_name_ctx.set("AgentRed")
        await cognee.add(["Ferrari is a car manufacturer."], "red_data")
        await cognee.cognify(["red_data"])
        
        # Verify Blue cannot see Red data
        agent_graph_name_ctx.set("AgentBlue")
        blue_results = await cognee.search(SearchType.CHUNKS, "Ferrari")
        # Vector search always returns something, but check content
        has_red_leak = any("ferrari" in str(r).lower() for r in blue_results)
        assert not has_red_leak, "AgentBlue leaked AgentRed data!"
        
        print("Isolation Test: [PASS]")

        # Cleanup
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

    finally:
        # Ensure cleanup even if tests fail
        for path in [data_directory_path, cognee_directory_path]:
            if os.path.exists(path):
                shutil.rmtree(path)

if __name__ == "__main__":
    asyncio.run(main())
