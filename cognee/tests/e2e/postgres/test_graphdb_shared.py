"""Shared graph database backend test.

Exercises the full add -> cognify -> search pipeline against whichever
graph backend is configured via `provider`. Requires:
  - A working LLM API key (LLM_API_KEY in .env)
  - A working vector database (default LanceDB is fine)
  - For postgres: DB_PROVIDER=postgres with valid connection credentials
"""

import os
import shutil
import pathlib

import cognee
from cognee.infrastructure.files.storage import get_storage_config
from cognee.modules.engine.models import NodeSet
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.search.types import SearchType
from cognee.modules.search.operations import get_history
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger

logger = get_logger()

TEST_DATA_DIR = pathlib.Path(__file__).parents[2] / "test_data"


async def run_graph_db_test(provider: str):
    """Run the full graph DB integration test for the given provider."""

    # Set up isolated directories for this test run
    base = pathlib.Path(__file__).parent
    data_dir = str((base / f".data_storage/test_{provider}").resolve())
    system_dir = str((base / f".cognee_system/test_{provider}").resolve())

    # Capture current config so we can restore after the test
    from cognee.infrastructure.databases.graph.config import get_graph_config
    from cognee.base_config import get_base_config

    graph_config = get_graph_config()
    base_config = get_base_config()
    prev_provider = graph_config.graph_database_provider
    prev_data_root = base_config.data_root_directory
    prev_system_root = base_config.system_root_directory

    try:
        cognee.config.set_graph_database_provider(provider)
        cognee.config.data_root_directory(data_dir)
        cognee.config.system_root_directory(system_dir)

        # Clean slate
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

        dataset_name = "cs_explanations"

        # Add first document
        nlp_path = str(TEST_DATA_DIR / "Natural_language_processing.txt")
        await cognee.add([nlp_path], dataset_name)

        # Add second document
        quantum_path = str(TEST_DATA_DIR / "Quantum_computers.txt")

        from cognee.infrastructure.databases.graph import get_graph_engine

        graph_engine = await get_graph_engine()

        # Graph should be empty before cognify
        is_empty = await graph_engine.is_empty()
        assert is_empty, f"{provider}: graph should be empty before cognify"

        await cognee.add([quantum_path], dataset_name)

        is_empty = await graph_engine.is_empty()
        assert is_empty, f"{provider}: graph should still be empty before cognify"

        # Run cognify (LLM extraction -> graph + vector indexing)
        await cognee.cognify([dataset_name])

        is_empty = await graph_engine.is_empty()
        assert not is_empty, f"{provider}: graph should not be empty after cognify"

        # Search via vector to get a node name for graph queries
        from cognee.infrastructure.databases.vector import get_vector_engine

        vector_engine = get_vector_engine()
        random_node = (
            await vector_engine.search("Entity_name", "Quantum computer", include_payload=True)
        )[0]
        random_node_name = random_node.payload["text"]

        # Test GRAPH_COMPLETION search (exercises graph adapter)
        search_results = await cognee.search(
            query_type=SearchType.GRAPH_COMPLETION, query_text=random_node_name
        )
        assert len(search_results) != 0, f"{provider}: GRAPH_COMPLETION returned no results"

        # Test CHUNKS search (vector-only, but confirms pipeline integrity)
        search_results = await cognee.search(
            query_type=SearchType.CHUNKS, query_text=random_node_name
        )
        assert len(search_results) != 0, f"{provider}: CHUNKS returned no results"

        # Test SUMMARIES search
        search_results = await cognee.search(
            query_type=SearchType.SUMMARIES, query_text=random_node_name
        )
        assert len(search_results) != 0, f"{provider}: SUMMARIES returned no results"

        # Check search history: 3 searches x 2 entries each (query + result) = 6
        user = await get_default_user()
        history = await get_history(user.id)
        assert len(history) == 6, f"{provider}: expected 6 history entries, got {len(history)}"

        # Test nodeset filtering via GraphCompletionRetriever
        nodeset_text = "Neo4j is a graph database that supports cypher."
        await cognee.add([nodeset_text], dataset_name, node_set=["first"])
        await cognee.cognify([dataset_name])

        # Existing nodeset should return results
        graph_retriever = GraphCompletionRetriever(node_type=NodeSet, node_name=["first"])
        objects = await graph_retriever.get_retrieved_objects("What is in the context?")
        context_nonempty = await graph_retriever.get_context_from_objects(
            query="What is in the context?", retrieved_objects=objects
        )

        # Nonexistent nodeset should return empty
        graph_retriever = GraphCompletionRetriever(node_type=NodeSet, node_name=["nonexistent"])
        objects = await graph_retriever.get_retrieved_objects("What is in the context?")
        context_empty = await graph_retriever.get_context_from_objects(
            query="What is in the context?", retrieved_objects=objects
        )

        assert isinstance(context_nonempty, str) and context_nonempty != "", (
            f"{provider}: expected non-empty context for existing nodeset, got: {context_nonempty!r}"
        )
        assert context_empty == "", (
            f"{provider}: expected empty context for nonexistent nodeset, got: {context_empty!r}"
        )

        # Clean up and verify
        await cognee.prune.prune_data()
        data_root_directory = get_storage_config()["data_root_directory"]
        assert not os.path.isdir(data_root_directory), f"{provider}: local data files not deleted"

        await cognee.prune.prune_system(metadata=True)

        is_empty = await graph_engine.is_empty()
        assert is_empty, f"{provider}: graph should be empty after prune"

    finally:
        # Restore previous config
        cognee.config.set_graph_database_provider(prev_provider)
        cognee.config.data_root_directory(prev_data_root)
        cognee.config.system_root_directory(prev_system_root)

        for path in [data_dir, system_dir]:
            if os.path.exists(path):
                shutil.rmtree(path)
