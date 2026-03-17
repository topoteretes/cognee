import os
import shutil
import cognee
import pathlib
from cognee.infrastructure.files.storage import get_storage_config
from cognee.shared.logging_utils import get_logger
from cognee.modules.search.types import SearchType
from cognee.modules.search.operations import get_history
from cognee.modules.users.methods import get_default_user

logger = get_logger()


async def main():
    # Clean up test directories before starting
    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_remote_kuzu")
        ).resolve()
    )
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_remote_kuzu")
        ).resolve()
    )

    try:
        # Set Kuzu as the graph database provider
        cognee.config.set_graph_database_provider("kuzu")
        cognee.config.data_root_directory(data_directory_path)
        cognee.config.system_root_directory(cognee_directory_path)

        # Configure remote Kuzu database using environment variables
        os.environ["KUZU_HOST"] = os.getenv("KUZU_HOST", "localhost")
        os.environ["KUZU_PORT"] = os.getenv("KUZU_PORT", "8000")
        os.environ["KUZU_USERNAME"] = os.getenv("KUZU_USERNAME", "kuzu")
        os.environ["KUZU_PASSWORD"] = os.getenv("KUZU_PASSWORD", "kuzu")
        os.environ["KUZU_DATABASE"] = os.getenv("KUZU_DATABASE", "cognee_test")

        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

        dataset_name = "cs_explanations"

        explanation_file_path_nlp = os.path.join(
            pathlib.Path(__file__).parent, "test_data/Natural_language_processing.txt"
        )
        await cognee.add([explanation_file_path_nlp], dataset_name)

        explanation_file_path_quantum = os.path.join(
            pathlib.Path(__file__).parent, "test_data/Quantum_computers.txt"
        )

        await cognee.add([explanation_file_path_quantum], dataset_name)

        await cognee.cognify([dataset_name])

        from cognee.infrastructure.databases.vector import get_vector_engine

        vector_engine = get_vector_engine()
        random_node = (
            await vector_engine.search("Entity_name", "Quantum computer", include_payload=True)
        )[0]
        random_node_name = random_node.payload["text"]

        search_results = await cognee.search(
            query_type=SearchType.GRAPH_COMPLETION, query_text=random_node_name
        )
        assert len(search_results) != 0, "The search results list is empty."
        print("\n\nExtracted sentences are:\n")
        for result in search_results:
            print(f"{result}\n")

        search_results = await cognee.search(
            query_type=SearchType.CHUNKS, query_text=random_node_name
        )
        assert len(search_results) != 0, "The search results list is empty."
        print("\n\nExtracted chunks are:\n")
        for result in search_results:
            print(f"{result}\n")

        search_results = await cognee.search(
            query_type=SearchType.SUMMARIES, query_text=random_node_name
        )
        assert len(search_results) != 0, "Query related summaries don't exist."
        print("\nExtracted summaries are:\n")
        for result in search_results:
            print(f"{result}\n")

        user = await get_default_user()
        history = await get_history(user.id)
        assert len(history) == 6, "Search history is not correct."

        await cognee.prune.prune_data()
        data_root_directory = get_storage_config()["data_root_directory"]
        assert not os.path.isdir(data_root_directory), "Local data files are not deleted"

        await cognee.prune.prune_system(metadata=True)
        from cognee.infrastructure.databases.graph import get_graph_engine

        graph_engine = await get_graph_engine()
        nodes, edges = await graph_engine.get_graph_data()
        assert len(nodes) == 0 and len(edges) == 0, "Remote Kuzu graph database is not empty"

    finally:
        # Ensure cleanup even if tests fail
        for path in [data_directory_path, cognee_directory_path]:
            if os.path.exists(path):
                shutil.rmtree(path)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
