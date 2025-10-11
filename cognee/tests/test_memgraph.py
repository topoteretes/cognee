import os

import pathlib
import cognee
from cognee.infrastructure.files.storage import get_storage_config
from cognee.modules.search.operations import get_history
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger
from cognee.modules.search.types import SearchType


logger = get_logger()


async def main():
    cognee.config.set_graph_database_provider("memgraph")
    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_memgraph")
        ).resolve()
    )
    cognee.config.data_root_directory(data_directory_path)
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_memgraph")
        ).resolve()
    )
    cognee.config.system_root_directory(cognee_directory_path)

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
    random_node = (await vector_engine.search("Entity_name", "Quantum computer"))[0]
    random_node_name = random_node.payload["text"]

    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION, query_text=random_node_name
    )
    assert len(search_results) != 0, "The search results list is empty."
    print("\n\nExtracted sentences are:\n")
    for result in search_results:
        print(f"{result}\n")

    search_results = await cognee.search(query_type=SearchType.CHUNKS, query_text=random_node_name)
    assert len(search_results) != 0, "The search results list is empty."
    print("\n\nExtracted chunks are:\n")
    for result in search_results:
        print(f"{result}\n")

    search_results = await cognee.search(
        query_type=SearchType.SUMMARIES, query_text=random_node_name
    )
    assert len(search_results) != 0, "Query related summaries don't exist."
    print("\nExtracted results are:\n")
    for result in search_results:
        print(f"{result}\n")

    search_results = await cognee.search(
        query_type=SearchType.NATURAL_LANGUAGE,
        query_text=f"Find nodes connected to node with name {random_node_name}",
    )
    assert len(search_results) != 0, "Query related natural language don't exist."
    print("\nExtracted results are:\n")
    for result in search_results:
        print(f"{result}\n")

    user = await get_default_user()
    history = await get_history(user.id)

    assert len(history) == 8, "Search history is not correct."

    await cognee.prune.prune_data()
    data_root_directory = get_storage_config()["data_root_directory"]
    assert not os.path.isdir(data_root_directory), "Local data files are not deleted"

    await cognee.prune.prune_system(metadata=True)
    from cognee.infrastructure.databases.graph import get_graph_engine

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()
    assert len(nodes) == 0 and len(edges) == 0, "Memgraph graph database is not empty"


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
