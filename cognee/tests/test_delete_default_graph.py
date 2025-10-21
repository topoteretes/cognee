import os
import pathlib

import cognee
from cognee.api.v1.datasets import datasets
from cognee.api.v1.visualize.visualize import visualize_graph
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.data.methods import get_dataset_data
from cognee.modules.engine.operations.setup import setup
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger

logger = get_logger()


async def main():
    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_delete_default_graph")
        ).resolve()
    )
    cognee.config.data_root_directory(data_directory_path)
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_delete_default_graph")
        ).resolve()
    )
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    vector_engine = get_vector_engine()

    assert not await vector_engine.has_collection("EdgeType_relationship_name")
    assert not await vector_engine.has_collection("Entity_name")

    await cognee.add(
        "John works for Apple. He is also affiliated with a non-profit organization called 'Food for Hungry'"
    )
    await cognee.add("Marie works for Apple as well. She is a software engineer on MacOS project.")

    cognify_result: dict = await cognee.cognify()
    dataset_id = list(cognify_result.keys())[0]

    dataset_data = await get_dataset_data(dataset_id)
    added_data = dataset_data[0]

    file_path = os.path.join(
        pathlib.Path(__file__).parent, ".artifacts", "graph_visualization_full.html"
    )
    await visualize_graph(file_path)

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()
    assert len(nodes) >= 12 and len(edges) >= 18, "Nodes and edges are not deleted."

    user = await get_default_user()
    await datasets.delete_data(dataset_id, added_data.id, user)  # type: ignore

    file_path = os.path.join(
        pathlib.Path(__file__).parent, ".artifacts", "graph_visualization_after_delete.html"
    )
    await visualize_graph(file_path)

    nodes, edges = await graph_engine.get_graph_data()
    assert len(nodes) >= 8 and len(nodes) < 12 and len(edges) >= 10 and len(edges) < 18, (
        "Nodes and edges are not deleted."
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
