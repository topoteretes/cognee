import json
import os
import pathlib

import cognee
from cognee.api.v1.datasets import datasets
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.engine.operations.setup import setup
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger

logger = get_logger()


async def main():
    data_directory_path = os.path.join(
        pathlib.Path(__file__).parent, ".data_storage/test_delete_default_graph_non_mocked"
    )
    cognee.config.data_root_directory(data_directory_path)

    cognee_directory_path = os.path.join(
        pathlib.Path(__file__).parent, ".cognee_system/test_delete_default_graph_non_mocked"
    )
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    add_result = await cognee.add(
        "John works for Apple. He is also affiliated with a non-profit organization called 'Food for Hungry'"
    )
    johns_data_id = add_result.data_ingestion_info[0]["data_id"]

    add_result = await cognee.add(
        "Marie works for Apple as well. She is a software engineer on MacOS project."
    )
    maries_data_id = add_result.data_ingestion_info[0]["data_id"]

    vector_engine = get_vector_engine()

    assert not await vector_engine.has_collection("EdgeType_relationship_name")
    assert not await vector_engine.has_collection("Entity_name")
    assert not await vector_engine.has_collection("DocumentChunk_text")
    assert not await vector_engine.has_collection("TextSummary_text")
    assert not await vector_engine.has_collection("TextDocument_text")

    cognify_result: dict = await cognee.cognify()
    dataset_id = list(cognify_result.keys())[0]

    graph_engine = await get_graph_engine()
    initial_nodes, initial_edges = await graph_engine.get_graph_data()
    assert len(initial_nodes) >= 14 and len(initial_edges) >= 18, (
        "Number of nodes and edges is not correct."
    )

    initial_nodes_by_vector_collection = {}

    for node in initial_nodes:
        node_data = node[1]
        node_metadata = (
            node_data["metadata"]
            if type(node_data["metadata"]) is dict
            else json.loads(node_data["metadata"])
        )
        collection_name = node_data["type"] + "_" + node_metadata["index_fields"][0]
        if collection_name not in initial_nodes_by_vector_collection:
            initial_nodes_by_vector_collection[collection_name] = []
        initial_nodes_by_vector_collection[collection_name].append(node)

    initial_node_ids = set([node[0] for node in initial_nodes])

    user = await get_default_user()
    await datasets.delete_data(dataset_id, johns_data_id, user)  # type: ignore

    nodes, edges = await graph_engine.get_graph_data()
    assert len(nodes) >= 9 and len(nodes) <= 11 and len(edges) >= 10 and len(edges) <= 12, (
        "Nodes and edges are not deleted."
    )
    assert not any(
        node[1]["name"] == "john" or node[1]["name"] == "food for hungry"
        for node in nodes
        if "name" in node[1]
    ), "Nodes are not deleted."

    after_first_delete_node_ids = set([node[0] for node in nodes])

    after_delete_nodes_by_vector_collection = {}
    for node in initial_nodes:
        node_data = node[1]
        node_metadata = (
            node_data["metadata"]
            if type(node_data["metadata"]) is dict
            else json.loads(node_data["metadata"])
        )
        collection_name = node_data["type"] + "_" + node_metadata["index_fields"][0]
        if collection_name not in after_delete_nodes_by_vector_collection:
            after_delete_nodes_by_vector_collection[collection_name] = []
        after_delete_nodes_by_vector_collection[collection_name].append(node)

    removed_node_ids = initial_node_ids - after_first_delete_node_ids

    for collection_name, initial_nodes in initial_nodes_by_vector_collection.items():
        query_node_ids = [node[0] for node in initial_nodes if node[0] in removed_node_ids]

        if query_node_ids:
            vector_items = await vector_engine.retrieve(collection_name, query_node_ids)
            assert len(vector_items) == 0, "Vector items are not deleted."

    await datasets.delete_data(dataset_id, maries_data_id, user)  # type: ignore

    final_nodes, final_edges = await graph_engine.get_graph_data()
    assert len(final_nodes) == 0 and len(final_edges) == 0, "Nodes and edges are not deleted."

    for collection_name, initial_nodes in initial_nodes_by_vector_collection.items():
        query_node_ids = [node[0] for node in initial_nodes]

        if query_node_ids:
            vector_items = await vector_engine.retrieve(collection_name, query_node_ids)
            assert len(vector_items) == 0, "Vector items are not deleted."

    query_edge_ids = [edge[0] for edge in initial_edges]

    vector_items = await vector_engine.retrieve("EdgeType_relationship_name", query_edge_ids)
    assert len(vector_items) == 0, "Vector items are not deleted."


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
