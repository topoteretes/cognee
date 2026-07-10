import json
import os
import pathlib

import cognee
from cognee.api.v1.datasets import datasets
from cognee.context_global_variables import backend_access_control_enabled
from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.provenance import make_source_ref_key
from cognee.infrastructure.databases.provenance.markers import stores_provenance_in_graph
from cognee.modules.engine.operations.setup import setup
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger

logger = get_logger()


async def _exclusive_node_ids_for_source_ref(graph_engine, source_ref_key):
    node_ids = await graph_engine.find_nodes_by_source_ref(source_ref_key)
    node_data = await graph_engine.get_node_delete_data(node_ids)
    return {
        node_id
        for node_id, data in node_data.items()
        if set(data.source_ref_keys) == {source_ref_key}
    }


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

    john_doc_person_name = "John"
    john_doc_exclusive_org_name = "Food for Hungry"
    john_document_text = (
        f"{john_doc_person_name} works for Apple. He is also affiliated with a non-profit "
        f"organization called '{john_doc_exclusive_org_name}'"
    )
    add_result = await cognee.add(john_document_text)
    johns_data_id = add_result.data_ingestion_info[0]["data_id"]

    add_result = await cognee.add(
        "Marie works for Apple as well. She is a software engineer on MacOS project."
    )
    maries_data_id = add_result.data_ingestion_info[0]["data_id"]

    vector_engine = await get_vector_engine_async()

    assert not await vector_engine.has_collection("EdgeType_relationship_name")
    assert not await vector_engine.has_collection("Entity_name")
    assert not await vector_engine.has_collection("DocumentChunk_text")
    assert not await vector_engine.has_collection("TextSummary_text")
    assert not await vector_engine.has_collection("TextDocument_text")

    cognify_result: dict = await cognee.cognify()
    dataset_id = list(cognify_result.keys())[0]

    graph_engine = await get_graph_engine()
    assert await stores_provenance_in_graph(graph_engine), (
        "Fresh default Ladybug graph should store delete provenance in the graph."
    )

    initial_nodes, initial_edges = await graph_engine.get_graph_data()
    initial_data_nodes = [n for n in initial_nodes if n[1].get("type") != "EdgeType"]
    assert len(initial_data_nodes) >= 14 and len(initial_edges) >= 18, (
        f"Expected >= 14 data nodes and >= 18 edges, got {len(initial_data_nodes)} and {len(initial_edges)}"
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

    initial_node_ids = {node[0] for node in initial_nodes}

    # Pre-delete: graph node ids removable with that data only (shared entities excluded).
    john_source_ref = make_source_ref_key(dataset_id, johns_data_id)
    marie_source_ref = make_source_ref_key(dataset_id, maries_data_id)
    john_node_ids = await _exclusive_node_ids_for_source_ref(graph_engine, john_source_ref)
    marie_node_ids = await _exclusive_node_ids_for_source_ref(graph_engine, marie_source_ref)

    assert john_node_ids, "John's doc must contribute at least one non-shared graph node."
    assert marie_node_ids, "Marie's doc must contribute at least one non-shared graph node."

    user = await get_default_user()

    # --- Delete John's data only ---
    await datasets.delete_data(dataset_id, johns_data_id, user)  # type: ignore

    still_john_nodes = await graph_engine.get_nodes(list(john_node_ids))
    assert len(still_john_nodes) == 0, "John-exclusive nodes should be removed from the graph."

    nodes, edges = await graph_engine.get_graph_data()
    assert not any(src in john_node_ids or tgt in john_node_ids for src, tgt, _, _ in edges), (
        "No graph edge should still attach to a removed John-exclusive node."
    )

    still_marie = await graph_engine.get_nodes(list(marie_node_ids))
    assert len(still_marie) == len(marie_node_ids), (
        "Marie-exclusive nodes must remain after deleting John's data only."
    )
    # Marie never mentions John or that org; they should be exclusive to John's data and gone now.
    john_only_entity_graph_names_lower = frozenset(
        {john_doc_person_name.lower(), john_doc_exclusive_org_name.lower()}
    )
    assert not any(
        node[1].get("name", "").lower() in john_only_entity_graph_names_lower for node in nodes
    ), (
        "Graph should not still name John or his org after deleting only John's document "
        f"({john_doc_person_name!r}, {john_doc_exclusive_org_name!r})."
    )

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

    # --- Delete Marie's data; graph and vectors should be empty ---
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
