import os
import json
import pathlib
import pytest
from unittest.mock import AsyncMock, patch

import cognee
from cognee.api.v1.datasets import datasets
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.llm import LLMGateway
from cognee.modules.engine.operations.setup import setup
from cognee.modules.users.methods import create_user, get_default_user
from cognee.shared.data_models import KnowledgeGraph, Node, Edge, SummarizedContent
from cognee.shared.logging_utils import get_logger

logger = get_logger()


@pytest.mark.asyncio
@patch.object(LLMGateway, "acreate_structured_output", new_callable=AsyncMock)
async def main(mock_create_structured_output: AsyncMock):
    data_directory_path = os.path.join(
        pathlib.Path(__file__).parent, ".data_storage/test_delete_dataset_neo4j"
    )
    cognee.config.data_root_directory(data_directory_path)

    cognee_directory_path = os.path.join(
        pathlib.Path(__file__).parent, ".cognee_system/test_delete_dataset_neo4j"
    )
    cognee.config.system_root_directory(cognee_directory_path)

    # Disable backend access control for this test
    os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    def mock_llm_output(text_input: str, system_prompt: str, response_model):
        if text_input == "test":  # LLM connection test
            return "test"

        if "John" in text_input and response_model == SummarizedContent:
            return SummarizedContent(
                summary="Summary of John's work.", description="Summary of John's work."
            )

        if "Marie" in text_input and response_model == SummarizedContent:
            return SummarizedContent(
                summary="Summary of Marie's work.", description="Summary of Marie's work."
            )

        if "Marie" in text_input and response_model == KnowledgeGraph:
            return KnowledgeGraph(
                nodes=[
                    Node(id="Marie", name="Marie", type="Person", description="Marie is a person"),
                    Node(
                        id="Apple",
                        name="Apple",
                        type="Company",
                        description="Apple is a company",
                    ),
                    Node(
                        id="MacOS",
                        name="MacOS",
                        type="Product",
                        description="MacOS is Apple's operating system",
                    ),
                ],
                edges=[
                    Edge(
                        source_node_id="Marie",
                        target_node_id="Apple",
                        relationship_name="works_for",
                    ),
                    Edge(
                        source_node_id="Marie", target_node_id="MacOS", relationship_name="works_on"
                    ),
                ],
            )

        if "John" in text_input and response_model == KnowledgeGraph:
            return KnowledgeGraph(
                nodes=[
                    Node(id="John", name="John", type="Person", description="John is a person"),
                    Node(
                        id="Apple",
                        name="Apple",
                        type="Company",
                        description="Apple is a company",
                    ),
                    Node(
                        id="Food for Hungry",
                        name="Food for Hungry",
                        type="Non-profit organization",
                        description="Food for Hungry is a non-profit organization",
                    ),
                ],
                edges=[
                    Edge(
                        source_node_id="John", target_node_id="Apple", relationship_name="works_for"
                    ),
                    Edge(
                        source_node_id="John",
                        target_node_id="Food for Hungry",
                        relationship_name="works_for",
                    ),
                ],
            )

    mock_create_structured_output.side_effect = mock_llm_output

    vector_engine = get_vector_engine()

    assert not await vector_engine.has_collection("EdgeType_relationship_name")
    assert not await vector_engine.has_collection("Entity_name")
    assert not await vector_engine.has_collection("DocumentChunk_text")
    assert not await vector_engine.has_collection("TextSummary_text")
    assert not await vector_engine.has_collection("TextDocument_text")

    new_user = await create_user(
        email="example@user.com",
        password="mypassword",
        is_superuser=True,
        is_active=True,
        is_verified=True,
        auto_login=True,
    )

    await cognee.add(
        "John works for Apple. He is also affiliated with a non-profit organization called 'Food for Hungry'"
    )

    await cognee.add(
        "Marie works for Apple as well. She is a software engineer on MacOS project.",
        user=new_user,
    )

    cognify_result: dict = await cognee.cognify()
    johns_dataset_id = list(cognify_result.keys())[0]

    cognify_result: dict = await cognee.cognify(user=new_user)
    maries_dataset_id = list(cognify_result.keys())[0]

    graph_engine = await get_graph_engine()
    initial_nodes, initial_edges = await graph_engine.get_graph_data()
    assert len(initial_nodes) == 15 and len(initial_edges) == 19, (
        "Number of nodes and edges is not correct."
    )

    initial_nodes_by_vector_collection = {}

    for node in initial_nodes:
        node_data = node[1]
        node_metadata = json.loads(node_data["metadata"])
        collection_name = node_data["type"] + "_" + node_metadata["index_fields"][0]
        if collection_name not in initial_nodes_by_vector_collection:
            initial_nodes_by_vector_collection[collection_name] = []
        initial_nodes_by_vector_collection[collection_name].append(node)

    initial_node_ids = set([node[0] for node in initial_nodes])

    default_user = await get_default_user()
    await datasets.empty_dataset(johns_dataset_id, default_user)  # type: ignore

    nodes, edges = await graph_engine.get_graph_data()
    assert len(nodes) == 9 and len(edges) == 10, "Nodes and edges are not deleted."
    assert not any(
        node[1]["name"] == "john" or node[1]["name"] == "food for hungry"
        for node in nodes
        if "name" in node[1]
    ), "Nodes are not deleted."

    after_first_delete_node_ids = set([node[0] for node in nodes])

    after_delete_nodes_by_vector_collection = {}
    for node in initial_nodes:
        node_data = node[1]
        node_metadata = json.loads(node_data["metadata"])
        collection_name = node_data["type"] + "_" + node_metadata["index_fields"][0]
        if collection_name not in after_delete_nodes_by_vector_collection:
            after_delete_nodes_by_vector_collection[collection_name] = []
        after_delete_nodes_by_vector_collection[collection_name].append(node)

    vector_engine = get_vector_engine()

    removed_node_ids = initial_node_ids - after_first_delete_node_ids

    for collection_name, initial_nodes in initial_nodes_by_vector_collection.items():
        query_node_ids = [node[0] for node in initial_nodes if node[0] in removed_node_ids]

        if query_node_ids:
            vector_items = await vector_engine.retrieve(collection_name, query_node_ids)
            assert len(vector_items) == 0, "Vector items are not deleted."

    await datasets.empty_dataset(maries_dataset_id, new_user)  # type: ignore

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
