import os
import pathlib
import pytest
from unittest.mock import AsyncMock, patch

import cognee
from cognee.api.v1.datasets import datasets
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.llm import LLMGateway
from cognee.modules.data.methods import get_dataset_data
from cognee.modules.engine.operations.setup import setup
from cognee.modules.users.methods import get_default_user
from cognee.shared.data_models import KnowledgeGraph, Node, Edge, SummarizedContent
from cognee.shared.logging_utils import get_logger

logger = get_logger()


@pytest.mark.asyncio
@patch.object(LLMGateway, "acreate_structured_output", new_callable=AsyncMock)
async def main(mock_create_structured_output: AsyncMock):
    data_directory_path = os.path.join(
        pathlib.Path(__file__).parent, ".data_storage/test_delete_default_graph"
    )
    cognee.config.data_root_directory(data_directory_path)

    cognee_directory_path = os.path.join(
        pathlib.Path(__file__).parent, ".cognee_system/test_delete_default_graph"
    )
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    vector_engine = get_vector_engine()

    assert not await vector_engine.has_collection("EdgeType_relationship_name")
    assert not await vector_engine.has_collection("Entity_name")

    mock_create_structured_output.side_effect = [
        "",  # For LLM connection test
        KnowledgeGraph(
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
                Edge(source_node_id="John", target_node_id="Apple", relationship_name="works_for"),
                Edge(
                    source_node_id="John",
                    target_node_id="Food for Hungry",
                    relationship_name="works_for",
                ),
            ],
        ),
        KnowledgeGraph(
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
                Edge(source_node_id="Marie", target_node_id="MacOS", relationship_name="works_on"),
            ],
        ),
        SummarizedContent(summary="Summary of John's work.", description="Summary of John's work."),
        SummarizedContent(
            summary="Summary of Marie's work.", description="Summary of Marie's work."
        ),
    ]

    await cognee.add(
        "John works for Apple. He is also affiliated with a non-profit organization called 'Food for Hungry'"
    )

    await cognee.add("Marie works for Apple as well. She is a software engineer on MacOS project.")

    cognify_result: dict = await cognee.cognify()
    dataset_id = list(cognify_result.keys())[0]

    dataset_data = await get_dataset_data(dataset_id)
    added_data = dataset_data[0]

    # file_path = os.path.join(
    #     pathlib.Path(__file__).parent, ".artifacts", "graph_visualization_full.html"
    # )
    # await visualize_graph(file_path)

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()
    assert len(nodes) == 15 and len(edges) == 19, "Number of nodes and edges is not correct."

    user = await get_default_user()
    await datasets.delete_data(dataset_id, added_data.id, user)  # type: ignore

    # file_path = os.path.join(
    #     pathlib.Path(__file__).parent, ".artifacts", "graph_visualization_after_delete.html"
    # )
    # await visualize_graph(file_path)

    nodes, edges = await graph_engine.get_graph_data()
    assert len(nodes) == 9 and len(edges) == 10, "Nodes and edges are not deleted."


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
