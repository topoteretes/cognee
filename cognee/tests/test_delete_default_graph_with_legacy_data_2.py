import os
import pathlib
from uuid import NAMESPACE_OID, uuid5
import pytest
from unittest.mock import AsyncMock, patch

import cognee
from cognee.api.v1.datasets import datasets
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.llm import LLMGateway
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.methods import create_authorized_dataset
from cognee.modules.data.models import Data
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.data.processing.document_types import TextDocument
from cognee.modules.engine.operations.setup import setup
from cognee.modules.engine.utils import generate_edge_id, generate_node_id
from cognee.modules.graph.legacy.record_data_in_legacy_ledger import record_data_in_legacy_ledger
from cognee.modules.pipelines.models import DataItemStatus
from cognee.modules.users.methods import get_default_user
from cognee.shared.data_models import KnowledgeGraph, Node, Edge, SummarizedContent
from cognee.tasks.storage import index_data_points, index_graph_edges


def get_nodes_and_edges():
    document = TextDocument(
        id=uuid5(NAMESPACE_OID, "text_test.txt"),
        name="text_test.txt",
        raw_data_location="git/cognee/examples/database_examples/data_storage/data/text_test.txt",
        external_metadata="{}",
        mime_type="text/plain",
    )
    document_chunk = DocumentChunk(
        id=uuid5(
            NAMESPACE_OID,
            "Neptune Analytics is an ideal choice for investigatory, exploratory, or data-science workloads \n    that require fast iteration for data, analytical and algorithmic processing, or vector search on graph data. It \n    complements Amazon Neptune Database, a popular managed graph database. To perform intensive analysis, you can load \n    the data from a Neptune Database graph or snapshot into Neptune Analytics. You can also load graph data that's \n    stored in Amazon S3.\n    ",
        ),
        text="Neptune Analytics is an ideal choice for investigatory, exploratory, or data-science workloads \n    that require fast iteration for data, analytical and algorithmic processing, or vector search on graph data. It \n    complements Amazon Neptune Database, a popular managed graph database. To perform intensive analysis, you can load \n    the data from a Neptune Database graph or snapshot into Neptune Analytics. You can also load graph data that's \n    stored in Amazon S3.\n    ",
        chunk_size=187,
        chunk_index=0,
        cut_type="paragraph_end",
        is_part_of=document,
    )

    graph_database = EntityType(
        id=uuid5(NAMESPACE_OID, "graph_database"),
        name="graph database",
        description="graph database",
    )
    neptune_analytics_entity = Entity(
        id=generate_node_id("neptune analytics"),
        name="neptune analytics",
        description="A memory-optimized graph database engine for analytics that processes large amounts of graph data quickly.",
    )
    neptune_database_entity = Entity(
        id=generate_node_id("amazon neptune database"),
        name="amazon neptune database",
        description="A popular managed graph database that complements Neptune Analytics.",
    )

    storage = EntityType(
        id=generate_node_id("storage"),
        name="storage",
        description="storage",
    )
    storage_entity = Entity(
        id=generate_node_id("amazon s3"),
        name="amazon s3",
        description="A storage service provided by Amazon Web Services that allows storing graph data.",
    )

    nodes_data = [
        document,
        document_chunk,
        graph_database,
        neptune_analytics_entity,
        neptune_database_entity,
        storage,
        storage_entity,
    ]

    edges_data = [
        (
            document_chunk.id,
            storage_entity.id,
            "contains",
            {
                "relationship_name": "contains",
            },
        ),
        (
            storage_entity.id,
            storage.id,
            "is_a",
            {
                "relationship_name": "is_a",
            },
        ),
        (
            document_chunk.id,
            neptune_database_entity.id,
            "contains",
            {
                "relationship_name": "contains",
            },
        ),
        (
            neptune_database_entity.id,
            graph_database.id,
            "is_a",
            {
                "relationship_name": "is_a",
            },
        ),
        (
            document_chunk.id,
            document.id,
            "is_part_of",
            {
                "relationship_name": "is_part_of",
            },
        ),
        (
            document_chunk.id,
            neptune_analytics_entity.id,
            "contains",
            {
                "relationship_name": "contains",
            },
        ),
        (
            neptune_analytics_entity.id,
            graph_database.id,
            "is_a",
            {
                "relationship_name": "is_a",
            },
        ),
    ]

    return nodes_data, edges_data


@pytest.mark.asyncio
@patch.object(LLMGateway, "acreate_structured_output", new_callable=AsyncMock)
async def main(mock_create_structured_output: AsyncMock):
    os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "False"

    data_directory_path = os.path.join(
        pathlib.Path(__file__).parent, ".data_storage/test_delete_default_graph_with_legacy_graph_2"
    )
    cognee.config.data_root_directory(data_directory_path)

    cognee_directory_path = os.path.join(
        pathlib.Path(__file__).parent,
        ".cognee_system/test_delete_default_graph_with_legacy_graph_2",
    )
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    vector_engine = get_vector_engine()

    assert not await vector_engine.has_collection("EdgeType_relationship_name")
    assert not await vector_engine.has_collection("Entity_name")
    assert not await vector_engine.has_collection("DocumentChunk_text")
    assert not await vector_engine.has_collection("TextSummary_text")
    assert not await vector_engine.has_collection("TextDocument_text")

    user = await get_default_user()

    old_document, old_nodes, old_edges = await add_mocked_legacy_data(user)

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

    add_john_result = await cognee.add(
        "John works for Apple. He is also affiliated with a non-profit organization called 'Food for Hungry'"
    )
    johns_data_id = add_john_result.data_ingestion_info[0]["data_id"]

    await cognee.add("Marie works for Apple as well. She is a software engineer on MacOS project.")

    cognify_result: dict = await cognee.cognify()
    dataset_id = list(cognify_result.keys())[0]

    graph_engine = await get_graph_engine()
    initial_nodes, initial_edges = await graph_engine.get_graph_data()
    assert len(initial_nodes) == 22 and len(initial_edges) == 26, (
        "Number of nodes and edges is not correct."
    )

    initial_nodes_by_vector_collection = {}

    for node in initial_nodes:
        node_data = node[1]
        collection_name = node_data["type"] + "_" + node_data["metadata"]["index_fields"][0]
        if collection_name not in initial_nodes_by_vector_collection:
            initial_nodes_by_vector_collection[collection_name] = []
        initial_nodes_by_vector_collection[collection_name].append(node)

    initial_node_ids = set([node[0] for node in initial_nodes])

    await datasets.delete_data(dataset_id, johns_data_id, user)  # type: ignore

    nodes, edges = await graph_engine.get_graph_data()
    assert len(nodes) == 16 and len(edges) == 17, "Nodes and edges are not deleted."
    assert not any(
        node[1]["name"] == "john" or node[1]["name"] == "food for hungry" for node in nodes
    ), "Nodes are not deleted."

    after_first_delete_node_ids = set([node[0] for node in nodes])

    after_delete_nodes_by_vector_collection = {}
    for node in initial_nodes:
        node_data = node[1]
        collection_name = node_data["type"] + "_" + node_data["metadata"]["index_fields"][0]
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

    # Delete old document
    await datasets.delete_data(dataset_id, old_document.id, user)  # type: ignore

    final_nodes, final_edges = await graph_engine.get_graph_data()
    assert len(final_nodes) == 9 and len(final_edges) == 10, "Nodes and edges are not deleted."

    old_nodes_by_vector_collection = {}
    for node in old_nodes:
        collection_name = node.type + "_" + node.metadata["index_fields"][0]
        if collection_name not in old_nodes_by_vector_collection:
            old_nodes_by_vector_collection[collection_name] = []
        old_nodes_by_vector_collection[collection_name].append(node)

    for collection_name, old_nodes in old_nodes_by_vector_collection.items():
        query_node_ids = [str(node.id) for node in old_nodes]

        if query_node_ids:
            vector_items = await vector_engine.retrieve(collection_name, query_node_ids)
            assert len(vector_items) == 0, "Vector items are not deleted."

    query_edge_ids = list(set([str(generate_edge_id(edge[2])) for edge in old_edges]))

    vector_items = await vector_engine.retrieve("EdgeType_relationship_name", query_edge_ids)
    assert len(vector_items) == len(query_edge_ids), "Vector items are not deleted."


async def add_mocked_legacy_data(user):
    graph_engine = await get_graph_engine()
    old_nodes, old_edges = get_nodes_and_edges()
    old_document = old_nodes[0]

    await graph_engine.add_nodes(old_nodes)
    await graph_engine.add_edges(old_edges)

    await index_data_points(old_nodes)
    await index_graph_edges(old_edges)

    await record_data_in_legacy_ledger(old_nodes, old_edges, user)

    db_engine = get_relational_engine()

    dataset = await create_authorized_dataset("main_dataset", user)

    async with db_engine.get_async_session() as session:
        old_data = Data(
            id=old_document.id,
            name=old_document.name,
            extension="txt",
            raw_data_location=old_document.raw_data_location,
            external_metadata=old_document.external_metadata,
            mime_type=old_document.mime_type,
            owner_id=user.id,
            pipeline_status={
                "cognify_pipeline": {
                    str(dataset.id): DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED,
                }
            },
        )
        session.add(old_data)

        dataset.data.append(old_data)
        session.add(dataset)

        await session.commit()

    return old_document, old_nodes, old_edges


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
