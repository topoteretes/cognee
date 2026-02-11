import os
import pytest
import pathlib
from uuid import NAMESPACE_OID, uuid5
from unittest.mock import AsyncMock, patch

import cognee
from cognee.api.v1.datasets import datasets
from cognee.context_global_variables import set_database_global_context_variables
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
from cognee.modules.engine.utils import generate_node_id
from cognee.modules.graph.legacy.record_data_in_legacy_ledger import record_data_in_legacy_ledger
from cognee.modules.graph.utils.deduplicate_nodes_and_edges import deduplicate_nodes_and_edges
from cognee.modules.graph.utils.get_graph_from_model import get_graph_from_model
from cognee.modules.pipelines.models import DataItemStatus
from cognee.modules.users.methods import get_default_user
from cognee.shared.data_models import KnowledgeGraph, Node, Edge, SummarizedContent
from cognee.tasks.storage import index_data_points, index_graph_edges
from cognee.tests.utils.assert_edges_vector_index_not_present import (
    assert_edges_vector_index_not_present,
)
from cognee.tests.utils.assert_edges_vector_index_present import assert_edges_vector_index_present
from cognee.tests.utils.assert_graph_edges_not_present import assert_graph_edges_not_present
from cognee.tests.utils.assert_graph_edges_present import assert_graph_edges_present
from cognee.tests.utils.assert_graph_nodes_not_present import assert_graph_nodes_not_present
from cognee.tests.utils.assert_graph_nodes_present import assert_graph_nodes_present
from cognee.tests.utils.assert_nodes_vector_index_not_present import (
    assert_nodes_vector_index_not_present,
)
from cognee.tests.utils.assert_nodes_vector_index_present import assert_nodes_vector_index_present
from cognee.tests.utils.extract_entities import extract_entities
from cognee.tests.utils.extract_relationships import extract_relationships
from cognee.tests.utils.extract_summary import extract_summary
from cognee.tests.utils.filter_overlapping_entities import filter_overlapping_entities
from cognee.tests.utils.filter_overlapping_relationships import filter_overlapping_relationships
from cognee.tests.utils.get_contains_edge_text import get_contains_edge_text


def create_legacy_data_points():
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
        is_a=graph_database,
    )
    neptune_database_entity = Entity(
        id=generate_node_id("amazon neptune database"),
        name="amazon neptune database",
        description="A popular managed graph database that complements Neptune Analytics.",
        is_a=graph_database,
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
        is_a=storage,
    )

    entities = [
        graph_database,
        neptune_analytics_entity,
        neptune_database_entity,
        storage,
        storage_entity,
    ]

    document_chunk.contains = entities

    data_points = [
        document,
        document_chunk,
    ]

    return data_points


@pytest.mark.asyncio
@patch.object(LLMGateway, "acreate_structured_output", new_callable=AsyncMock)
async def main(mock_create_structured_output: AsyncMock):
    # Disable backend access control for this legacy test
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

    user = await get_default_user()
    await set_database_global_context_variables("main_dataset", user.id)

    vector_engine = get_vector_engine()

    assert not await vector_engine.has_collection("EdgeType_relationship_name")
    assert not await vector_engine.has_collection("Entity_name")
    assert not await vector_engine.has_collection("DocumentChunk_text")
    assert not await vector_engine.has_collection("TextSummary_text")
    assert not await vector_engine.has_collection("TextDocument_text")

    # Add legacy data to the system
    legacy_document, legacy_data_points, legacy_relationships = await create_mocked_legacy_data(
        user
    )

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

    johns_text = "John works for Apple. He is also affiliated with a non-profit organization called 'Food for Hungry'"
    add_john_result = await cognee.add(johns_text)
    johns_data_id = add_john_result.data_ingestion_info[0]["data_id"]

    maries_text = "Marie works for Apple as well. She is a software engineer on MacOS project."
    add_marie_result = await cognee.add(maries_text)
    maries_data_id = add_marie_result.data_ingestion_info[0]["data_id"]

    cognify_result: dict = await cognee.cognify()
    dataset_id = list(cognify_result.keys())[0]

    johns_document = TextDocument(
        id=johns_data_id,
        name="John's Work",
        raw_data_location="johns_data_location",
        external_metadata="",
    )
    johns_chunk = DocumentChunk(
        id=uuid5(NAMESPACE_OID, f"{str(johns_data_id)}-0"),
        text=johns_text,
        chunk_size=14,
        chunk_index=0,
        cut_type="sentence_end",
        is_part_of=johns_document,
    )
    johns_summary = extract_summary(johns_chunk, mock_llm_output("John", "", SummarizedContent))  # type: ignore

    maries_document = TextDocument(
        id=maries_data_id,
        name="Maries's Work",
        raw_data_location="maries_data_location",
        external_metadata="",
    )
    maries_chunk = DocumentChunk(
        id=uuid5(NAMESPACE_OID, f"{str(maries_data_id)}-0"),
        text=maries_text,
        chunk_size=14,
        chunk_index=0,
        cut_type="sentence_end",
        is_part_of=maries_document,
    )
    maries_summary = extract_summary(maries_chunk, mock_llm_output("Marie", "", SummarizedContent))  # type: ignore

    johns_entities = extract_entities(mock_llm_output("John", "", KnowledgeGraph))  # type: ignore
    maries_entities = extract_entities(mock_llm_output("Marie", "", KnowledgeGraph))  # type: ignore
    (overlapping_entities, johns_entities, maries_entities) = filter_overlapping_entities(
        johns_entities, maries_entities
    )

    johns_data = [
        johns_document,
        johns_chunk,
        johns_summary,
        *johns_entities,
    ]
    maries_data = [
        maries_document,
        maries_chunk,
        maries_summary,
        *maries_entities,
    ]

    expected_data_points = johns_data + maries_data + overlapping_entities + legacy_data_points

    # Assert data points presence in the graph, vector collections and nodes table
    await assert_graph_nodes_present(expected_data_points)
    await assert_nodes_vector_index_present(expected_data_points)

    johns_relationships = extract_relationships(
        johns_chunk,
        mock_llm_output("John", "", KnowledgeGraph),  # type: ignore
    )
    maries_relationships = extract_relationships(
        maries_chunk,
        mock_llm_output("Marie", "", KnowledgeGraph),  # type: ignore
    )
    (overlapping_relationships, johns_relationships, maries_relationships, legacy_relationships) = (
        filter_overlapping_relationships(
            johns_relationships, maries_relationships, legacy_relationships
        )
    )

    johns_relationships = [
        (johns_chunk.id, johns_document.id, "is_part_of"),
        (johns_summary.id, johns_chunk.id, "made_from"),
        *johns_relationships,
    ]
    maries_relationships = [
        (maries_chunk.id, maries_document.id, "is_part_of"),
        (maries_summary.id, maries_chunk.id, "made_from"),
        *maries_relationships,
    ]

    expected_relationships = (
        johns_relationships
        + maries_relationships
        + overlapping_relationships
        + legacy_relationships
    )

    await assert_graph_edges_present(expected_relationships)

    await assert_edges_vector_index_present(expected_relationships)

    # Delete John's data
    await datasets.delete_data(dataset_id, johns_data_id, user)  # type: ignore

    # Assert data points presence in the graph, vector collections and nodes table
    await assert_graph_nodes_present(maries_data + overlapping_entities + legacy_data_points)
    await assert_nodes_vector_index_present(maries_data + overlapping_entities + legacy_data_points)

    await assert_graph_nodes_not_present(johns_data)
    await assert_nodes_vector_index_not_present(johns_data)

    # Assert relationships presence in the graph, vector collections and nodes table
    await assert_graph_edges_present(
        maries_relationships + overlapping_relationships + legacy_relationships
    )
    await assert_edges_vector_index_present(maries_relationships + legacy_relationships)

    await assert_graph_edges_not_present(johns_relationships)

    johns_contains_relationships = [
        (
            johns_chunk.id,
            entity.id,
            get_contains_edge_text(entity.name, entity.description),
            {
                "relationship_name": get_contains_edge_text(entity.name, entity.description),
            },
        )
        for entity in johns_entities
        if isinstance(entity, Entity)
    ]

    # We check only by relationship name and we need edges that are created by John's data and no other.
    await assert_edges_vector_index_not_present(johns_contains_relationships)

    # Delete legacy data
    await datasets.delete_data(dataset_id, legacy_document.id, user)  # type: ignore

    # Assert data points presence in the graph, vector collections and nodes table
    await assert_graph_nodes_present(maries_data + overlapping_entities)
    await assert_nodes_vector_index_present(maries_data + overlapping_entities)

    await assert_graph_nodes_not_present(johns_data + legacy_data_points)
    await assert_nodes_vector_index_not_present(johns_data + legacy_data_points)

    # Assert relationships presence in the graph, vector collections and nodes table
    await assert_graph_edges_present(maries_relationships + overlapping_relationships)
    await assert_edges_vector_index_present(maries_relationships)

    await assert_graph_edges_not_present(johns_relationships + legacy_relationships)

    # Vector index didn't change after deleting legacy data


async def create_mocked_legacy_data(user):
    graph_engine = await get_graph_engine()
    legacy_data_points = create_legacy_data_points()
    legacy_document = legacy_data_points[0]

    nodes = []
    edges = []

    added_nodes = {}
    added_edges = {}
    visited_properties = {}

    results = await asyncio.gather(
        *[
            get_graph_from_model(
                data_point,
                added_nodes=added_nodes,
                added_edges=added_edges,
                visited_properties=visited_properties,
            )
            for data_point in legacy_data_points
        ]
    )

    for result_nodes, result_edges in results:
        nodes.extend(result_nodes)
        edges.extend(result_edges)

    graph_nodes, graph_edges = deduplicate_nodes_and_edges(nodes, edges)

    await graph_engine.add_nodes(graph_nodes)
    await graph_engine.add_edges(graph_edges)

    nodes_by_id = {node.id: node for node in graph_nodes}

    def format_relationship_name(relationship):
        if relationship[2] == "contains":
            node = nodes_by_id[relationship[1]]
            return get_contains_edge_text(node.name, node.description)
        return relationship[2]

    await index_data_points(graph_nodes)
    await index_graph_edges(
        [
            (
                edge[0],
                edge[1],
                format_relationship_name(edge),
                {
                    **(edge[3] or {}),
                    "relationship_name": format_relationship_name(edge),
                },
            )
            for edge in graph_edges
        ]  # type: ignore
    )

    await record_data_in_legacy_ledger(graph_nodes, graph_edges)

    db_engine = get_relational_engine()

    dataset = await create_authorized_dataset("main_dataset", user)

    async with db_engine.get_async_session() as session:
        old_data = Data(
            id=legacy_document.id,
            name=legacy_document.name,
            extension="txt",
            raw_data_location=legacy_document.raw_data_location,
            external_metadata=legacy_document.external_metadata,
            mime_type=legacy_document.mime_type,
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

    return legacy_document, graph_nodes, graph_edges


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
