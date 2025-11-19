import os
import pytest
import pathlib
from uuid import NAMESPACE_OID, uuid5
from unittest.mock import AsyncMock, patch

import cognee
from cognee.api.v1.datasets import datasets
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.llm import LLMGateway
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.processing.document_types.TextDocument import TextDocument
from cognee.modules.engine.models import Entity
from cognee.modules.engine.operations.setup import setup
from cognee.modules.users.methods import get_default_user
from cognee.shared.data_models import KnowledgeGraph, Node, Edge, SummarizedContent
from cognee.shared.logging_utils import get_logger
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

    user = await get_default_user()

    await set_database_global_context_variables("main_dataset", user.id)

    vector_engine = get_vector_engine()

    assert not await vector_engine.has_collection("Entity_name")
    assert not await vector_engine.has_collection("DocumentChunk_text")
    assert not await vector_engine.has_collection("TextSummary_text")
    assert not await vector_engine.has_collection("TextDocument_text")
    assert not await vector_engine.has_collection("EdgeType_relationship_name")

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

    # Assert data points presence in the graph, vector collections and nodes table
    await assert_graph_nodes_present(johns_data + maries_data + overlapping_entities)
    await assert_nodes_vector_index_present(johns_data + maries_data + overlapping_entities)

    johns_relationships = extract_relationships(
        johns_chunk,
        mock_llm_output("John", "", KnowledgeGraph),  # type: ignore
    )
    maries_relationships = extract_relationships(
        maries_chunk,
        mock_llm_output("Marie", "", KnowledgeGraph),  # type: ignore
    )
    (overlapping_relationships, johns_relationships, maries_relationships) = (
        filter_overlapping_relationships(johns_relationships, maries_relationships)
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

    expected_relationships = johns_relationships + maries_relationships + overlapping_relationships

    await assert_graph_edges_present(expected_relationships)

    await assert_edges_vector_index_present(expected_relationships)

    # Delete John's data from cognee
    await datasets.delete_data(dataset_id, johns_data_id, user)  # type: ignore

    # Assert data points presence in the graph, vector collections and nodes table
    await assert_graph_nodes_present(maries_data + overlapping_entities)
    await assert_nodes_vector_index_present(maries_data + overlapping_entities)

    await assert_graph_nodes_not_present(johns_data)
    await assert_nodes_vector_index_not_present(johns_data)

    # Assert relationships presence in the graph, vector collections and nodes table
    await assert_graph_edges_present(maries_relationships + overlapping_relationships)
    await assert_edges_vector_index_present(maries_relationships)

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

    # Delete Marie's data from cognee
    await datasets.delete_data(dataset_id, maries_data_id, user)  # type: ignore

    await assert_graph_nodes_not_present(johns_data + maries_data + overlapping_entities)
    await assert_nodes_vector_index_not_present(johns_data + maries_data + overlapping_entities)

    # Assert relationships presence in the graph, vector collections and nodes table
    await assert_graph_edges_not_present(
        johns_relationships + maries_relationships + overlapping_relationships
    )

    await assert_edges_vector_index_not_present(maries_relationships)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
