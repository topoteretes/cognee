"""End-to-end delete test (mocked LLM): two documents, incremental delete.

Scenario:
  1. Add two documents that share one entity ("Apple") and otherwise have
     their own exclusive entities/relationships.
  2. cognify() (LLM output mocked → deterministic graph). On a fresh graph this
     marks it graph-native, so delete routes through the unified provenance
     planner.
  3. Delete document 1. Assert that ONLY document-1-exclusive nodes/edges are
     removed, and that EVERY document-2 node/edge (its exclusive set AND the
     shared "Apple") is still present in both the graph and the vector store —
     i.e. nothing relevant is missing and nothing shared is over-deleted.
  4. Delete document 2. Assert the graph AND every vector collection are EMPTY
     — no orphaned nodes, edges, EdgeType nodes/vectors, or triplet rows.

Step 4's hard emptiness check is the regression guard for orphan leaks (the
shared-EdgeType-vector over-deletion class of bug).
"""

import os
import pathlib
from uuid import NAMESPACE_OID, uuid5
from unittest.mock import AsyncMock, patch

import cognee
from cognee.api.v1.datasets import datasets
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.llm import LLMGateway
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.processing.document_types.TextDocument import TextDocument
from cognee.modules.engine.operations.setup import setup
from cognee.modules.users.methods import get_default_user
from cognee.shared.data_models import KnowledgeGraph, Node, Edge, SummarizedContent
from cognee.shared.logging_utils import get_logger
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

logger = get_logger()


async def _all_vector_collection_row_counts(vector_engine) -> dict:
    """Return ``{collection_name: row_count}`` for every vector collection.

    Backend-agnostic so the emptiness guard runs on whatever ``.env`` selects:
    LanceDB exposes its collections via ``get_connection().table_names()``;
    pgvector has no such connection, but every collection is a table carrying a
    ``vector``-typed column, so they are discovered from ``information_schema`` on
    the adapter's SQL engine.
    """
    connection = await vector_engine.get_connection()
    if connection is not None and hasattr(connection, "table_names"):
        counts = {}
        for name in await connection.table_names():
            collection = await vector_engine.get_collection(name)
            counts[name] = await collection.count_rows()
        return counts

    from sqlalchemy import text

    counts = {}
    async with vector_engine.engine.begin() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT table_name FROM information_schema.columns "
                    "WHERE udt_name = 'vector' AND table_schema = 'public'"
                )
            )
        ).fetchall()
        for (table_name,) in rows:
            count = (await conn.execute(text(f'SELECT count(*) FROM "{table_name}"'))).scalar()
            counts[table_name] = count
    return counts


async def assert_store_is_empty():
    """Assert the graph (nodes + edges) and every vector collection are empty.

    Uses :Node-scoped graph queries (so the GraphMetadata marker is ignored but
    EdgeType nodes are counted) and per-collection row counts, so any orphaned
    artifact — node, edge, EdgeType node/vector, or triplet — is surfaced.
    """
    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()
    assert len(nodes) == 0, (
        f"Graph still has {len(nodes)} node(s) after deleting all documents: "
        f"{[n[0] for n in nodes][:10]}"
    )
    assert len(edges) == 0, (
        f"Graph still has {len(edges)} edge(s) after deleting all documents: "
        f"{[(e[0], e[2], e[1]) for e in edges][:10]}"
    )
    assert await graph_engine.is_empty(), "graph is_empty() is False after deleting all documents"

    vector_engine = get_vector_engine()
    for name, count in (await _all_vector_collection_row_counts(vector_engine)).items():
        assert count == 0, (
            f"Vector collection '{name}' still has {count} row(s) after deleting all documents"
        )


@patch.object(LLMGateway, "acreate_structured_output", new_callable=AsyncMock)
async def main(mock_create_structured_output: AsyncMock):
    data_directory_path = os.path.join(
        pathlib.Path(__file__).parent, ".data_storage/test_delete_two_documents_empty_store"
    )
    cognee.config.data_root_directory(data_directory_path)

    cognee_directory_path = os.path.join(
        pathlib.Path(__file__).parent, ".cognee_system/test_delete_two_documents_empty_store"
    )
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    # ------------------------------------------------------------------
    # Mock LLM: two documents sharing the "Apple" entity.
    #   doc1 (John):  John,  Apple(shared), "Food for Hungry"
    #   doc2 (Marie): Marie, Apple(shared), MacOS
    # ------------------------------------------------------------------
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

        if "John" in text_input and response_model == KnowledgeGraph:
            return KnowledgeGraph(
                nodes=[
                    Node(id="John", name="John", type="Person", description="John is a person"),
                    Node(
                        id="Apple", name="Apple", type="Company", description="Apple is a company"
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
        if "Marie" in text_input and response_model == KnowledgeGraph:
            return KnowledgeGraph(
                nodes=[
                    Node(id="Marie", name="Marie", type="Person", description="Marie is a person"),
                    Node(
                        id="Apple", name="Apple", type="Company", description="Apple is a company"
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

    mock_create_structured_output.side_effect = mock_llm_output

    user = await get_default_user()
    await set_database_global_context_variables("main_dataset", user.id)

    vector_engine = get_vector_engine()
    assert not await vector_engine.has_collection("Entity_name")
    assert not await vector_engine.has_collection("EdgeType_relationship_name")

    # ------------------------------------------------------------------
    # Add + cognify two documents.
    # ------------------------------------------------------------------
    doc1_text = "John works for Apple. He is also affiliated with a non-profit organization called 'Food for Hungry'"
    add_doc1 = await cognee.add(doc1_text)
    doc1_data_id = add_doc1.data_ingestion_info[0]["data_id"]

    doc2_text = "Marie works for Apple as well. She is a software engineer on MacOS project."
    add_doc2 = await cognee.add(doc2_text)
    doc2_data_id = add_doc2.data_ingestion_info[0]["data_id"]

    cognify_result: dict = await cognee.cognify()
    dataset_id = list(cognify_result.keys())[0]

    # Reconstruct the expected graph artifacts the same way the pipeline built them.
    doc1_document = TextDocument(
        id=doc1_data_id, name="Doc1", raw_data_location="doc1_location", external_metadata=""
    )
    doc1_chunk = DocumentChunk(
        id=uuid5(NAMESPACE_OID, f"{str(doc1_data_id)}-0"),
        text=doc1_text,
        chunk_size=14,
        chunk_index=0,
        cut_type="sentence_end",
        is_part_of=doc1_document,
    )
    doc1_summary = extract_summary(doc1_chunk, mock_llm_output("John", "", SummarizedContent))  # type: ignore

    doc2_document = TextDocument(
        id=doc2_data_id, name="Doc2", raw_data_location="doc2_location", external_metadata=""
    )
    doc2_chunk = DocumentChunk(
        id=uuid5(NAMESPACE_OID, f"{str(doc2_data_id)}-0"),
        text=doc2_text,
        chunk_size=14,
        chunk_index=0,
        cut_type="sentence_end",
        is_part_of=doc2_document,
    )
    doc2_summary = extract_summary(doc2_chunk, mock_llm_output("Marie", "", SummarizedContent))  # type: ignore

    doc1_entities = extract_entities(mock_llm_output("John", "", KnowledgeGraph))  # type: ignore
    doc2_entities = extract_entities(mock_llm_output("Marie", "", KnowledgeGraph))  # type: ignore
    (shared_entities, doc1_entities, doc2_entities) = filter_overlapping_entities(
        doc1_entities, doc2_entities
    )

    doc1_data = [doc1_document, doc1_chunk, doc1_summary, *doc1_entities]
    doc2_data = [doc2_document, doc2_chunk, doc2_summary, *doc2_entities]

    # Everything present after cognify.
    await assert_graph_nodes_present(doc1_data + doc2_data + shared_entities)
    await assert_nodes_vector_index_present(doc1_data + doc2_data + shared_entities)

    doc1_relationships = extract_relationships(
        doc1_chunk,
        mock_llm_output("John", "", KnowledgeGraph),  # type: ignore
    )
    doc2_relationships = extract_relationships(
        doc2_chunk,
        mock_llm_output("Marie", "", KnowledgeGraph),  # type: ignore
    )
    (shared_relationships, doc1_relationships, doc2_relationships) = (
        filter_overlapping_relationships(doc1_relationships, doc2_relationships)
    )
    doc1_relationships = [
        (doc1_chunk.id, doc1_document.id, "is_part_of"),
        (doc1_summary.id, doc1_chunk.id, "made_from"),
        *doc1_relationships,
    ]
    doc2_relationships = [
        (doc2_chunk.id, doc2_document.id, "is_part_of"),
        (doc2_summary.id, doc2_chunk.id, "made_from"),
        *doc2_relationships,
    ]
    await assert_graph_edges_present(doc1_relationships + doc2_relationships + shared_relationships)

    # ------------------------------------------------------------------
    # Delete document 1: only doc1-exclusive artifacts go; doc2 + shared stay.
    # ------------------------------------------------------------------
    await datasets.delete_data(dataset_id, doc1_data_id, user)  # type: ignore

    # Nothing relevant missing: every doc2 node/edge + the shared "Apple" survive.
    await assert_graph_nodes_present(doc2_data + shared_entities)
    await assert_nodes_vector_index_present(doc2_data + shared_entities)
    await assert_graph_edges_present(doc2_relationships + shared_relationships)
    await assert_edges_vector_index_present(doc2_relationships)

    # Only doc1-exclusive artifacts removed.
    await assert_graph_nodes_not_present(doc1_data)
    await assert_nodes_vector_index_not_present(doc1_data)
    await assert_graph_edges_not_present(doc1_relationships)

    # ------------------------------------------------------------------
    # Delete document 2: the graph and vector store must be completely empty.
    # ------------------------------------------------------------------
    await datasets.delete_data(dataset_id, doc2_data_id, user)  # type: ignore

    await assert_graph_nodes_not_present(doc1_data + doc2_data + shared_entities)
    await assert_graph_edges_not_present(
        doc1_relationships + doc2_relationships + shared_relationships
    )
    await assert_store_is_empty()

    logger.info("test_delete_two_documents_empty_store passed")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
