"""End-to-end provenance lineage test on the default graph (mocked LLM).

Runs a real ``add`` + ``cognify`` with a mocked LLM (deterministic, per the
#3601 harness) over two documents that share an entity ("Apple"), then asserts
the provenance lineage layer added by ``cognee.tasks.storage.provenance_lineage``:

* every extracted node has a ``derived_from`` edge to its source Document;
* each Document has an ``in_dataset`` edge to a single shared ``DatasetNode``;
* the shared entity has ``derived_from`` edges to BOTH documents (many-to-many);
* deleting one document removes exactly that document's lineage while the shared
  entity (co-owned) and the other document's lineage survive;
* deleting the last document removes the remaining lineage and the DatasetNode.

Like the sibling ``test_delete_default_graph.py`` this is a standalone script
(``main`` is not collected by pytest) and runs in the e2e CI workflow, which
provides the embedding backend.
"""

import os
import pathlib
from uuid import NAMESPACE_OID, uuid5
from unittest.mock import AsyncMock, patch

import pytest

import cognee
from cognee.api.v1.datasets import datasets
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.llm import LLMGateway
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.processing.document_types.TextDocument import TextDocument
from cognee.modules.engine.models import Entity
from cognee.modules.engine.models.DatasetNode import DatasetNode
from cognee.modules.engine.operations.setup import setup
from cognee.modules.users.methods import get_default_user
from cognee.shared.data_models import KnowledgeGraph, Node, Edge, SummarizedContent
from cognee.shared.logging_utils import get_logger
from cognee.tasks.storage.provenance_lineage import (
    DERIVED_FROM_RELATIONSHIP,
    IN_DATASET_RELATIONSHIP,
    dataset_lineage_node_id,
)
from cognee.tests.utils.assert_graph_edges_not_present import assert_graph_edges_not_present
from cognee.tests.utils.assert_graph_edges_present import assert_graph_edges_present
from cognee.tests.utils.assert_graph_nodes_not_present import assert_graph_nodes_not_present
from cognee.tests.utils.assert_graph_nodes_present import assert_graph_nodes_present
from cognee.tests.utils.extract_entities import extract_entities
from cognee.tests.utils.extract_summary import extract_summary
from cognee.tests.utils.filter_overlapping_entities import filter_overlapping_entities

logger = get_logger()


def _derived_from(nodes, document_id):
    """Provenance edges: each content node -derived_from-> its Document."""
    return [(node.id, document_id, DERIVED_FROM_RELATIONSHIP) for node in nodes]


@pytest.mark.asyncio
@patch.object(LLMGateway, "acreate_structured_output", new_callable=AsyncMock)
async def main(mock_create_structured_output: AsyncMock):
    data_directory_path = os.path.join(
        pathlib.Path(__file__).parent, ".data_storage/test_provenance_lineage_default_graph"
    )
    cognee.config.data_root_directory(data_directory_path)

    cognee_directory_path = os.path.join(
        pathlib.Path(__file__).parent, ".cognee_system/test_provenance_lineage_default_graph"
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

    mock_create_structured_output.side_effect = mock_llm_output

    user = await get_default_user()
    await set_database_global_context_variables("main_dataset", user.id)

    johns_text = "John works for Apple. He is also affiliated with a non-profit organization called 'Food for Hungry'"
    johns_data_id = (await cognee.add(johns_text)).data_ingestion_info[0]["data_id"]

    maries_text = "Marie works for Apple as well. She is a software engineer on MacOS project."
    maries_data_id = (await cognee.add(maries_text)).data_ingestion_info[0]["data_id"]

    cognify_result: dict = await cognee.cognify()
    dataset_id = list(cognify_result.keys())[0]
    dataset_node_id = dataset_lineage_node_id(dataset_id)

    # Reconstruct the graph objects deterministically (same ids the pipeline used).
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
        name="Marie's Work",
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
    overlapping_entities, johns_entities, maries_entities = filter_overlapping_entities(
        johns_entities, maries_entities
    )
    # "Apple" is extracted from both documents, so it is a shared/merged node.
    shared_entities = [e for e in overlapping_entities if isinstance(e, Entity)]

    johns_nodes = [johns_chunk, johns_summary, *johns_entities]
    maries_nodes = [maries_chunk, maries_summary, *maries_entities]

    # ── After cognify: full lineage present ──────────────────────────────────

    # One shared DatasetNode, reachable from both documents.
    await assert_graph_nodes_present([DatasetNode(id=dataset_node_id, name="main_dataset")])
    await assert_graph_edges_present(
        [
            (johns_document.id, dataset_node_id, IN_DATASET_RELATIONSHIP),
            (maries_document.id, dataset_node_id, IN_DATASET_RELATIONSHIP),
        ]
    )

    # Every extracted node traces to its source Document.
    await assert_graph_edges_present(_derived_from(johns_nodes, johns_document.id))
    await assert_graph_edges_present(_derived_from(maries_nodes, maries_document.id))

    # Many-to-many: the shared entity derives from BOTH documents.
    await assert_graph_edges_present(_derived_from(shared_entities, johns_document.id))
    await assert_graph_edges_present(_derived_from(shared_entities, maries_document.id))

    # ── Delete John's data: John's lineage gone, shared + Marie survive ───────

    await datasets.delete_data(dataset_id, johns_data_id, user)  # type: ignore

    await assert_graph_edges_not_present(_derived_from(johns_nodes, johns_document.id))
    await assert_graph_edges_not_present(
        [(johns_document.id, dataset_node_id, IN_DATASET_RELATIONSHIP)]
    )
    # The shared entity loses its John anchor but keeps its Marie anchor.
    await assert_graph_edges_not_present(_derived_from(shared_entities, johns_document.id))
    await assert_graph_nodes_present(shared_entities)
    await assert_graph_edges_present(_derived_from(shared_entities, maries_document.id))

    # Marie's lineage and the shared DatasetNode remain intact.
    await assert_graph_edges_present(_derived_from(maries_nodes, maries_document.id))
    await assert_graph_edges_present(
        [(maries_document.id, dataset_node_id, IN_DATASET_RELATIONSHIP)]
    )
    await assert_graph_nodes_present([DatasetNode(id=dataset_node_id, name="main_dataset")])

    # ── Delete Marie's data: everything provenance-related gone ───────────────

    await datasets.delete_data(dataset_id, maries_data_id, user)  # type: ignore

    await assert_graph_edges_not_present(_derived_from(maries_nodes, maries_document.id))
    await assert_graph_edges_not_present(
        [(maries_document.id, dataset_node_id, IN_DATASET_RELATIONSHIP)]
    )
    await assert_graph_nodes_not_present([DatasetNode(id=dataset_node_id, name="main_dataset")])


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
