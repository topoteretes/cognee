"""Task 1 of the ``from_entities`` global context index: raw graph rows to ``DatasetContent``.

The graph engine and the dataset's data records are faked; rows are given as
every adapter returns them.
"""

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.modules.pipelines.models import PipelineContext
from cognee.tasks.memify.global_context_index.models import DatasetEntity

load_dataset_content_module = import_module(
    "cognee.tasks.memify.global_context_index.load_dataset_content"
)


def node(node_id, **attributes):
    """A raw adapter row: ``(id, properties)``, every property the node carries."""
    return (node_id, attributes)


def edge(source, target, relationship_name, **attributes):
    return (source, target, relationship_name, attributes)


async def load(monkeypatch, nodes, edges, *, dataset_id, document_ids):
    """Run the task for a dataset whose data records are ``document_ids``."""
    graph_engine = SimpleNamespace(get_filtered_graph_data=AsyncMock(return_value=(nodes, edges)))
    monkeypatch.setattr(
        load_dataset_content_module, "get_graph_engine", AsyncMock(return_value=graph_engine)
    )
    monkeypatch.setattr(
        load_dataset_content_module,
        "get_dataset_data",
        AsyncMock(return_value=[SimpleNamespace(id=document_id) for document_id in document_ids]),
    )
    ctx = PipelineContext(dataset=SimpleNamespace(id=dataset_id))
    return await load_dataset_content_module.load_dataset_content(None, ctx)


@pytest.mark.asyncio
async def test_graph_rows_are_reshaped_into_content_and_skip_unusable_nodes(monkeypatch):
    dataset_id = str(uuid4())
    nodes = [
        node("c1", type="DocumentChunk", document_id="d1", document_name="alpha.md"),
        node("c2", type="DocumentChunk"),  # no document, so it counts nowhere
        node("e_acme", type="Entity", name="  Acme  "),
        node("e_bob", type="Entity", name="Bob"),
        node("e_blank", type="Entity", name="   "),
        node("e_unmentioned", type="Entity", name="Only in an ontology"),
        node("t_org", type="EntityType", name="Organization"),
        node("mine", type="GlobalContextSummary", dataset_id=dataset_id, text="kept"),
        node("theirs", type="GlobalContextSummary", dataset_id=str(uuid4()), text="dropped"),
        node("legacy", type="GlobalContextSummary", dataset_id=None, text="ignored"),
        node("hidden", type="Entity", name="Internal", is_internal=True),
    ]
    edges = [
        edge("c1", "e_acme", "contains"),
        edge("c1", "e_bob", "contains"),
        edge("c2", "e_acme", "contains"),
        edge("e_acme", "t_org", "is_a"),
        edge("e_bob", "e_acme", "works_at", edge_text="  Bob works at   Acme "),
        edge("e_acme", "e_blank", "related_to", edge_text="Acme relates to nothing"),
    ]

    content = await load(monkeypatch, nodes, edges, dataset_id=dataset_id, document_ids={"d1"})

    assert content.dataset_id == dataset_id
    assert content.document_by_chunk == {"c1": "d1"}
    assert content.document_names == {"d1": "alpha.md"}
    # The chunk with no document counts nowhere, an entity no chunk mentions is not
    # the dataset's, and a relation to the unnamed endpoint is not kept either.
    assert content.entities == {
        "e_acme": DatasetEntity(name="Acme", type_name="Organization", chunk_ids=frozenset({"c1"})),
        "e_bob": DatasetEntity(name="Bob", type_name="Entity", chunk_ids=frozenset({"c1"})),
    }
    assert content.relations == {("e_bob", "e_acme"): ("Bob works at Acme",)}
    assert content.existing_summary_texts == {"mine": "kept"}


@pytest.mark.asyncio
async def test_chunks_of_another_datasets_documents_are_left_out_with_their_entities(monkeypatch):
    # Datasets sharing one database (backend access control off) share its chunks.
    nodes = [
        node("c1", type="DocumentChunk", document_id="d1", document_name="ours.md"),
        node("c2", type="DocumentChunk", document_id="d2", document_name="theirs.md"),
        node("e_shared", type="Entity", name="Shared"),
        node("e_theirs", type="Entity", name="Theirs"),
    ]
    edges = [
        edge("c1", "e_shared", "contains"),
        edge("c2", "e_shared", "contains"),
        edge("c2", "e_theirs", "contains"),
        edge("e_shared", "e_theirs", "related_to", edge_text="Shared meets Theirs"),
    ]

    content = await load(monkeypatch, nodes, edges, dataset_id=str(uuid4()), document_ids={"d1"})

    assert content.document_by_chunk == {"c1": "d1"}
    assert content.entities == {
        "e_shared": DatasetEntity(name="Shared", type_name="Entity", chunk_ids=frozenset({"c1"}))
    }
    # Their entity is not ours, so neither is the relation to it.
    assert content.relations == {}


@pytest.mark.asyncio
async def test_entity_without_a_type_edge_falls_back_to_the_default_type(monkeypatch):
    nodes = [
        node("c1", type="DocumentChunk", document_id="d1"),
        node("e_acme", type="Entity", name="Acme"),
    ]
    edges = [edge("c1", "e_acme", "contains")]

    content = await load(monkeypatch, nodes, edges, dataset_id=str(uuid4()), document_ids={"d1"})

    assert content.entities["e_acme"].type_name == "Entity"
