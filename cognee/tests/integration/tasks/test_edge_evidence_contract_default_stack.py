"""Edge-evidence lifecycle contract on the real default stack.

Runs add -> cognify -> lookup -> delete -> forget(memory_only) -> re-cognify
against real Ladybug + LanceDB + SQLite. The LLM is mocked for a deterministic
graph and embeddings are stubbed (nothing here retrieves by similarity), so the
test is offline. It proves the contract the unit tests can only fake:

1. cognify writes one evidence row per (chunk, edge) it stored, tagged with the
   real dataset, data item, and pipeline run;
2. the lookup resolves the graph's own edge ids back to the source document;
3. deleting one document removes its rows and leaves its neighbours' rows;
4. ``forget(memory_only=True)`` sweeps the dataset's rows, and re-cognifying
   the preserved files captures them again.

Other graph backends follow the same shape as ``test_belongs_to_set_neo4j.py``:
the evidence path is backend-agnostic (relational sidecar + graph edge ids), so
a backend variant only needs the graph engine swapped.
"""

from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest
from sqlalchemy import select

import cognee
from cognee.api.v1.datasets import datasets
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.locks import dataset_lock
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)
from cognee.infrastructure.llm import LLMGateway
from cognee.modules.data.methods import create_authorized_dataset
from cognee.modules.engine.operations.setup import setup as setup_cognee
from cognee.modules.engine.utils import generate_edge_object_id
from cognee.modules.provenance.edge_evidence.lookup import get_edge_evidence_records
from cognee.modules.provenance.edge_evidence.models import ProvenanceEdgeEvidence
from cognee.modules.users.methods import get_default_user
from cognee.shared.data_models import Edge, KnowledgeGraph, Node, SummarizedContent

DOC1 = "Alice knows Bob."
DOC2 = "Alice lives in New York. She is from Berlin."


def _kg(nodes, edges):
    return KnowledgeGraph(
        summary="s",
        description="s",
        nodes=[Node(id=n, name=n, type=t, description=f"{n} is a {t}", label=n) for n, t in nodes],
        edges=[Edge(source_node_id=s, target_node_id=d, relationship_name=r) for s, d, r in edges],
    )


def _mock_llm_output(text_input, system_prompt, response_model):
    if text_input == "test":
        return "test"
    if response_model == SummarizedContent:
        return SummarizedContent(summary="s", description="s")
    if response_model == KnowledgeGraph:
        if "Berlin" in text_input:
            return _kg(
                [("Alice", "Person"), ("New York", "City"), ("Berlin", "City")],
                [("Alice", "New York", "lives_in"), ("Alice", "Berlin", "from")],
            )
        if "knows" in text_input:
            return _kg([("Alice", "Person"), ("Bob", "Person")], [("Alice", "Bob", "knows")])
    return _kg([], [])


async def _fake_embed(self, text):
    return [[0.0] * (self.dimensions or 8) for _ in text]


async def _setup(tmp_path):
    cognee.config.data_root_directory(str(tmp_path / "data"))
    cognee.config.system_root_directory(str(tmp_path / "system"))
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup_cognee()
    return await get_default_user()


async def _ingest():
    r1 = await cognee.add(DOC1)
    r2 = await cognee.add(DOC2)
    d1 = UUID(str(r1.data_ingestion_info[0]["data_id"]))
    d2 = UUID(str(r2.data_ingestion_info[0]["data_id"]))
    cognify_result = await cognee.cognify()
    dataset_id = UUID(str(list(cognify_result.keys())[0]))
    return dataset_id, d1, d2


async def _evidence_rows():
    async with get_relational_engine().get_async_session() as session:
        return (await session.execute(select(ProvenanceEdgeEvidence))).scalars().all()


async def _graph_edge_ids_by_relationship():
    graph = await get_graph_engine()
    _nodes, edges = await graph.get_graph_data()
    return {
        relationship: UUID(
            generate_edge_object_id(UUID(str(source)), UUID(str(target)), relationship)
        )
        for source, target, relationship, *_rest in edges
    }


@pytest.mark.asyncio
@patch.object(LiteLLMEmbeddingEngine, "embed_text", _fake_embed)
@patch.object(LLMGateway, "acreate_structured_output", new_callable=AsyncMock)
async def test_edge_evidence_follows_the_graph_through_its_lifecycle(mock_struct, tmp_path):
    mock_struct.side_effect = _mock_llm_output
    user = await _setup(tmp_path)
    authorized_dataset = await create_authorized_dataset("main_dataset", user)

    # Canonical lock order (SDK-483): hold the dataset lock before the legacy
    # context call below acquires its queue slot; nested add/cognify/delete
    # re-enter via held_datasets instead of re-acquiring the lock. The scope
    # releases the lock at test end so later tests in this pytest process can
    # operate on the same dataset id.
    async with dataset_lock(authorized_dataset.id):
        await set_database_global_context_variables(authorized_dataset.id, user.id)
        dataset_id, d1, d2 = await _ingest()

        # 1. Capture: every row is scoped to the real dataset/data item/run.
        rows = await _evidence_rows()
        assert rows, "cognify must persist edge evidence on the default stack"
        assert {row.dataset_id for row in rows} == {dataset_id}
        assert {row.data_id for row in rows} <= {d1, d2}
        assert all(row.pipeline_run_id is not None for row in rows)
        extracted = {row.relationship_name for row in rows if row.evidence_kind == "extracted"}
        assert {"knows", "lives_in", "from"} <= extracted

        # 2. Lookup: the graph's own edge ids resolve back to the source document.
        edge_ids = await _graph_edge_ids_by_relationship()
        records = await get_edge_evidence_records([edge_ids["knows"], edge_ids["from"]], dataset_id)
        by_edge = {record.edge_id: record for record in records}
        assert by_edge[edge_ids["knows"]].data_id == d1
        assert by_edge[edge_ids["from"]].data_id == d2
        assert all(record.document_name for record in records)

        # 3. Deleting one document sweeps its rows only.
        await datasets.delete_data(dataset_id, d2, user)
        remaining = await _evidence_rows()
        assert {row.data_id for row in remaining} == {d1}
        assert await get_edge_evidence_records([edge_ids["from"]], dataset_id) == []
        assert await get_edge_evidence_records([edge_ids["knows"]], dataset_id)

        # 4. Dropping memory sweeps the dataset; re-cognify recaptures under a new run.
        await cognee.forget(dataset_id=dataset_id, memory_only=True)
        assert await _evidence_rows() == []

        await cognee.cognify()
        recaptured = await _evidence_rows()
        assert recaptured
        assert {row.data_id for row in recaptured} == {d1}
        assert "knows" in {row.relationship_name for row in recaptured}
