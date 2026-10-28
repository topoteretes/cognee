"""A persisted user decision changes retrieval without an LLM or re-embedding sources."""

import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter
from cognee.infrastructure.databases.provenance import make_source_ref_key
from cognee.infrastructure.databases.provenance.markers import mark_graph_provenance_if_empty
from cognee.infrastructure.databases.unified.capabilities import EngineCapability
from cognee.infrastructure.databases.unified.unified_store_engine import UnifiedStoreEngine
from cognee.modules.engine.models import Entity, FactConflict
from cognee.modules.engine.utils.generate_edge_object_id import generate_edge_object_id
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.retrieval.utils.conflict_context import attach_chunk_conflicts
from cognee.modules.retrieval.utils.resolved_context import (
    apply_fact_hit_resolutions,
    read_user_resolutions,
)
from cognee.tasks.memify.review_conflicts import write_conflicts
from cognee.tasks.memify.review_conflicts.models import (
    AcceptedConflict,
    ConflictsToFinalize,
    ConflictsToWrite,
    ReviewScope,
)
from cognee.tests.integration.tasks.test_review_conflicts import RecordingVectors


@pytest.mark.asyncio
async def test_resolution_survives_real_graph_writes_and_overrides_old_vector_hit(
    tmp_path, monkeypatch
):
    graph = LadybugAdapter(str(tmp_path / "graph"), kuzu_buffer_pool_size=128 * 1024 * 1024)
    vector = RecordingVectors()
    unified = UnifiedStoreEngine(
        graph_engine=graph,
        vector_engine=vector,
        capabilities=EngineCapability.GRAPH | EngineCapability.VECTOR,
    )
    storage = importlib.import_module("cognee.tasks.storage.add_data_points")

    async def engine():
        return unified

    monkeypatch.setattr(storage, "get_unified_engine", engine)
    monkeypatch.setattr(write_conflicts, "get_unified_engine", engine)
    dataset = uuid4()
    ctx = PipelineContext(dataset=SimpleNamespace(id=dataset), data_item={})
    try:
        await mark_graph_provenance_if_empty(graph)
        company, alice, bob, chunk = [
            Entity(name=n, description=n) for n in ["Company", "Alice", "Bob", "Source"]
        ]
        await graph.add_nodes(
            [company, alice, bob, chunk], source_ref_key=make_source_ref_key(dataset, uuid4())
        )
        facts = {}
        for value in (alice, bob):
            fid = str(generate_edge_object_id(str(company.id), str(value.id), "has_ceo"))
            facts[fid] = {
                "id": fid,
                "source": str(company.id),
                "target": str(value.id),
                "relationship": "has_ceo",
                "properties": {"edge_object_id": fid, "edge_text": f"Company has CEO {value.name}"},
                "effective_date": None,
                "sources": [
                    {"chunk_id": str(chunk.id), "document": "Report", "effective_date": None}
                ],
            }
        await graph.add_edges(
            [(f["source"], f["target"], f["relationship"], f["properties"]) for f in facts.values()]
        )
        winner, loser = facts
        conflict = FactConflict(
            dataset_id=str(dataset),
            about_id=str(company.id),
            attribute="has_ceo",
            kind="fixed",
            status="resolved",
            text="Resolved: Company has CEO Alice",
            values=[str(alice.id), str(bob.id)],
            sources=[str(chunk.id)],
            resolution_source="user",
            selected_fact_id=winner,
            selected_fact="Company has CEO Alice",
            rejected_fact_ids=[loser],
            rejected_fact_texts=["Company has CEO Bob"],
            resolved_by=str(uuid4()),
            resolved_at="2026-10-28T12:00:00+00:00",
        )
        scope = ReviewScope(str(dataset), {}, facts)
        accepted = AcceptedConflict(conflict, {winner: "current", loser: "superseded"})
        await write_conflicts.write_conflicts(
            [ConflictsToWrite(scope, [accepted], shown_fact_ids=set(facts))], ctx
        )
        await write_conflicts.write_conflicts([ConflictsToFinalize([accepted])], ctx)
        resolutions = await read_user_resolutions(graph)
        assert len(resolutions) == 1 and resolutions[0]["selected_fact_id"] == winner
        assert not resolutions[0]["write_pending"]
        old_hit = SimpleNamespace(
            id=str(EdgeType.id_for("Company has CEO Bob")),
            payload={"relationship_name": "Company has CEO Bob"},
        )
        assert (
            apply_fact_hit_resolutions([old_hit], resolutions)[0].payload["text"]
            == "Company has CEO Alice"
        )
        passage = "Company has CEO Bob. Revenue grew 20 percent."
        chunks = [{"id": str(chunk.id), "text": passage}]
        await attach_chunk_conflicts(chunks, graph_engine=graph)
        assert chunks[0]["text"] == passage
        assert chunks[0]["resolved_facts"] == {str(conflict.id): "Company has CEO Alice"}
        assert "Accepted by user: Company has CEO Alice" in chunks[0]["conflicts"][0]
        # Both source facts still exist for audit/history.
        _, edges = await graph.get_neighborhood([str(company.id)], depth=1)
        assert len([e for e in edges if e[2] == "has_ceo"]) == 2
    finally:
        await graph.close()
