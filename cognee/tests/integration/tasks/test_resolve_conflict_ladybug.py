"""A persisted user decision changes retrieval without an LLM or re-embedding sources."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter
from cognee.infrastructure.databases.provenance import make_source_ref_key
from cognee.infrastructure.databases.provenance.markers import mark_graph_provenance_if_empty
from cognee.infrastructure.databases.unified.capabilities import EngineCapability
from cognee.infrastructure.databases.unified.unified_store_engine import UnifiedStoreEngine
from cognee.modules.engine.models import Entity, FactConflict
from cognee.modules.engine.utils.generate_edge_object_id import generate_edge_object_id
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.graph.utils.fact_conflicts import is_rejected_fact
from cognee.modules.graph.utils.resolve_edges_to_text import resolve_edges_to_text
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.retrieval.utils.conflict_context import attach_chunk_conflicts
from cognee.modules.retrieval.utils.resolved_context import (
    apply_fact_hit_resolutions,
    apply_graph_resolutions,
    read_user_resolutions,
)
from cognee.tasks.memify.review_conflicts import read_facts, write_conflicts
from cognee.tasks.memify.review_conflicts.models import (
    AcceptedConflict,
    ConflictsToFinalize,
    ConflictsToWrite,
    ReviewScope,
)
from cognee.tasks.memify.review_conflicts.schema import ReviewResponse
from cognee.tests.integration.tasks.test_review_conflicts import RecordingVectors


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "forget_selected, review_drops",
    # After the forget, the review either drops the decision itself or leaves it out of
    # its response; either way the decision retires.
    [(False, False), (True, True), (True, False)],
)
async def test_resolution_survives_real_graph_writes_and_overrides_old_vector_hit(
    tmp_path, monkeypatch, forget_selected, review_drops
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
        facts = {}
        documents = {}
        source_chunks = []
        for value in (alice, bob):
            source_chunk = (
                chunk if value is alice else Entity(name="Other source", description="Other source")
            )
            source_chunks.append(str(source_chunk.id))
            data_id = uuid4()
            owner = make_source_ref_key(dataset, data_id)
            await graph.add_nodes([company, value, source_chunk], source_ref_key=owner)
            fid = str(generate_edge_object_id(str(company.id), str(value.id), "has_ceo"))
            facts[fid] = {
                "id": fid,
                "source": str(company.id),
                "target": str(value.id),
                "relationship": "has_ceo",
                "properties": {"edge_object_id": fid, "edge_text": f"Company has CEO {value.name}"},
                "effective_date": None,
                "sources": [
                    {"chunk_id": str(source_chunk.id), "document": "Report", "effective_date": None}
                ],
            }
            f = facts[fid]
            documents[fid] = data_id
            await graph.add_edges(
                [(f["source"], f["target"], f["relationship"], f["properties"])],
                source_ref_key=owner,
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
            sources=source_chunks,
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
        if forget_selected:
            # Use the same provenance deletion as forget(data_id=...), retaining the
            # shared subject and the other document's fact.
            await unified.delete_by_document(str(dataset), str(documents[winner]))
            monkeypatch.setattr(read_facts, "get_graph_engine", AsyncMock(return_value=graph))
            monkeypatch.setattr(read_facts, "get_edge_sources", AsyncMock(return_value=[]))
            monkeypatch.setattr(read_facts, "backend_access_control_enabled", lambda: True)
            scope = await read_facts.read_entity_facts(None, entity_ids=[str(company.id)], ctx=ctx)
            assert winner not in scope.facts and loser in scope.facts
            review = importlib.import_module("cognee.tasks.memify.review_conflicts.review_entities")
            monkeypatch.setattr(
                review,
                "get_llm_tokenizer",
                lambda: SimpleNamespace(count_tokens=lambda text: len(text.split())),
            )
            monkeypatch.setattr(review, "get_llm_token_ceiling", lambda: 16384)

            async def review_subject(prompt, system_prompt):
                labels = [
                    label
                    for label, node_id in prompt.node_labels.items()
                    if node_id in prompt.shown_facts
                ]
                drop = list(prompt.conflict_labels) if review_drops else []
                return ReviewResponse(reviewed=labels, drop=drop)

            monkeypatch.setattr(review, "_ask_llm", review_subject)
            async for batch in review.review_entities(scope, [], []):
                await write_conflicts.write_conflicts([batch], ctx)
            assert await read_user_resolutions(graph) == []
            _, edges = await graph.get_neighborhood([str(company.id)], depth=1)
            surviving = [e for e in edges if e[2] == "has_ceo"]
            assert len(surviving) == 1 and not is_rejected_fact(surviving[0][3])
            triplets = [
                Edge(Node(s), Node(t), attributes={**props, "relationship_name": rel})
                for s, t, rel, props in surviving
            ]
            context = await resolve_edges_to_text(
                apply_graph_resolutions(triplets, await read_user_resolutions(graph))
            )
            assert "Company has CEO Bob" in context
            assert "Company has CEO Alice" not in context
            assert not any(
                "Company has CEO Alice" in text
                for text in vector.points["EdgeType_relationship_name"].values()
            )

    finally:
        await graph.close()


async def _decided_conflict(tmp_path, monkeypatch, document_counts):
    """A Ladybug graph where the user chose Alice over Bob as Company's CEO.

    ``document_counts`` is how many documents state Alice's and Bob's fact. Returns the
    graph, the unified engine, the conflicts module, the dataset and user, the chosen
    and rejected fact ids, the documents behind each fact, and the stored conflict.
    """
    graph = LadybugAdapter(str(tmp_path / "graph"), kuzu_buffer_pool_size=128 * 1024 * 1024)
    unified = UnifiedStoreEngine(
        graph_engine=graph,
        vector_engine=RecordingVectors(),
        capabilities=EngineCapability.GRAPH | EngineCapability.VECTOR,
    )
    storage = importlib.import_module("cognee.tasks.storage.add_data_points")
    methods = importlib.import_module("cognee.modules.conflicts.methods")

    async def engine():
        return unified

    monkeypatch.setattr(storage, "get_unified_engine", engine)
    monkeypatch.setattr(write_conflicts, "get_unified_engine", engine)
    monkeypatch.setattr(methods, "get_graph_engine", AsyncMock(return_value=graph))
    monkeypatch.setattr(read_facts, "get_graph_engine", AsyncMock(return_value=graph))
    monkeypatch.setattr(read_facts, "get_edge_sources", AsyncMock(return_value=[]))
    monkeypatch.setattr(read_facts, "backend_access_control_enabled", lambda: True)
    dataset = SimpleNamespace(id=uuid4())
    user = SimpleNamespace(id=uuid4())
    ctx = PipelineContext(dataset=dataset, data_item={})
    await mark_graph_provenance_if_empty(graph)
    company, alice, bob = [Entity(name=n, description=n) for n in ["Company", "Alice", "Bob"]]
    facts, documents = {}, {}
    for value, document_count in zip((alice, bob), document_counts):
        fid = str(generate_edge_object_id(str(company.id), str(value.id), "has_ceo"))
        properties = {"edge_object_id": fid, "edge_text": f"Company has CEO {value.name}"}
        facts[fid] = {
            "id": fid,
            "source": str(company.id),
            "target": str(value.id),
            "relationship": "has_ceo",
            "properties": properties,
            "effective_date": None,
            "sources": [],
        }
        documents[fid] = [uuid4() for _ in range(document_count)]
        for data_id in documents[fid]:
            owner = make_source_ref_key(dataset.id, data_id)
            await graph.add_nodes([company, value], source_ref_key=owner)
            await graph.add_edges(
                [(str(company.id), str(value.id), "has_ceo", properties)],
                source_ref_key=owner,
            )
    chosen, rejected = facts
    conflict = FactConflict(
        dataset_id=str(dataset.id),
        about_id=str(company.id),
        attribute="has_ceo",
        kind="fixed",
        status="resolved",
        text="Resolved by user: Company has CEO Alice",
        values=[str(alice.id), str(bob.id)],
        resolution_source="user",
        selected_fact_id=chosen,
        selected_fact="Company has CEO Alice",
        rejected_fact_ids=[rejected],
        rejected_fact_texts=["Company has CEO Bob"],
    )
    accepted = AcceptedConflict(conflict, {chosen: "current", rejected: "superseded"})
    scope = ReviewScope(str(dataset.id), {}, facts)
    await write_conflicts.write_conflicts(
        [ConflictsToWrite(scope, [accepted], shown_fact_ids=set(facts))], ctx
    )
    await write_conflicts.write_conflicts([ConflictsToFinalize([accepted])], ctx)
    return SimpleNamespace(
        graph=graph,
        unified=unified,
        methods=methods,
        dataset=dataset,
        user=user,
        company=company,
        alice=alice,
        chosen=chosen,
        documents=documents,
        conflict=conflict,
    )


async def _remaining_ceo_edges(graph, company):
    _, edges = await graph.get_neighborhood([str(company.id)], depth=1)
    return [edge for edge in edges if edge[2] in ("has_ceo", "ceo_is")]


@pytest.mark.asyncio
async def test_deleting_every_document_behind_the_chosen_fact_retires_the_decision(
    tmp_path, monkeypatch
):
    """The decision stays while any document still supports the chosen fact."""
    world = await _decided_conflict(tmp_path, monkeypatch, document_counts=(2, 1))
    try:
        first, second = world.documents[world.chosen]
        await world.unified.delete_by_document(str(world.dataset.id), str(first))
        assert await world.methods.retire_forgotten_decisions(world.dataset, world.user) == set()
        assert len(await read_user_resolutions(world.graph)) == 1

        await world.unified.delete_by_document(str(world.dataset.id), str(second))
        assert await world.methods.retire_forgotten_decisions(world.dataset, world.user) == {
            str(world.conflict.id)
        }
        assert await read_user_resolutions(world.graph) == []
        remaining = await _remaining_ceo_edges(world.graph, world.company)
        assert len(remaining) == 1 and not is_rejected_fact(remaining[0][3])
    finally:
        await world.graph.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rebuilt_relationship, retired",
    # The rebuild extracts the chosen fact again, or states it under another relationship,
    # which gives it another id.
    [("has_ceo", False), ("ceo_is", True)],
)
async def test_a_rebuild_retires_the_decision_only_when_the_chosen_fact_did_not_come_back(
    tmp_path, monkeypatch, rebuilt_relationship, retired
):
    world = await _decided_conflict(tmp_path, monkeypatch, document_counts=(1, 1))
    invalidate = AsyncMock()
    monkeypatch.setattr(world.methods, "invalidate_sessions_for_deleted_data", invalidate)
    try:
        (document,) = world.documents[world.chosen]
        owner = make_source_ref_key(world.dataset.id, document)
        # The rebuild drops the document's memory, then extracts it again.
        await world.unified.delete_by_document(str(world.dataset.id), str(document))
        fid = str(
            generate_edge_object_id(
                str(world.company.id), str(world.alice.id), rebuilt_relationship
            )
        )
        await world.graph.add_nodes([world.company, world.alice], source_ref_key=owner)
        await world.graph.add_edges(
            [
                (
                    str(world.company.id),
                    str(world.alice.id),
                    rebuilt_relationship,
                    {"edge_object_id": fid, "edge_text": "Company has CEO Alice"},
                )
            ],
            source_ref_key=owner,
        )

        result = await world.methods.retire_decisions_after_rebuild(world.dataset, world.user)

        if retired:
            assert result == {str(world.conflict.id)}
            assert await read_user_resolutions(world.graph) == []
            # Bob's fact is no longer hidden as the rejected one.
            assert not any(
                is_rejected_fact(edge[3])
                for edge in await _remaining_ceo_edges(world.graph, world.company)
            )
            invalidate.assert_awaited_once_with(
                world.dataset.id, {str(world.conflict.id)}, set(), user_id=world.user.id
            )
        else:
            assert result == set()
            assert len(await read_user_resolutions(world.graph)) == 1
            invalidate.assert_not_awaited()
    finally:
        await world.graph.close()
