import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import UUID, uuid4

import pytest

from cognee.modules.engine.models import Entity, FactConflict
from cognee.modules.engine.utils.generate_edge_object_id import generate_edge_object_id
from cognee.modules.graph.utils.fact_conflicts import (
    effective_date_display,
    is_conflict_edge,
    read_conflict_marks,
)
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.provenance.edge_evidence.lookup import EdgeEvidenceRecord
from cognee.tasks.memify.review_conflicts import read_facts


class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.neighborhood_calls = []
        self.filter_calls = []

    def node(self, name, type="Entity", **properties):
        node_id = str(uuid4())
        self.nodes[node_id] = {"id": node_id, "name": name, "type": type, **properties}
        return node_id

    def edge(self, source, target, relationship, **properties):
        edge_id = generate_edge_object_id(source, target, relationship)
        self.edges.append((source, target, relationship, {"edge_object_id": edge_id, **properties}))
        return edge_id

    async def get_nodes(self, ids):
        return [dict(self.nodes[node_id]) for node_id in ids if node_id in self.nodes]

    async def get_filtered_graph_data(self, filters):
        self.filter_calls.append(filters)
        if "id" in filters[0]:
            return [
                (key, dict(value)) for key, value in self.nodes.items() if key in filters[0]["id"]
            ], []
        types = filters[0]["type"]
        return [
            (key, dict(value)) for key, value in self.nodes.items() if value["type"] in types
        ], []

    async def get_neighborhood(self, ids, depth=1):
        self.neighborhood_calls.append(set(ids))
        selected = set(ids)
        for source, target, _, _ in self.edges:
            if source in ids or target in ids:
                selected.update((source, target))
        return (
            [(key, dict(value)) for key, value in self.nodes.items() if key in selected],
            [edge for edge in self.edges if edge[0] in selected and edge[1] in selected],
        )


@pytest.fixture
def setup(monkeypatch):
    graph = Graph()
    dataset_id = str(uuid4())
    monkeypatch.setattr(read_facts, "get_graph_engine", AsyncMock(return_value=graph))
    monkeypatch.setattr(read_facts, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(read_facts, "get_edge_sources", AsyncMock(return_value=[]))
    return graph, dataset_id, PipelineContext(dataset=SimpleNamespace(id=dataset_id))


def conflict(graph, dataset_id, subject, values, sources, **properties):
    return graph.node(
        "CEO dispute",
        "FactConflict",
        dataset_id=dataset_id,
        about_id=subject,
        values=values,
        sources=sources,
        **properties,
    )


@pytest.mark.asyncio
async def test_reads_competing_entities_and_full_sources_without_internal_nodes(setup, monkeypatch):
    graph, dataset_id, ctx = setup
    acme, alice, bob = [graph.node(name, description=name) for name in ("Acme", "Alice", "Bob")]
    alice_type = graph.node("Person", "EntityType")
    chunk = graph.node("source", "DocumentChunk")
    preference = graph.node("private", "UserPreference", is_internal=True)
    graph.edge(alice, alice_type, "is_a")
    first = graph.edge(
        acme,
        alice,
        "has_ceo",
        edge_text="Alice leads Acme",
        weight=0.8,
        conflict_marks_json='[{"conflict_id":"old","status":"current"}]',
    )
    second = graph.edge(acme, bob, "has_ceo", edge_text="Bob leads Acme")
    statement = graph.edge(chunk, alice, "contains", edge_text="Alice works at Acme")
    graph.edge(chunk, bob, "contains")
    graph.edge(preference, acme, "prefers", edge_text="private")
    graph.edge(alice, bob, "contradicts", edge_text="legacy")
    old_conflict = conflict(graph, dataset_id, acme, [alice, bob], [chunk])
    graph.edge(old_conflict, acme, "conflict_about", edge_text="disputed")
    now = datetime(2025, 2, 1, tzinfo=timezone.utc)
    records = [
        EdgeEvidenceRecord(
            UUID(first), uuid4(), UUID(chunk), 0, "one", {"effective_date": "2024-01-01"}, now
        ),
        EdgeEvidenceRecord(
            UUID(first),
            uuid4(),
            uuid4(),
            0,
            "two",
            {"effective_date": "2024-12-31T23:30:00-02:00"},
            now,
        ),
        EdgeEvidenceRecord(
            UUID(second), uuid4(), uuid4(), 0, "bad", {"effective_date": "invalid"}, now
        ),
        EdgeEvidenceRecord(
            UUID(statement), uuid4(), UUID(chunk), 0, "bad", {"effective_date": "invalid"}, now
        ),
    ]
    monkeypatch.setattr(read_facts, "get_edge_sources", AsyncMock(return_value=records))
    warning = Mock()
    monkeypatch.setattr(read_facts.logger, "warning", warning)

    result = await read_facts.read_entity_facts([{}], entity_ids=[acme, bob, chunk], ctx=ctx)

    assert list(result.entities) == [*sorted([acme, bob]), alice]
    assert graph.neighborhood_calls == [{acme, bob}, {alice}]
    assert result.entities[alice]["types"] == ["Person"]
    assert set(result.facts) == {first, second, statement}
    assert result.facts[first]["properties"]["weight"] == 0.8
    assert result.facts[first]["properties"]["conflict_marks"] == [
        {"conflict_id": "old", "status": "current"}
    ]
    assert "conflict_marks_json" not in result.facts[first]["properties"]
    assert result.facts[first]["effective_date"] == "2025-01-01T01:30:00+00:00"
    assert result.facts[first]["sources"][0]["effective_date"] == "2024-01-01"
    assert result.facts[first]["observed_at"] == now
    assert result.facts[second]["effective_date"] is None
    assert len(result.facts[first]["sources"]) == 2
    assert warning.call_count == 1
    assert old_conflict in result.conflicts
    assert preference not in result.nodes


@pytest.mark.asyncio
async def test_lost_citations_pending_writes_and_missing_subjects_are_recoverable(
    setup, monkeypatch
):
    graph, dataset_id, ctx = setup
    acme, alice, bob = [graph.node(name) for name in ("Acme", "Alice", "Bob")]
    chunk = graph.node("surviving source", "DocumentChunk")
    lost = conflict(graph, dataset_id, acme, [alice], [str(uuid4())])
    pending = conflict(graph, dataset_id, bob, [alice], [chunk], review_pending=True)
    dropped = conflict(graph, dataset_id, str(uuid4()), [alice], [chunk])
    other_dataset = conflict(graph, str(uuid4()), acme, [], [str(uuid4())])
    untouched = graph.node("untouched")
    # Neo4j list aliases are equivalent to native list properties.
    for conflict_id in (lost, dropped):
        graph.nodes[conflict_id]["values_json"] = json.dumps(graph.nodes[conflict_id].pop("values"))
        graph.nodes[conflict_id]["sources_json"] = json.dumps(
            graph.nodes[conflict_id].pop("sources")
        )

    surviving, drops = await read_facts.find_conflicts_with_lost_citations(graph, dataset_id)
    assert {node["id"] for node in surviving} == {lost, pending}
    assert [node["id"] for node in drops] == [dropped]
    assert drops[0]["values"] == [alice]
    assert drops[0]["sources"] == [chunk]
    assert "values_json" not in drops[0]

    # The typed read already returned these properties; an ID reread is unnecessary.
    filtered = graph.get_filtered_graph_data

    async def no_conflict_reread(filters):
        assert dropped not in filters[0].get("id", [])
        return await filtered(filters)

    monkeypatch.setattr(graph, "get_filtered_graph_data", no_conflict_reread)

    result = await read_facts.read_entity_facts([{}], entity_ids=[], ctx=ctx)
    assert set(result.entities) == {acme, alice, bob}
    assert set(result.conflicts) == {lost, pending, dropped}
    assert result.drop_conflict_ids == [dropped]
    assert result.conflicts[dropped]["values"] == [alice]
    assert untouched not in result.entities
    assert other_dataset not in result.conflicts


@pytest.mark.asyncio
@pytest.mark.parametrize("graph_provenance", [False, True])
async def test_full_review_scopes_shared_graph_and_keeps_legacy_entities(
    setup, monkeypatch, graph_provenance
):
    graph, _, ctx = setup
    owned = graph.node("legacy", description="no evidence yet")
    unowned = graph.node("another dataset")
    graph.edge(owned, unowned, "knows", edge_text="outside scope")
    monkeypatch.setattr(read_facts, "backend_access_control_enabled", lambda: False)
    monkeypatch.setattr(
        read_facts, "stores_provenance_in_graph", AsyncMock(return_value=graph_provenance)
    )
    graph.find_node_source_refs_by_dataset = AsyncMock(return_value={owned: ["ref"]})
    monkeypatch.setattr(
        read_facts,
        "get_dataset_related_nodes",
        AsyncMock(return_value=[SimpleNamespace(slug=owned)]),
    )

    result = await read_facts.read_entity_facts([{}], entity_ids=None, ctx=ctx)

    assert list(result.entities) == [owned]
    assert not result.facts
    assert unowned not in result.nodes
    assert not any("id" in call[0] for call in graph.filter_calls)


@pytest.mark.asyncio
async def test_full_review_survives_a_deleted_conflict_subject(setup):
    graph, dataset_id, ctx = setup
    acme = graph.node("Acme")
    dropped = conflict(graph, dataset_id, str(uuid4()), [], [str(uuid4())])

    result = await read_facts.read_entity_facts([{}], entity_ids=None, ctx=ctx)

    assert set(result.entities) == {acme}
    assert result.drop_conflict_ids == [dropped]


@pytest.mark.asyncio
@pytest.mark.parametrize("has_conflict", [False, True])
async def test_failed_node_reads_cannot_be_mistaken_for_deleted_entities(
    setup, monkeypatch, has_conflict
):
    graph, dataset_id, ctx = setup
    subject = graph.node("Acme")
    if has_conflict:
        conflict(graph, dataset_id, subject, [], [str(uuid4())])
    permissive_read = AsyncMock(return_value=[])
    monkeypatch.setattr(graph, "get_nodes", permissive_read)
    filtered = graph.get_filtered_graph_data

    async def failing_id_read(filters):
        if "id" in filters[0]:
            raise RuntimeError("node read failed")
        return await filtered(filters)

    monkeypatch.setattr(graph, "get_filtered_graph_data", failing_id_read)
    with pytest.raises(RuntimeError, match="node read failed"):
        await read_facts.read_entity_facts([{}], entity_ids=[subject], ctx=ctx)
    permissive_read.assert_not_awaited()


@pytest.mark.asyncio
async def test_successful_empty_id_read_skips_a_deleted_selection(setup):
    _, _, ctx = setup
    result = await read_facts.read_entity_facts([{}], entity_ids=[str(uuid4())], ctx=ctx)
    assert not result.entities
    assert not result.drop_conflict_ids


@pytest.mark.parametrize(
    "properties, expected",
    [
        ({"conflict_marks": [], "conflict_marks_json": '[{"status":"current"}]'}, []),
        (
            {"conflict_marks": None, "conflict_marks_json": '[{"status":"current"}]'},
            [{"status": "current"}],
        ),
        ({"conflict_marks": "invalid", "conflict_marks_json": "[{}]"}, []),
        ({"conflict_marks_json": "invalid"}, []),
        ({"conflict_marks_json": "{}"}, []),
    ],
)
def test_conflict_mark_aliases(properties, expected):
    assert read_conflict_marks(properties) == expected


def test_model_identity_and_extraction_reset():
    fields = {
        "dataset_id": str(uuid4()),
        "about_id": str(uuid4()),
        "kind": "fixed",
        "status": "unresolved",
        "text": "two values",
    }
    first = FactConflict(attribute="CEO", **fields)
    second = FactConflict(attribute="ceo", **fields)
    assert first.id == second.id
    assert first.metadata["index_fields"] == []
    assert first.review_pending is False
    assert (
        Entity(name="Alice", description="new extraction").model_dump()["conflicts_reviewed_at"]
        is None
    )
    assert effective_date_display("2025-01-01T12:00:00+00:00") == "2025-01-01"
    assert effective_date_display(None) == ""
    assert is_conflict_edge("conflict_value")
    assert not is_conflict_edge("has_ceo")
