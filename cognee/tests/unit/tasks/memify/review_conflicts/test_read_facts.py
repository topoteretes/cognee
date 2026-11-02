import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import UUID, uuid4

import pytest

from cognee.modules.engine.models import FactConflict
from cognee.modules.engine.utils.generate_edge_object_id import generate_edge_object_id
from cognee.modules.graph.utils.fact_conflicts import (
    effective_date_display,
    is_conflict_edge,
    read_conflict_marks,
)
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.provenance.edge_evidence.lookup import EdgeEvidenceRecord
from cognee.tasks.memify.review_conflicts import read_facts
from cognee.tasks.memify.review_conflicts.select_review_facts import (
    find_fact_comparisons,
    select_subject_ids,
)


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

    async def get_neighborhood(self, ids, depth=1, edge_types=None):
        # Like the adapters: edge_types limits the neighbours reached, and every edge
        # among the reached nodes comes back.
        self.neighborhood_calls.append((set(ids), edge_types))
        selected = set(ids)
        for source, target, relationship, _ in self.edges:
            if edge_types and relationship not in edge_types:
                continue
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


def answer_for_requested_edges(records):
    """get_edge_sources answering, like the real lookup, only for the edges asked about."""

    async def get_edge_sources(dataset_id, edge_ids):
        requested = set(edge_ids)
        return [record for record in records if record.edge_id in requested]

    return get_edge_sources


def select_subjects(scope):
    """The subjects the reviewer makes of a scope's entities."""
    comparisons = find_fact_comparisons(scope.facts, scope.nodes, scope.since)
    return select_subject_ids(scope, comparisons)


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
    acme, alice, bob = [graph.node(name) for name in ("Acme", "Alice", "Bob")]
    person = graph.node("Person", "EntityType")
    chunk = graph.node("source", "DocumentChunk")
    preference = graph.node("private", "UserPreference", is_internal=True)
    graph.edge(alice, person, "is_a")
    graph.edge(bob, person, "is_a")
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
    monkeypatch.setattr(read_facts, "get_edge_sources", answer_for_requested_edges(records))
    warning = Mock()
    monkeypatch.setattr(read_facts.logger, "warning", warning)

    result = await read_facts.read_entity_facts([{}], entity_ids=[acme, bob, chunk], ctx=ctx)

    assert set(result.entities) == {acme, bob}
    assert select_subjects(result) == [acme]  # Bob's one claim names nothing to compare
    # Alice is only a value: the second read follows her is_a edges alone.
    assert graph.neighborhood_calls == [({acme, bob}, None), ({alice}, ["is_a"])]
    assert result.nodes[alice]["types"] == result.nodes[bob]["types"] == ["Person"]
    # The chunk's statement about Alice touches neither selected entity.
    assert set(result.facts) == {first, second}
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
    pending = conflict(graph, dataset_id, bob, [alice], [chunk], write_pending=True)
    dropped = conflict(graph, dataset_id, str(uuid4()), [alice], [chunk])
    other_dataset = conflict(graph, str(uuid4()), acme, [], [str(uuid4())])
    untouched = graph.node("untouched")
    # Neo4j list aliases are equivalent to native list properties.
    for conflict_id in (lost, dropped):
        graph.nodes[conflict_id]["values_json"] = json.dumps(graph.nodes[conflict_id].pop("values"))
        graph.nodes[conflict_id]["sources_json"] = json.dumps(
            graph.nodes[conflict_id].pop("sources")
        )

    stored = await read_facts.read_stored_conflicts(graph, dataset_id, None)
    assert set(stored.ids_to_review) == {lost, pending}
    assert stored.ids_to_drop == [dropped]
    assert stored.by_id[dropped]["values"] == [alice]
    assert stored.by_id[dropped]["sources"] == [chunk]
    assert "values_json" not in stored.by_id[dropped]
    assert other_dataset not in stored.by_id

    # The typed read already returned these properties; an ID reread is unnecessary.
    filtered = graph.get_filtered_graph_data

    async def no_conflict_reread(filters):
        assert dropped not in filters[0].get("id", [])
        return await filtered(filters)

    monkeypatch.setattr(graph, "get_filtered_graph_data", no_conflict_reread)

    result = await read_facts.read_entity_facts([{}], entity_ids=[], ctx=ctx)
    assert set(result.entities) == {acme, bob}
    assert set(result.conflicts) == {lost, pending, dropped}
    assert result.conflict_ids_to_drop == [dropped]
    assert result.conflicts[dropped]["values"] == [alice]
    assert untouched not in result.entities
    assert other_dataset not in result.conflicts


@pytest.mark.asyncio
async def test_a_run_reads_the_stored_conflicts_touching_its_entities_or_their_values(setup):
    graph, dataset_id, ctx = setup
    acme, alice, carol, far = [graph.node(name) for name in ("Acme", "Alice", "Carol", "Far")]
    chunk = graph.node("source", "DocumentChunk")
    graph.edge(acme, alice, "has_ceo")
    about_acme = conflict(graph, dataset_id, acme, [alice], [chunk])
    about_its_value = conflict(graph, dataset_id, alice, [carol], [chunk])
    valued_at_acme = conflict(graph, dataset_id, carol, [acme], [chunk])
    unrelated = conflict(graph, dataset_id, far, [carol], [chunk])

    result = await read_facts.read_entity_facts([{}], entity_ids=[acme], ctx=ctx)

    assert set(result.conflicts) == {about_acme, about_its_value, valued_at_acme}
    assert unrelated not in result.conflicts


@pytest.mark.asyncio
@pytest.mark.parametrize("graph_provenance", [False, True])
async def test_full_review_scopes_shared_graph_to_owned_nodes(setup, monkeypatch, graph_provenance):
    graph, _, ctx = setup
    owned = graph.node("legacy")
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
    assert owned in result.nodes and unowned not in result.nodes
    assert not any("id" in call[0] for call in graph.filter_calls)


@pytest.mark.asyncio
async def test_shared_graph_drops_a_conflict_whose_subject_left_the_dataset(setup, monkeypatch):
    graph, dataset_id, ctx = setup
    acme, departed = graph.node("Acme"), graph.node("Now only in another dataset")
    chunk = graph.node("source", "DocumentChunk")
    kept = conflict(graph, dataset_id, acme, [], [chunk])
    stuck = conflict(graph, dataset_id, departed, [], [str(uuid4())], write_pending=True)
    monkeypatch.setattr(read_facts, "backend_access_control_enabled", lambda: False)
    monkeypatch.setattr(read_facts, "stores_provenance_in_graph", AsyncMock(return_value=True))
    graph.find_node_source_refs_by_dataset = AsyncMock(
        return_value={node_id: ["ref"] for node_id in (acme, chunk, kept, stuck)}
    )

    result = await read_facts.read_entity_facts([{}], entity_ids=None, ctx=ctx)

    # The run reads only this dataset's nodes, so it could never review the stuck one again.
    assert list(result.entities) == [acme]
    assert result.conflict_ids_to_drop == [stuck]
    assert kept in result.conflicts


@pytest.mark.asyncio
async def test_shared_graph_scopes_a_dataset_that_owns_no_nodes_to_nothing(setup, monkeypatch):
    graph, dataset_id, ctx = setup
    acme, alice = graph.node("Acme"), graph.node("Alice")
    chunk = graph.node("source", "DocumentChunk")
    graph.edge(acme, alice, "has_ceo")
    stored = conflict(graph, dataset_id, acme, [alice], [chunk])
    monkeypatch.setattr(read_facts, "backend_access_control_enabled", lambda: False)
    monkeypatch.setattr(read_facts, "stores_provenance_in_graph", AsyncMock(return_value=True))
    graph.find_node_source_refs_by_dataset = AsyncMock(return_value={})

    result = await read_facts.read_entity_facts([{}], entity_ids=None, ctx=ctx)

    # An empty set is a filter that keeps no node; only None means no filter.
    assert not result.entities and not result.facts
    assert result.conflict_ids_to_drop == [stored]


@pytest.mark.asyncio
async def test_full_review_survives_a_deleted_conflict_subject(setup):
    graph, dataset_id, ctx = setup
    acme = graph.node("Acme")
    dropped = conflict(graph, dataset_id, str(uuid4()), [], [str(uuid4())])

    result = await read_facts.read_entity_facts([{}], entity_ids=None, ctx=ctx)

    assert list(result.entities) == [acme] and not select_subjects(result)
    assert result.conflict_ids_to_drop == [dropped]


@pytest.mark.asyncio
async def test_a_conflict_whose_subject_is_no_longer_an_entity_is_dropped(setup):
    graph, dataset_id, ctx = setup
    subject = graph.node("Acme", "EntityType")
    chunk = graph.node("source", "DocumentChunk")
    stored = conflict(graph, dataset_id, subject, [], [chunk], write_pending=True)

    result = await read_facts.read_entity_facts([{}], entity_ids=[], ctx=ctx)

    # A run selects only entities, so no later run could finish its pending write.
    assert result.conflict_ids_to_drop == [stored]


@pytest.mark.asyncio
async def test_subjects_need_a_value_group_or_a_stored_conflict(setup, monkeypatch):
    graph, dataset_id, ctx = setup
    person = graph.node("Person", "EntityType")
    dated, disputed, paired, one_document, mixed_types = [
        graph.node(name) for name in ("Dated", "Disputed", "Paired", "One document", "Mixed types")
    ]
    alice, bob, carol, dave, erin = [
        graph.node(name) for name in ("Alice", "Bob", "Carol", "Dave", "Erin")
    ]
    for value in (alice, bob, carol, dave):
        graph.edge(value, person, "is_a")
    days = [graph.node(name, "Timestamp") for name in ("2014-05-12", "2015-05-12")]
    claims = [
        graph.edge(dated, days[0], "founded_at"),
        graph.edge(dated, days[1], "founded_at"),
        graph.edge(disputed, alice, "has_ceo"),
        graph.edge(one_document, bob, "has_ceo"),
        graph.edge(one_document, carol, "has_ceo"),
        graph.edge(mixed_types, dave, "has_ceo"),
        graph.edge(mixed_types, erin, "has_ceo"),
    ]
    chunks = [graph.node(f"chunk {index}", "DocumentChunk") for index in (1, 2)]
    statements = [
        graph.edge(chunk, paired, "contains", edge_text=f"Document chunk mentions paired: {text}")
        for chunk, text in zip(chunks, ("Paired has 120 staff", "Paired has 150 staff"))
    ]
    documents = [uuid4() for _ in claims + statements]
    documents[4] = documents[3]
    records = [
        EdgeEvidenceRecord(UUID(edge_id), data_id, uuid4(), 0, "doc")
        for edge_id, data_id in zip(claims + statements, documents)
    ]
    monkeypatch.setattr(read_facts, "get_edge_sources", AsyncMock(return_value=records))
    chunk = graph.node("cited", "DocumentChunk")
    stored = conflict(graph, dataset_id, disputed, [alice], [chunk])
    graph.edge(stored, disputed, "conflict_about")

    result = await read_facts.read_entity_facts(
        [{}], entity_ids=[dated, disputed, paired, one_document, mixed_types], ctx=ctx
    )

    assert set(select_subjects(result)) == {dated, disputed, paired}
    assert graph.neighborhood_calls[1] == ({alice, bob, carol, dave, erin}, ["is_a"])
    assert result.nodes[dave]["types"] == ["Person"] and result.nodes[erin]["types"] == []
    assert set(result.facts) == set(claims) | set(statements)

    # After a review, Dated's only new fact compares with nothing, so Dated gets no
    # call; Paired's new statement is in its pair, so Paired is reviewed again.
    since = datetime(2026, 1, 1, tzinfo=timezone.utc)
    later = since + timedelta(days=1)
    new_claim = graph.edge(dated, alice, "has_ceo")
    records += [
        EdgeEvidenceRecord(UUID(new_claim), uuid4(), uuid4(), 0, "doc", observed_at=later),
        EdgeEvidenceRecord(UUID(statements[0]), uuid4(), uuid4(), 0, "doc", observed_at=later),
    ]
    result = await read_facts.read_entity_facts(
        [{}], entity_ids=[dated, disputed, paired, one_document, mixed_types], since=since, ctx=ctx
    )
    assert set(select_subjects(result)) == {disputed, paired}


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
    assert not result.conflict_ids_to_drop


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


def test_conflict_model_identity_and_helpers():
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
    assert first.write_pending is False
    assert effective_date_display("2025-01-01T12:00:00+00:00") == "2025-01-01"
    assert effective_date_display(None) == ""
    assert is_conflict_edge("conflict_value")
    assert not is_conflict_edge("has_ceo")
