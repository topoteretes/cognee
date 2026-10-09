"""Partial review writes remain discoverable and safe to repeat."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from cognee.infrastructure.databases.exceptions import UnsupportedProvenanceCapability
from cognee.modules.engine.models import Entity, FactConflict
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.provenance.edge_evidence.lookup import EdgeEvidenceRecord
from cognee.tasks.memify.review_conflicts import read_facts, write_conflicts
from cognee.tasks.memify.review_conflicts.models import AcceptedConflict, ReviewBatch, ReviewScope


@pytest.fixture
def store(monkeypatch):
    subject, alice, bob, chunk, missing, dataset = [str(uuid4()) for _ in range(6)]
    entities = {
        node_id: Entity(id=node_id, name=name, description="Old description").model_dump(
            mode="json"
        )
        for node_id, name in [(subject, "Acme"), (alice, "Alice"), (bob, "Bob")]
    }
    conflict = FactConflict(
        dataset_id=dataset,
        about_id=subject,
        attribute="has_ceo",
        kind="time_varying",
        status="resolved",
        text="Alice was CEO; Bob is CEO.",
        values=[alice, bob],
        sources=[chunk, missing],
    )
    cid = str(conflict.id)
    facts = {}
    for value, status in [(alice, "superseded"), (bob, "current")]:
        fid = str(uuid4())
        facts[fid] = {
            "id": fid,
            "source": subject,
            "target": value,
            "relationship": "has_ceo",
            "properties": {
                "edge_text": "CEO fact",
                "weight": 3,
                "conflict_marks": [
                    {"conflict_id": cid, "status": status},
                    {"conflict_id": "other", "status": "conflicting"},
                ],
            },
            "sources": [
                {
                    "chunk_id": chunk,
                    "data_id": "data",
                    "document": "Report",
                    "effective_date": "2026-01-10",
                }
            ],
            "effective_date": "2026-01-10",
            "observed_at": None,
        }
    scope = ReviewScope(dataset, entities, facts, {cid: conflict.model_dump(mode="json")})
    nodes = {**entities, cid: conflict.model_dump(mode="json")}
    edges = {(cid, alice, "conflict_value"): {}, (cid, chunk, "conflict_cites"): {}}
    vectors = {str(EdgeType.id_for(conflict.text))}
    calls = []
    fail = SimpleNamespace(at=None)

    async def add_nodes(items):
        for node in items:
            nodes[str(node.id)] = node.model_dump(mode="json")

    async def add_edges(items):
        if fail.at == "marks":
            raise RuntimeError("marks")
        for source, target, relationship, properties in items:
            edges[(source, target, relationship)] = properties

    async def neighborhood(ids, **kwargs):
        selected = [(s, t, r, p) for (s, t, r), p in edges.items() if s in ids or t in ids]
        return list(nodes.items()), selected

    async def delete_triples(items):
        for edge in items:
            edges.pop((edge.source_id, edge.target_id, edge.relationship_name), None)

    async def delete_nodes(ids):
        for node_id in ids:
            nodes.pop(node_id, None)
        for key in list(edges):
            if key[0] in ids or key[1] in ids:
                del edges[key]

    async def delete_vectors(collection, ids):
        if fail.at == "cleanup":
            raise RuntimeError("cleanup")
        vectors.difference_update(map(str, ids))

    graph = SimpleNamespace(
        add_nodes=AsyncMock(side_effect=add_nodes),
        add_edges=AsyncMock(side_effect=add_edges),
        get_neighborhood=AsyncMock(side_effect=neighborhood),
        delete_edge_triples=AsyncMock(side_effect=delete_triples),
        delete_nodes=AsyncMock(side_effect=delete_nodes),
    )
    vector = SimpleNamespace(delete_data_points=AsyncMock(side_effect=delete_vectors))

    async def storage(items, custom_edges=None, ctx=None):
        calls.append((items, custom_edges, ctx))
        if items and isinstance(items[0], Entity) and fail.at == "entity":
            raise RuntimeError("entity")
        await add_nodes(items)
        if custom_edges:
            if fail.at == "links":
                raise RuntimeError("links")
            for source, target, relationship, properties in custom_edges:
                edges[(source, target, relationship)] = properties
            if fail.at == "index":
                raise RuntimeError("index")
            vectors.update(str(EdgeType.id_for(props["edge_text"])) for *_, props in custom_edges)

    monkeypatch.setattr(
        write_conflicts,
        "get_unified_engine",
        AsyncMock(return_value=SimpleNamespace(graph=graph, vector=vector)),
    )
    monkeypatch.setattr(write_conflicts, "add_data_points", storage)
    new = conflict.model_copy(update={"text": "Bob is the current CEO.", "sources": [chunk]})
    accepted = AcceptedConflict(
        new,
        {
            fid: ("superseded" if fact["target"] == alice else "current")
            for fid, fact in facts.items()
        },
    )
    batch = ReviewBatch(
        scope,
        {eid: "Reviewed description" for eid in entities},
        [accepted],
        shown_fact_ids=set(facts),
    )
    ctx = PipelineContext(dataset=SimpleNamespace(id=dataset), data_item={})
    return SimpleNamespace(
        scope=scope,
        batch=batch,
        cid=cid,
        nodes=nodes,
        edges=edges,
        vectors=vectors,
        calls=calls,
        fail=fail,
        ctx=ctx,
        graph=graph,
        vector=vector,
        alice=alice,
        bob=bob,
    )


async def finish(store, state):
    await write_conflicts.write_review_batch(
        [ReviewBatch(store.scope, final=True)], store.ctx, state
    )


@pytest.mark.asyncio
async def test_review_replaces_links_preserves_marks_and_finalizes_last(store):
    state = write_conflicts.ReviewWriteState()
    await write_conflicts.write_review_batch([store.batch], store.ctx, state)
    assert store.nodes[store.cid]["review_pending"] is True
    assert store.ctx.data_item == {}
    conflict_calls = [call for call in store.calls if isinstance(call[0][0], FactConflict)]
    assert all(call[2] is not store.ctx and call[2].data_item.id for call in conflict_calls)
    assert all(call[2] is store.ctx for call in store.calls if isinstance(call[0][0], Entity))
    for fact in store.scope.facts.values():
        properties = store.edges[(fact["source"], fact["target"], "has_ceo")]
        assert properties["weight"] == 3
        assert {"conflict_id": "other", "status": "conflicting"} in properties["conflict_marks"]
    await finish(store, state)
    assert store.nodes[store.cid]["review_pending"] is False
    assert store.nodes[store.cid]["text"] == "Bob is the current CEO."
    assert store.vectors == {str(EdgeType.id_for("Bob is the current CEO."))}


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["marks", "entity", "links", "index", "cleanup"])
async def test_storage_failure_keeps_pending_node_and_rerun_repairs_it(store, failure):
    store.fail.at = failure
    with pytest.raises(RuntimeError, match=failure):
        await write_conflicts.write_review_batch(
            [store.batch], store.ctx, write_conflicts.ReviewWriteState()
        )
    assert store.nodes[store.cid]["review_pending"] is True
    # A fresh read sees the stored pending node even after its sources were replaced.
    store.scope.conflicts[store.cid] = dict(store.nodes[store.cid])
    store.fail.at = None
    state = write_conflicts.ReviewWriteState()
    await write_conflicts.write_review_batch([store.batch], store.ctx, state)
    await finish(store, state)
    assert store.nodes[store.cid]["review_pending"] is False
    assert (store.cid, store.bob, "conflict_value") in store.edges
    assert str(EdgeType.id_for("Bob is the current CEO.")) in store.vectors


@pytest.mark.asyncio
async def test_drop_waits_for_failed_value_description_and_cleanup(store):
    batch = ReviewBatch(
        store.scope,
        {eid: "Reviewed" for eid in store.scope.entities if eid != store.bob},
        dropped_conflict_ids=[store.cid],
    )
    state = write_conflicts.ReviewWriteState()
    await write_conflicts.write_review_batch([batch], store.ctx, state)
    await write_conflicts.write_review_batch(
        [ReviewBatch(store.scope, final=True, unreviewed_entity_ids=[store.bob])], store.ctx, state
    )
    assert state.unreviewed_entity_ids == [store.bob]
    assert store.nodes[store.cid]["review_pending"] is True
    store.graph.delete_nodes.assert_not_awaited()
    # Finishing the value in another batch allows the final deletion.
    await write_conflicts.write_review_batch(
        [ReviewBatch(store.scope, {store.bob: "Reviewed Bob"})], store.ctx, state
    )
    await finish(store, state)
    assert store.cid not in store.nodes
    assert not store.vectors


@pytest.mark.asyncio
async def test_drop_cleanup_failure_preserves_retry_marker(store):
    store.fail.at = "cleanup"
    batch = ReviewBatch(store.scope, store.batch.descriptions, dropped_conflict_ids=[store.cid])
    with pytest.raises(RuntimeError, match="cleanup"):
        await write_conflicts.write_review_batch(
            [batch], store.ctx, write_conflicts.ReviewWriteState()
        )
    assert store.nodes[store.cid]["review_pending"] is True
    store.graph.delete_nodes.assert_not_awaited()


@pytest.mark.asyncio
async def test_removed_value_is_retained_for_retry_until_its_description_succeeds(store):
    store.batch.conflicts[0].conflict.values = [store.bob]
    store.batch.descriptions.pop(store.alice)
    state = write_conflicts.ReviewWriteState()
    await write_conflicts.write_review_batch([store.batch], store.ctx, state)
    await finish(store, state)
    assert store.nodes[store.cid]["review_pending"] is True
    assert store.alice in store.nodes[store.cid]["values"]
    await write_conflicts.write_review_batch(
        [ReviewBatch(store.scope, {store.alice: "Updated Alice"})], store.ctx, state
    )
    await finish(store, state)
    assert store.nodes[store.cid]["values"] == [store.bob]


@pytest.mark.asyncio
async def test_context_only_value_is_selected_on_next_run_before_finalizing(store, monkeypatch):
    store.scope.nodes = dict(store.scope.entities)
    store.scope.entities.pop(store.alice)
    store.batch.descriptions.pop(store.alice)
    chunk = store.batch.conflicts[0].conflict.sources[0]
    store.nodes[chunk] = {"id": chunk, "type": "DocumentChunk"}
    evidence = []
    for fact_id, fact in store.scope.facts.items():
        fact["properties"]["edge_object_id"] = fact_id
        evidence.append(EdgeEvidenceRecord(UUID(fact_id), uuid4(), UUID(chunk), 0, "Report"))

    state = write_conflicts.ReviewWriteState()
    await write_conflicts.write_review_batch([store.batch], store.ctx, state)
    await finish(store, state)
    assert store.nodes[store.cid]["review_pending"] is True
    assert store.nodes[store.alice]["description"] == "Old description"

    store.graph.get_nodes = AsyncMock(
        side_effect=lambda ids: [store.nodes[key] for key in ids if key in store.nodes]
    )
    store.graph.get_filtered_graph_data = AsyncMock(
        side_effect=lambda filters: (
            [
                (key, node)
                for key, node in store.nodes.items()
                if (
                    key in filters[0]["id"]
                    if "id" in filters[0]
                    else node["type"] in filters[0]["type"]
                )
            ],
            [],
        )
    )
    monkeypatch.setattr(read_facts, "get_graph_engine", AsyncMock(return_value=store.graph))
    monkeypatch.setattr(read_facts, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(read_facts, "get_edge_sources", AsyncMock(return_value=evidence))
    recovered = await read_facts.read_entity_facts([{}], entity_ids=[], ctx=store.ctx)
    assert store.alice in recovered.entities
    batch = ReviewBatch(
        recovered,
        {key: "Reviewed again" for key in recovered.entities},
        store.batch.conflicts,
        shown_fact_ids=set(recovered.facts),
    )
    state = write_conflicts.ReviewWriteState()
    await write_conflicts.write_review_batch([batch], store.ctx, state)
    await finish(store, state)
    assert store.nodes[store.cid]["review_pending"] is False
    assert store.nodes[store.alice]["description"] == "Reviewed again"


@pytest.mark.asyncio
async def test_deleted_old_value_does_not_block_finalization(store):
    store.scope.conflicts[store.cid]["values"].append(str(uuid4()))
    state = write_conflicts.ReviewWriteState()
    await write_conflicts.write_review_batch([store.batch], store.ctx, state)
    await finish(store, state)
    assert store.nodes[store.cid]["review_pending"] is False


@pytest.mark.asyncio
async def test_unsupported_edge_delete_is_not_swallowed(store):
    """A backend that cannot delete edges fails the batch instead of completing.

    Finishing would leave the conflict's old value and citation links beside the
    new ones, and search annotates de-cited chunks and dropped values from them.
    """
    store.graph.delete_edge_triples = AsyncMock(side_effect=UnsupportedProvenanceCapability())
    with pytest.raises(UnsupportedProvenanceCapability):
        await write_conflicts.write_review_batch(
            [store.batch], store.ctx, write_conflicts.ReviewWriteState()
        )
