"""What one review batch writes, and partial writes that stay discoverable and safe to repeat."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.exceptions import UnsupportedProvenanceCapability
from cognee.modules.engine.models import FactConflict
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.pipelines.models import PipelineContext
from cognee.tasks.memify.review_conflicts import write_conflicts
from cognee.tasks.memify.review_conflicts.facts import select_supporting_facts
from cognee.tasks.memify.review_conflicts.models import (
    AcceptedConflict,
    ConflictsToFinalize,
    ConflictsToWrite,
    ReviewScope,
)


@pytest.fixture
def store(monkeypatch):
    subject, alice, bob, chunk, missing, dataset = [str(uuid4()) for _ in range(6)]
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
    scope = ReviewScope(dataset, {}, facts, {cid: conflict.model_dump(mode="json")})
    nodes = {cid: conflict.model_dump(mode="json")}
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
    batch = ConflictsToWrite(scope, [accepted], shown_fact_ids=set(facts))
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


async def finish(store, *batches):
    """End the stream as the reviewer does, with every conflict these batches decided."""
    end = ConflictsToFinalize(
        accepted_conflicts=[item for batch in batches for item in batch.accepted_conflicts],
        dropped_conflict_ids=[cid for batch in batches for cid in batch.dropped_conflict_ids],
    )
    await write_conflicts.write_conflicts([end], store.ctx)


def find_fact_id(store, value: str) -> str:
    """The fixture's fact that links the subject to this value."""
    return next(fid for fid, fact in store.scope.facts.items() if fact["target"] == value)


def read_written_properties(store, fact_id: str) -> dict | None:
    """The properties the writer stored on this fact's edge; None when it did not write it."""
    fact = store.scope.facts[fact_id]
    return store.edges.get((fact["source"], fact["target"], fact["relationship"]))


@pytest.mark.asyncio
async def test_review_replaces_links_preserves_marks_and_finalizes_last(store):
    await write_conflicts.write_conflicts([store.batch], store.ctx)
    assert store.nodes[store.cid]["write_pending"] is True
    # Later calls of the run see the conflict as the run leaves it, not as a pending write.
    assert store.scope.conflicts[store.cid]["write_pending"] is False
    assert store.ctx.data_item == {}
    assert all(isinstance(call[0][0], FactConflict) for call in store.calls)
    assert all(call[2] is not store.ctx and call[2].data_item.id for call in store.calls)
    for fact in store.scope.facts.values():
        properties = store.edges[(fact["source"], fact["target"], "has_ceo")]
        assert properties["weight"] == 3
        assert {"conflict_id": "other", "status": "conflicting"} in properties["conflict_marks"]
    await finish(store, store.batch)
    assert store.nodes[store.cid]["write_pending"] is False
    assert store.nodes[store.cid]["text"] == "Bob is the current CEO."
    assert store.vectors == {str(EdgeType.id_for("Bob is the current CEO."))}


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["marks", "links", "index", "cleanup"])
async def test_storage_failure_keeps_pending_node_and_rerun_repairs_it(store, failure):
    store.fail.at = failure
    with pytest.raises(RuntimeError, match=failure):
        await write_conflicts.write_conflicts([store.batch], store.ctx)
    assert store.nodes[store.cid]["write_pending"] is True
    # A fresh read sees the stored pending node even after its sources were replaced.
    store.scope.conflicts[store.cid] = dict(store.nodes[store.cid])
    store.fail.at = None
    await write_conflicts.write_conflicts([store.batch], store.ctx)
    await finish(store, store.batch)
    assert store.nodes[store.cid]["write_pending"] is False
    assert (store.cid, store.bob, "conflict_value") in store.edges
    assert str(EdgeType.id_for("Bob is the current CEO.")) in store.vectors


@pytest.mark.asyncio
async def test_drop_is_deleted_only_when_the_stream_ends(store):
    batch = ConflictsToWrite(store.scope, dropped_conflict_ids=[store.cid])
    await write_conflicts.write_conflicts([batch], store.ctx)
    assert store.nodes[store.cid]["write_pending"] is True
    store.graph.delete_nodes.assert_not_awaited()
    await finish(store, batch)
    assert store.cid not in store.nodes
    assert not store.vectors


@pytest.mark.asyncio
async def test_drop_leaves_the_run_scope_at_once(store):
    """Later calls of the run must not show a dropped conflict or count its citations."""
    batch = ConflictsToWrite(store.scope, dropped_conflict_ids=[store.cid])
    await write_conflicts.write_conflicts([batch], store.ctx)
    assert store.cid not in store.scope.conflicts
    # Both facts come from a chunk the dropped conflict cites; neither supports a conflict now.
    supporting = select_supporting_facts(
        store.scope.facts, store.scope.facts, store.scope.conflicts.values()
    )
    assert supporting == set()


@pytest.mark.asyncio
async def test_drop_cleanup_failure_preserves_retry_marker(store):
    store.fail.at = "cleanup"
    batch = ConflictsToWrite(store.scope, dropped_conflict_ids=[store.cid])
    with pytest.raises(RuntimeError, match="cleanup"):
        await write_conflicts.write_conflicts([batch], store.ctx)
    assert store.nodes[store.cid]["write_pending"] is True
    store.graph.delete_nodes.assert_not_awaited()


@pytest.mark.asyncio
async def test_removed_value_leaves_the_stored_conflict_at_once(store):
    store.batch.accepted_conflicts[0].conflict.values = [store.bob]
    await write_conflicts.write_conflicts([store.batch], store.ctx)
    await finish(store, store.batch)
    assert store.nodes[store.cid]["values"] == [store.bob]
    assert store.nodes[store.cid]["write_pending"] is False


@pytest.mark.asyncio
async def test_unsupported_edge_delete_is_not_swallowed(store):
    """A backend that cannot delete edges fails the batch instead of completing.

    Finishing would leave the conflict's old value and citation links beside the
    new ones, and search annotates de-cited chunks and dropped values from them.
    """
    store.graph.delete_edge_triples = AsyncMock(side_effect=UnsupportedProvenanceCapability())
    with pytest.raises(UnsupportedProvenanceCapability):
        await write_conflicts.write_conflicts([store.batch], store.ctx)


@pytest.mark.asyncio
@pytest.mark.parametrize("shown", [True, False])
async def test_unlabelled_fact_loses_its_mark_only_when_shown(store, shown):
    """A shown fact the answer did not label loses the conflict's mark.

    The answer judged that fact and left it out of the conflict. An unshown fact was not judged.
    """
    alice_fact, bob_fact = find_fact_id(store, store.alice), find_fact_id(store, store.bob)
    stored_marks = store.scope.facts[alice_fact]["properties"]["conflict_marks"]
    store.batch.accepted_conflicts[0].fact_statuses = {bob_fact: "current"}
    store.batch.shown_fact_ids = {alice_fact, bob_fact} if shown else {bob_fact}
    await write_conflicts.write_conflicts([store.batch], store.ctx)
    written = read_written_properties(store, alice_fact)
    if shown:
        assert written["conflict_marks"] == [{"conflict_id": "other", "status": "conflicting"}]
    else:
        assert written is None
        assert store.scope.facts[alice_fact]["properties"]["conflict_marks"] == stored_marks


@pytest.mark.asyncio
async def test_answer_with_no_conflicts_rewrites_a_changed_date(store):
    """A shown fact is stored with the date it was judged with, even when nothing else changed."""
    alice_fact, bob_fact = find_fact_id(store, store.alice), find_fact_id(store, store.bob)
    store.scope.facts[alice_fact]["properties"]["effective_date"] = "2025-06-01"
    store.scope.facts[bob_fact]["properties"]["effective_date"] = "2026-01-10"
    stored_marks = store.scope.facts[alice_fact]["properties"]["conflict_marks"]
    batch = ConflictsToWrite(store.scope, shown_fact_ids={alice_fact, bob_fact})
    await write_conflicts.write_conflicts([batch], store.ctx)
    written = read_written_properties(store, alice_fact)
    assert written["effective_date"] == "2026-01-10"
    assert written["conflict_marks"] == stored_marks
    # Its stored date is the date read now, so there is nothing to rewrite.
    assert read_written_properties(store, bob_fact) is None


@pytest.mark.asyncio
async def test_value_link_takes_the_strongest_status_of_its_facts(store):
    """A value's facts can be given different statuses; its link stores the strongest."""
    facts = store.scope.facts
    fact_statuses = {}
    # Alice's stronger status comes second, Bob's weaker one comes second.
    for value, first_status, second_status in [
        (store.alice, "superseded", "conflicting"),
        (store.bob, "superseded", "current"),
    ]:
        first_fact, second_fact = find_fact_id(store, value), str(uuid4())
        facts[second_fact] = copy.deepcopy(facts[first_fact])
        facts[second_fact].update(id=second_fact, relationship="led_by")
        fact_statuses.update({first_fact: first_status, second_fact: second_status})
    store.batch.accepted_conflicts[0].fact_statuses = fact_statuses
    store.batch.shown_fact_ids = set(facts)
    await write_conflicts.write_conflicts([store.batch], store.ctx)
    assert store.edges[(store.cid, store.alice, "conflict_value")]["status"] == "conflicting"
    assert store.edges[(store.cid, store.bob, "conflict_value")]["status"] == "superseded"


@pytest.mark.asyncio
async def test_stored_metadata_read_as_json_text_is_decoded(store):
    """Neo4j returns a node's metadata as JSON text; the pending copy needs it as a dict."""
    stored = store.scope.conflicts[store.cid]
    metadata = stored["metadata"]
    stored["metadata"] = json.dumps(metadata)
    await write_conflicts.write_conflicts([store.batch], store.ctx)
    pending_nodes = store.calls[0][0]  # marking pending is the batch's first write
    assert [node.metadata for node in pending_nodes] == [metadata]
