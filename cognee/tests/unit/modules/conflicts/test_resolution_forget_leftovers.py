"""A forgotten document leaves no trace in a stored user decision (SDK-1328).

Scenario. The user resolved the conflict "who is Company's CEO" by accepting the
Alice fact over the Bob fact. Bob's document is then forgotten: his fact edge, his
value node and the chunk that cited him are gone. The next automatic review of the
subject reports the conflict with Alice alone.

What happens on the branch: ``_preserve_user_resolutions`` prunes ``rejected_fact_ids``
and (since 5403540d3) ``sources`` to what still exists, but it unions
``rejected_fact_texts`` and ``values`` with the stored lists, so "Company has CEO Bob"
and Bob's deleted node id stay on the FactConflict node for good. The stale text keeps
rewriting standalone fact hits with that sentence to the accepted fact
(``apply_fact_hit_resolutions``), and the stale value id keeps the conflict in review
scope for a node that no longer exists. The test fails until every list on the node
describes only facts, values and chunks that still exist.
"""

import pytest

from cognee.tasks.memify.review_conflicts import write_conflicts
from cognee.tasks.memify.review_conflicts.models import ConflictsToFinalize
from cognee.tests.unit.modules.conflicts.test_resolve_conflict import (
    find_fact_id,
    lane,
    resolve,
)
from cognee.tests.unit.tasks.memify.review_conflicts.test_write_conflicts import (
    store,
)


@pytest.mark.asyncio
async def test_forgetting_the_rejected_document_leaves_no_trace_in_the_decision(lane):  # noqa: F811
    alice_fact = find_fact_id(lane.store, lane.store.alice)
    bob_fact = find_fact_id(lane.store, lane.store.bob)
    selected = await resolve(lane, fact=alice_fact)
    stored = lane.store.nodes[lane.store.cid]
    assert stored["rejected_fact_ids"] == [bob_fact]
    assert stored["rejected_fact_texts"] == ["Company has CEO Bob"]
    assert set(stored["values"]) == {lane.store.alice, lane.store.bob}

    # Bob's document is forgotten: his fact, his value node and his chunk are gone.
    del lane.store.scope.facts[bob_fact]
    lane.store.scope.entities.pop(lane.store.bob, None)
    alice_chunks = {s["chunk_id"] for s in lane.store.scope.facts[alice_fact]["sources"]}

    # The next automatic review of the subject sees Alice alone.
    batch = lane.store.batch
    incoming = batch.accepted_conflicts[0]
    incoming.fact_statuses = {alice_fact: "current"}
    incoming.conflict.values = [lane.store.alice]
    incoming.conflict.sources = sorted(alice_chunks)
    batch.shown_fact_ids = {alice_fact}
    lane.store.scope.conflicts[lane.store.cid] = dict(stored)
    await write_conflicts.write_conflicts([batch], lane.store.ctx)
    await write_conflicts.write_conflicts(
        [ConflictsToFinalize(batch.accepted_conflicts, batch.dropped_conflict_ids)], lane.store.ctx
    )

    saved = lane.store.nodes[lane.store.cid]
    assert saved["selected_fact_id"] == str(selected.selected_fact_id)
    leftovers = {
        "rejected_fact_ids": [fid for fid in saved["rejected_fact_ids"] if fid == bob_fact],
        "rejected_fact_texts": [t for t in saved["rejected_fact_texts"] if "Bob" in t],
        "values": [v for v in saved["values"] if v == lane.store.bob],
        "sources": [s for s in saved["sources"] if s not in alice_chunks],
    }
    assert all(not stale for stale in leftovers.values()), leftovers
