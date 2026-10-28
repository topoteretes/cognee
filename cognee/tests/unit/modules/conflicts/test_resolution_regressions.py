"""Reproductions of the review findings on the conflict resolution API (SDK-1328).

Each test encodes one scenario from the review and fails on the branch today, with
no expected-failure marker, so the failures are visible in CI. A test turns green
once its finding is fixed.
"""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.modules.conflicts import methods
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.graph.utils.fact_conflicts import is_rejected_fact
from cognee.modules.graph.utils.resolve_edges_to_text import resolve_edges_to_text
from cognee.modules.retrieval.utils import conflict_context
from cognee.modules.retrieval.utils.conflict_context import attach_chunk_conflicts
from cognee.modules.retrieval.utils.resolved_context import (
    accepted_facts_context,
    apply_graph_resolutions,
    user_resolution_text,
)
from cognee.modules.session_lifecycle import invalidate_sessions
from cognee.tasks.memify.review_conflicts import write_conflicts
from cognee.tasks.memify.review_conflicts.models import ConflictsToFinalize
from cognee.tests.unit.modules.conflicts.test_resolve_conflict import lane, resolve
from cognee.tests.unit.tasks.memify.review_conflicts.test_write_conflicts import store


def _fake_session_manager(entries, *, delete_qa_result=True):
    return SimpleNamespace(
        is_available=True,
        get_session=AsyncMock(return_value=entries),
        get_session_context_entries=AsyncMock(return_value=[]),
        delete_qa=AsyncMock(return_value=delete_qa_result),
        delete_session_context_entry=AsyncMock(return_value=True),
    )


def _use_real_invalidation(monkeypatch, manager, sessions):
    @asynccontextmanager
    async def no_lock(*args, **kwargs):
        yield

    monkeypatch.setattr(invalidate_sessions, "get_session_manager", lambda: manager)
    monkeypatch.setattr(
        invalidate_sessions, "list_sessions_for_dataset", AsyncMock(return_value=sessions)
    )
    monkeypatch.setattr(
        invalidate_sessions, "list_unattributed_sessions", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(invalidate_sessions, "get_persisted_qa_count", AsyncMock(return_value=0))
    monkeypatch.setattr(invalidate_sessions, "save_persisted_qa_count", AsyncMock())
    monkeypatch.setattr(invalidate_sessions, "session_lock", no_lock)


# Finding 1: write_conflicts.py `_preserve_user_resolutions` -----------------------------


@pytest.mark.asyncio
async def test_review_does_not_reject_new_evidence_the_user_never_saw(lane):  # noqa: F811
    """The user picked Alice over Bob. A later review reports the same conflict with a new
    fact, Carol, the user never saw: preserving the choice must not mark Carol superseded.
    """
    selected = await resolve(lane)
    facts = lane.store.scope.facts
    carol = str(uuid4())
    facts[carol] = {
        **next(iter(facts.values())),
        "id": carol,
        "target": str(uuid4()),
        "properties": {"edge_text": "Company has CEO Carol", "weight": 1, "conflict_marks": []},
    }
    batch = lane.store.batch  # an automatic review, resolution_source != "user"
    incoming = batch.accepted_conflicts[0]
    incoming.fact_statuses = {fid: ("current" if fid == carol else "superseded") for fid in facts}
    batch.shown_fact_ids = set(facts)
    await write_conflicts.write_conflicts([batch], lane.store.ctx)
    await write_conflicts.write_conflicts(
        [ConflictsToFinalize(batch.accepted_conflicts, batch.dropped_conflict_ids)], lane.store.ctx
    )
    saved = lane.store.nodes[lane.store.cid]
    assert saved["selected_fact_id"] == str(selected.selected_fact_id)
    assert carol not in saved["rejected_fact_ids"]
    assert "Company has CEO Carol" not in saved["rejected_fact_texts"]
    assert not is_rejected_fact(facts[carol]["properties"])


# Finding 2: methods.py `resolve_conflict` session invalidation --------------------------


@pytest.mark.asyncio
async def test_resolving_a_conflict_keeps_cached_answers_about_other_attributes(
    lane,  # noqa: F811
    monkeypatch,
):
    """A cached answer about revenue cites the Company node but neither competing CEO
    fact: resolving the CEO conflict must leave it in the session cache.
    """
    about_id = lane.store.nodes[lane.store.cid]["about_id"]
    revenue_turn = SimpleNamespace(
        qa_id="revenue",
        used_graph_element_ids={"node_ids": [about_id], "edge_ids": ["revenue-edge"]},
        used_session_context_ids=[],
    )
    manager = _fake_session_manager([revenue_turn])
    _use_real_invalidation(monkeypatch, manager, [(lane.user.id, "s1")])
    monkeypatch.setattr(
        methods,
        "invalidate_sessions_for_deleted_data",
        invalidate_sessions.invalidate_sessions_for_deleted_data,
    )
    await resolve(lane)
    manager.delete_qa.assert_not_awaited()


# Finding 3: conflict_context.py `attach_chunk_conflicts` guard ---------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("engine_given", [True, False])
async def test_chunk_search_survives_an_unreadable_graph(monkeypatch, engine_given):
    """A deployment whose graph engine cannot be read (vector-only, adapter without
    get_filtered_graph_data) still returns its chunk hits, as get_chunk_conflicts does.
    """
    hits = [{"id": "chunk", "text": "Bob is CEO. Revenue grew."}]
    graph = SimpleNamespace(
        get_filtered_graph_data=AsyncMock(side_effect=NotImplementedError("no graph")),
        get_neighborhood=AsyncMock(side_effect=NotImplementedError("no graph")),
    )
    monkeypatch.setattr(
        conflict_context, "get_graph_engine", AsyncMock(side_effect=RuntimeError("no graph"))
    )
    await attach_chunk_conflicts(hits, graph_engine=graph if engine_given else None)
    assert hits == [{"id": "chunk", "text": "Bob is CEO. Revenue grew."}]


# Finding 6: resolved_context.py `accepted_facts_context` cutoff ---------------------------


def test_every_relevant_user_decision_reaches_the_prelude():
    """Six decisions about retrieved nodes with global_context_index_top_k=5: a decision
    the user made about the retrieved subject is never dropped by a cutoff.
    """
    resolutions = [
        {"about_id": "retrieved", "values": [], "sources": [], "selected_fact": f"Fact {i}"}
        for i in range(6)
    ]
    context = accepted_facts_context(resolutions, ["retrieved"], limit=5)
    missing = [f"Fact {i}" for i in range(6) if user_resolution_text(f"Fact {i}") not in context]
    assert missing == []


# Finding 7: invalidate_sessions.py strict mode -------------------------------------------


@pytest.mark.asyncio
async def test_strict_invalidation_tolerates_an_entry_that_is_already_gone(monkeypatch):
    """A contaminated answer removed by a concurrent cleanup makes delete_qa return False;
    strict cleanup must treat that as done, not as a failure of the resolution.
    """
    stale_turn = SimpleNamespace(
        qa_id="stale",
        used_graph_element_ids={"node_ids": ["conflict"], "edge_ids": []},
        used_session_context_ids=[],
    )
    manager = _fake_session_manager([stale_turn], delete_qa_result=False)
    _use_real_invalidation(monkeypatch, manager, [(uuid4(), "s1")])
    totals = await invalidate_sessions.invalidate_sessions_for_deleted_data(
        uuid4(), deleted_node_ids={"conflict"}, deleted_edge_ids=set(), strict=True
    )
    assert totals["qa_entries_deleted"] == 0
    manager.delete_qa.assert_awaited_once()


# Finding 8: methods.py SDK entry points ignore serve() ----------------------------------


@pytest.mark.asyncio
async def test_sdk_conflict_calls_follow_serve_to_the_remote_instance(monkeypatch):
    """After cognee.serve(url), list_conflicts and resolve_conflict go to the remote
    instance like every other SDK entry point, instead of the empty local stores.
    """
    import cognee
    from cognee.api.v1.serve import state

    client = SimpleNamespace(
        list_conflicts=AsyncMock(return_value=[]), resolve_conflict=AsyncMock(return_value={})
    )
    monkeypatch.setattr(state, "get_remote_client", lambda: client)
    monkeypatch.setattr(methods, "get_default_user", AsyncMock(return_value=SimpleNamespace(id=1)))
    monkeypatch.setattr(methods, "get_authorized_existing_datasets", AsyncMock(return_value=[]))
    dataset_id, conflict_id, fact_id = uuid4(), uuid4(), uuid4()
    await cognee.list_conflicts(dataset_id)
    client.list_conflicts.assert_awaited_once()
    await cognee.resolve_conflict(conflict_id, dataset_id=dataset_id, selected_fact_id=fact_id)
    client.resolve_conflict.assert_awaited_once()


# Finding 10: two mechanisms render one user decision ------------------------------------


@pytest.mark.asyncio
async def test_a_user_decision_is_stated_once_in_graph_context():
    """The surviving Alice edge carries the user's mark and the rejected Bob edge is
    replaced by the resolution: the decision text must be stated once, not twice.
    """
    company = Node("company", {"name": "Company"})
    decision = {
        "id": "conflict",
        "about_id": "company",
        "selected_fact": "Company has CEO Alice",
        "selected_fact_id": "alice-fact",
        "rejected_fact_ids": ["bob-fact"],
        "sources": [],
        "values": [],
    }
    mark = {
        "conflict_id": "conflict",
        "resolution_source": "user",
        "selected_fact": "Company has CEO Alice",
    }
    alice = Edge(
        company,
        Node("alice", {"name": "Alice"}),
        attributes={
            "edge_object_id": "alice-fact",
            "edge_text": "Company has CEO Alice",
            "conflict_marks": [{**mark, "status": "current"}],
        },
    )
    bob = Edge(
        company,
        Node("bob", {"name": "Bob"}),
        attributes={
            "edge_object_id": "bob-fact",
            "edge_text": "Company has CEO Bob",
            "conflict_marks": [{**mark, "status": "superseded"}],
        },
    )
    text = await resolve_edges_to_text(apply_graph_resolutions([alice, bob], [decision]))
    assert "Company has CEO Bob" not in text
    synthetic = "Accepted fact" in text
    section = "## Accepted facts (user resolved)" in text
    assert not (synthetic and section), text
