"""Integration test: reconcile_memory against a REAL ladybug/kuzu graph (no graph mocks).

Inserts two contradictory claims about the same subject into a temp graph, runs the task with a STUBBED
LLM verdict (so the test is deterministic + offline — only the contradiction *decision* is stubbed; the
edge write + weight demotion + persistence are all real), and asserts the GRAPH-LEVEL outcome:
  - a `supersedes` edge is added from the CURRENT claim to the STALE one,
  - the stale node's feedback_weight is demoted (and persisted below the current's),
  - dry_run mutates nothing.

The acceptance bullet "a subsequent recall surfaces the current claim" is NOT asserted here — it follows
from feedback-weighted retrieval once `feedback_influence > 0` (off by default); this test proves the
demotion that produces that preference, not a live search ranking.
"""

import pytest

from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter
from cognee.infrastructure.engine import DataPoint
import cognee.tasks.memify.reconcile_memory as _reconcile_mod
from cognee.tasks.memify.reconcile_memory import (
    reconcile_memory,
    ContradictionVerdict,
    SUPERSEDES,
    PREFER_FEEDBACK,
)


class _ClaimTs(DataPoint):
    name: str = ""
    updated_at: int = (
        0  # epoch ms; lets the recency path be tested with distinct, controlled timestamps
    )
    metadata: dict = {"index_fields": ["name"]}


class _Claim(DataPoint):
    name: str = ""
    metadata: dict = {"index_fields": ["name"]}


async def _stub_judge(text_a, text_b):
    """Two 'reports to <manager>' claims contradict; anything else does not. Deterministic, no LLM call."""
    both = "reports to" in text_a.lower() and "reports to" in text_b.lower()
    return ContradictionVerdict(
        contradicts=both,
        confidence=0.95 if both else 0.0,
        reason="different manager" if both else "",
    )


async def _seed(tmp_path):
    adapter = LadybugAdapter(str(tmp_path / "kuzu_reconcile"))
    subject = _Claim(name="Alice")
    old = _Claim(name="Alice reports to Bob")
    new = _Claim(name="Alice reports to Carol")
    await adapter.add_nodes([subject, old, new])
    # both claims are 'about' the subject -> they share a neighbour -> candidate contradiction pair
    await adapter.add_edge(str(old.id), str(subject.id), "about", {})
    await adapter.add_edge(str(new.id), str(subject.id), "about", {})
    # set authoritative feedback weights: the 'new' claim is the trusted/current one
    await adapter.set_node_feedback_weights({str(old.id): 0.5, str(new.id): 0.9})
    return adapter, subject, old, new


@pytest.mark.asyncio
async def test_reconcile_supersedes_on_real_graph(tmp_path, monkeypatch):
    adapter, subject, old, new = await _seed(tmp_path)

    async def _engine():
        return adapter

    monkeypatch.setattr(_reconcile_mod, "get_graph_engine", _engine)

    result = await reconcile_memory(
        prefer=PREFER_FEEDBACK,
        confidence_threshold=0.6,
        demote_factor=0.5,
        dry_run=False,
        judge=_stub_judge,
    )

    assert result["contradictions"] == 1
    assert result["superseded"] == 1

    # a real `supersedes` edge now exists, current (new) -> stale (old)
    _, edges = await adapter.get_graph_data()
    supersedes_edges = [(s, t) for (s, t, rel, _p) in edges if rel == SUPERSEDES]
    assert (str(new.id), str(old.id)) in supersedes_edges

    # the stale node's feedback_weight was demoted (0.5 * 0.5 = 0.25) and PERSISTED; current untouched
    weights = await adapter.get_node_feedback_weights([str(old.id), str(new.id)])
    assert abs(weights[str(old.id)] - 0.25) < 1e-6
    assert abs(weights[str(new.id)] - 0.9) < 1e-6


@pytest.mark.asyncio
async def test_reconcile_prefers_recency_on_real_graph(tmp_path, monkeypatch):
    # the DEFAULT prefer="recency" path, end-to-end on a real graph: with EQUAL feedback weights, the
    # newer claim (higher updated_at) must supersede the older one — so recency reads persisted int-ms
    # timestamps off the real graph, not just the in-memory mocks.
    adapter = LadybugAdapter(str(tmp_path / "kuzu_reconcile_recency"))
    subject = _ClaimTs(name="Bob", updated_at=1)
    older = _ClaimTs(name="Bob reports to Dave", updated_at=1_000)
    newer = _ClaimTs(name="Bob reports to Erin", updated_at=2_000)
    await adapter.add_nodes([subject, older, newer])
    await adapter.add_edge(str(older.id), str(subject.id), "about", {})
    await adapter.add_edge(str(newer.id), str(subject.id), "about", {})
    # EQUAL feedback weights -> only recency can pick the winner
    await adapter.set_node_feedback_weights({str(older.id): 0.5, str(newer.id): 0.5})

    async def _engine():
        return adapter

    monkeypatch.setattr(_reconcile_mod, "get_graph_engine", _engine)

    result = await reconcile_memory(
        dry_run=False, judge=_stub_judge
    )  # prefer defaults to "recency"
    assert result["superseded"] == 1

    _, edges = await adapter.get_graph_data()
    supersedes_edges = [(s, t) for (s, t, rel, _p) in edges if rel == SUPERSEDES]
    assert (
        str(newer.id),
        str(older.id),
    ) in supersedes_edges  # newer (updated_at=2000) supersedes older


@pytest.mark.asyncio
async def test_dry_run_does_not_mutate_real_graph(tmp_path, monkeypatch):
    adapter, subject, old, new = await _seed(tmp_path)

    async def _engine():
        return adapter

    monkeypatch.setattr(_reconcile_mod, "get_graph_engine", _engine)

    result = await reconcile_memory(prefer=PREFER_FEEDBACK, dry_run=True, judge=_stub_judge)

    assert result["contradictions"] == 1  # detected
    assert result["superseded"] == 1  # reported as intended
    # ...but nothing was written
    _, edges = await adapter.get_graph_data()
    assert not [e for e in edges if e[2] == SUPERSEDES]
    weights = await adapter.get_node_feedback_weights([str(old.id)])
    assert abs(weights[str(old.id)] - 0.5) < 1e-6  # unchanged
