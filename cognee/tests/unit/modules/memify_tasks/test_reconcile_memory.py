from unittest.mock import AsyncMock, patch

import pytest

from cognee.tasks.memify.reconcile_memory import (
    reconcile_memory,
    candidate_pairs,
    pick_current,
    _node_text,
    ContradictionVerdict,
    SUPERSEDES,
    PREFER_RECENCY,
    PREFER_FEEDBACK,
)


# ----------------------------- graph mock -----------------------------
class InMemoryGraph:
    """Minimal graph engine mock — same shape reconcile_memory consumes (get_graph_data) / writes."""

    def __init__(self, nodes, edges):
        self._nodes = nodes
        self._edges = edges
        self._weights = {nid: props.get("feedback_weight", 0.5) for nid, props in nodes}
        self.weight_writes = {}
        self.edge_writes = []

    async def get_graph_data(self):
        return self._nodes, self._edges

    async def get_node_feedback_weights(self, node_ids):
        return {nid: self._weights[nid] for nid in node_ids if nid in self._weights}

    async def set_node_feedback_weights(self, updates):
        self.weight_writes = dict(updates)
        return {nid: True for nid in updates}

    async def add_edge(self, source, target, relationship_name, edge_properties=None):
        self.edge_writes.append((source, target, relationship_name, edge_properties or {}))


def _patch_engine(mock):
    return patch(
        "cognee.tasks.memify.reconcile_memory.get_graph_engine",
        AsyncMock(return_value=mock),
    )


def _node(node_id, name, weight=0.5, updated_at=None, node_type=None):
    props = {"name": name, "feedback_weight": weight}
    if updated_at is not None:
        props["updated_at"] = updated_at
    if node_type:
        props["type"] = node_type
    return (node_id, props)


def _stub_judge(contradict_when_both_contain="reports to"):
    async def judge(text_a, text_b):
        both = (
            contradict_when_both_contain in text_a.lower()
            and contradict_when_both_contain in text_b.lower()
        )
        return ContradictionVerdict(
            contradicts=both, confidence=0.95 if both else 0.0, reason="conflict" if both else ""
        )

    return judge


# ----------------------------- pure core -----------------------------
def test_candidate_pairs_share_a_neighbour():
    # old and new both point at 'subject' -> they are a candidate pair; 'other' shares nothing -> not paired
    nodes_ids = ["old", "new", "subject", "other"]
    edges = [("old", "subject", "about", {}), ("new", "subject", "about", {})]
    pairs = candidate_pairs(nodes_ids, edges, max_pairs=10)
    assert ("new", "old") in pairs or ("old", "new") in pairs
    assert all("other" not in p for p in pairs)


def test_candidate_pairs_ignores_self_loops_and_caps():
    nodes_ids = ["a", "b", "c", "hub"]
    edges = [
        ("a", "a", "SELF", {}),
        ("a", "hub", "x", {}),
        ("b", "hub", "x", {}),
        ("c", "hub", "x", {}),
    ]
    pairs = candidate_pairs(nodes_ids, edges, max_pairs=2)
    assert len(pairs) == 2  # capped (a-b, a-c, b-c would be 3 uncapped)
    assert ("a", "a") not in pairs


def test_pick_current_recency_then_feedback_fallback():
    # recency: higher updated_at wins
    assert pick_current({"updated_at": 200}, {"updated_at": 100}, PREFER_RECENCY) == "a"
    assert pick_current({"updated_at": 100}, {"updated_at": 200}, PREFER_RECENCY) == "b"
    # recency falls back to feedback when timestamps are missing
    assert pick_current({"feedback_weight": 0.9}, {"feedback_weight": 0.2}, PREFER_RECENCY) == "a"


def test_pick_current_feedback():
    assert pick_current({"feedback_weight": 0.2}, {"feedback_weight": 0.8}, PREFER_FEEDBACK) == "b"
    assert pick_current({"feedback_weight": 0.8}, {"feedback_weight": 0.2}, PREFER_FEEDBACK) == "a"


def test_node_text_builds_from_fields_and_skips_empty():
    assert (
        _node_text({"name": "Alice", "description": "reports to Bob"}) == "Alice — reports to Bob"
    )
    assert _node_text({"type": "EntityType"}) == ""


# ----------------------------- task: dry-run safety -----------------------------
@pytest.mark.asyncio
async def test_dry_run_reports_without_mutating():
    nodes = [
        _node("old", "Alice reports to Bob", weight=0.5),
        _node("new", "Alice reports to Carol", weight=0.9),
        _node("subject", "Alice"),
    ]
    edges = [("old", "subject", "about", {}), ("new", "subject", "about", {})]
    graph = InMemoryGraph(nodes, edges)
    with _patch_engine(graph):
        result = await reconcile_memory(prefer=PREFER_FEEDBACK, dry_run=True, judge=_stub_judge())
    assert result["contradictions"] == 1
    assert result["superseded"] == 1
    assert graph.edge_writes == []  # dry-run -> nothing written
    assert graph.weight_writes == {}


# ----------------------------- task: real supersession -----------------------------
@pytest.mark.asyncio
async def test_supersedes_edge_current_to_stale_and_demotes():
    nodes = [
        _node("old", "Alice reports to Bob", weight=0.5),
        _node("new", "Alice reports to Carol", weight=0.9),
        _node("subject", "Alice"),
    ]
    edges = [("old", "subject", "about", {}), ("new", "subject", "about", {})]
    graph = InMemoryGraph(nodes, edges)
    with _patch_engine(graph):
        result = await reconcile_memory(
            prefer=PREFER_FEEDBACK, demote_factor=0.5, dry_run=False, judge=_stub_judge()
        )
    assert result["superseded"] == 1
    # edge: current (new, higher feedback) -> stale (old), relationship 'supersedes'
    assert len(graph.edge_writes) == 1
    src, tgt, rel, _props = graph.edge_writes[0]
    assert (src, tgt, rel) == ("new", "old", SUPERSEDES)
    # stale node demoted: 0.5 * 0.5 = 0.25; current untouched
    assert graph.weight_writes == {"old": 0.25}


@pytest.mark.asyncio
async def test_below_confidence_threshold_is_skipped():
    nodes = [
        _node("old", "Alice reports to Bob", weight=0.5),
        _node("new", "Alice reports to Carol", weight=0.9),
        _node("subject", "Alice"),
    ]
    edges = [("old", "subject", "about", {}), ("new", "subject", "about", {})]
    graph = InMemoryGraph(nodes, edges)

    async def weak_judge(a, b):
        return ContradictionVerdict(contradicts=True, confidence=0.3, reason="maybe")

    with _patch_engine(graph):
        result = await reconcile_memory(confidence_threshold=0.6, dry_run=False, judge=weak_judge)
    assert result["superseded"] == 0
    assert graph.edge_writes == []
    assert graph.weight_writes == {}


# ---------------- candidate_pairs via a TEXT-LESS shared neighbour (the recall-gap fix) ----------------
def test_candidate_pairs_via_textless_shared_neighbour():
    # two claims linked only through a structural node ('chunk') that has no text and is NOT scannable;
    # they must still be paired (the shared-neighbour universe is the whole graph, endpoints stay scannable)
    pairs = candidate_pairs(
        [
            "old",
            "new",
        ],  # scannable claim endpoints only ('chunk' intentionally absent from this set)
        [("old", "chunk", "mentioned_in", {}), ("new", "chunk", "mentioned_in", {})],
        max_pairs=10,
    )
    assert ("new", "old") in pairs or ("old", "new") in pairs


# ---------------- task: prefer="recency" (the DEFAULT path), end-to-end ----------------
@pytest.mark.asyncio
async def test_prefer_recency_supersedes_newer_over_older():
    nodes = [
        _node("old", "Alice reports to Bob", weight=0.5, updated_at=100),
        _node(
            "new", "Alice reports to Carol", weight=0.5, updated_at=200
        ),  # equal weight -> recency decides
        _node("subject", "Alice"),
    ]
    edges = [("old", "subject", "about", {}), ("new", "subject", "about", {})]
    graph = InMemoryGraph(nodes, edges)
    with _patch_engine(graph):
        result = await reconcile_memory(prefer=PREFER_RECENCY, dry_run=False, judge=_stub_judge())
    assert result["superseded"] == 1
    src, tgt, rel, _props = graph.edge_writes[0]
    assert (src, tgt, rel) == ("new", "old", SUPERSEDES)  # newer (updated_at=200) supersedes older
    assert graph.weight_writes == {"old": 0.25}


# ---------------- task: protect_node_types excludes a protected node ----------------
@pytest.mark.asyncio
async def test_protect_node_types_skips_protected_claim():
    nodes = [
        _node("old", "Alice reports to Bob", weight=0.5, node_type="Protected"),
        _node("new", "Alice reports to Carol", weight=0.9),
        _node("subject", "Alice"),
    ]
    edges = [("old", "subject", "about", {}), ("new", "subject", "about", {})]
    graph = InMemoryGraph(nodes, edges)
    with _patch_engine(graph):
        result = await reconcile_memory(
            prefer=PREFER_FEEDBACK,
            dry_run=False,
            protect_node_types=["Protected"],
            judge=_stub_judge(),
        )
    # the protected node is excluded -> no candidate pair -> nothing superseded
    assert result["superseded"] == 0
    assert graph.edge_writes == []
    assert graph.weight_writes == {}


# ---------------- task: a stale node is superseded ONCE even if it loses multiple pairs ----------------
@pytest.mark.asyncio
async def test_stale_node_is_superseded_only_once():
    # 'old' contradicts BOTH 'new1' and 'new2' (all three share 'subject'); both newer claims outrank it.
    # Without the resolved_stale guard, 'old' would get two supersedes edges and be demoted twice (0.125).
    nodes = [
        _node("old", "Alice reports to Bob", weight=0.5),
        _node("new1", "Alice reports to Carol", weight=0.9),
        _node("new2", "Alice reports to Dave", weight=0.8),
        _node("subject", "Alice"),
    ]
    edges = [
        ("old", "subject", "about", {}),
        ("new1", "subject", "about", {}),
        ("new2", "subject", "about", {}),
    ]
    graph = InMemoryGraph(nodes, edges)
    with _patch_engine(graph):
        result = await reconcile_memory(prefer=PREFER_FEEDBACK, dry_run=False, judge=_stub_judge())

    # all three pairs are judged contradictory, but 'old' is superseded only once
    assert result["contradictions"] == 3
    assert (
        result["superseded"] == 2
    )  # (new1 -> old) once, plus (new1 -> new2); NOT (new2 -> old) again
    stale_targets = [tgt for _src, tgt, _rel, _props in graph.edge_writes]
    assert stale_targets.count("old") == 1  # exactly one supersedes edge into 'old'
    assert graph.weight_writes["old"] == 0.25  # demoted once (0.5*0.5), not twice (0.125)
