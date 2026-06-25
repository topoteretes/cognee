from unittest.mock import AsyncMock, patch

import pytest

from cognee.tasks.memify.decay_memory import (
    decay_memory,
    decay_weight,
    node_degrees,
    select_prune,
    _MS_PER_DAY,
)

NOW_MS = 1_000_000_000_000  # fixed clock for deterministic tests
HALF_LIFE_DAYS = 30


class InMemoryGraph:
    """Minimal graph engine mock — same shape decay_memory consumes (get_graph_data) / writes."""

    def __init__(self, nodes, edges):
        self._nodes = nodes
        self._edges = edges
        # authoritative weights, derived from each node's feedback_weight prop
        self._weights = {nid: props.get("feedback_weight", 0.5) for nid, props in nodes}
        self.weight_writes = {}
        self.deleted = []

    async def get_graph_data(self):
        return self._nodes, self._edges

    async def get_node_feedback_weights(self, node_ids):
        return {nid: self._weights[nid] for nid in node_ids if nid in self._weights}

    async def set_node_feedback_weights(self, updates):
        self.weight_writes = dict(updates)
        return {nid: True for nid in updates}

    async def delete_nodes(self, node_ids):
        self.deleted = list(node_ids)


def _node(node_id, weight, age_days, node_type=None):
    props = {"feedback_weight": weight, "updated_at": NOW_MS - int(age_days * _MS_PER_DAY)}
    if node_type:
        props["type"] = node_type
    return (node_id, props)


def _patch_engine(mock):
    return patch(
        "cognee.tasks.memify.decay_memory.get_graph_engine",
        AsyncMock(return_value=mock),
    )


# ----------------------- pure core -----------------------
def test_decay_is_deterministic_half_life():
    one_hl = HALF_LIFE_DAYS * _MS_PER_DAY
    assert decay_weight(0.8, 0, one_hl) == 0.8                  # no age -> unchanged
    assert abs(decay_weight(0.8, one_hl, one_hl) - 0.4) < 1e-9  # one half-life -> halves
    assert abs(decay_weight(0.8, 2 * one_hl, one_hl) - 0.2) < 1e-9
    assert decay_weight(0.5, 5, 0) == 0.5                       # guard: half_life 0 -> no decay


def test_degrees_and_prune_selection():
    edges = [("hub", "leaf", "REL", {}), ("hub", "mid", "REL", {}), ("mid", "out", "REL", {})]
    degrees = node_degrees(["orphan", "leaf", "mid", "hub"], edges)
    assert degrees == {"orphan": 0, "leaf": 1, "mid": 2, "hub": 2}
    # a self-loop (Cognee writes a SELF edge per node) does not count as a connection -> orphan
    assert node_degrees(["a"], [("a", "a", "SELF", {})]) == {"a": 0}
    pruned = select_prune({"orphan": 0.01, "leaf": 0.01, "mid": 0.9}, degrees, min_weight=0.05)
    assert "mid" not in pruned          # above threshold -> kept
    assert pruned[0] == "orphan"        # orphan-first (lowest degree)
    # a below-threshold but well-connected node is NEVER pruned (can't orphan a live subgraph)
    pruned_hub = select_prune({"orphan": 0.01, "hub": 0.01}, {"orphan": 0, "hub": 3}, min_weight=0.05)
    assert pruned_hub == ["orphan"]


# ----------------------- task (mocked graph) -----------------------
@pytest.mark.asyncio
async def test_dry_run_reports_without_mutating():
    nodes = [_node("fresh", 0.8, 0), _node("old", 0.8, HALF_LIFE_DAYS)]  # 'old' aged exactly one half-life
    graph = InMemoryGraph(nodes, [])
    with _patch_engine(graph):
        result = await decay_memory(
            half_life_days=HALF_LIFE_DAYS, min_weight=0.05, dry_run=True, now_ms=NOW_MS
        )
    assert result["scanned"] == 2
    assert result["decayed"] == 1          # only 'old' changed
    assert result["dry_run"] is True
    assert graph.weight_writes == {}       # dry_run -> graph untouched
    assert graph.deleted == []


@pytest.mark.asyncio
async def test_prune_orphans_below_threshold_and_protect_types():
    nodes = [
        _node("stale_orphan", 0.1, 300),                       # very old + small -> below min, orphan
        _node("protected", 0.1, 300, node_type="EntityType"),  # would qualify, but type is protected
        _node("fresh", 0.9, 0),                                # stays
    ]
    graph = InMemoryGraph(nodes, edges=[])  # stale_orphan has degree 0
    with _patch_engine(graph):
        result = await decay_memory(
            half_life_days=HALF_LIFE_DAYS,
            min_weight=0.05,
            protect_node_types=["EntityType"],
            dry_run=False,
            now_ms=NOW_MS,
        )
    assert "protected" not in result["pruned_ids"]   # protected type never scanned/pruned
    assert "stale_orphan" in result["pruned_ids"]
    assert "fresh" not in result["pruned_ids"]
    assert graph.deleted == result["pruned_ids"]     # real delete happened (dry_run=False)
