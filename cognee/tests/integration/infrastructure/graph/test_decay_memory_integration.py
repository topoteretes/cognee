"""Integration test: decay_memory against a REAL kuzu graph (no graph mocks).

Inserts DataPoint nodes with controlled feedback_weight + age into a temp kuzu DB, runs the task,
and asserts weights decayed (and persisted) + a stale orphan was actually deleted from the graph.
"""

import pytest

from cognee.infrastructure.databases.graph.kuzu.adapter import KuzuAdapter
from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.tasks.memify.decay_memory import decay_memory, _MS_PER_DAY

NOW_MS = 2_000_000_000_000


class _Node(DataPoint):
    name: str = ""
    metadata: dict = {"index_fields": ["name"]}


@pytest.mark.asyncio
async def test_decay_memory_against_real_kuzu_graph(tmp_path, monkeypatch):
    adapter = KuzuAdapter(db_path=str(tmp_path / "kuzu_decay"))

    fresh = _Node(name="fresh", feedback_weight=0.8, updated_at=NOW_MS)
    old = _Node(name="old", feedback_weight=0.8, updated_at=NOW_MS - 30 * _MS_PER_DAY)      # one half-life
    stale = _Node(name="stale", feedback_weight=0.1, updated_at=NOW_MS - 300 * _MS_PER_DAY)  # old + low -> prune
    await adapter.add_nodes([fresh, old, stale])

    async def _engine():
        return adapter

    monkeypatch.setattr("cognee.tasks.memify.decay_memory.get_graph_engine", _engine)

    result = await decay_memory(half_life_days=30, min_weight=0.05, dry_run=False, now_ms=NOW_MS)

    assert result["scanned"] == 3
    assert result["pruned"] == 1
    assert str(stale.id) in result["pruned_ids"]

    weights = await adapter.get_node_feedback_weights([str(fresh.id), str(old.id)])
    assert abs(weights[str(fresh.id)] - 0.8) < 1e-9   # age 0 -> unchanged
    assert abs(weights[str(old.id)] - 0.4) < 1e-9     # one half-life -> halved, persisted to the DB

    remaining = {node[1].get("name") for node in (await adapter.get_graph_data())[0]}
    assert remaining == {"fresh", "old"}              # the stale orphan was deleted from the real graph
