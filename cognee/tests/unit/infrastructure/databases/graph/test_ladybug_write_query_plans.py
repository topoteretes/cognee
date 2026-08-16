"""The bulk-write statements must reach nodes through the primary-key index.

This is the test that would have caught COG-6216. ``Node.id`` is declared
``STRING PRIMARY KEY``, so the writes look like index seeks — but Kuzu plans
``MERGE`` as SCAN_NODE_TABLE + HASH_JOIN and ignores the index entirely. One
scanning statement per 2000-row chunk turns a bulk ingest into O(N^2): the
100k-node nightly performance suite ran for over three hours without
finishing, while the same graph on Postgres took seven minutes.

Wall-clock assertions cannot catch this — on a small graph a scan and a seek
are indistinguishable, which is exactly the regime unit tests run in. So
assert the PLAN instead. A Kuzu upgrade that changes the planner fails here
rather than silently reintroducing the quadratic.
"""

import re

import pytest

from cognee.infrastructure.databases.graph.ladybug.adapter import (
    LadybugAdapter,
    _EDGE_CREATE_QUERY,
    _EDGE_PROBE_QUERY,
    _EDGE_UPDATE_QUERY,
    _NODE_CREATE_QUERY,
    _NODE_PROBE_QUERY,
    _NODE_UPDATE_QUERY,
)

NOW = "2026-01-01 00:00:00.000000"
IDS = [f"00000000-0000-0000-0000-{index:012d}" for index in range(20)]
NODES = [
    {
        "id": node_id,
        "name": "n",
        "type": "T",
        "properties": "{}",
        "created_at": NOW,
        "updated_at": NOW,
    }
    for node_id in IDS
]
EDGES = [
    {
        "from_id": IDS[index],
        "to_id": IDS[index + 1],
        "relationship_name": "rel",
        "properties": "{}",
        "created_at": NOW,
        "updated_at": NOW,
    }
    for index in range(0, len(IDS) - 1, 2)
]


@pytest.fixture
def adapter(tmp_path):
    return LadybugAdapter(str(tmp_path / "graph_db"))


async def _plan(adapter, query, params):
    rows = await adapter.query("EXPLAIN " + query, params)
    return "\n".join(str(cell) for row in rows for cell in row)


def _count(plan, operator):
    # Guard the prefix so SCAN_NODE_TABLE does not also match
    # PRIMARY_KEY_SCAN_NODE_TABLE.
    return len(re.findall(rf"(?<![A-Z_]){operator}\[", plan))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name, query, params",
    [
        ("node probe", _NODE_PROBE_QUERY, {"ids": IDS}),
        ("node update", _NODE_UPDATE_QUERY, {"nodes": NODES}),
        ("edge create", _EDGE_CREATE_QUERY, {"edges": EDGES}),
    ],
)
async def test_node_lookups_are_primary_key_seeks(adapter, name, query, params):
    plan = await _plan(adapter, query, params)

    assert _count(plan, "QUERY_PRIMARY_KEY_LOOKUP") > 0, f"{name} lost the index:\n{plan}"
    assert _count(plan, "SCAN_NODE_TABLE") == 0, f"{name} scans the node table:\n{plan}"


@pytest.mark.asyncio
async def test_node_create_touches_no_node_table(adapter):
    """A pure insert needs neither a seek nor a scan."""
    plan = await _plan(adapter, _NODE_CREATE_QUERY, {"nodes": NODES})

    assert _count(plan, "SCAN_NODE_TABLE") == 0, plan


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query", [_NODE_CREATE_QUERY, _NODE_UPDATE_QUERY, _EDGE_CREATE_QUERY, _EDGE_UPDATE_QUERY]
)
async def test_writes_never_use_merge(query):
    """MERGE is the operator that drops the primary-key index. Keeping it out
    of the write statements is the whole fix, so state it directly — a plan
    assertion alone would not explain the failure to whoever reintroduces it.
    """
    assert "MERGE" not in query


@pytest.mark.asyncio
async def test_edge_endpoints_use_separate_match_clauses(adapter):
    """One comma-separated MATCH of both endpoints plans as a cartesian product
    of two scans; separate clauses plan as two seeks."""
    plan = await _plan(adapter, _EDGE_CREATE_QUERY, {"edges": EDGES})

    assert _count(plan, "QUERY_PRIMARY_KEY_LOOKUP") == 2, plan
    assert _count(plan, "CROSS_PRODUCT") == 0, plan


@pytest.mark.asyncio
async def test_edge_probe_expands_outgoing_edges_only(adapter):
    """The probe walks the source nodes' outgoing edges instead of matching
    both endpoints plus the relationship, which plans as a scan even with both
    endpoints bound. Cost is the sum of those out-degrees — and cognee's hub
    nodes (EntityType, via `is_a`) are hubs by IN-degree, which this never
    touches.
    """
    assert "-[r:EDGE]->(to:Node)" in _EDGE_PROBE_QUERY
    plan = await _plan(adapter, _EDGE_PROBE_QUERY, {"ids": IDS})
    assert _count(plan, "SCAN_REL_TABLE") <= 1, plan
