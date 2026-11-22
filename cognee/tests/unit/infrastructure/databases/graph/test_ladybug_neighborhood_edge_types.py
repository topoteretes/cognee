"""Ladybug ``get_neighborhood`` with ``edge_types`` at one hop.

An ontology class has an ``is_a`` edge from every entity of its type. Hybrid recall
reads a pinned class with ``edge_types=["realizes"]``, so those edges must be
filtered in the query, not returned to Python and dropped there.
"""

import pytest

MEMBERS = [f"member{index}" for index in range(20)]
EDGES = [
    *((member, "customer", "is_a") for member in MEMBERS),
    ("crm", "customer", "realizes"),
    ("legacy", "customer", "realizes"),
    ("customer", "party", "is_a"),
]


async def _class_with_members(tmp_path):
    """A Ladybug graph with ``customer``, its members, two tables and a parent."""
    pytest.importorskip("ladybug")
    from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter

    adapter = LadybugAdapter(db_path=str(tmp_path / "neighborhood.lbug"))
    for node_id in ["customer", "party", "crm", "legacy", *MEMBERS]:
        await adapter.query(
            "CREATE (:Node {id: $id, name: $id, type: 'Entity', properties: '{}'})",
            {"id": node_id},
        )
    for source, target, relation in EDGES:
        await adapter.query(
            "MATCH (a:Node {id: $s}), (b:Node {id: $t}) "
            "CREATE (a)-[:EDGE {relationship_name: $r, properties: '{}'}]->(b)",
            {"s": source, "t": target, "r": relation},
        )
    return adapter


@pytest.mark.asyncio
async def test_one_hop_returns_only_neighbours_over_the_given_edge_types(tmp_path):
    adapter = await _class_with_members(tmp_path)

    nodes, edges = await adapter.get_neighborhood(["customer"], depth=1, edge_types=["realizes"])

    assert {node_id for node_id, _ in nodes} == {"customer", "crm", "legacy"}
    assert sorted((source, target, relation) for source, target, relation, _ in edges) == [
        ("crm", "customer", "realizes"),
        ("legacy", "customer", "realizes"),
    ]


@pytest.mark.asyncio
async def test_one_hop_never_reads_the_edges_of_other_types(tmp_path):
    adapter = await _class_with_members(tmp_path)
    row_counts = []
    query = adapter.query

    async def counting_query(cypher, params=None):
        rows = await query(cypher, params)
        row_counts.append(len(rows))
        return rows

    adapter.query = counting_query
    await adapter.get_neighborhood(["customer"], depth=1, edge_types=["realizes"])

    # Twenty members point at the class; no query brings their rows back.
    assert max(row_counts) < len(MEMBERS)
