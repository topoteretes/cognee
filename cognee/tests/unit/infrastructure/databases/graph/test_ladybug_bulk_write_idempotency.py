"""Writing the same rows twice leaves one row per id or triple, on a real database.

The bulk writes are probe / CREATE-missing / SET-existing. A probe that returns
too few rows sends existing rows to CREATE: a node raises on its primary key,
an edge is silently duplicated because EDGE has no unique key. The plan tests
cannot see that (an empty lookup still prints as QUERY_PRIMARY_KEY_LOOKUP, and
ladybug 0.19.0 has shapes that plan as a seek yet return nothing,
LadybugDB/ladybug#861), so this test writes, writes again, and counts.
"""

from uuid import NAMESPACE_OID, uuid5

import pytest

from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.modules.graph.utils.prepare_edges_for_storage import ensure_default_edge_properties


class Thing(DataPoint):
    name: str
    description: str = ""
    metadata: dict = {"index_fields": ["name"]}


def _things(description: str) -> list[Thing]:
    return [
        Thing(id=uuid5(NAMESPACE_OID, name), name=name, description=description)
        for name in ("alpha", "beta", "gamma")
    ]


def _edges(things, text: str):
    a, b, c = (str(thing.id) for thing in things)
    return ensure_default_edge_properties(
        [
            (a, b, "knows", {"edge_text": text}),
            (b, c, "knows", {"edge_text": text}),
            (a, c, "likes", {}),
        ]
    )


@pytest.fixture
def ladybug(tmp_path):
    pytest.importorskip("ladybug")
    from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter

    return LadybugAdapter(db_path=str(tmp_path / "twice.lbug"))


async def _nodes(ladybug):
    rows = await ladybug.query("MATCH (n:Node) RETURN n.id, n.created_at, n.properties")
    return {row[0]: (row[1], row[2]) for row in rows}


async def _edges_stored(ladybug):
    rows = await ladybug.query(
        "MATCH (a:Node)-[r:EDGE]->(b:Node) "
        "RETURN a.id, b.id, r.relationship_name, r.created_at, r.properties"
    )
    return [((row[0], row[1], row[2]), row[3], row[4]) for row in rows]


@pytest.mark.asyncio
async def test_writing_the_same_rows_twice_updates_them_in_place(ladybug):
    first = _things("first")
    await ladybug.add_nodes(first)
    await ladybug.add_edges(_edges(first, "first"))
    nodes_before = await _nodes(ladybug)
    edges_before = await _edges_stored(ladybug)
    assert len(nodes_before) == 3 and len(edges_before) == 3

    second = _things("second")
    await ladybug.add_nodes(second)
    await ladybug.add_edges(_edges(second, "second"))
    nodes_after = await _nodes(ladybug)
    edges_after = await _edges_stored(ladybug)

    assert set(nodes_after) == set(nodes_before), "a node was created twice or lost"
    for node_id, (created_at, properties) in nodes_after.items():
        assert created_at == nodes_before[node_id][0], "SET must keep created_at"
        assert '"second"' in properties and '"first"' not in properties

    assert sorted(key for key, *_ in edges_after) == sorted(key for key, *_ in edges_before), (
        "an edge was created twice or lost"
    )
    created_before = {key: created_at for key, created_at, _ in edges_before}
    for key, created_at, properties in edges_after:
        assert created_at == created_before[key], "SET must keep created_at"
        if key[2] == "knows":
            assert '"second"' in properties and '"first"' not in properties


@pytest.mark.asyncio
async def test_single_edge_write_is_the_same_upsert(ladybug):
    things = _things("x")
    await ladybug.add_nodes(things)
    a, b = str(things[0].id), str(things[1].id)
    for text in ("one", "two"):
        await ladybug.add_edge(a, b, "knows", {"edge_text": text})

    edges = await _edges_stored(ladybug)
    assert len(edges) == 1
    assert '"two"' in edges[0][2]


@pytest.mark.asyncio
async def test_has_edges_reports_exactly_the_stored_edges(ladybug):
    things = _things("x")
    await ladybug.add_nodes(things)
    await ladybug.add_edges(_edges(things, "x"))
    a, b, c = (str(thing.id) for thing in things)

    found = await ladybug.has_edges(
        [(a, b, "knows"), (b, c, "knows"), (a, c, "likes"), (c, a, "knows")]
    )

    assert sorted(found) == sorted([(a, b, "knows"), (b, c, "knows"), (a, c, "likes")])


@pytest.mark.asyncio
async def test_edge_lookup_split_over_statements_is_still_one_upsert(ladybug, monkeypatch):
    """With the per-statement cap below the call's source count, the lookup runs
    several statements and add_edges / has_edges behave exactly as with one:
    no duplicate edge, learned weights kept, has_edges exact."""
    import cognee.infrastructure.databases.graph.ladybug.adapter as mod

    monkeypatch.setattr(mod, "_EDGE_LOOKUP_IDS_PER_STATEMENT", 7)
    things = [
        Thing(id=uuid5(NAMESPACE_OID, f"t{i}"), name=f"t{i}", description="x") for i in range(30)
    ]
    await ladybug.add_nodes(things)
    ids = [str(thing.id) for thing in things]
    edges = ensure_default_edge_properties(
        [(ids[i], ids[(i * 7 + 3) % 30], "knows", {}) for i in range(30)]
        + [(ids[i], ids[0], "is_a", {}) for i in range(1, 30)]
    )
    await ladybug.add_edges(edges)
    weighted = {edges[i][3]["edge_object_id"]: 0.8 for i in (0, 10, 40)}
    await ladybug.set_edge_feedback_weights(weighted)

    statements = []
    real_query = ladybug.query

    async def counting_query(query, params=None, *args, **kwargs):
        if isinstance(params, dict) and "source_ids" in params:
            statements.append(len(params["source_ids"]))
        return await real_query(query, params, *args, **kwargs)

    monkeypatch.setattr(ladybug, "query", counting_query)
    await ladybug.add_edges(edges)

    assert len(statements) == 5 and max(statements) == 7, statements
    assert len(await _edges_stored(ladybug)) == len(edges)
    assert await ladybug.get_edge_feedback_weights(list(weighted)) == weighted
    probe = [(f, t, r) for f, t, r, _ in edges] + [(ids[0], ids[1], "missing")]
    found = await ladybug.has_edges(probe)
    assert sorted(found) == sorted(probe[:-1])
