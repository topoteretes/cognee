"""``get_node_ids_by_property`` and ``get_outgoing_edge_keys`` answer what
``get_graph_data`` holds, as a filter over that whole read computes it.

Runs against every in-tree adapter it can reach: Ladybug in-process and in
subprocess mode and Turso always (local files); Postgres and Neo4j when
``.env`` configures them (see conftest). Neptune has no live instance in CI.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
import pytest_asyncio

from cognee.infrastructure.engine import DataPoint
from cognee.tests.integration.infrastructure.graph.conftest import (
    _make_neo4j_adapter,
    _make_postgres_adapter,
)
from cognee.tests.utils.graph_lookups import node_ids_by_property, outgoing_edge_keys

pytestmark = pytest.mark.asyncio


class _Symbol(DataPoint):
    name: str
    repo: str
    metadata: dict = {"index_fields": ["name"]}


class _Line(DataPoint):
    name: str
    line: int
    metadata: dict = {"index_fields": ["name"]}


class _Module(DataPoint):
    name: str
    repo: str
    metadata: dict = {"index_fields": ["name"]}


@pytest_asyncio.fixture(params=["ladybug", "ladybug_subprocess", "turso", "postgres", "neo4j"])
async def lookup_adapter(request, tmp_path):
    if request.param == "ladybug":
        module = pytest.importorskip("cognee.infrastructure.databases.graph.ladybug.adapter")
        adapter = module.LadybugAdapter(str(tmp_path / "graph_db"))
    elif request.param == "ladybug_subprocess":
        module = pytest.importorskip("cognee.infrastructure.databases.graph.ladybug.adapter")
        adapter = module.LadybugAdapter.create_subprocess(db_path=str(tmp_path / "graph_db"))
    elif request.param == "turso":
        pytest.importorskip("turso", reason="pyturso not installed")
        from cognee.infrastructure.databases.graph.turso.adapter import TursoAdapter

        adapter = TursoAdapter(database_path=str(tmp_path / "graph.db"))
        await adapter.initialize()
    elif request.param == "postgres":
        adapter = await _make_postgres_adapter()
    else:
        adapter = await _make_neo4j_adapter()

    try:
        yield adapter
    finally:
        await adapter.close()


async def _seed(adapter) -> dict[str, str]:
    """Symbols and modules of two repos, one name in both; a hub whose three edges
    to one target differ by relationship only. Returns the node ids by key."""
    nodes = {
        "a_sym": _Symbol(id=uuid4(), name="shared", repo="A"),
        "a_mod": _Module(id=uuid4(), name="mod", repo="A"),
        "b_sym": _Symbol(id=uuid4(), name="shared", repo="B"),
        "b_mod": _Module(id=uuid4(), name="other", repo="B"),
    }
    await adapter.add_nodes(list(nodes.values()))
    ids = {key: str(node.id) for key, node in nodes.items()}

    def edge(source, target, relationship):
        source_id, target_id = ids[source], ids[target]
        properties = {"source_node_id": source_id, "target_node_id": target_id}
        return (
            source_id,
            target_id,
            relationship,
            {**properties, "relationship_name": relationship},
        )

    await adapter.add_edges(
        [
            edge("a_sym", "a_mod", "calls"),
            edge("a_sym", "a_mod", "imports"),
            edge("a_sym", "a_mod", "declares"),
            edge("a_mod", "b_sym", "calls"),
            edge("b_sym", "b_mod", "calls"),
        ]
    )
    return ids


@pytest.mark.parametrize(
    ("node_types", "property_name", "values", "expected"),
    [
        (["_Symbol"], "repo", ["A"], ["a_sym"]),
        (["_Symbol", "_Module"], "repo", ["A", "B"], ["a_sym", "a_mod", "b_sym", "b_mod"]),
        (["_Module"], "name", ["mod", "other"], ["a_mod", "b_mod"]),
        (["_Symbol"], "repo", ["C"], []),
    ],
)
async def test_node_ids_by_property_match_the_full_read(
    lookup_adapter, node_types, property_name, values, expected
):
    ids = await _seed(lookup_adapter)

    found = await lookup_adapter.get_node_ids_by_property(node_types, property_name, values)
    fallback = await node_ids_by_property(lookup_adapter, node_types, property_name, values)

    assert sorted(found) == sorted(fallback) == sorted(ids[key] for key in expected)


async def test_node_ids_by_property_match_values_that_need_escaping(lookup_adapter):
    """A stored value with a quote or a backslash is matched as written, not escaped."""
    names = ['a"b', "c\\d", "plain"]
    nodes = [_Symbol(id=uuid4(), name=name, repo=name) for name in names]
    await lookup_adapter.add_nodes(nodes)

    for node in nodes:
        found = await lookup_adapter.get_node_ids_by_property(["_Symbol"], "repo", [node.repo])
        assert found == [str(node.id)], node.repo


async def test_node_ids_by_property_compare_values_as_text(lookup_adapter):
    """A numeric property matches its value given as a number or as text."""
    node = _Line(id=uuid4(), name="at-42", line=42)
    await lookup_adapter.add_nodes([node, _Line(id=uuid4(), name="at-7", line=7)])

    for wanted in (42, "42"):
        found = await lookup_adapter.get_node_ids_by_property(["_Line"], "line", [wanted])
        assert found == [str(node.id)], repr(wanted)


async def test_outgoing_edge_keys_match_the_full_read(lookup_adapter):
    ids = await _seed(lookup_adapter)
    sources = [ids["a_sym"], ids["b_sym"], str(uuid4())]

    found = await lookup_adapter.get_outgoing_edge_keys(sources)
    fallback = await outgoing_edge_keys(lookup_adapter, sources)

    expected = [
        (ids["a_sym"], ids["a_mod"], "calls"),
        (ids["a_sym"], ids["a_mod"], "declares"),
        (ids["a_sym"], ids["a_mod"], "imports"),
        (ids["b_sym"], ids["b_mod"], "calls"),
    ]
    assert sorted(found) == sorted(fallback) == sorted(expected)
