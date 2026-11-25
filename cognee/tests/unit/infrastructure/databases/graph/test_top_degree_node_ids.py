"""Seed ranking must not read the whole graph.

The default graph visualization has no query and no explicit seeds, so it takes
the degree path. That path used to call ``get_graph_data()`` and count degree in
Python: on a 5.6M-node / 35.6M-edge graph, tens of gigabytes of Python objects
built in order to keep ten ids, and the worker was OOM-killed before answering
(``Killed process (gunicorn) anon-rss:20,845,748kB``). Adapters that can
aggregate now do the ranking in the store.

The in-memory count survives as the interface's inherited default so a
community adapter keeps working — which is exactly why it needs a test of its
own rather than being deleted.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface


class _FullReadAdapter:
    """An adapter with no native ranking: exercises the inherited default.

    Not a GraphDBInterface subclass on purpose. The interface declares ~20
    abstract methods that this behaviour does not touch, and stubbing all of
    them to satisfy the ABC would bury what is being tested. The default is
    invoked unbound against this object instead, which runs exactly the code a
    real adapter would inherit.
    """

    def __init__(self, nodes, edges):
        self._nodes = nodes
        self._edges = edges
        self.full_reads = 0

    async def get_graph_data(self):
        self.full_reads += 1
        return self._nodes, self._edges

    async def get_top_degree_node_ids(self, top_k: int) -> list[str]:
        return await GraphDBInterface.get_top_degree_node_ids(self, top_k)


def _star(spokes: int = 4):
    nodes = [("hub", {})] + [(f"s{i}", {}) for i in range(spokes)]
    edges = [("hub", f"s{i}", "rel", {}) for i in range(spokes)]
    return nodes, edges


@pytest.mark.asyncio
async def test_default_ranks_by_degree():
    """The hub has degree 4; every spoke has degree 1."""
    nodes, edges = _star()
    adapter = _FullReadAdapter(nodes, edges)

    assert await adapter.get_top_degree_node_ids(1) == ["hub"]
    assert adapter.full_reads == 1


@pytest.mark.asyncio
async def test_default_counts_both_endpoints():
    """Degree counts an edge at both ends, so a chain's middle outranks its tips."""
    nodes = [(str(i), {}) for i in range(3)]
    edges = [("0", "1", "rel", {}), ("1", "2", "rel", {})]

    top = await _FullReadAdapter(nodes, edges).get_top_degree_node_ids(1)

    assert top == ["1"]


@pytest.mark.asyncio
async def test_default_on_empty_graph_returns_no_seeds():
    assert await _FullReadAdapter([], []).get_top_degree_node_ids(5) == []


@pytest.mark.asyncio
async def test_default_includes_isolated_nodes():
    adapter = _FullReadAdapter([("a", {}), ("b", {})], [])

    assert set(await adapter.get_top_degree_node_ids(5)) == {"a", "b"}


@pytest.mark.asyncio
@pytest.mark.parametrize("top_k", [0, -1])
async def test_default_rejects_invalid_limit_before_reading(top_k):
    adapter = _FullReadAdapter(*_star())

    with pytest.raises(ValueError, match="top_k must be >= 1"):
        await adapter.get_top_degree_node_ids(top_k)

    assert adapter.full_reads == 0


@pytest.mark.asyncio
async def test_default_tolerates_top_k_larger_than_the_graph():
    nodes, edges = _star(2)

    top = await _FullReadAdapter(nodes, edges).get_top_degree_node_ids(50)

    assert len(top) == 3
    assert top[0] == "hub"


@pytest.mark.asyncio
async def test_default_ignores_edges_pointing_outside_the_node_set():
    """A dangling edge must not invent a seed that has no node."""
    nodes = [("a", {})]
    edges = [("a", "ghost", "rel", {}), ("ghost", "ghost2", "rel", {})]

    assert await _FullReadAdapter(nodes, edges).get_top_degree_node_ids(5) == ["a"]


def test_the_default_is_inherited_not_abstract():
    """A community adapter must keep loading without implementing this."""
    assert "get_top_degree_node_ids" not in getattr(
        GraphDBInterface, "__abstractmethods__", frozenset()
    )


@pytest.mark.parametrize(
    "adapter_module",
    [
        "cognee.infrastructure.databases.graph.postgres_demo.adapter",
        "cognee.infrastructure.databases.graph.neo4j_driver.adapter",
        "cognee.infrastructure.databases.graph.ladybug.adapter",
        "cognee.infrastructure.databases.graph.turso.adapter",
        "cognee.infrastructure.databases.graph.neptune_driver.adapter",
    ],
)
def test_in_tree_adapters_override_the_expensive_default(adapter_module):
    """These are the adapters cognee ships; none should be on the full-read path."""
    import inspect
    import types

    # Each adapter's driver is an optional extra, so an uninstalled one is
    # skipped rather than reported as a missing override.
    module = pytest.importorskip(adapter_module)
    # On Python <= 3.10, isinstance(tuple[int], type) is True (changed in
    # 3.11), so inspect.isclass lets module-level builtin-generic aliases
    # like ``tuple[str, str, str, dict[str, Any]]`` through — and issubclass
    # then raises TypeError on them. Filter the aliases out explicitly.
    adapters = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if not isinstance(obj, types.GenericAlias)
        and issubclass(obj, GraphDBInterface)
        and obj is not GraphDBInterface
    ]

    assert adapters, f"no GraphDBInterface subclass found in {adapter_module}"
    for adapter in adapters:
        assert "get_top_degree_node_ids" in adapter.__dict__, (
            f"{adapter.__name__} inherits the full-graph-read default"
        )


# --- the Postgres override must stay bounded --------------------------------


@pytest.mark.asyncio
async def test_postgres_ranking_is_bounded_not_an_exact_aggregate():
    """Guards against "fixing" the sampling into an exact count.

    An exact aggregate is the obvious-looking improvement and the wrong one: on
    a 5.59M-node / 35.6M-edge graph it measured 57 s with an ~8.5 GB temp spill,
    because it must group all 71M endpoint rows. The sampled form returned the
    same top five in 1.14 s.

    Asserted on the SQL the adapter actually emits, and on the parameters bound
    with it — not on the module source, which a comment could satisfy.
    """
    from contextlib import asynccontextmanager

    from cognee.infrastructure.databases.graph.postgres_demo.adapter import (
        PostgresDemoAdapter,
    )

    captured = {}

    class _Session:
        async def execute(self, statement, params=None):
            captured["sql"] = " ".join(str(statement).split())
            captured["params"] = params

            class _Result:
                def all(self_inner):
                    return [("node-a",), ("node-b",)]

            return _Result()

    adapter = object.__new__(PostgresDemoAdapter)

    @asynccontextmanager
    async def _sessionmaker():
        yield _Session()

    adapter.sessionmaker = _sessionmaker

    assert await adapter.get_top_degree_node_ids(2) == ["node-a", "node-b"]

    sql = captured["sql"]
    params = captured["params"]

    # Both endpoint directions use one bounded edge sample.
    assert sql.count("LIMIT :sample") == 1, "one coherent edge sample feeds both endpoints"
    assert params["sample"] == PostgresDemoAdapter._SEED_SAMPLE_ROWS
    assert params["top_k"] == 2

    # A full seed sample needs no node lookup. Sparse graphs use a separate
    # LIMIT-bounded id lookup to include isolated nodes.
    assert "graph_node" not in sql
    assert sql.count("graph_edge") == 1
    assert "AS MATERIALIZED" in sql


@pytest.mark.asyncio
@pytest.mark.parametrize("top_k", [0, -1])
async def test_postgres_ranking_rejects_a_meaningless_top_k(top_k):
    from cognee.infrastructure.databases.graph.postgres_demo.adapter import (
        PostgresDemoAdapter,
    )

    adapter = object.__new__(PostgresDemoAdapter)

    with pytest.raises(ValueError):
        await adapter.get_top_degree_node_ids(top_k)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sampled", "extra", "top_k"),
    [([], [], 5), ([], ["a", "b"], 5), (["hub", "spoke"], ["isolated"], 3)],
)
async def test_postgres_fills_sparse_samples_with_bounded_node_ids(sampled, extra, top_k):
    from cognee.infrastructure.databases.graph.postgres_demo.adapter import PostgresDemoAdapter

    adapter = object.__new__(PostgresDemoAdapter)
    adapter.sessionmaker = MagicMock()
    session = adapter.sessionmaker.return_value.__aenter__.return_value
    results = [MagicMock(), MagicMock()]
    results[0].all.return_value = [(node_id,) for node_id in sampled]
    results[1].all.return_value = [(node_id,) for node_id in extra]
    session.execute = AsyncMock(side_effect=results)

    assert await adapter.get_top_degree_node_ids(top_k) == sampled + extra

    assert session.execute.await_count == 2
    statement, params = session.execute.await_args.args
    assert str(statement) == (
        "SELECT id FROM graph_node WHERE NOT (id = ANY(:seed_ids)) LIMIT :remaining"
    )
    assert params == {"seed_ids": sampled, "remaining": top_k - len(sampled)}


@pytest.mark.asyncio
@pytest.mark.parametrize("top_k", [0, -1])
@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [("ladybug", "LadybugAdapter"), ("neo4j_driver", "Neo4jAdapter")],
)
async def test_cypher_adapters_reject_invalid_limit_before_query(module_name, class_name, top_k):
    module = pytest.importorskip(f"cognee.infrastructure.databases.graph.{module_name}.adapter")
    adapter = object.__new__(getattr(module, class_name))
    adapter.query = AsyncMock()

    with pytest.raises(ValueError, match="top_k must be >= 1"):
        await adapter.get_top_degree_node_ids(top_k)

    adapter.query.assert_not_awaited()


@pytest.mark.asyncio
async def test_duck_typed_adapter_without_new_method_keeps_working(monkeypatch):
    from types import SimpleNamespace

    from cognee.modules.visualization.subgraph_data import resolve_seeds_by_degree

    adapter = SimpleNamespace(get_graph_data=AsyncMock(return_value=_star()))
    assert await resolve_seeds_by_degree(adapter, 1) == ["hub"]


@pytest.mark.asyncio
async def test_full_read_warning_is_once_per_adapter_type(monkeypatch):
    import cognee.infrastructure.databases.graph.graph_db_interface as module

    monkeypatch.setattr(module, "_warned_degree_fallbacks", set())
    warning = MagicMock()
    monkeypatch.setattr(module.logger, "warning", warning)
    for _ in range(2):
        await _FullReadAdapter(*_star()).get_top_degree_node_ids(1)
    warning.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("typed", [False, True])
async def test_cypher_limits_edges_before_aggregating(typed):
    from types import SimpleNamespace

    from cognee.infrastructure.databases.graph.degree_seeds import (
        EDGE_SAMPLE_ROWS,
        cypher_degree_seeds,
    )

    row = ("hub", 2) if typed else {"id": "hub", "degree": 2}
    adapter = SimpleNamespace(query=AsyncMock(return_value=[row]))
    assert await cypher_degree_seeds(adapter, 1, typed=typed) == ["hub"]
    sql, params = adapter.query.await_args.args
    assert sql.index("LIMIT $sample") < sql.index("count(*)")
    assert params == {"sample": EDGE_SAMPLE_ROWS, "top_k": 1}
    assert "COUNT {" not in sql


@pytest.mark.asyncio
async def test_turso_native_seed_selection(tmp_path):
    pytest.importorskip("turso", reason="pyturso not installed")
    from types import SimpleNamespace

    from cognee.infrastructure.databases.graph.turso.adapter import TursoAdapter

    adapter = TursoAdapter(database_path=str(tmp_path / "seed-test.db"))
    adapter.get_graph_data = AsyncMock(side_effect=AssertionError("unexpected full graph read"))
    try:
        assert await adapter.get_top_degree_node_ids(5) == []
        for node_id in ["hub", "incoming", "outgoing", "isolated"]:
            await adapter.add_node(
                SimpleNamespace(
                    model_dump=lambda node_id=node_id: {
                        "id": node_id,
                        "name": node_id,
                        "type": "Node",
                    }
                )
            )
        assert len(await adapter.get_top_degree_node_ids(5)) == 4
        await adapter.add_edge("incoming", "hub", "rel")
        await adapter.add_edge("hub", "outgoing", "rel")
        assert await adapter.get_top_degree_node_ids(1) == ["hub"]
        assert set(await adapter.get_top_degree_node_ids(5)) == {
            "hub",
            "incoming",
            "outgoing",
            "isolated",
        }
        adapter.get_graph_data.assert_not_awaited()
    finally:
        await adapter.close()
