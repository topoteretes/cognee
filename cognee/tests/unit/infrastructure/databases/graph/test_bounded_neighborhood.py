"""A graph view must not read the whole neighbourhood to draw part of it.

The default view seeds from the highest-degree nodes and expands two hops,
and two hops from a hub reach almost the whole graph. ``get_neighborhood``
returned all of it and Python kept ``max_nodes``: on a 13959-node / 33488-edge
dataset, every Mindmap refresh read ~10k nodes with full properties to draw
1000. ``iter_bounded_neighborhood`` bounds the read in adapters that can, and
hands the result over in chunks under one contract for every adapter.

The in-memory cut survives as the interface's inherited default, so a
community adapter keeps working, which is why it has tests of its own.
"""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from cognee.infrastructure.databases.graph.bounded_neighborhood import (
    chunk_members,
    hop_distances,
)
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.tests.utils.assert_bounded_neighborhood_contract import (
    assert_bounded_neighborhood_contract,
)


class _NeighborhoodOnlyAdapter:
    """An adapter with no native bounded read: exercises the inherited default.

    Not a GraphDBInterface subclass, for the same reason as the test double in
    test_top_degree_node_ids.py: the ABC's abstract methods are irrelevant here.
    ``get_neighborhood`` answers from an in-memory graph the way real adapters
    do, returning every node within ``depth`` hops and the edges among them.
    """

    def __init__(self, nodes, edges):
        self._nodes = nodes
        self._edges = edges
        self.neighborhood_reads = 0

    async def get_neighborhood(self, node_ids, depth=1, edge_types=None):
        self.neighborhood_reads += 1
        known = {node_id for node_id, _ in self._nodes}
        distance = hop_distances(self._edges, [seed for seed in node_ids if seed in known])
        reached = {node_id for node_id, hops in distance.items() if hops <= depth}
        nodes = [node for node in self._nodes if node[0] in reached]
        edges = [edge for edge in self._edges if edge[0] in reached and edge[1] in reached]
        return nodes, edges

    def iter_bounded_neighborhood(self, *args, **kwargs):
        return GraphDBInterface.iter_bounded_neighborhood(self, *args, **kwargs)


def _two_hubs(spokes: int = 30):
    """Two hubs, each with its own spokes, and one spoke of each linked to a tail."""
    nodes = [("hub-a", {"name": "a", "type": "Hub"}), ("hub-b", {"name": "b", "type": "Hub"})]
    edges = []
    for hub in ("a", "b"):
        for index in range(spokes):
            spoke = f"{hub}-{index}"
            nodes.append((spoke, {"name": spoke, "type": "Spoke", "text": "x" * 100}))
            edges.append((f"hub-{hub}", spoke, "has", {}))
    nodes.append(("tail", {"name": "tail", "type": "Tail"}))
    edges += [("a-0", "tail", "next", {}), ("b-0", "tail", "next", {})]
    return nodes, edges


def _chain(length: int):
    nodes = [(str(index), {"name": str(index), "type": "Link"}) for index in range(length)]
    edges = [(str(index), str(index + 1), "next", {}) for index in range(length - 1)]
    return nodes, edges


async def _collect(adapter, *args, **kwargs):
    return [chunk async for chunk in adapter.iter_bounded_neighborhood(*args, **kwargs)]


# --- the inherited default --------------------------------------------------


@pytest.mark.asyncio
async def test_default_caps_a_hub_heavy_graph_nearest_hop_first():
    nodes, edges = _two_hubs()
    adapter = _NeighborhoodOnlyAdapter(nodes, edges)

    chunks = await _collect(adapter, ["hub-a", "hub-b"], 2, 21, chunk_size=5)

    members, _ = assert_bounded_neighborhood_contract(
        chunks, max_nodes=21, chunk_size=5, seed_ids=["hub-a", "hub-b"], graph_edges=edges
    )
    assert len(members) == 21
    assert "tail" not in members  # hop 2 gets nothing while hop 1 fills the budget


@pytest.mark.asyncio
@pytest.mark.parametrize(("depth", "expected"), [(1, {"0", "1"}), (2, {"0", "1", "2"})])
async def test_default_respects_depth(depth, expected):
    nodes, edges = _chain(6)
    adapter = _NeighborhoodOnlyAdapter(nodes, edges)

    chunks = await _collect(adapter, ["0"], depth, 100)

    members, identities = assert_bounded_neighborhood_contract(
        chunks, max_nodes=100, chunk_size=2000, seed_ids=["0"], graph_edges=edges
    )
    assert set(members) == expected
    assert identities == {edge[:3] for edge in edges if {edge[0], edge[1]} <= expected}


@pytest.mark.asyncio
async def test_default_with_a_budget_larger_than_the_graph_returns_the_neighbourhood():
    nodes, edges = _two_hubs(spokes=4)
    adapter = _NeighborhoodOnlyAdapter(nodes, edges)
    reference_nodes, reference_edges = await adapter.get_neighborhood(["hub-a"], depth=2)

    chunks = await _collect(adapter, ["hub-a"], 2, 10_000, chunk_size=3)

    members, identities = assert_bounded_neighborhood_contract(
        chunks, max_nodes=10_000, chunk_size=3, seed_ids=["hub-a"], graph_edges=edges
    )
    assert set(members) == {node_id for node_id, _ in reference_nodes}
    assert identities == {edge[:3] for edge in reference_edges}


@pytest.mark.asyncio
async def test_default_yields_an_isolated_seed_alone():
    adapter = _NeighborhoodOnlyAdapter([("alone", {"name": "alone", "type": "T"})], [])

    chunks = await _collect(adapter, ["alone"], 2, 10)

    assert chunks == [([("alone", {"name": "alone", "type": "T"})], [])]


@pytest.mark.asyncio
async def test_default_seed_order_survives_and_missing_duplicate_and_uuid_seeds_take_no_slot():
    uuid_seed = UUID("00000000-0000-0000-0000-000000000001")
    nodes = [(str(uuid_seed), {"name": "u", "type": "T"}), ("b", {"name": "b", "type": "T"})]
    nodes += [(f"n{index}", {"name": str(index), "type": "T"}) for index in range(5)]
    edges = [("b", f"n{index}", "rel", {}) for index in range(5)]
    adapter = _NeighborhoodOnlyAdapter(nodes, edges)

    chunks = await _collect(adapter, ["b", "missing", uuid_seed, "b"], 1, 4)

    members, _ = assert_bounded_neighborhood_contract(
        chunks, max_nodes=4, chunk_size=2000, seed_ids=["b", str(uuid_seed)]
    )
    assert len(members) == 4  # the missing seed and the repeated one cost nothing


@pytest.mark.asyncio
async def test_default_projects_properties():
    nodes, edges = _two_hubs(spokes=2)
    adapter = _NeighborhoodOnlyAdapter(nodes, edges)

    chunks = await _collect(adapter, ["hub-a"], 1, 10, property_keys=["missing_key"])

    for chunk_nodes, _ in chunks:
        for _, properties in chunk_nodes:
            assert set(properties) <= {"name", "type"}
            assert "text" not in properties


@pytest.mark.asyncio
async def test_default_reads_nothing_for_no_seeds():
    adapter = _NeighborhoodOnlyAdapter(*_chain(3))

    assert await _collect(adapter, [], 2, 10) == []
    assert adapter.neighborhood_reads == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bounds",
    [
        {"depth": 0, "max_nodes": 10, "chunk_size": 5},
        {"depth": 1, "max_nodes": 0, "chunk_size": 5},
        {"depth": 1, "max_nodes": 10, "chunk_size": 0},
        {"depth": -1, "max_nodes": -1, "chunk_size": -1},
    ],
)
async def test_default_rejects_meaningless_bounds_before_reading(bounds):
    adapter = _NeighborhoodOnlyAdapter(*_chain(3))

    with pytest.raises(ValueError, match="must be >= 1"):
        await _collect(adapter, ["0"], **bounds)
    assert adapter.neighborhood_reads == 0


def test_the_default_is_inherited_not_abstract():
    """A community adapter must keep loading without implementing this."""
    assert "iter_bounded_neighborhood" not in getattr(
        GraphDBInterface, "__abstractmethods__", frozenset()
    )


@pytest.mark.asyncio
async def test_neighbourhood_read_warning_is_once_per_adapter_type(monkeypatch):
    import cognee.infrastructure.databases.graph.graph_db_interface as module

    monkeypatch.setattr(module, "_warned_neighborhood_fallbacks", set())
    warning = MagicMock()
    monkeypatch.setattr(module.logger, "warning", warning)
    for _ in range(2):
        await _collect(_NeighborhoodOnlyAdapter(*_chain(3)), ["0"], 1, 10)
    warning.assert_called_once()


# --- the chunking rule ------------------------------------------------------


def test_each_edge_lands_in_the_chunk_of_its_later_endpoint():
    members = [(node_id, {}) for node_id in ["a", "b", "c", "d"]]
    edges = [
        ("d", "a", "back", {}),
        ("a", "b", "same", {}),
        ("b", "c", "cross", {}),
        ("c", "outside", "dropped", {}),
        ("c", "c", "loop", {}),
    ]

    chunks = list(chunk_members(members, edges, chunk_size=2))

    assert [[node_id for node_id, _ in nodes] for nodes, _ in chunks] == [["a", "b"], ["c", "d"]]
    assert [[edge[2] for edge in chunk_edges] for _, chunk_edges in chunks] == [
        ["same"],
        ["back", "cross", "loop"],
    ]


def test_projection_keeps_name_and_type_and_the_requested_keys():
    members = [("n", {"name": "n", "type": "T", "text": "long", "belongs_to_set": ["s"]})]

    ((nodes, _),) = chunk_members(members, [], chunk_size=1, property_keys=["belongs_to_set"])

    assert nodes == [("n", {"name": "n", "type": "T", "belongs_to_set": ["s"]})]


# --- callers ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_duck_typed_adapter_without_the_method_keeps_working():
    from cognee.modules.visualization.subgraph_data import expand_seed_neighborhood

    nodes, edges = _chain(4)
    helper = _NeighborhoodOnlyAdapter(nodes, edges)
    adapter = SimpleNamespace(get_neighborhood=helper.get_neighborhood)

    kept_nodes, kept_edges = await expand_seed_neighborhood(adapter, ["0"], 1, 10)

    assert [node_id for node_id, _ in kept_nodes] == ["0", "1"]
    assert kept_edges == [("0", "1", "next", {})]


@pytest.mark.asyncio
async def test_visualization_never_asks_a_native_adapter_for_the_whole_neighbourhood():
    from cognee.modules.visualization.subgraph_data import fetch_visualization_graph_data

    requests = []

    class _NativeAdapter:
        get_neighborhood = AsyncMock(side_effect=AssertionError("unbounded neighbourhood read"))

        async def iter_bounded_neighborhood(self, node_ids, depth, max_nodes, chunk_size=2000):
            requests.append((node_ids, depth, max_nodes, chunk_size))
            yield [("s", {"name": "s", "type": "T"})], []

    graph = await fetch_visualization_graph_data(_NativeAdapter(), seed_node_ids=["s"], max_nodes=7)

    assert graph == ([("s", {"name": "s", "type": "T"})], [])
    assert requests == [(["s"], 2, 7, 7)]


# --- in-tree adapters ---------------------------------------------------------

_NO_NATIVE_YET = "no native iter_bounded_neighborhood yet; follow-up to SDK-786"


@pytest.mark.parametrize(
    "adapter_module",
    [
        "cognee.infrastructure.databases.graph.postgres_demo.adapter",
        pytest.param(
            "cognee.infrastructure.databases.graph.neo4j_driver.adapter",
            marks=pytest.mark.xfail(strict=True, reason=_NO_NATIVE_YET),
        ),
        pytest.param(
            "cognee.infrastructure.databases.graph.ladybug.adapter",
            marks=pytest.mark.xfail(strict=True, reason=_NO_NATIVE_YET),
        ),
        pytest.param(
            "cognee.infrastructure.databases.graph.turso.adapter",
            marks=pytest.mark.xfail(strict=True, reason=_NO_NATIVE_YET),
        ),
        pytest.param(
            "cognee.infrastructure.databases.graph.neptune_driver.adapter",
            marks=pytest.mark.xfail(strict=True, reason=_NO_NATIVE_YET),
        ),
    ],
)
def test_in_tree_adapters_bound_the_neighbourhood_read(adapter_module):
    """The adapters cognee ships. The xfails are known gaps, not passing tests."""
    import inspect
    import types

    module = pytest.importorskip(adapter_module)
    adapters = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if not isinstance(obj, types.GenericAlias)
        and issubclass(obj, GraphDBInterface)
        and obj is not GraphDBInterface
    ]

    assert adapters, f"no GraphDBInterface subclass found in {adapter_module}"
    for adapter in adapters:
        assert "iter_bounded_neighborhood" in adapter.__dict__, (
            f"{adapter.__name__} inherits the whole-neighbourhood default"
        )


# --- the Postgres override ----------------------------------------------------


def _postgres_adapter_with_session(results):
    """A PostgresDemoAdapter whose sessions record every statement they run."""
    from cognee.infrastructure.databases.graph.postgres_demo.adapter import PostgresDemoAdapter

    adapter = object.__new__(PostgresDemoAdapter)
    sessions = []
    pending = iter(results)

    class _Session:
        def __init__(self):
            self.statements = []

        async def execute(self, statement, params=None):
            sql = " ".join(str(statement).split())
            self.statements.append((sql, params))
            if sql.startswith("SET LOCAL"):
                return MagicMock()
            return next(pending)

    @asynccontextmanager
    async def _sessionmaker():
        session = _Session()
        sessions.append(session)
        yield session

    adapter.sessionmaker = _sessionmaker
    return adapter, sessions


def _rows(*rows):
    result = MagicMock()
    result.all.return_value = list(rows)
    result.mappings.return_value.all.return_value = list(rows)
    return result


@pytest.mark.asyncio
async def test_postgres_every_session_pins_custom_plans_and_a_timeout():
    adapter, sessions = _postgres_adapter_with_session(
        [
            _rows(("seed",)),  # existing seeds
            _rows(("seed",), ("n1",)),  # hop 1: the seed comes back and is dropped
            _rows({"id": "seed", "name": "s", "type": "T", "properties": {}}),
            _rows(),  # no edge ends in the first chunk
            _rows({"id": "n1", "name": "n", "type": "T", "properties": {}}),
            _rows(
                {"source_id": "seed", "target_id": "n1", "relationship_name": "r", "properties": {}}
            ),
        ]
    )

    chunks = await _collect(adapter, ["seed"], 1, 10, chunk_size=1)

    assert [[node_id for node_id, _ in nodes] for nodes, _ in chunks] == [["seed"], ["n1"]]
    assert chunks[1][1] == [("seed", "n1", "r", {})]
    assert len(sessions) == 3  # membership, then one short session per chunk
    for session in sessions:
        assert session.statements[0][0] == "SET LOCAL plan_cache_mode = force_custom_plan"
        assert session.statements[1][0].startswith("SET LOCAL statement_timeout = ")


@pytest.mark.asyncio
async def test_postgres_hop_excludes_reached_ids_in_python_not_in_sql():
    """Guards against moving the exclusion back into SQL.

    ``<> ALL(:reached)`` is linear in the reached count and measured 270 ms at
    4000 reached ids; the Python set costs nothing.
    """
    adapter, sessions = _postgres_adapter_with_session(
        [
            _rows(("seed",)),
            _rows(("n1",)),
            _rows(("n2",)),
            _rows(
                *(
                    {"id": node_id, "name": node_id, "type": "T", "properties": {}}
                    for node_id in ["seed", "n1", "n2"]
                )
            ),
            _rows(),
        ]
    )

    chunks = await _collect(adapter, ["seed"], 2, 10)

    assert [node_id for node_id, _ in chunks[0][0]] == ["seed", "n1", "n2"]
    hop_statements = [(sql, params) for sql, params in sessions[0].statements if "LATERAL" in sql]
    assert len(hop_statements) == 2
    for sql, params in hop_statements:
        assert "reached" not in sql
        assert set(params) == {"frontier", "max_nodes"}


@pytest.mark.asyncio
@pytest.mark.parametrize("bounds", [(0, 10, 5), (1, 0, 5), (1, 10, 0)])
async def test_postgres_rejects_meaningless_bounds_before_opening_a_session(bounds):
    adapter, sessions = _postgres_adapter_with_session([])
    depth, max_nodes, chunk_size = bounds

    with pytest.raises(ValueError, match="must be >= 1"):
        await _collect(adapter, ["seed"], depth, max_nodes, chunk_size=chunk_size)
    assert sessions == []
