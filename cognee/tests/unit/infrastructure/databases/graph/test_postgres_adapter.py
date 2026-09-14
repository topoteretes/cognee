import asyncio
import importlib.util
import json
import sys
import types
from copy import deepcopy

import pytest

_created_asyncpg_stub = False
if importlib.util.find_spec("asyncpg") is None:
    asyncpg_stub = types.ModuleType("asyncpg")

    class DeadlockDetectedError(Exception):
        pass

    asyncpg_stub.DeadlockDetectedError = DeadlockDetectedError
    sys.modules["asyncpg"] = asyncpg_stub
    _created_asyncpg_stub = True

from sqlalchemy import NullPool  # noqa: E402

from cognee.infrastructure.databases.graph.postgres_demo.adapter import (  # noqa: E402
    PostgresDemoAdapter,
    _component_sizes,
    _prepare_edge_rows,
    _prepare_node_rows,
    _resolve_engine_args,
    _select_nodeset_neighbor_ids,
)

if _created_asyncpg_stub:
    del sys.modules["asyncpg"]


def _stored_properties(row):
    """Decode the JSON blob a prepared row carries for the jsonb bind."""
    return json.loads(row["properties"])


def test_prepare_node_rows_splits_core_columns_from_stored_properties():
    (row,) = _prepare_node_rows([("n1", {"name": "Alice", "type": "Person", "age": 30})])

    assert (row["id"], row["name"], row["type"]) == ("n1", "Alice", "Person")
    assert _stored_properties(row) == {"age": 30}


def test_prepare_node_rows_treats_the_tuple_id_as_authoritative():
    (row,) = _prepare_node_rows([("authoritative", {"id": "ignored", "name": "N"})])

    assert row["id"] == "authoritative"
    assert "id" not in _stored_properties(row)


def test_prepare_node_rows_accepts_datapoint_like_objects():
    class _Point:
        def model_dump(self):
            return {"id": "dp1", "name": "Point", "type": "T", "extra": True}

    (row,) = _prepare_node_rows([_Point()])

    assert (row["id"], row["name"], row["type"]) == ("dp1", "Point", "T")
    assert _stored_properties(row) == {"extra": True}


def test_prepare_node_rows_keeps_the_last_duplicate_and_sorts_by_id():
    rows = _prepare_node_rows(
        [("b", {"name": "first"}), ("a", {"name": "only"}), ("b", {"name": "last"})]
    )

    # Sorted output is the lock-ordering rule: concurrent batches touching the
    # same ids must take their row locks in one order.
    assert [row["id"] for row in rows] == ["a", "b"]
    assert [row["name"] for row in rows] == ["only", "last"]


def test_prepare_node_rows_strips_nul_bytes_from_columns_and_nested_values():
    (row,) = _prepare_node_rows(
        [("i\0d", {"name": "na\0me", "type": "t\0", "nested": {"k\0": ["v\0"]}})]
    )

    assert (row["id"], row["name"], row["type"]) == ("id", "name", "t")
    assert _stored_properties(row) == {"nested": {"k": ["v"]}}


def test_prepare_edge_rows_keeps_the_last_duplicate_triple_and_sorts_by_identity():
    rows = _prepare_edge_rows(
        [
            ("s2", "t", "R", {"value": "other"}),
            ("s1", "t", "R", {"value": "first"}),
            ("s1", "t", "R", {"value": "last"}),
        ]
    )

    assert [(row["source_id"], row["target_id"], row["relationship_name"]) for row in rows] == [
        ("s1", "t", "R"),
        ("s2", "t", "R"),
    ]
    assert _stored_properties(rows[0]) == {"value": "last"}


def test_prepare_edge_rows_sanitizes_identity_and_defaults_absent_properties():
    rows = _prepare_edge_rows([("s\0", "t\0", "RE\0L", None), ("s2", "t2", "R2")])

    assert (rows[0]["source_id"], rows[0]["target_id"], rows[0]["relationship_name"]) == (
        "s",
        "t",
        "REL",
    )
    assert _stored_properties(rows[0]) == {}
    assert _stored_properties(rows[1]) == {}


def test_prepare_rows_do_not_mutate_caller_data():
    """Sanitizing rewrites values the caller still owns, so preparation must copy."""
    node_properties = {"name": "Ca\0ller", "nested": {"value": "a\0b"}}
    edge_properties = {"nested": {"value": "c\0d"}}
    expected_node = deepcopy(node_properties)
    expected_edge = deepcopy(edge_properties)

    _prepare_node_rows([("n\0", node_properties)])
    _prepare_edge_rows([("s\0", "t\0", "R\0", edge_properties)])

    assert node_properties == expected_node
    assert edge_properties == expected_edge


def test_component_sizes_treats_edges_as_undirected():
    assert _component_sizes(
        ["a", "b", "c", "d", "e", "isolated"],
        [("a", "b"), ("c", "b"), ("d", "e")],
    ) == [3, 2, 1]
    assert _component_sizes(["self"], [("self", "self")]) == [1]
    assert _component_sizes([], []) == []


def test_nodeset_neighbor_selection_supports_or_and_missing_primaries():
    primaries = {"a", "b"}
    edges = [("a", "shared"), ("b", "shared"), ("a", "one-side")]

    assert _select_nodeset_neighbor_ids(primaries, edges, "OR") == {"shared", "one-side"}
    assert _select_nodeset_neighbor_ids(primaries, edges, "AND") == {"shared"}
    assert _select_nodeset_neighbor_ids(set(), edges, "OR") == set()


def test_nodeset_neighbor_selection_rejects_unknown_operator():
    with pytest.raises(ValueError, match="must be 'OR' or 'AND'"):
        _select_nodeset_neighbor_ids({"a"}, [("a", "b")], "XOR")


# --- pool exhaustion (SDK-533) -------------------------------------------------
#
# The datasheets real-LLM postgres arm failed with "QueuePool limit of size 5
# overflow 10 reached". Not a leak: every acquisition is `async with`. It is
# 164 concurrent writers each checking a connection out and THEN blocking on
# pg_advisory_xact_lock, plus hundreds of create_all() reflections per batch,
# against a pool whose sizing config was silently discarded.


def _bare_adapter():
    adapter = PostgresDemoAdapter.__new__(PostgresDemoAdapter)
    adapter._write_gate = None
    adapter._write_gate_loop = None
    adapter._initialized = False
    adapter._init_lock = None
    adapter._init_lock_loop = None
    return adapter


@pytest.mark.asyncio
async def test_write_gate_serializes_graph_writes():
    """Fifty concurrent add_nodes must hold at most one session at a time.

    Without the gate each writer pins a pooled connection while waiting on the
    advisory lock, so peak == min(50, pool ceiling); on origin/dev this is 50.
    """
    adapter = _bare_adapter()
    live = 0
    peak = 0

    class _Session:
        async def __aenter__(self):
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.01)
            return self

        async def __aexit__(self, *exc):
            nonlocal live
            live -= 1

        async def execute(self, *a, **k):
            return None

        async def commit(self):
            return None

    adapter.sessionmaker = lambda: _Session()
    await asyncio.gather(*[adapter.add_nodes([(f"n{i}", {})]) for i in range(50)])
    assert peak == 1


@pytest.mark.asyncio
async def test_initialize_runs_create_all_once_per_adapter():
    """Regression for the read-path QueuePool timeout (job 98411831471).

    Every get_graph_engine() and every get_graph_metadata() called initialize(),
    each a full create_all(checkfirst=True) reflection on a pooled connection.
    Fifty concurrent calls must reflect exactly once; on origin/dev it is 50.
    """
    adapter = _bare_adapter()
    calls = 0

    class _Conn:
        async def __aenter__(self):
            nonlocal calls
            calls += 1
            await asyncio.sleep(0)
            return self

        async def __aexit__(self, *exc):
            pass

        async def run_sync(self, *a, **k):
            pass

    adapter.engine = types.SimpleNamespace(begin=lambda: _Conn())
    await asyncio.gather(*[adapter.initialize() for _ in range(50)])
    assert calls == 1


@pytest.mark.asyncio
async def test_write_gate_is_rebuilt_for_a_new_event_loop():
    """The adapter is process-cached and outlives the loop it was built on."""
    adapter = _bare_adapter()
    first = adapter._get_write_gate()
    assert adapter._get_write_gate() is first

    def _other_loop():
        return asyncio.run(_probe())

    async def _probe():
        return adapter._get_write_gate()

    import threading

    result = []
    t = threading.Thread(target=lambda: result.append(_other_loop()))
    t.start()
    t.join()
    assert result[0] is not first


def test_resolve_engine_args_defaults_match_the_per_dataset_pgvector_sizing():
    args = _resolve_engine_args({})
    assert (args["pool_size"], args["max_overflow"]) == (2, 20)
    assert args["pool_timeout"] == 280  # stock 30s before this change
    assert args["pool_recycle"] == 280
    assert args["pool_pre_ping"] is True
    assert "poolclass" not in args


def test_resolve_engine_args_honors_configured_pool_args():
    assert _resolve_engine_args({"pool_size": 7})["pool_size"] == 7
    assert _resolve_engine_args({"pool_size": 7})["max_overflow"] == 20


def test_resolve_engine_args_keeps_the_nullpool_escape_hatch():
    assert _resolve_engine_args({"poolclass": "NullPool"}) == {"poolclass": NullPool}
    assert _resolve_engine_args({"poolclass": "nullpool", "pool_size": 9}) == {
        "poolclass": NullPool
    }


def test_resolve_engine_args_accepts_the_configs_tuple_of_pairs_form():
    # relational/config.py stores POOL_ARGS as tuple(sorted(parsed.items()))
    args = _resolve_engine_args((("pool_size", 2), ("pool_timeout", 5)))
    assert args["pool_timeout"] == 5
    assert args["pool_size"] == 2
