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

from cognee.infrastructure.databases.graph.postgres_demo.adapter import (  # noqa: E402
    _component_sizes,
    _prepare_edge_rows,
    _prepare_node_rows,
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
