"""The one edge-row dedup helper shared by the task layer (first-wins, via
deduplicate_nodes_and_edges) and the Ladybug adapter guard (last-wins)."""

from uuid import UUID

import pytest

from cognee.infrastructure.databases.graph.deduplicate_edges import deduplicate_edges, edge_key
from cognee.modules.graph.utils.deduplicate_nodes_and_edges import deduplicate_nodes_and_edges

ROWS = [
    ("a", "b", "rel", {"k": 1}),
    ("b", "c", "rel", {"k": 2}),
    ("a", "b", "rel", {"k": 3}),
    ("a", "b", "other", {"k": 4}),
]


def test_first_wins_keeps_first_row_in_first_seen_order():
    assert deduplicate_edges(ROWS, keep="first") == [ROWS[0], ROWS[1], ROWS[3]]


def test_last_wins_keeps_last_row_at_first_seen_position():
    assert deduplicate_edges(ROWS, keep="last") == [ROWS[2], ROWS[1], ROWS[3]]


def test_key_normalizes_ids_with_str():
    uid = UUID("00000000-0000-0000-0000-000000000001")
    rows = [(uid, "b", "rel", {"k": 1}), (str(uid), "b", "rel", {"k": 2})]

    assert edge_key(rows[0]) == edge_key(rows[1]) == (str(uid), "b", "rel")
    assert deduplicate_edges(rows, keep="last") == [rows[1]]


def test_empty_and_unknown_policy():
    assert deduplicate_edges([], keep="first") == []
    with pytest.raises(ValueError):
        deduplicate_edges(ROWS, keep="most_confident")  # type: ignore[arg-type]


def test_deduplicate_nodes_and_edges_uses_first_wins():
    _nodes, edges = deduplicate_nodes_and_edges([], ROWS)

    assert edges == deduplicate_edges(ROWS, keep="first")


def test_every_edge_key_consumer_imports_the_one_definition():
    """The dedup, the chunk-ownership map, add_data_points' per-row provenance
    mapping and the Ladybug per-row lookup must all key edges identically, or a
    ref lookup can KeyError on a row the collapse kept. Pin that they share the
    object rather than each spelling the tuple out."""
    from importlib import import_module

    from cognee.infrastructure.databases.graph.deduplicate_edges import EdgeKey

    adapter = import_module("cognee.infrastructure.databases.graph.ladybug.adapter")
    chunk_ownership = import_module("cognee.tasks.storage.chunk_ownership")
    # The package re-exports the function under the module's name; go via sys.modules.
    add_data_points = import_module("cognee.tasks.storage.add_data_points")

    assert chunk_ownership.edge_key is edge_key
    assert chunk_ownership.EdgeKey is EdgeKey
    assert add_data_points.edge_key is edge_key
    assert adapter.edge_key is edge_key


def test_chunk_ownership_edge_owners_are_keyed_by_edge_key():
    from uuid import uuid4

    from cognee.tasks.storage.chunk_ownership import ChunkOwnership

    ownership = ChunkOwnership()
    edge = (uuid4(), uuid4(), "relates_to", {})
    ownership.edge_owners[edge_key(edge)] = ["ref"]

    assert ownership.edge_owners[(str(edge[0]), str(edge[1]), "relates_to")] == ["ref"]
