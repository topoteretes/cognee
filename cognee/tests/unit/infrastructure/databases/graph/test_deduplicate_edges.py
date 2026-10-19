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
