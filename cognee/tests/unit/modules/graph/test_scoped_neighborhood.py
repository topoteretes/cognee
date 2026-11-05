"""Selection invariants for scoped, undirected depth expansion."""

from uuid import UUID

import pytest

from cognee.modules.graph.cognee_graph.scoped_neighborhood import select_scoped_neighborhood


def test_neighborhood_does_not_reenter_scope_through_excluded_nodes():
    nodes = [("a", {}), ("c", {}), ("d", {})]
    edges = [
        ("a", "b", "related", {}),
        ("b", "c", "related", {}),
        ("c", "d", "related", {}),
    ]

    assert select_scoped_neighborhood(nodes, edges, ["a"], 3) == ([("a", {})], [])


def test_neighborhood_normalizes_seed_ids_and_keeps_adapter_order():
    a = UUID("00000000-0000-0000-0000-000000000001")
    b = UUID("00000000-0000-0000-0000-000000000002")
    c = UUID("00000000-0000-0000-0000-000000000003")
    nodes = [(c, {"name": "c"}), (b, {"name": "b"}), (a, {"name": "a"})]
    edges = [(c, b, "related", {"weight": 2}), (b, a, "related", {"weight": 1})]

    selected_nodes, selected_edges = select_scoped_neighborhood(
        nodes, edges, [str(a), a, "missing"], 1
    )

    assert selected_nodes == [(b, {"name": "b"}), (a, {"name": "a"})]
    assert selected_edges == [(b, a, "related", {"weight": 1})]


def test_edge_type_filter_limits_traversal_but_keeps_induced_edges():
    nodes = [("a", {}), ("b", {}), ("c", {})]
    edges = [
        ("a", "b", "related", {}),
        ("a", "b", "context", {"text": "retained"}),
        ("b", "c", "context", {}),
    ]

    selected_nodes, selected_edges = select_scoped_neighborhood(nodes, edges, ["a"], 2, ["related"])

    assert selected_nodes == [("a", {}), ("b", {})]
    assert selected_edges == [
        ("a", "b", "related", {}),
        ("a", "b", "context", {"text": "retained"}),
    ]


@pytest.mark.parametrize("seed_ids", [[], ["missing"]])
def test_missing_seeds_do_not_fall_back_to_the_scoped_graph(seed_ids):
    assert select_scoped_neighborhood([("a", {})], [], seed_ids, 1) == ([], [])


@pytest.mark.parametrize("edge_types", [None, []])
def test_unrestricted_traversal_handles_cycles_and_multiple_seeds(edge_types):
    nodes = [("a", {}), ("b", {}), ("c", {}), ("isolated", {})]
    edges = [
        ("a", "b", "related", {}),
        ("b", "c", "context", {}),
        ("c", "a", "related", {}),
    ]

    assert select_scoped_neighborhood(nodes, edges, ["a", "isolated"], 1, edge_types) == (
        nodes,
        edges,
    )
