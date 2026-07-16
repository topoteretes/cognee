"""Unit tests for structural (graph-topology) dedup — issue #3630, Approach D.

Deterministic and pure-Python: no LLM, no database, no network. These import
the REAL resolver from cognee.modules.graph.utils and run it on real DataPoint
nodes, so the shipped code — not a copy — is under test.
"""

import copy

from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.utils import resolve_structural_duplicates


class _Entity(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


class _Chunk(DataPoint):
    """A different node type, for type-gating tests."""

    text: str
    metadata: dict = {"index_fields": ["text"]}


def _edge(source, target, relationship_name):
    return (source.id, target.id, relationship_name, {})


def _apple_fixture():
    """ "Apple" and "Apple Inc." share an identical typed neighbourhood; "Tim
    Cook" and "iPhone" are the shared neighbours."""
    apple = _Entity(name="Apple")
    apple_inc = _Entity(name="Apple Inc")
    tim = _Entity(name="Tim Cook")
    iphone = _Entity(name="iPhone")
    nodes = [apple, apple_inc, tim, iphone]
    edges = [
        _edge(apple, tim, "ceo"),
        _edge(apple_inc, tim, "ceo"),
        _edge(apple, iphone, "makes"),
        _edge(apple_inc, iphone, "makes"),
    ]
    return nodes, edges, apple, apple_inc, tim, iphone


def test_empty_and_single_node_return_empty():
    assert resolve_structural_duplicates([], []) == []
    assert resolve_structural_duplicates([_Entity(name="Solo")], []) == []


def test_no_edges_returns_empty():
    nodes = [_Entity(name="Apple"), _Entity(name="Apple Inc")]
    assert resolve_structural_duplicates(nodes, []) == []


def test_structural_duplicates_are_linked_with_merged_into_edge():
    nodes, edges, apple, apple_inc, _tim, _iphone = _apple_fixture()

    merge_edges = resolve_structural_duplicates(nodes, edges)

    assert len(merge_edges) == 1
    source, target, relationship_name, properties = merge_edges[0]
    assert relationship_name == "merged_into"
    assert {source, target} == {apple.id, apple_inc.id}
    assert properties["similarity_score"] == 1.0
    assert properties["resolution"] == "structural_overlap"


def test_distinct_node_is_not_linked():
    """ "Google" shares only one neighbour (Tim Cook) via a different relationship
    — it must not be linked to Apple."""
    nodes, edges, apple, apple_inc, tim, _iphone = _apple_fixture()
    google = _Entity(name="Google")
    nodes.append(google)
    edges.append(_edge(google, tim, "board_member"))

    merge_edges = resolve_structural_duplicates(nodes, edges)

    involved = {node_id for edge in merge_edges for node_id in edge[:2]}
    assert google.id not in involved
    # Apple / Apple Inc. still merge.
    assert {merge_edges[0][0], merge_edges[0][1]} == {apple.id, apple_inc.id}


def test_type_gating_prevents_cross_type_merge():
    """An Entity and a Chunk with an identical typed neighbourhood must never be
    linked, even though same-type nodes with that neighbourhood would."""
    entity = _Entity(name="Doc entity")
    chunk = _Chunk(text="a chunk")
    n1 = _Entity(name="N1")
    n2 = _Entity(name="N2")
    nodes = [entity, chunk, n1, n2]
    # Distinct relationships per neighbour so n1/n2 don't become duplicates of
    # each other — isolating the cross-type (entity, chunk) pair.
    edges = [
        _edge(entity, n1, "rel_a"),
        _edge(chunk, n1, "rel_a"),
        _edge(entity, n2, "rel_b"),
        _edge(chunk, n2, "rel_b"),
    ]

    merge_edges = resolve_structural_duplicates(nodes, edges)

    involved_pairs = [{edge[0], edge[1]} for edge in merge_edges]
    assert {entity.id, chunk.id} not in involved_pairs


def test_typed_edges_prevent_false_positive():
    """Two nodes touching the same two neighbours through DIFFERENT relationships
    have zero typed overlap and must not merge."""
    a = _Entity(name="A")
    b = _Entity(name="B")
    x = _Entity(name="X")
    y = _Entity(name="Y")
    nodes = [a, b, x, y]
    edges = [
        _edge(a, x, "makes"),
        _edge(b, x, "competes_with"),
        _edge(a, y, "owns"),
        _edge(b, y, "sues"),
    ]

    assert resolve_structural_duplicates(nodes, edges) == []


def test_min_shared_neighbors_blocks_single_overlap():
    """Sharing a single neighbour is below the default min_shared_neighbors=2, so
    the pair is never even scored."""
    a = _Entity(name="A")
    b = _Entity(name="B")
    x = _Entity(name="X")
    nodes = [a, b, x]
    edges = [_edge(a, x, "rel"), _edge(b, x, "rel")]

    assert resolve_structural_duplicates(nodes, edges) == []
    # Lowering the blocking threshold lets the (fully overlapping) pair through.
    linked = resolve_structural_duplicates(nodes, edges, min_shared_neighbors=1)
    assert len(linked) == 1


def test_threshold_is_respected():
    """A partial-overlap pair links below a permissive threshold but not above a
    strict one."""
    a = _Entity(name="A")
    b = _Entity(name="B")
    x = _Entity(name="X")
    y = _Entity(name="Y")
    z = _Entity(name="Z")
    nodes = [a, b, x, y, z]
    # a: {x, y, z}; b: {x, y}  ->  typed Jaccard = 2/3 ≈ 0.67. Distinct
    # relationships keep the shared neighbours x/y from merging with each other.
    edges = [
        _edge(a, x, "rel_x"),
        _edge(b, x, "rel_x"),
        _edge(a, y, "rel_y"),
        _edge(b, y, "rel_y"),
        _edge(a, z, "rel_z"),
    ]

    assert resolve_structural_duplicates(nodes, edges, similarity_threshold=0.7) == []
    assert len(resolve_structural_duplicates(nodes, edges, similarity_threshold=0.6)) == 1


def test_canonical_is_the_better_connected_node():
    """When degrees differ, the higher-degree node is kept as canonical and the
    merged_into edge points from the duplicate to it."""
    a = _Entity(name="A")
    b = _Entity(name="B")
    x = _Entity(name="X")
    y = _Entity(name="Y")
    z = _Entity(name="Z")
    nodes = [a, b, x, y, z]
    # a has degree 3 (x, y, z); b has degree 2 (x, y). Jaccard = 2/3.
    edges = [
        _edge(a, x, "rel_x"),
        _edge(b, x, "rel_x"),
        _edge(a, y, "rel_y"),
        _edge(b, y, "rel_y"),
        _edge(a, z, "rel_z"),
    ]

    merge_edges = resolve_structural_duplicates(nodes, edges, similarity_threshold=0.6)

    assert len(merge_edges) == 1
    duplicate_id, canonical_id, _rel, _props = merge_edges[0]
    assert canonical_id == a.id  # better connected
    assert duplicate_id == b.id


def test_star_merge_one_canonical_absorbs_many_without_chaining():
    """Three mutually-identical nodes collapse onto a single canonical (two
    merged_into edges), never forming a chain."""
    a = _Entity(name="A")
    b = _Entity(name="B")
    c = _Entity(name="C")
    x = _Entity(name="X")
    y = _Entity(name="Y")
    nodes = [a, b, c, x, y]
    edges = []
    for node in (a, b, c):
        edges.append(_edge(node, x, "rel_x"))
        edges.append(_edge(node, y, "rel_y"))

    merge_edges = resolve_structural_duplicates(nodes, edges)

    # All degrees equal, so canonical is the smallest id among the three.
    expected_canonical = min((a.id, b.id, c.id), key=str)
    assert len(merge_edges) == 2
    assert all(edge[1] == expected_canonical for edge in merge_edges)
    duplicates = {edge[0] for edge in merge_edges}
    assert duplicates == {a.id, b.id, c.id} - {expected_canonical}


def test_self_loop_is_not_adjacency_evidence():
    """A self-loop must not contribute to a node's neighbourhood."""
    a = _Entity(name="A")
    b = _Entity(name="B")
    x = _Entity(name="X")
    y = _Entity(name="Y")
    nodes = [a, b, x, y]
    edges = [
        _edge(a, a, "loops"),
        _edge(b, b, "loops"),
        _edge(a, x, "rel_x"),
        _edge(b, x, "rel_x"),
        _edge(a, y, "rel_y"),
        _edge(b, y, "rel_y"),
    ]

    merge_edges = resolve_structural_duplicates(nodes, edges)

    assert len(merge_edges) == 1
    assert {merge_edges[0][0], merge_edges[0][1]} == {a.id, b.id}
    assert merge_edges[0][3]["similarity_score"] == 1.0  # self-loops excluded → identical


def test_input_is_not_mutated():
    nodes, edges, *_ = _apple_fixture()
    edges_before = copy.deepcopy(edges)
    node_ids_before = [node.id for node in nodes]

    resolve_structural_duplicates(nodes, edges)

    assert edges == edges_before
    assert [node.id for node in nodes] == node_ids_before  # no nodes added/removed
