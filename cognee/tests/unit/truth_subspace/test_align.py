import math

import cognee.modules.truth_subspace.align as align


def test_cosine_identical_vectors():
    assert align.cosine([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_orthogonal_vectors():
    assert align.cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_opposite_vectors():
    assert align.cosine([1.0, 0.0], [-1.0, 0.0]) == -1.0


def test_cosine_scale_invariant():
    assert math.isclose(align.cosine([1.0, 1.0], [2.0, 2.0]), 1.0, rel_tol=1e-9)


def test_cosine_zero_vector_returns_zero():
    assert align.cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert align.cosine([1.0, 1.0], [0.0, 0.0]) == 0.0


def test_cosine_empty_vector_returns_zero():
    assert align.cosine([], [1.0]) == 0.0
    assert align.cosine([1.0], []) == 0.0


def test_node_coords_per_anchor():
    anchors = [[1.0, 0.0], [0.0, 1.0]]
    coords = align.node_coords([1.0, 0.0], anchors)
    assert coords == [1.0, 0.0]


def test_node_coords_zero_pad_to_anchor_count():
    anchors = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    coords = align.node_coords([0.0, 0.0], anchors)
    assert len(coords) == len(anchors)
    assert coords == [0.0, 0.0, 0.0]


def test_query_coords_matches_node_coords():
    anchors = [[1.0, 0.0], [0.0, 1.0]]
    assert align.query_coords([0.0, 1.0], anchors) == align.node_coords([0.0, 1.0], anchors)


def test_truth_factor_neutral_when_coords_missing():
    # NEUTRAL: empty coords => factor exactly 1.0
    assert align.truth_factor([], []) == 1.0
    assert align.truth_factor([0.1], []) == 1.0
    assert align.truth_factor([], [0.1]) == 1.0


def test_truth_factor_neutral_when_coords_zero():
    # Zero coord vectors => cosine 0.0 => score 0.5 => factor 1.0
    assert align.truth_factor([0.0, 0.0], [0.0, 0.0]) == 1.0


def test_truth_score_neutral_cases():
    # NEUTRAL (0.5): empty coords, or a query with no positive weight to spread.
    assert align.truth_score([], []) == 0.5
    assert align.truth_score([0.0, 0.0], [0.0, 0.0]) == 0.5
    assert align.truth_score([1.0, 1.0], [0.0, 0.0]) == 0.5
    # Negative query coords contribute no weight -> still neutral.
    assert align.truth_score([1.0, 1.0], [-1.0, 0.0]) == 0.5


def test_truth_score_weighted_alignment():
    # Query-relevance-weighted average of the node's per-anchor alignment.
    # Only the first anchor has weight here, so the node's first coord wins.
    assert math.isclose(align.truth_score([1.0, 0.0], [1.0, 0.0]), 1.0, rel_tol=1e-9)
    assert math.isclose(align.truth_score([0.5, 0.0], [1.0, 0.0]), 0.5, rel_tol=1e-9)
    # Equal weights -> plain mean of the node coords.
    assert math.isclose(align.truth_score([0.2, 0.8], [0.5, 0.5]), 0.5, rel_tol=1e-9)


def test_truth_score_is_magnitude_sensitive():
    # The whole point: a node aligned MORE strongly with the anchors scores higher,
    # even though both point the same direction (cosine would call them equal).
    q = [0.3, 0.3]
    assert align.truth_score([0.4, 0.4], q) > align.truth_score([0.2, 0.2], q)


def test_truth_factor_within_bounds():
    cases = [
        ([1.0, 0.0], [1.0, 0.0]),
        ([0.0, 0.0], [1.0, 1.0]),
        ([1.0, 1.0], [1.0, 0.0]),
        ([0.3, 0.7, 0.1], [0.2, 0.9, 0.4]),
    ]
    for nc, qc in cases:
        factor = align.truth_factor(nc, qc)
        assert 0.75 <= factor <= 1.25

    # Extremes hit the bounds exactly.
    assert math.isclose(align.truth_factor([1.0, 0.0], [1.0, 0.0]), 1.25, rel_tol=1e-9)
    assert math.isclose(align.truth_factor([0.0, 0.0], [1.0, 1.0]), 0.75, rel_tol=1e-9)


def test_active_anchor_order_recency():
    anchors = [
        {"id": "a", "created_at": 100},
        {"id": "b", "created_at": 300},
        {"id": "c", "created_at": 200},
    ]
    ordered = align.active_anchor_order(anchors, 2)
    assert [a["id"] for a in ordered] == ["b", "c"]


def test_active_anchor_order_tiebreak_by_id():
    anchors = [
        {"id": "z", "created_at": 100},
        {"id": "a", "created_at": 100},
        {"id": "m", "created_at": 100},
    ]
    ordered = align.active_anchor_order(anchors, 3)
    assert [a["id"] for a in ordered] == ["a", "m", "z"]


def test_active_anchor_order_takes_k():
    anchors = [{"id": str(i), "created_at": i} for i in range(20)]
    ordered = align.active_anchor_order(anchors, 8)
    assert len(ordered) == 8


def test_active_anchor_order_supports_attribute_objects():
    class _Anchor:
        def __init__(self, anchor_id, created_at):
            self.id = anchor_id
            self.created_at = created_at

    anchors = [_Anchor("a", 100), _Anchor("b", 200)]
    ordered = align.active_anchor_order(anchors, 1)
    assert ordered[0].id == "b"


def test_anchor_signature_stability():
    ids = ["a", "b", "c"]
    sig1 = align.anchor_signature(ids)
    sig2 = align.anchor_signature(ids)
    assert sig1 == sig2
    assert len(sig1) == 40  # sha1 hex digest


def test_anchor_signature_order_sensitive():
    assert align.anchor_signature(["a", "b"]) != align.anchor_signature(["b", "a"])
