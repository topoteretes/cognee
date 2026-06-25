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


def test_node_coords_per_basis_vector():
    basis = [[1.0, 0.0], [0.0, 1.0]]
    coords = align.node_coords([1.0, 0.0], basis)
    assert coords == [1.0, 0.0]


def test_node_coords_zero_pad_to_basis_count():
    basis = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    coords = align.node_coords([0.0, 0.0], basis)
    assert len(coords) == len(basis)
    assert coords == [0.0, 0.0, 0.0]


def test_query_coords_matches_node_coords():
    basis = [[1.0, 0.0], [0.0, 1.0]]
    assert align.query_coords([0.0, 1.0], basis) == align.node_coords([0.0, 1.0], basis)


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
    # Query-relevance-weighted average of the node's per-direction alignment.
    # Only the first direction has weight here, so the node's first coord wins.
    assert math.isclose(align.truth_score([1.0, 0.0], [1.0, 0.0]), 1.0, rel_tol=1e-9)
    assert math.isclose(align.truth_score([0.5, 0.0], [1.0, 0.0]), 0.5, rel_tol=1e-9)
    # Equal weights -> plain mean of the node coords.
    assert math.isclose(align.truth_score([0.2, 0.8], [0.5, 0.5]), 0.5, rel_tol=1e-9)


def test_truth_score_is_magnitude_sensitive():
    # The whole point: a node aligned MORE strongly with the directions scores higher,
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


def test_stable_signature_stability():
    ids = ["a", "b", "c"]
    sig1 = align.stable_signature(ids)
    sig2 = align.stable_signature(ids)
    assert sig1 == sig2
    assert len(sig1) == 40  # sha1 hex digest


def test_stable_signature_order_sensitive():
    assert align.stable_signature(["a", "b"]) != align.stable_signature(["b", "a"])
