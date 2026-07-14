"""Unit tests for the semantic layout — deterministic offline projection.

No LLM, no vector store: embeddings are handed in directly. Covers PCA
determinism + sign convention, [-1,1] normalization, the no-vector neighbor
fallback, isolated-node ring placement, de-overlap separation, and the UMAP
ImportError fallback (CI has no umap-learn).
"""

import logging

import numpy as np

from cognee.modules.visualization.layouts import semantic_layout
from cognee.modules.visualization.layouts.semantic_layout import (
    MIN_SEPARATION,
    SPREAD,
    compute_positions,
)


def _nodes(ids, ntype="Entity"):
    return [{"id": i, "type": ntype, "name": f"n-{i}"} for i in ids]


# Fixed 4-point embedding in 3-D: two tight pairs separated along axis 0.
FIXED_EMB = {
    "a": [1.0, 0.0, 0.0],
    "b": [1.1, 0.05, 0.0],
    "c": [-1.0, 0.0, 0.0],
    "d": [-1.1, -0.05, 0.0],
}


def test_pca_deterministic_across_runs():
    nodes = _nodes(["a", "b", "c", "d"])
    r1 = compute_positions(nodes, [], FIXED_EMB, seed=42)
    r2 = compute_positions(nodes, [], FIXED_EMB, seed=42)
    assert r1 == r2  # exact equality — pinned, deterministic


def test_all_nodes_positioned_and_within_spread():
    nodes = _nodes(["a", "b", "c", "d"])
    pos = compute_positions(nodes, [], FIXED_EMB, seed=42)
    assert set(pos) == {"a", "b", "c", "d"}
    for p in pos.values():
        # De-overlap can nudge slightly past the box; allow a small margin.
        assert -SPREAD - 0.2 <= p["x"] <= SPREAD + 0.2
        assert -SPREAD - 0.2 <= p["y"] <= SPREAD + 0.2


def test_positions_invariant_to_node_order():
    # The sign convention pins SVD's arbitrary internal sign, so the projection
    # is a pure function of the embedding set — independent of node ordering.
    fwd = _nodes(["a", "b", "c", "d"])
    rev = _nodes(["d", "c", "b", "a"])
    p_fwd = compute_positions(fwd, [], FIXED_EMB, seed=42)
    p_rev = compute_positions(rev, [], FIXED_EMB, seed=42)
    assert p_fwd == p_rev


def test_primary_axis_separates_clusters():
    # The two tight pairs are far apart on embedding axis 0 -> they must land on
    # opposite sides of the projection's primary axis.
    nodes = _nodes(["a", "b", "c", "d"])
    pos = compute_positions(nodes, [], FIXED_EMB, seed=42)
    left = {"a", "b"}
    right = {"c", "d"}
    left_x = np.mean([pos[i]["x"] for i in left])
    right_x = np.mean([pos[i]["x"] for i in right])
    assert abs(left_x - right_x) > 0.5  # clearly separated


def test_no_vector_node_placed_at_neighbor_centroid():
    # 'x' has no embedding but links to 'a' and 'c'; it should sit near their
    # midpoint (small seeded jitter aside), not on the ring.
    nodes = _nodes(["a", "b", "c", "d", "x"])
    links = [{"source": "x", "target": "a"}, {"source": "x", "target": "c"}]
    pos = compute_positions(nodes, links, FIXED_EMB, seed=42)
    midpoint = np.array([(pos["a"]["x"] + pos["c"]["x"]) / 2, (pos["a"]["y"] + pos["c"]["y"]) / 2])
    placed = np.array([pos["x"]["x"], pos["x"]["y"]])
    assert np.linalg.norm(placed - midpoint) < 0.25  # jitter + de-overlap tolerance


def test_isolated_no_vector_node_on_ring():
    # 'iso' has no embedding and no links -> deterministic ring, outside the box.
    nodes = _nodes(["a", "b", "c", "d", "iso"])
    pos = compute_positions(nodes, [], FIXED_EMB, seed=42)
    r = np.hypot(pos["iso"]["x"], pos["iso"]["y"])
    assert r > SPREAD  # pushed to the ring radius (1.15 * spread)


def test_deoverlap_separates_coincident_points():
    # Four identical embeddings would collapse to one point; de-overlap must
    # spread them to at least ~MIN_SEPARATION apart.
    nodes = _nodes(["a", "b", "c", "d"])
    same = {k: [0.5, 0.5, 0.5] for k in ["a", "b", "c", "d"]}
    pos = compute_positions(nodes, [], same, seed=42)
    pts = np.array([[pos[i]["x"], pos[i]["y"]] for i in ["a", "b", "c", "d"]])
    dists = [
        np.linalg.norm(pts[i] - pts[j]) for i in range(len(pts)) for j in range(i + 1, len(pts))
    ]
    assert min(dists) >= MIN_SEPARATION * SPREAD * 0.9


def test_umap_method_falls_back_to_pca(monkeypatch, caplog):
    # Force the umap import to fail; method="umap" must fall back to PCA and log.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "umap":
            raise ImportError("no umap")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    nodes = _nodes(["a", "b", "c", "d"])
    with caplog.at_level(logging.INFO):
        umap_pos = compute_positions(nodes, [], FIXED_EMB, method="umap", seed=42)
    pca_pos = compute_positions(nodes, [], FIXED_EMB, method="pca", seed=42)
    assert umap_pos == pca_pos  # identical -> fell back to PCA
    assert "falling back to PCA" in caplog.text  # log half of the contract


def test_emit_js_carries_position_token():
    assert "__SEMANTIC_POSITIONS__" in semantic_layout.emit_js()


def test_empty_and_single_node_graphs():
    assert compute_positions([], [], {}) == {}
    single = compute_positions(_nodes(["a"]), [], {"a": [1.0, 2.0, 3.0]}, seed=42)
    assert set(single) == {"a"}
