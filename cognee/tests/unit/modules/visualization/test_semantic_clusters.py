"""Unit tests for clustering + NN precompute — deterministic, offline, no LLM.

Covers seeded k-means determinism and k bounds, cosine nearest-neighbor
correctness, the deterministic top-entity label path, and the optional
(mocked) summary seam.
"""

from unittest.mock import Mock

import numpy as np

from cognee.modules.visualization.semantic_clusters import (
    compute_clusters,
    default_k,
    kmeans,
)

# Two well-separated blobs in 3-D.
BLOB = {
    "a": [10.0, 0.0, 0.0],
    "b": [10.1, 0.1, 0.0],
    "c": [10.0, -0.1, 0.1],
    "d": [-10.0, 0.0, 0.0],
    "e": [-10.1, 0.1, 0.0],
    "f": [-10.0, -0.1, 0.1],
}


def _nodes(ids, degree=None):
    degree = degree or {}
    return [{"id": i, "type": "Entity", "name": f"N{i}", "degree": degree.get(i, 0)} for i in ids]


def test_default_k_bounds():
    assert default_k(0) == 1
    assert default_k(1) == 1
    assert default_k(2) == 2  # max(2, ...)
    assert default_k(10_000) == 12  # capped at 12
    assert default_k(8) == 2  # round(sqrt(4)) == 2


def test_kmeans_deterministic_and_separates_blobs():
    x = np.array([BLOB[i] for i in ["a", "b", "c", "d", "e", "f"]])
    l1 = kmeans(x, 2, seed=42)
    l2 = kmeans(x, 2, seed=42)
    assert np.array_equal(l1, l2)  # identical across runs
    # First three (positive blob) share a label; last three share the other.
    assert l1[0] == l1[1] == l1[2]
    assert l1[3] == l1[4] == l1[5]
    assert l1[0] != l1[3]


def test_compute_clusters_two_groups():
    nodes = _nodes(["a", "b", "c", "d", "e", "f"])
    result = compute_clusters(nodes, BLOB, k=2, seed=42)
    assert len(result["clusters"]) == 2
    # Every embedded node is assigned.
    assert set(result["node_cluster"]) == set(BLOB)
    # The two positive-blob members land in the same cluster.
    assert result["node_cluster"]["a"] == result["node_cluster"]["b"]
    assert result["node_cluster"]["a"] != result["node_cluster"]["d"]


def test_nearest_neighbors_cosine_topk():
    nodes = _nodes(["a", "b", "c", "d", "e", "f"])
    result = compute_clusters(nodes, BLOB, k=2, seed=42)
    # 'a' neighbors must be its blob-mates (b, c) ahead of the far blob.
    nbrs = result["neighbors"]["a"]
    assert len(nbrs) == 5  # top-5, self excluded
    assert nbrs[0] in {"b", "c"}
    assert nbrs[1] in {"b", "c"}
    assert "a" not in nbrs


def test_labels_use_top_degree_entities():
    # 'b' has the highest degree in the positive blob -> leads its cluster label.
    nodes = _nodes(["a", "b", "c", "d", "e", "f"], degree={"b": 9, "a": 1, "d": 9})
    result = compute_clusters(nodes, BLOB, k=2, seed=42)
    labels = {c["id"]: c["label"] for c in result["clusters"]}
    pos_cluster = result["node_cluster"]["b"]
    assert labels[pos_cluster].startswith("Nb")  # name of highest-degree member


def test_summarize_seam_is_used_when_provided():
    nodes = _nodes(["a", "b", "c", "d", "e", "f"])
    summarize = Mock(return_value="SUMMARY")
    result = compute_clusters(nodes, BLOB, k=2, seed=42, summarize=summarize)
    assert all(c["label"] == "SUMMARY" for c in result["clusters"])
    assert summarize.call_count == 2  # once per cluster


def test_nodes_without_vectors_are_absent():
    nodes = _nodes(["a", "b", "c", "d", "e", "f", "novec"])
    result = compute_clusters(nodes, BLOB, k=2, seed=42)
    assert "novec" not in result["node_cluster"]
    assert "novec" not in result["neighbors"]


def test_empty_embeddings():
    result = compute_clusters(_nodes(["a"]), {}, seed=42)
    assert result == {"clusters": [], "node_cluster": {}, "neighbors": {}}
