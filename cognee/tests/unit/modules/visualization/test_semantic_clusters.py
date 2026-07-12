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


def test_nearest_neighbors_excludes_self_on_small_graphs():
    # With <= TOP_NEIGHBORS (5) embedded nodes the self index would fall inside
    # the top-k slice unless it is explicitly dropped. Every node's neighbor list
    # must exclude itself regardless of graph size.
    ids = ["a", "b", "c"]  # 3 nodes < top-5
    result = compute_clusters(_nodes(ids), {i: BLOB[i] for i in ids}, k=2, seed=42)
    for nid in ids:
        nbrs = result["neighbors"][nid]
        assert nid not in nbrs
        assert len(nbrs) == len(ids) - 1  # every other node, self excluded


def test_labels_use_top_degree_entities():
    # 'b' has the highest degree in the positive blob -> leads its cluster label.
    nodes = _nodes(["a", "b", "c", "d", "e", "f"], degree={"b": 9, "a": 1, "d": 9})
    result = compute_clusters(nodes, BLOB, k=2, seed=42)
    labels = {c["id"]: c["label"] for c in result["clusters"]}
    pos_cluster = result["node_cluster"]["b"]
    assert labels[pos_cluster].startswith("Nb")  # name of highest-degree member


def test_label_prefers_entities_over_chunk_text():
    # One blob mixes a low-degree Entity with a high-degree DocumentChunk whose
    # "name" is a long text blob. The label must use the entity, never the chunk.
    chunk_text = "This is a long document chunk sentence that should never become a label."
    nodes = [
        {"id": "a", "type": "DocumentChunk", "name": chunk_text, "degree": 99},
        {"id": "b", "type": "Entity", "name": "Ada Lovelace", "degree": 1},
        {"id": "c", "type": "Entity", "name": "deadbeefdeadbeefdeadbeefdeadbeef", "degree": 5},
        {"id": "d", "type": "Entity", "name": "Turing", "degree": 2},
        {"id": "e", "type": "Entity", "name": "Babbage", "degree": 2},
        {"id": "f", "type": "Entity", "name": "London", "degree": 2},
    ]
    result = compute_clusters(nodes, BLOB, k=2, seed=42)
    labels = [c["label"] for c in result["clusters"]]
    pos_label = labels[result["node_cluster"]["a"]]
    assert "document chunk" not in pos_label.lower()  # chunk text excluded
    assert "deadbeef" not in pos_label  # identifier-shaped name skipped
    assert "Ada Lovelace" in pos_label  # real entity used despite low degree


def test_label_skips_preprocessor_flagged_unnamed_nodes():
    # The preprocessor renames UUID-named nodes to "Unnamed Entity (ab12cd34)"
    # and flags them is_unnamed; those placeholders must never become labels.
    nodes = [
        {
            "id": "a",
            "type": "Entity",
            "name": "Unnamed Entity (ab12cd34)",
            "is_unnamed": True,
            "degree": 99,
        },
        {"id": "b", "type": "Entity", "name": "Ada Lovelace", "degree": 1},
        {"id": "c", "type": "Entity", "name": "Turing", "degree": 1},
        {
            "id": "d",
            "type": "Entity",
            "name": "Unnamed Entity (ee55ff66)",
            "is_unnamed": True,
            "degree": 99,
        },
        {"id": "e", "type": "Entity", "name": "Babbage", "degree": 1},
        {"id": "f", "type": "Entity", "name": "London", "degree": 1},
    ]
    result = compute_clusters(nodes, BLOB, k=2, seed=42)
    for cluster in result["clusters"]:
        assert "Unnamed" not in cluster["label"]


def test_label_falls_back_to_dominant_type_when_no_names():
    # A cluster of nameless chunks/summaries: no usable name, so the label is the
    # dominant node type rather than the generic "cluster".
    nodes = [
        {"id": "a", "type": "TextSummary", "text": "summary one"},
        {"id": "b", "type": "TextSummary", "text": "summary two"},
        {"id": "c", "type": "DocumentChunk", "text": "a chunk"},
        {"id": "d", "type": "TextSummary", "text": "summary three"},
        {"id": "e", "type": "TextSummary", "text": "summary four"},
        {"id": "f", "type": "DocumentChunk", "text": "another chunk"},
    ]
    result = compute_clusters(nodes, BLOB, k=2, seed=42)
    labels = [c["label"] for c in result["clusters"]]
    assert labels  # every cluster is labelled
    assert all(lbl in {"TextSummary", "DocumentChunk"} for lbl in labels)
    assert "cluster" not in labels  # generic fallback no longer reached


def test_label_fn_seam_overrides_and_receives_member_nodes():
    # A custom label_fn (e.g. an LLM summarizer) fully replaces the default and is
    # called once per cluster with that cluster's member nodes — no discarded
    # default label is computed first.
    nodes = _nodes(["a", "b", "c", "d", "e", "f"])
    label_fn = Mock(return_value="SUMMARY")
    result = compute_clusters(nodes, BLOB, k=2, seed=42, label_fn=label_fn)
    assert all(c["label"] == "SUMMARY" for c in result["clusters"])
    assert label_fn.call_count == 2  # once per cluster
    # Each call gets a list of member-node dicts, not the default label string.
    for call in label_fn.call_args_list:
        (member_nodes,) = call.args
        assert isinstance(member_nodes, list)
        assert all(isinstance(nd, dict) and "id" in nd for nd in member_nodes)


def test_nodes_without_vectors_are_absent():
    nodes = _nodes(["a", "b", "c", "d", "e", "f", "novec"])
    result = compute_clusters(nodes, BLOB, k=2, seed=42)
    assert "novec" not in result["node_cluster"]
    assert "novec" not in result["neighbors"]


def test_empty_embeddings():
    result = compute_clusters(_nodes(["a"]), {}, seed=42)
    assert result == {"clusters": [], "node_cluster": {}, "neighbors": {}}
