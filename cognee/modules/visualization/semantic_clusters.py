"""Cluster node embeddings and precompute nearest neighbors for the semantic map.

Clustering runs on the *full-dimensional* embeddings (not the 2-D projection),
so groupings reflect real semantic structure rather than a lossy layout. Output
feeds the ``__SEMANTIC_CLUSTERS__`` data token: per-node cluster id + top-5 cosine
neighbors (powering the hover panel without shipping raw vectors), plus a label
per cluster.

k-means is pure numpy (seeded k-means++ init, fixed iteration order) so results
are identical across runs — no scikit-learn, which lives only in the evals extra.
HDBSCAN is an optional lazy path for density clustering.
"""

import math
from collections import Counter
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from cognee.modules.visualization.preprocessor import looks_like_identifier
from cognee.shared.logging_utils import get_logger

logger = get_logger("semantic_clusters")

CLUSTER_SEED = 42
TOP_NEIGHBORS = 5
_EPS = 1e-12
# Names longer than this read as chunk/summary text, not entity labels.
_MAX_LABEL_NAME = 40


def default_k(n: int) -> int:
    """k = min(12, max(2, round(sqrt(n/2)))), clamped to a sane range."""
    if n < 2:
        return 1
    return min(12, max(2, round(math.sqrt(n / 2))))


def _kmeans_pp_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Seeded k-means++ center selection."""
    n = len(x)
    first = int(rng.integers(n))
    chosen = [first]
    d2 = ((x - x[first]) ** 2).sum(axis=1)
    for _ in range(1, k):
        total = d2.sum()
        probs = d2 / total if total > 0 else np.full(n, 1.0 / n)
        idx = int(rng.choice(n, p=probs))
        chosen.append(idx)
        d2 = np.minimum(d2, ((x - x[idx]) ** 2).sum(axis=1))
    return x[chosen].copy()


def kmeans(x: np.ndarray, k: int, seed: int = CLUSTER_SEED, max_iter: int = 50) -> np.ndarray:
    """Pure-numpy Lloyd's k-means. Deterministic given (x, k, seed).

    Ties in assignment go to the lowest cluster index (argmin); empty clusters
    keep their center rather than being re-seeded, so iteration order is fixed.
    """
    n = len(x)
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    centers = _kmeans_pp_init(x, k, rng)
    labels = np.full(n, -1, dtype=int)
    for _ in range(max_iter):
        dists = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if mask.any():
                centers[c] = x[mask].mean(axis=0)
    return labels


def _nearest_neighbors(ids: List[str], x: np.ndarray, top: int) -> Dict[str, List[str]]:
    """Top-``top`` cosine neighbors per node (self excluded, stable tie order)."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    unit = x / (norms + _EPS)
    sim = unit @ unit.T
    np.fill_diagonal(sim, -np.inf)
    out: Dict[str, List[str]] = {}
    for i, nid in enumerate(ids):
        order = np.argsort(-sim[i], kind="stable")[:top]
        out[nid] = [ids[j] for j in order]
    return out


def _usable_name(nd: Dict[str, Any]) -> Optional[str]:
    """A node's name if it reads as a clean label — not a UUID/hash or a text blob."""
    name = nd.get("name")
    if not isinstance(name, str):
        return None
    name = name.strip()
    if not name or len(name) > _MAX_LABEL_NAME or looks_like_identifier(name):
        return None
    return name


def _cluster_label(member_nodes: List[Dict[str, Any]]) -> str:
    """Label a cluster by its top-3 real ``Entity`` nodes (by degree, importance).

    Entities win over DocumentChunk/TextSummary/EntityType so labels read as
    concepts, not chunk text or type names. Identifier-shaped or over-long names
    are skipped; a cluster with no usable name falls back to its dominant node
    type (e.g. ``"TextSummary"``), then to ``"cluster"``.
    """
    ranked = sorted(
        member_nodes,
        key=lambda nd: (nd.get("type") == "Entity", nd.get("degree", 0), nd.get("importance", 0.0)),
        reverse=True,
    )
    names = [n for nd in ranked if (n := _usable_name(nd))]
    if names:
        return ", ".join(names[:3])
    # No usable names (e.g. a cluster of chunks/summaries): name it by dominant type.
    types = [nd.get("type") for nd in member_nodes if nd.get("type")]
    return Counter(types).most_common(1)[0][0] if types else "cluster"


def compute_clusters(
    nodes: List[Dict[str, Any]],
    embeddings: Dict[str, List[float]],
    *,
    k: Optional[int] = None,
    seed: int = CLUSTER_SEED,
    summarize: Optional[Callable[[str, List[str]], str]] = None,
) -> Dict[str, Any]:
    """Cluster embedded nodes and precompute neighbors.

    Returns ``{"clusters": [...], "node_cluster": {id: cluster_id},
    "neighbors": {id: [neighbor_id, ...]}}``. Nodes without a vector are absent
    from all three (the view leaves them uncolored).

    ``summarize`` is an optional seam for a one-line LLM cluster summary; when
    None (default and in CI) the deterministic top-entity label is used. Tests
    inject a plain mock here — this does not depend on the #3601 LLM harness.
    """
    node_by_id = {str(n["id"]): n for n in nodes}
    ids = sorted(nid for nid in node_by_id if nid in embeddings)
    if not ids:
        return {"clusters": [], "node_cluster": {}, "neighbors": {}}

    x = np.array([embeddings[nid] for nid in ids], dtype=float)
    k_effective = default_k(len(ids)) if k is None else k
    k_effective = max(1, min(k_effective, len(ids)))

    labels = kmeans(x, k_effective, seed)
    neighbors = _nearest_neighbors(ids, x, TOP_NEIGHBORS)

    clusters: List[Dict[str, Any]] = []
    node_cluster: Dict[str, int] = {}
    for c in range(k_effective):
        members = [ids[i] for i in range(len(ids)) if labels[i] == c]
        if not members:
            continue
        member_nodes = [node_by_id[m] for m in members]
        label = _cluster_label(member_nodes)
        if summarize is not None:
            top_names = [nd.get("name") for nd in member_nodes[:3] if nd.get("name")]
            label = summarize(label, top_names)
        for m in members:
            node_cluster[m] = c
        clusters.append({"id": c, "label": label, "node_ids": members, "size": len(members)})

    return {"clusters": clusters, "node_cluster": node_cluster, "neighbors": neighbors}
