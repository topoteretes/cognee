"""Semantic layout: project node embeddings to 2-D pinned positions.

Given ``{node_id: vector}`` from ``embedding_join.fetch_node_embeddings`` and the
graph's links, this computes one 2-D coordinate per node so semantically similar
nodes land near each other. Positions are a deterministic pure function of the
inputs — no force simulation, no un-seeded randomness — so they can be pinned and
rendered layout-once (the repo's rule, see ``memory_map.js`` header).

Pipeline:
  * PCA via numpy SVD (default), with a deterministic sign convention so snapshot
    tests are stable — raw SVD sign is arbitrary. Optional UMAP via lazy import;
    ImportError falls back to PCA.
  * Normalize to [-1, 1] per axis, scaled by ``spread``.
  * Nodes without a vector are placed at the seeded-jittered centroid of their
    positioned neighbors; nodes with no positioned neighbor land on a
    deterministic ring.
  * A deterministic seeded de-overlap pass spreads coincident points so dense
    clusters stay legible.
"""

import math
from typing import Any, Dict, List, Optional

import numpy as np

from cognee.shared.logging_utils import get_logger

logger = get_logger("semantic_layout")

# Half-width of the normalized coordinate box; the renderer scales this to canvas.
SPREAD = 1.0
# Minimum separation (in normalized units) enforced by the de-overlap pass.
MIN_SEPARATION = 0.02
LAYOUT_SEED = 42


def _pca_2d(matrix: np.ndarray) -> np.ndarray:
    """Project rows of ``matrix`` onto their first two principal components.

    Sign convention: for each component, the feature loading with the largest
    magnitude is forced positive. Raw SVD sign is arbitrary, so without this the
    same input could flip across runs and break snapshot equality.
    """
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    # full_matrices=False keeps Vt at (min(n, d), d); deterministic given input.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]
    if components.shape[0] < 2:
        # Degenerate (single component): pad the second axis with zeros.
        pad = np.zeros((2 - components.shape[0], components.shape[1]))
        components = np.vstack([components, pad])
    for i in range(2):
        loading = components[i]
        j = int(np.argmax(np.abs(loading)))
        if loading[j] < 0:
            components[i] = -loading
    return centered @ components.T


def _umap_2d(matrix: np.ndarray, seed: int) -> Optional[np.ndarray]:
    """Optional UMAP projection. Returns None if UMAP isn't installed."""
    try:
        import umap  # lazy: umap-learn is not a cognee dependency
    except ImportError:
        logger.info("UMAP not installed; falling back to PCA for semantic layout.")
        return None
    reducer = umap.UMAP(n_components=2, random_state=seed, n_jobs=1)
    return reducer.fit_transform(matrix)


def _normalize(coords: np.ndarray, spread: float) -> np.ndarray:
    """Min-max normalize each axis into [-spread, spread]."""
    out = np.zeros_like(coords, dtype=float)
    for axis in range(coords.shape[1]):
        col = coords[:, axis]
        lo, hi = float(col.min()), float(col.max())
        if hi > lo:
            out[:, axis] = (2.0 * (col - lo) / (hi - lo) - 1.0) * spread
        # else: constant axis -> leave at 0
    return out


def _place_missing(
    node_ids: List[str],
    embedded_pos: Dict[str, np.ndarray],
    adjacency: Dict[str, set],
    spread: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Position nodes without a vector.

    Neighbor-centroid with seeded jitter, iterated so chains of vector-less nodes
    resolve; nodes with no positioned neighbor land on a deterministic ring.
    """
    positioned = dict(embedded_pos)
    missing = [nid for nid in node_ids if nid not in positioned]

    changed = True
    while changed and missing:
        changed = False
        still_missing = []
        for nid in missing:  # node_ids is pre-sorted -> deterministic order
            neighbor_pts = [positioned[n] for n in adjacency.get(nid, ()) if n in positioned]
            if neighbor_pts:
                centroid = np.mean(neighbor_pts, axis=0)
                jitter = rng.uniform(-0.03, 0.03, size=2) * spread
                positioned[nid] = centroid + jitter
                changed = True
            else:
                still_missing.append(nid)
        missing = still_missing

    # Whatever remains is disconnected from every positioned node: ring it.
    for k, nid in enumerate(missing):
        angle = 2.0 * math.pi * k / max(1, len(missing))
        positioned[nid] = np.array(
            [1.15 * spread * math.cos(angle), 1.15 * spread * math.sin(angle)]
        )
    return positioned


def _deoverlap(
    ordered_ids: List[str],
    positioned: Dict[str, np.ndarray],
    min_dist: float,
    rng: np.random.Generator,
    iterations: int = 40,
) -> Dict[str, np.ndarray]:
    """Deterministic seeded relaxation: push apart points closer than ``min_dist``.

    ponytail: O(n²) per iteration; fine at the 2000-node fetch cap. Swap in a
    spatial grid if the cap ever grows.
    """
    if len(ordered_ids) < 2:
        return positioned
    pts = np.array([positioned[nid] for nid in ordered_ids], dtype=float)
    # Seeded tie-breaking nudge so exactly-coincident points separate deterministically.
    pts = pts + rng.uniform(-min_dist / 4, min_dist / 4, size=pts.shape)
    for _ in range(iterations):
        diff = pts[:, None, :] - pts[None, :, :]  # (n, n, 2)
        dist = np.sqrt((diff**2).sum(axis=2))  # (n, n)
        np.fill_diagonal(dist, np.inf)
        too_close = dist < min_dist
        if not too_close.any():
            break
        safe = np.where(dist == 0, 1.0, dist)
        push = (min_dist - dist) / safe
        push = np.where(too_close, push, 0.0)
        shift = (diff * push[:, :, None]).sum(axis=1) * 0.5
        pts = pts + shift
    return {nid: pts[i] for i, nid in enumerate(ordered_ids)}


def compute_positions(
    nodes: List[Dict[str, Any]],
    links: List[Dict[str, Any]],
    embeddings: Dict[str, List[float]],
    *,
    method: str = "pca",
    seed: int = LAYOUT_SEED,
    spread: float = SPREAD,
) -> Dict[str, Dict[str, float]]:
    """Return ``{node_id: {"x": float, "y": float}}`` for every node.

    Deterministic given identical inputs. ``method`` is ``"pca"`` (default) or
    ``"umap"`` (falls back to PCA when umap-learn is absent).
    """
    node_ids = sorted(str(n["id"]) for n in nodes)
    rng = np.random.default_rng(seed)

    adjacency: Dict[str, set] = {nid: set() for nid in node_ids}
    for link in links:
        s, t = str(link["source"]), str(link["target"])
        if s in adjacency and t in adjacency:
            adjacency[s].add(t)
            adjacency[t].add(s)

    # Embedded nodes with a consistent dimension.
    embedded_ids = [nid for nid in node_ids if nid in embeddings]
    embedded_pos: Dict[str, np.ndarray] = {}
    if len(embedded_ids) >= 2:
        matrix = np.array([embeddings[nid] for nid in embedded_ids], dtype=float)
        coords = None
        if method == "umap":
            coords = _umap_2d(matrix, seed)
        if coords is None:
            coords = _pca_2d(matrix)
        coords = _normalize(np.asarray(coords, dtype=float), spread)
        embedded_pos = {nid: coords[i] for i, nid in enumerate(embedded_ids)}
    elif len(embedded_ids) == 1:
        embedded_pos = {embedded_ids[0]: np.zeros(2)}

    positioned = _place_missing(node_ids, embedded_pos, adjacency, spread, rng)
    positioned = _deoverlap(node_ids, positioned, MIN_SEPARATION * spread, rng)

    return {nid: {"x": float(p[0]), "y": float(p[1])} for nid, p in positioned.items()}


def emit_js(_preprocessed=None) -> str:
    """Expose the pinned semantic positions to the renderer via a data token.

    The orchestrator substitutes ``__SEMANTIC_POSITIONS__`` with the JSON produced
    by ``compute_positions``; the semantic view reads ``window._semanticPositions``.
    """
    return "window._semanticPositions = __SEMANTIC_POSITIONS__;"
