"""Semantic layout: position nodes by embedding similarity in 2D space.

Uses PCA by default (no extra deps — numpy is already required by cognee).
Automatically upgrades to UMAP when the ``umap-learn`` package is installed.

The layout is exposed as a drop-in alongside ``pipeline_layout``:

    from cognee.modules.visualization.layouts import semantic_layout
    html = html.replace("__SEMANTIC_LAYOUT_JS__", semantic_layout.emit_js(pre))

``pre`` is the ``PreprocessedGraph`` produced by the preprocessor.
Node embeddings are looked up from the vector store by node id; nodes with
no embedding fall back to the graph-topology centroid of their neighbours.
"""

from __future__ import annotations

import json
import logging
import math
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cognee.modules.visualization.preprocessor import PreprocessedGraph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Cap on nodes sent through the projection (performance guard for huge graphs).
_MAX_NODES: int = 2_000

# How much to scale the projected coordinates so D3 spreads nodes nicely.
_SPREAD: float = 800.0

# Minimum number of nodes needed before projection makes sense.
_MIN_NODES_FOR_PROJECTION: int = 3


# ---------------------------------------------------------------------------
# Embedding retrieval
# ---------------------------------------------------------------------------

async def _fetch_embeddings(node_ids: list[str]) -> dict[str, list[float]]:
    """Return {node_id: embedding_vector} for as many ids as possible.

    Silently skips nodes whose embeddings cannot be found — the caller
    handles missing entries with a fallback.
    """
    result: dict[str, list[float]] = {}
    if not node_ids:
        return result

    try:
        from cognee.infrastructure.databases.vector import get_vector_engine
        vector_engine = await get_vector_engine()

        for node_id in node_ids:
            try:
                # Try fetching the stored embedding by node id
                records = await vector_engine.search(
                    collection_name="entities",
                    query_text=None,
                    query_vector=None,
                    limit=1,
                    with_payload=True,
                    filter={"must": [{"key": "node_id", "match": {"value": node_id}}]},
                )
                if records:
                    vec = getattr(records[0], "vector", None)
                    if vec is not None:
                        result[node_id] = list(vec)
            except Exception:
                pass  # Missing embedding — handled by fallback below

    except Exception as exc:
        logger.debug("Could not fetch embeddings from vector store: %s", exc)

    return result


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def _pca_2d(matrix: list[list[float]]) -> list[tuple[float, float]]:
    """Pure-numpy PCA projection to 2D.  Always available."""
    import numpy as np

    X = np.array(matrix, dtype=float)
    X -= X.mean(axis=0)

    # Covariance via SVD (more numerically stable than eig on cov matrix)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    components = Vt[:2]  # shape (min(2,d), d)
    projected = X @ components.T  # shape (n, min(2,d))

    # Pad to (n, 2) if embedding dimension < 2
    if projected.shape[1] < 2:
        pad = np.zeros((projected.shape[0], 2 - projected.shape[1]))
        projected = np.hstack([projected, pad])

    # Normalise to [-1, 1]
    for dim in range(2):
        col = projected[:, dim]
        span = col.max() - col.min()
        if span > 1e-9:
            projected[:, dim] = (col - col.min()) / span * 2 - 1

    return [(float(row[0]), float(row[1])) for row in projected]


def _umap_2d(matrix: list[list[float]], seed: int = 42) -> list[tuple[float, float]]:
    """UMAP projection — only called when umap-learn is installed."""
    import numpy as np
    import umap  # type: ignore[import]

    X = np.array(matrix, dtype=float)
    reducer = umap.UMAP(n_components=2, random_state=seed, n_jobs=1)
    projected = reducer.fit_transform(X)

    for dim in range(2):
        col = projected[:, dim]
        span = col.max() - col.min()
        if span > 1e-9:
            projected[:, dim] = (col - col.min()) / span * 2 - 1

    return [(float(row[0]), float(row[1])) for row in projected]


def _project(matrix: list[list[float]]) -> list[tuple[float, float]]:
    """Choose UMAP if available, fall back to PCA."""
    try:
        import umap  # noqa: F401
        logger.debug("semantic_layout: using UMAP projection")
        return _umap_2d(matrix)
    except ImportError:
        logger.debug("semantic_layout: umap-learn not found, falling back to PCA")
        return _pca_2d(matrix)


# ---------------------------------------------------------------------------
# Fallback positions
# ---------------------------------------------------------------------------

def _topology_fallback(
    missing_ids: list[str],
    id_to_pos: dict[str, tuple[float, float]],
    links: list[dict],
) -> dict[str, tuple[float, float]]:
    """Place nodes with no embedding at the centroid of their neighbours,
    or randomly if they have no neighbours with known positions.
    """
    positions: dict[str, tuple[float, float]] = {}
    rng = random.Random(0)

    # Build neighbour lookup from links
    neighbours: dict[str, list[str]] = {nid: [] for nid in missing_ids}
    for link in links:
        s, t = str(link.get("source", "")), str(link.get("target", ""))
        if s in neighbours:
            neighbours[s].append(t)
        if t in neighbours:
            neighbours[t].append(s)

    for nid in missing_ids:
        known = [id_to_pos[nb] for nb in neighbours[nid] if nb in id_to_pos]
        if known:
            x = sum(p[0] for p in known) / len(known)
            y = sum(p[1] for p in known) / len(known)
            # Small jitter so overlapping nodes separate
            x += rng.uniform(-0.05, 0.05)
            y += rng.uniform(-0.05, 0.05)
        else:
            x = rng.uniform(-1, 1)
            y = rng.uniform(-1, 1)
        positions[nid] = (x, y)

    return positions


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------

async def compute_semantic_positions(
    preprocessed: "PreprocessedGraph",
) -> dict[str, tuple[float, float]]:
    """Return {node_id: (x, y)} with coordinates in [-SPREAD, +SPREAD].

    Steps:
      1. Collect node ids (capped at _MAX_NODES).
      2. Fetch embeddings from the vector store.
      3. Run PCA / UMAP on nodes that have embeddings.
      4. Fall back to topology-centroid for nodes without embeddings.
      5. Scale to pixel coordinates.
    """
    nodes = preprocessed.nodes
    if not nodes:
        return {}

    # Sample if graph is very large
    if len(nodes) > _MAX_NODES:
        logger.warning(
            "semantic_layout: graph has %d nodes — sampling %d for projection",
            len(nodes),
            _MAX_NODES,
        )
        nodes = nodes[:_MAX_NODES]

    node_ids = [str(n["id"]) for n in nodes]

    # Fetch embeddings
    embeddings = await _fetch_embeddings(node_ids)

    # Split into nodes-with and nodes-without embeddings
    embedded_ids = [nid for nid in node_ids if nid in embeddings]
    missing_ids  = [nid for nid in node_ids if nid not in embeddings]

    id_to_pos: dict[str, tuple[float, float]] = {}

    if len(embedded_ids) >= _MIN_NODES_FOR_PROJECTION:
        matrix = [embeddings[nid] for nid in embedded_ids]
        coords = _project(matrix)
        for nid, (x, y) in zip(embedded_ids, coords):
            id_to_pos[nid] = (x * _SPREAD, y * _SPREAD)
    else:
        # Not enough embeddings — place all nodes randomly (D3 will refine)
        logger.debug(
            "semantic_layout: only %d embedded nodes (need %d) — using random layout",
            len(embedded_ids),
            _MIN_NODES_FOR_PROJECTION,
        )
        rng = random.Random(0)
        for nid in embedded_ids:
            id_to_pos[nid] = (rng.uniform(-_SPREAD, _SPREAD), rng.uniform(-_SPREAD, _SPREAD))

    # Fallback positions for un-embedded nodes
    if missing_ids:
        fallback = _topology_fallback(missing_ids, id_to_pos, preprocessed.links)
        for nid, pos in fallback.items():
            id_to_pos[nid] = (pos[0] * _SPREAD, pos[1] * _SPREAD)

    return id_to_pos


# ---------------------------------------------------------------------------
# JS emitter (called by cognee_network_visualization)
# ---------------------------------------------------------------------------

def emit_js(preprocessed: "PreprocessedGraph | None" = None) -> str:
    """Return a synchronous JS stub.

    The actual semantic positions are injected via the
    ``__SEMANTIC_POSITIONS__`` token which the orchestrator replaces with
    the JSON produced by ``compute_semantic_positions``.  The JS reads
    this data and overrides D3's initial node positions before the
    simulation starts, so the force layout refines rather than rebuilds
    the semantic arrangement.
    """
    return r"""
// ── Semantic Layout ──────────────────────────────────────────────────────────
(function () {
  "use strict";

  /** Pre-computed semantic positions: {node_id: {x, y}} or null */
  var SEMANTIC_POSITIONS = __SEMANTIC_POSITIONS__;

  if (!SEMANTIC_POSITIONS) return;

  /**
   * Seed D3 node objects with semantic positions before the simulation
   * starts.  Call this after nodes are bound to D3 data but before
   * simulation.alpha(1).restart().
   *
   * @param {Array}  d3Nodes  - array of D3 node datum objects (must have .id)
   * @param {number} [alpha]  - initial alpha for seeded nodes (default 0.3)
   */
  window._applySemanticLayout = function (d3Nodes, alpha) {
    if (!SEMANTIC_POSITIONS) return;
    alpha = (alpha === undefined) ? 0.3 : alpha;
    d3Nodes.forEach(function (n) {
      var pos = SEMANTIC_POSITIONS[String(n.id)];
      if (pos) {
        // Fix the node at the semantic position initially;
        // the force simulation will gently unfix as alpha cools.
        n.x  = pos.x;
        n.y  = pos.y;
        n.fx = pos.x;
        n.fy = pos.y;
        // Release the fix after a short warm-up so nodes can drift slightly
        // to resolve edge-length constraints while preserving global layout.
        setTimeout(function () { n.fx = null; n.fy = null; }, 800);
      }
    });
  };

  /** True when semantic positions are available for at least one node. */
  window._hasSemanticLayout = true;
})();
"""
