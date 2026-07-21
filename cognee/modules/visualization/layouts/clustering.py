"""Semantic clustering: group graph nodes by embedding similarity.

Uses k-means (pure numpy — no extra deps) by default.
Automatically upgrades to HDBSCAN when the ``hdbscan`` package is installed,
which produces better natural clusters without requiring a fixed *k*.

Public API
----------
compute_clusters(preprocessed, positions) -> ClusterResult
    Given a PreprocessedGraph and 2D semantic positions, return cluster
    assignments, per-cluster labels, and colour codes.

emit_js(cluster_result) -> str
    Return a JS snippet that colours D3 nodes by cluster and renders
    labelled cluster hulls (convex boundaries) in the background.

Typical usage (after semantic_layout):
    from cognee.modules.visualization.layouts import semantic_layout, clustering

    positions = await semantic_layout.compute_semantic_positions(pre)
    clusters  = await clustering.compute_clusters(pre, positions)
    html = html.replace("__SEMANTIC_LAYOUT_JS__",  semantic_layout.emit_js())
    html = html.replace("__CLUSTER_JS__", clustering.emit_js(clusters))
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cognee.modules.visualization.preprocessor import PreprocessedGraph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default number of clusters when using k-means.
_DEFAULT_K: int = 5

#: Maximum clusters (capped so labels don't overwhelm the canvas).
_MAX_CLUSTERS: int = 12

#: Minimum nodes per cluster before it's merged into the "Other" group.
_MIN_CLUSTER_SIZE: int = 2

#: Palette for cluster colours (colour-blind-friendly).
_PALETTE: list[str] = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    "#D4A6C8", "#86BCB6",
]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ClusterResult:
    """Output of :func:`compute_clusters`."""

    #: {node_id: cluster_id}  (cluster_id is an int ≥ 0; -1 = noise/unclustered)
    assignments: dict[str, int] = field(default_factory=dict)

    #: {cluster_id: human-readable label}
    labels: dict[int, str] = field(default_factory=dict)

    #: {cluster_id: hex colour string}
    colors: dict[int, str] = field(default_factory=dict)

    #: {cluster_id: [node_ids]}  membership lists
    members: dict[int, list[str]] = field(default_factory=dict)

    #: Number of meaningful clusters (excludes noise cluster -1)
    n_clusters: int = 0


# ---------------------------------------------------------------------------
# K-means (pure numpy)
# ---------------------------------------------------------------------------

def _kmeans(points: list[tuple[float, float]], k: int, seed: int = 42,
            max_iter: int = 100) -> list[int]:
    """Minimal k-means returning cluster index per point."""
    import numpy as np

    X = np.array(points, dtype=float)
    n = len(X)
    if n <= k:
        return list(range(n))

    rng = np.random.default_rng(seed)
    # k-means++ initialisation
    centres = [X[rng.integers(n)]]
    for _ in range(1, k):
        dists = np.array([min(np.linalg.norm(x - c) ** 2 for c in centres) for x in X])
        probs = dists / dists.sum()
        centres.append(X[rng.choice(n, p=probs)])
    centres = np.array(centres)

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # Assignment
        dists = np.linalg.norm(X[:, None] - centres[None, :], axis=2)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # Update
        for j in range(k):
            members = X[labels == j]
            if len(members):
                centres[j] = members.mean(axis=0)

    return labels.tolist()


# ---------------------------------------------------------------------------
# HDBSCAN (optional)
# ---------------------------------------------------------------------------

def _hdbscan(points: list[tuple[float, float]]) -> list[int]:
    """HDBSCAN clustering — only called when hdbscan is installed."""
    import numpy as np
    import hdbscan  # type: ignore[import]

    X = np.array(points, dtype=float)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, len(X) // 10))
    return clusterer.fit_predict(X).tolist()


# ---------------------------------------------------------------------------
# Cluster labelling
# ---------------------------------------------------------------------------

def _label_cluster(node_ids: list[str], nodes_by_id: dict[str, dict]) -> str:
    """Pick a short human-readable label for a cluster.

    Uses the most common node *type* plus top-2 entity names.
    """
    types: list[str] = []
    names: list[str] = []
    for nid in node_ids:
        node = nodes_by_id.get(nid, {})
        t = str(node.get("type", ""))
        if t:
            types.append(t)
        label = str(node.get("label", node.get("name", "")))
        if label and label != t:
            names.append(label)

    # Most common type
    type_label = ""
    if types:
        from collections import Counter
        type_label = Counter(types).most_common(1)[0][0]

    # Up to 2 representative names
    name_part = ", ".join(names[:2])
    if name_part:
        return f"{type_label}: {name_part}" if type_label else name_part
    return type_label or f"Cluster ({len(node_ids)} nodes)"


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------

async def compute_clusters(
    preprocessed: "PreprocessedGraph",
    positions: dict[str, tuple[float, float]],
    k: int | None = None,
) -> ClusterResult:
    """Cluster nodes by their 2D semantic positions.

    Parameters
    ----------
    preprocessed:
        The preprocessed graph (nodes + links).
    positions:
        Output of ``semantic_layout.compute_semantic_positions``.
        Only nodes present in *positions* are clustered; others get -1.
    k:
        Number of clusters for k-means.  If *None*, chosen automatically
        as ``min(_MAX_CLUSTERS, max(2, sqrt(n_nodes) // 2))``.

    Returns
    -------
    ClusterResult
    """
    result = ClusterResult()

    if not positions:
        return result

    node_ids = list(positions.keys())
    points = [positions[nid] for nid in node_ids]

    # Choose algorithm
    n = len(node_ids)
    raw_labels: list[int]

    if n < _MIN_CLUSTER_SIZE * 2:
        # Too few nodes to cluster meaningfully
        raw_labels = [0] * n
    else:
        try:
            import hdbscan  # noqa: F401
            logger.debug("clustering: using HDBSCAN")
            raw_labels = _hdbscan(points)
        except ImportError:
            logger.debug("clustering: hdbscan not found, using k-means")
            if k is None:
                k = min(_MAX_CLUSTERS, max(2, int(math.sqrt(n)) // 2 + 1))
            raw_labels = _kmeans(points, k=k)

    # Map node_ids → raw cluster labels
    for nid, lbl in zip(node_ids, raw_labels):
        result.assignments[nid] = lbl

    # Assign -1 to nodes without positions
    all_node_ids = {str(n["id"]) for n in preprocessed.nodes}
    for nid in all_node_ids - set(result.assignments):
        result.assignments[nid] = -1

    # Build membership lists (merge tiny clusters into -1)
    raw_members: dict[int, list[str]] = {}
    for nid, lbl in result.assignments.items():
        raw_members.setdefault(lbl, []).append(nid)

    # Re-label sequentially (skip noise -1)
    nodes_by_id = {str(n["id"]): n for n in preprocessed.nodes}
    new_id = 0
    remap: dict[int, int] = {}

    for old_lbl, members in sorted(raw_members.items()):
        if old_lbl == -1:
            continue
        if len(members) < _MIN_CLUSTER_SIZE:
            # Merge tiny clusters into noise
            for nid in members:
                result.assignments[nid] = -1
            continue
        remap[old_lbl] = new_id
        result.members[new_id] = members
        result.labels[new_id] = _label_cluster(members, nodes_by_id)
        result.colors[new_id] = _PALETTE[new_id % len(_PALETTE)]
        new_id += 1

    # Apply remap
    for nid, old_lbl in list(result.assignments.items()):
        result.assignments[nid] = remap.get(old_lbl, -1)

    result.n_clusters = new_id
    logger.debug("clustering: %d clusters from %d nodes", new_id, n)
    return result


# ---------------------------------------------------------------------------
# JS emitter
# ---------------------------------------------------------------------------

def emit_js(cluster_result: ClusterResult | None = None) -> str:
    """Return a JS snippet that colours nodes by cluster and draws hull overlays.

    The snippet reads ``__CLUSTER_DATA__`` (replaced by the orchestrator) and:
    - Overrides each D3 node's fill colour with the cluster colour.
    - Draws convex-hull polygons per cluster in a background ``<g>`` layer.
    - Adds a legend chip per cluster (label + colour).

    If ``cluster_result`` is provided the data is inlined directly; otherwise
    the ``__CLUSTER_DATA__`` token must be replaced by the caller.
    """
    _token = "CLUSTER_DATA_TOKEN"  # avoid literal in template when inlining
    if cluster_result is not None:
        data_json = json.dumps({
            "assignments": cluster_result.assignments,
            "labels": {str(k): v for k, v in cluster_result.labels.items()},
            "colors": {str(k): v for k, v in cluster_result.colors.items()},
            "n_clusters": cluster_result.n_clusters,
        })
        guard = "if (!CLUSTER_DATA) return;"
    else:
        data_json = f"__{_token}__"  # placeholder replaced by orchestrator
        guard = f'if (!CLUSTER_DATA || typeof CLUSTER_DATA === "string") return;'

    return f"""
// ── Semantic Clustering ───────────────────────────────────────────────────────
(function () {{
  "use strict";

  var CLUSTER_DATA = {data_json};

  {guard}

  var assignments = CLUSTER_DATA.assignments || {{}};
  var labels      = CLUSTER_DATA.labels      || {{}};
  var colors      = CLUSTER_DATA.colors      || {{}};

  /**
   * Apply cluster colours to D3 node elements.
   * Call after nodes are rendered: _applyClusterColors(nodeSelection)
   *
   * @param {{d3.Selection}} nodeSelection  - D3 selection of node <circle>/<g> elements
   */
  window._applyClusterColors = function (nodeSelection) {{
    nodeSelection.each(function (d) {{
      var cid = assignments[String(d.id)];
      if (cid !== undefined && cid >= 0) {{
        var color = colors[String(cid)];
        if (color) {{
          d3.select(this).select("circle").style("fill", color);
        }}
      }}
    }});
  }};

  /**
   * Draw convex-hull outlines for each cluster on a background <g>.
   * Call after the simulation has settled:
   *   _drawClusterHulls(svg, d3Nodes, margin)
   *
   * @param {{SVGElement}} svg      - the root SVG element
   * @param {{Array}}      d3Nodes - array of D3 node datum objects with .x/.y
   * @param {{number}}     [pad=30] - hull padding in pixels
   */
  window._drawClusterHulls = function (svg, d3Nodes, pad) {{
    pad = pad === undefined ? 30 : pad;

    // Group nodes by cluster
    var buckets = {{}};
    d3Nodes.forEach(function (n) {{
      var cid = assignments[String(n.id)];
      if (cid === undefined || cid < 0) return;
      if (!buckets[cid]) buckets[cid] = [];
      buckets[cid].push([n.x, n.y]);
    }});

    // Remove any previous hull layer
    d3.select(svg).select(".cluster-hulls").remove();
    var hullG = d3.select(svg).insert("g", ":first-child")
      .attr("class", "cluster-hulls")
      .style("pointer-events", "none");

    Object.keys(buckets).forEach(function (cid) {{
      var pts  = buckets[cid];
      var color = colors[String(cid)] || "#aaa";
      var label = labels[String(cid)] || ("Cluster " + cid);

      // Need at least 3 points for a hull; use bounding box for fewer
      var hull;
      if (pts.length >= 3) {{
        hull = d3.polygonHull(pts);
      }} else {{
        var xs = pts.map(function (p) {{ return p[0]; }});
        var ys = pts.map(function (p) {{ return p[1]; }});
        var x0 = Math.min.apply(null, xs) - pad;
        var y0 = Math.min.apply(null, ys) - pad;
        var x1 = Math.max.apply(null, xs) + pad;
        var y1 = Math.max.apply(null, ys) + pad;
        hull = [[x0,y0],[x1,y0],[x1,y1],[x0,y1]];
      }}

      if (!hull) return;

      // Pad hull outward
      var cx = hull.reduce(function (s, p) {{ return s + p[0]; }}, 0) / hull.length;
      var cy = hull.reduce(function (s, p) {{ return s + p[1]; }}, 0) / hull.length;
      var padded = hull.map(function (p) {{
        var dx = p[0] - cx, dy = p[1] - cy;
        var len = Math.sqrt(dx*dx + dy*dy) || 1;
        return [p[0] + dx/len*pad, p[1] + dy/len*pad];
      }});

      hullG.append("path")
        .datum(padded)
        .attr("d", function (d) {{
          return "M" + d.map(function (p) {{ return p[0]+","+p[1]; }}).join("L") + "Z";
        }})
        .style("fill", color)
        .style("fill-opacity", 0.08)
        .style("stroke", color)
        .style("stroke-opacity", 0.35)
        .style("stroke-width", 1.5)
        .style("stroke-dasharray", "4 3");

      // Cluster label at centroid
      var labelX = padded.reduce(function (s, p) {{ return s + p[0]; }}, 0) / padded.length;
      var labelY = Math.min.apply(null, padded.map(function (p) {{ return p[1]; }})) - 6;
      hullG.append("text")
        .attr("x", labelX)
        .attr("y", labelY)
        .attr("text-anchor", "middle")
        .style("font-size", "11px")
        .style("fill", color)
        .style("font-weight", "600")
        .style("pointer-events", "none")
        .text(label.length > 30 ? label.slice(0, 28) + "…" : label);
    }});
  }};

  /** True when cluster data is available. */
  window._hasClusterLayout = true;
}})();
"""
