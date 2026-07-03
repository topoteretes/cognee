// Semantic-map pure core: the view's decision layer.
//
// DOM-free and d3-free by design, so it runs in plain Node for unit tests and,
// concatenated ahead of semantic_map.js, as window.SemanticCore in the browser.
// Everything here is a pure function: plain data in, plain data out, no reads of
// window._viz* and no d3. The shell (semantic_map.js) owns all d3/DOM mechanics
// (force sim, turbo palette, painting) and hands this core precomputed maps.
(function (root) {
  'use strict';

  // Linear map with no clamping — matches d3.scaleLinear's default extrapolation.
  function mapLinear(v, d0, d1, r0, r1) {
    return r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
  }

  // Pinned embedding coords ([-1.2, 1.2] box) -> screen pixels. y is inverted.
  function screenPositions(positions, width, height, pad) {
    const out = {};
    Object.keys(positions).forEach((id) => {
      out[id] = {
        x: mapLinear(positions[id].x, -1.2, 1.2, pad, width - pad),
        y: mapLinear(positions[id].y, -1.2, 1.2, height - pad, pad),
      };
    });
    return out;
  }

  function isolationOpacity(id, state, ctx) {
    if (state.colorBy === 'type') {
      if (state.isolatedType == null) return 1;
      return ctx.typeById[id] === state.isolatedType ? 1 : 0.12;
    }
    return state.isolatedCluster == null || ctx.nodeCluster[id] === state.isolatedCluster
      ? 1
      : 0.12;
  }

  // Per-node visual attributes. The recall overlay (state.recall is a Set) wins:
  // retrieved nodes get a red ring and full opacity, everything else dims. With
  // no overlay, the legend isolation applies.
  function styleFor(id, state, ctx) {
    const hit = !!(state.recall && state.recall.has(id));
    return {
      opacity: state.recall ? (hit ? 1 : 0.06) : isolationOpacity(id, state, ctx),
      stroke: hit ? '#ff3b3b' : 'rgba(0,0,0,0.25)',
      strokeWidth: hit ? 2.5 : 0.5,
      r: hit ? 7 : 5,
    };
  }

  function fillFor(id, state, ctx) {
    if (state.colorBy === 'type') return ctx.colorById[id] || '#8a8a8a';
    const cid = ctx.nodeCluster[id];
    return cid == null ? '#8a8a8a' : ctx.clusterColors[cid];
  }

  function truncate(s, n) {
    return s.length > n ? s.slice(0, n - 1) + '…' : s;
  }

  // On-screen centroid + display label per non-empty cluster.
  function clusterCentroids(clusters, screenPos) {
    const out = [];
    clusters.forEach((c) => {
      const pts = (c.node_ids || []).map((id) => screenPos[id]).filter(Boolean);
      if (!pts.length) return;
      let sx = 0;
      let sy = 0;
      pts.forEach((p) => {
        sx += p.x;
        sy += p.y;
      });
      out.push({ id: c.id, x: sx / pts.length, y: sy / pts.length, label: truncate(c.label, 34) });
    });
    return out;
  }

  // Legend rows follow the color mode. Each row carries what the shell needs to
  // wire a click: kind ('type'|'cluster') + value (the type name or cluster id).
  function legendModel(state, clusters, ids, ctx) {
    if (state.colorBy === 'type') {
      const typeColor = {};
      ids.forEach((id) => {
        const t = ctx.typeById[id];
        if (t && !(t in typeColor)) typeColor[t] = ctx.colorById[id] || '#8a8a8a';
      });
      return Object.keys(typeColor)
        .sort()
        .map((t) => ({ kind: 'type', value: t, color: typeColor[t], text: t }));
    }
    return clusters.map((c) => ({
      kind: 'cluster',
      value: c.id,
      color: ctx.clusterColors[c.id],
      text: c.label,
    }));
  }

  function recallQueries(events) {
    return (events || []).filter((e) => e && e.kind === 'search' && (e.node_ids || []).length);
  }

  function recallOnMap(query, positions) {
    return (query.node_ids || []).filter((id) => positions[String(id)]).length;
  }

  const api = {
    screenPositions,
    isolationOpacity,
    styleFor,
    fillFor,
    clusterCentroids,
    legendModel,
    recallQueries,
    recallOnMap,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.SemanticCore = api;
})(typeof self !== 'undefined' ? self : this);
