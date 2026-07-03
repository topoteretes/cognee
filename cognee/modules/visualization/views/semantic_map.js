// Semantic-map view — the d3/DOM shell.
//
// A meaning-space scatter of the graph: each node is pinned at the 2-D
// projection of its embedding (window._semanticPositions, from the layout's
// __SEMANTIC_POSITIONS__ token), shaded by cluster (__SEMANTIC_CLUSTERS__).
// Positions are computed once in Python and never simulated here (layout-once
// rule) except under the explicit Structural toggle. Hovering a node lights up
// its precomputed nearest neighbors and lists its graph relations.
//
// All view *decisions* (styling, isolation, recall overlay, legend model, screen
// mapping) live in the pure, d3-free SemanticCore (semantic_core.js, concatenated
// ahead of this file). This shell owns only d3/DOM mechanics: the force sim, the
// turbo palette, painting, zoom, and event binding. Node detail is read from
// window._vizNodeById / window._vizLinks (exposed by story_view.js).
(function () {
  'use strict';

  const Core = window.SemanticCore;
  const CLUSTERS = __SEMANTIC_CLUSTERS__;
  // Recall events from the session layer (same token the Memory tab uses). Each
  // 'search' event carries the node_ids a recall query retrieved — the overlay
  // lights up where that query landed in meaning-space.
  const SEARCH_EVENTS = __SEARCH_EVENTS__;

  function nodeById(id) {
    return (window._vizNodeById && window._vizNodeById[id]) || null;
  }
  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  // Single mutable view state; the core reads it, never mutates it.
  const state = {
    colorBy: 'cluster',
    layoutMode: 'semantic',
    isolatedCluster: null,
    isolatedType: null,
    recall: null, // Set of node_ids from the active recall overlay, or null
  };

  let zoomBehavior = null;
  // Retained render artifacts so overlay/isolation changes restyle in place
  // (no full re-render, no force-sim re-run) instead of via a back-pointer.
  let currentCircles = null;
  let currentCtx = null;

  // Structural layout: relax the same nodes over graph topology (window._vizLinks)
  // with a bounded, synchronous force sim. This is the one place a sim is allowed —
  // it explicitly leaves the pinned semantic view. Bounded by the 2000-node payload
  // and a fixed tick count, seeded from the semantic screen positions.
  function structuralPositions(ids, semanticScreen, width, height) {
    const idSet = new Set(ids);
    const nodes = ids.map((id) => ({ id, x: semanticScreen[id].x, y: semanticScreen[id].y }));
    const links = (window._vizLinks || [])
      .filter((l) => idSet.has(String(l.source)) && idSet.has(String(l.target)))
      .map((l) => ({ source: String(l.source), target: String(l.target) }));
    const sim = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id((d) => d.id).distance(40))
      .force('charge', d3.forceManyBody().strength(-30))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collide', d3.forceCollide(8))
      .stop();
    for (let i = 0; i < 200; i++) sim.tick();
    const out = {};
    nodes.forEach((n) => { out[n.id] = { x: n.x, y: n.y }; });
    return out;
  }

  function clusterColor(clusters) {
    // Deterministic palette indexed by cluster id.
    const map = {};
    const n = Math.max(1, clusters.length);
    clusters.forEach((c, i) => {
      map[c.id] = d3.interpolateTurbo((i + 0.5) / n);
    });
    return map;
  }

  function relationsFor(id) {
    const links = window._vizLinks || [];
    const out = [];
    for (const l of links) {
      const s = String(l.source), t = String(l.target);
      if (s === id) out.push({ other: t, rel: l.relation || l.label || 'related' });
      else if (t === id) out.push({ other: s, rel: l.relation || l.label || 'related' });
      if (out.length >= 8) break;
    }
    return out;
  }

  function showPanel(id, clusterColors, nodeCluster) {
    const panel = document.getElementById('semantic-panel');
    if (!panel) return;
    const nd = nodeById(id) || {};
    const cid = nodeCluster[id];
    const cluster = (CLUSTERS.clusters || []).find((c) => c.id === cid);
    const neighbors = (CLUSTERS.neighbors && CLUSTERS.neighbors[id]) || [];
    const rels = relationsFor(id);

    let html = '<div style="font-size:15px;font-weight:700;margin-bottom:6px;">' + esc(nd.name || id) + '</div>';
    html += '<div style="font-size:12px;color:var(--text2);margin-bottom:12px;">' + esc(nd.type || 'node') + '</div>';
    if (cluster) {
      html += '<div style="display:flex;align-items:center;gap:8px;margin-bottom:14px;">'
        + '<span style="width:11px;height:11px;border-radius:50%;background:' + clusterColors[cid] + ';display:inline-block;"></span>'
        + '<span style="font-size:12.5px;">' + esc(cluster.label) + '</span></div>';
    }
    if (neighbors.length) {
      html += '<div style="font-size:11px;text-transform:uppercase;letter-spacing:0.04em;color:var(--text2);margin-bottom:6px;">Nearest in meaning</div>';
      html += '<div style="margin-bottom:14px;">';
      neighbors.forEach((nid) => {
        const nn = nodeById(nid) || {};
        html += '<div style="font-size:12.5px;padding:2px 0;">' + esc(nn.name || nid) + '</div>';
      });
      html += '</div>';
    }
    if (rels.length) {
      html += '<div style="font-size:11px;text-transform:uppercase;letter-spacing:0.04em;color:var(--text2);margin-bottom:6px;">Relations</div>';
      rels.forEach((r) => {
        const on = nodeById(r.other) || {};
        html += '<div style="font-size:12px;padding:2px 0;color:var(--text2);">'
          + esc(r.rel) + ' &rarr; <span style="color:var(--text);">' + esc(on.name || r.other) + '</span></div>';
      });
    }
    panel.innerHTML = html;
    panel.style.display = 'block';
  }

  // Restyle the retained circles from current state + the core's decisions.
  // A no-op before the first render (currentCircles is null); render() calls it
  // at the end, so a state change made while hidden still lands on next render.
  function repaint() {
    if (!currentCircles) return;
    currentCircles.each(function (id) {
      const s = Core.styleFor(id, state, currentCtx);
      d3.select(this)
        .attr('opacity', s.opacity)
        .attr('stroke', s.stroke)
        .attr('stroke-width', s.strokeWidth)
        .attr('r', s.r);
    });
  }

  function render(preserve) {
    const svgEl = document.getElementById('semantic-svg');
    const empty = document.getElementById('semantic-empty');
    const positions = window._semanticPositions || null;
    const ids = positions ? Object.keys(positions) : [];

    if (!positions || ids.length === 0 || !CLUSTERS) {
      if (empty) empty.style.display = 'flex';
      if (svgEl) svgEl.style.display = 'none';
      return;
    }
    if (empty) empty.style.display = 'none';
    if (svgEl) svgEl.style.display = 'block';

    const svg = d3.select(svgEl);
    const width = svgEl.clientWidth || 800;
    const height = svgEl.clientHeight || 600;
    const pad = 60;
    const clusters = CLUSTERS.clusters || [];
    const nodeCluster = CLUSTERS.node_cluster || {};
    const clusterColors = clusterColor(clusters);

    // Precomputed maps handed to the pure core — it never reads window._viz*.
    const typeById = {};
    const colorById = {};
    ids.forEach((id) => {
      const nd = nodeById(id) || {};
      typeById[id] = nd.type;
      colorById[id] = nd.color;
    });
    const ctx = { nodeCluster, typeById, colorById, clusterColors };

    const semanticScreen = Core.screenPositions(positions, width, height, pad);
    const screenPos =
      state.layoutMode === 'structural'
        ? structuralPositions(ids, semanticScreen, width, height)
        : semanticScreen;

    const prevTransform = preserve ? d3.zoomTransform(svgEl) : d3.zoomIdentity;
    svg.selectAll('*').remove();
    const g = svg.append('g');

    // Cluster labels at the on-screen centroid of each cluster's members.
    Core.clusterCentroids(clusters, screenPos).forEach((c) => {
      g.append('text')
        .attr('x', c.x).attr('y', c.y)
        .attr('text-anchor', 'middle')
        .attr('fill', clusterColors[c.id])
        .attr('font-size', 13).attr('font-weight', 700)
        .attr('opacity', 0.85)
        .attr('pointer-events', 'none')
        .text(c.label);
    });

    const circles = g.selectAll('circle').data(ids, (d) => d).enter().append('circle')
      .attr('cx', (id) => screenPos[id].x)
      .attr('cy', (id) => screenPos[id].y)
      .attr('r', 5)
      .attr('fill', (id) => Core.fillFor(id, state, ctx))
      .attr('stroke', 'rgba(0,0,0,0.25)').attr('stroke-width', 0.5)
      .style('cursor', 'pointer');
    currentCircles = circles;
    currentCtx = ctx;

    circles.on('mouseover', function (event, id) {
      const nbrs = new Set([(id), ...((CLUSTERS.neighbors && CLUSTERS.neighbors[id]) || [])]);
      circles.attr('opacity', (d) => (nbrs.has(d) ? 1 : 0.15));
      d3.select(this).attr('r', 8);
      showPanel(id, clusterColors, nodeCluster);
    }).on('mouseout', function () {
      repaint();
      const panel = document.getElementById('semantic-panel');
      if (panel) panel.style.display = 'none';
    });

    zoomBehavior = d3.zoom().scaleExtent([0.2, 12]).on('zoom', (event) => {
      g.attr('transform', event.transform);
    });
    svg.call(zoomBehavior);
    svg.call(zoomBehavior.transform, prevTransform);

    repaint();
    renderLegend(clusters, ids, ctx);

    const status = document.getElementById('semantic-status');
    if (status) {
      status.textContent = ids.length + ' nodes · ' + clusters.length + ' clusters';
    }
  }

  function legendRow(color, text, onClick) {
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;align-items:center;gap:8px;cursor:pointer;padding:1px 0;';
    row.innerHTML = '<span style="width:10px;height:10px;border-radius:50%;background:'
      + color + ';display:inline-block;flex:0 0 auto;"></span>'
      + '<span>' + esc(text.length > 30 ? text.slice(0, 29) + '…' : text) + '</span>';
    row.addEventListener('click', onClick);
    return row;
  }

  function renderLegend(clusters, ids, ctx) {
    const legend = document.getElementById('semantic-legend');
    if (!legend) return;
    legend.innerHTML = '';
    // Rows follow the color mode (cluster vs type); clicking one isolates it.
    Core.legendModel(state, clusters, ids, ctx).forEach((rowModel) => {
      legend.appendChild(legendRow(rowModel.color, rowModel.text, () => {
        if (rowModel.kind === 'type') {
          state.isolatedType = state.isolatedType === rowModel.value ? null : rowModel.value;
        } else {
          state.isolatedCluster = state.isolatedCluster === rowModel.value ? null : rowModel.value;
        }
        repaint();
      }));
    });
  }

  window._renderSemanticView = function (preserve) {
    render(preserve);
  };

  function setActive(group, btn) {
    document.querySelectorAll(group).forEach((b) => {
      b.classList.remove('active');
      b.style.background = 'var(--surface)'; b.style.color = 'var(--text2)'; b.style.border = '1px solid var(--border)';
    });
    btn.classList.add('active');
    btn.style.background = 'var(--accent)'; btn.style.color = '#fff'; btn.style.border = 'none';
  }

  // Color-mode toggle (Cluster / Type).
  document.querySelectorAll('.sem-color-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      setActive('.sem-color-btn', btn);
      state.colorBy = btn.dataset.colorby;
      state.isolatedCluster = null; state.isolatedType = null;  // isolation is per-mode
      render(true);
    });
  });

  // Layout toggle (Semantic / Structural).
  document.querySelectorAll('.sem-layout-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      setActive('.sem-layout-btn', btn);
      state.layoutMode = btn.dataset.layout;
      render(true);
    });
  });

  // Zoom controls.
  function zoomBy(factor) {
    const svgEl = document.getElementById('semantic-svg');
    if (svgEl && zoomBehavior) d3.select(svgEl).transition().duration(150).call(zoomBehavior.scaleBy, factor);
  }
  const zi = document.getElementById('semantic-zoom-in');
  const zo = document.getElementById('semantic-zoom-out');
  const zf = document.getElementById('semantic-zoom-fit');
  if (zi) zi.addEventListener('click', () => zoomBy(1.4));
  if (zo) zo.addEventListener('click', () => zoomBy(1 / 1.4));
  if (zf) zf.addEventListener('click', () => {
    const svgEl = document.getElementById('semantic-svg');
    if (svgEl && zoomBehavior) d3.select(svgEl).transition().duration(200).call(zoomBehavior.transform, d3.zoomIdentity);
  });

  // Recall overlay: pick a past recall query, light up the nodes it retrieved.
  (function initRecall() {
    const wrap = document.getElementById('semantic-recall-wrap');
    const select = document.getElementById('semantic-recall');
    const note = document.getElementById('semantic-recall-note');
    if (!select) return;
    const queries = Core.recallQueries(SEARCH_EVENTS);
    if (!queries.length) {
      if (wrap) wrap.style.display = 'none';  // nothing to overlay
      return;
    }
    queries.forEach((e, i) => {
      const opt = document.createElement('option');
      opt.value = String(i);
      const q = (e.question || 'recall ' + (i + 1)).trim();
      opt.textContent = q.length > 40 ? q.slice(0, 39) + '…' : q;
      select.appendChild(opt);
    });
    select.addEventListener('change', () => {
      const idx = select.value;
      if (idx === '') {
        state.recall = null;
        if (note) note.textContent = '';
      } else {
        const evt = queries[+idx];
        state.recall = new Set((evt.node_ids || []).map(String));
        const positions = window._semanticPositions || {};
        if (note) note.textContent = Core.recallOnMap(evt, positions) + ' of ' + (evt.node_ids || []).length + ' on map';
      }
      repaint();
    });
  })();

  // Deep link: #semantic opens the tab on load.
  if (window.location.hash === '#semantic') {
    const btn = document.querySelector('.tab-btn[data-view="semantic"]');
    if (btn) btn.click();
  }
})();
