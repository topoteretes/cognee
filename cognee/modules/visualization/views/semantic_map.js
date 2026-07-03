// Semantic-map view.
//
// A meaning-space scatter of the graph: each node is pinned at the 2-D
// projection of its embedding (window._semanticPositions, from the layout's
// __SEMANTIC_POSITIONS__ token), shaded by cluster (__SEMANTIC_CLUSTERS__).
// Positions are computed once in Python and never simulated here (layout-once
// rule). Hovering a node lights up its precomputed nearest neighbors and lists
// its graph relations. Node/cluster payloads carry structure only; node detail
// is read from window._vizNodeById / window._vizLinks (exposed by story_view.js).
(function () {
  'use strict';

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

  let colorBy = 'cluster';
  let layoutMode = 'semantic';
  let zoomBehavior = null;
  let isolatedCluster = null;
  let isolatedType = null;
  let recallSet = null;       // Set of node_ids from the active recall overlay, or null
  let refreshStyles = null;   // set by render() so overlay changes restyle without a re-render

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

  function colorForNode(id, clusterColors, nodeCluster) {
    if (colorBy === 'type') {
      const nd = nodeById(id);
      return (nd && nd.color) || '#8a8a8a';
    }
    const cid = nodeCluster[id];
    return cid == null ? '#8a8a8a' : clusterColors[cid];
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

    const sx = d3.scaleLinear().domain([-1.2, 1.2]).range([pad, width - pad]);
    const sy = d3.scaleLinear().domain([-1.2, 1.2]).range([height - pad, pad]);

    // Screen positions: pinned embedding projection (semantic) or a bounded
    // force relaxation over the graph topology (structural). Structural seeds
    // from the semantic screen positions so the toggle reads as a relax, not a
    // jump; Semantic snaps straight back to the pinned coordinates.
    const semanticScreen = {};
    ids.forEach((id) => {
      semanticScreen[id] = { x: sx(positions[id].x), y: sy(positions[id].y) };
    });
    const screenPos =
      layoutMode === 'structural'
        ? structuralPositions(ids, semanticScreen, width, height)
        : semanticScreen;

    const prevTransform = preserve ? d3.zoomTransform(svgEl) : d3.zoomIdentity;
    svg.selectAll('*').remove();
    const g = svg.append('g');

    // Cluster labels at the on-screen centroid of each cluster's members.
    clusters.forEach((c) => {
      const pts = (c.node_ids || []).map((id) => screenPos[id]).filter(Boolean);
      if (!pts.length) return;
      g.append('text')
        .attr('x', d3.mean(pts, (p) => p.x)).attr('y', d3.mean(pts, (p) => p.y))
        .attr('text-anchor', 'middle')
        .attr('fill', clusterColors[c.id])
        .attr('font-size', 13).attr('font-weight', 700)
        .attr('opacity', 0.85)
        .attr('pointer-events', 'none')
        .text(c.label.length > 34 ? c.label.slice(0, 33) + '…' : c.label);
    });

    const circles = g.selectAll('circle').data(ids, (d) => d).enter().append('circle')
      .attr('cx', (id) => screenPos[id].x)
      .attr('cy', (id) => screenPos[id].y)
      .attr('r', 5)
      .attr('fill', (id) => colorForNode(id, clusterColors, nodeCluster))
      .attr('stroke', 'rgba(0,0,0,0.25)').attr('stroke-width', 0.5)
      .style('cursor', 'pointer');

    function isolationOpacity(id) {
      if (colorBy === 'type') {
        if (isolatedType == null) return 1;
        const nd = nodeById(id);
        return nd && nd.type === isolatedType ? 1 : 0.12;
      }
      return isolatedCluster == null || nodeCluster[id] === isolatedCluster ? 1 : 0.12;
    }

    // Base styling: the recall overlay (when active) dims everything except the
    // retrieved nodes and rings them; otherwise the legend isolation applies.
    function refresh() {
      circles
        .attr('opacity', (id) => (recallSet ? (recallSet.has(id) ? 1 : 0.06) : isolationOpacity(id)))
        .attr('stroke', (id) => (recallSet && recallSet.has(id) ? '#ff3b3b' : 'rgba(0,0,0,0.25)'))
        .attr('stroke-width', (id) => (recallSet && recallSet.has(id) ? 2.5 : 0.5))
        .attr('r', (id) => (recallSet && recallSet.has(id) ? 7 : 5));
    }
    refreshStyles = refresh;

    circles.on('mouseover', function (event, id) {
      const nbrs = new Set([(id), ...((CLUSTERS.neighbors && CLUSTERS.neighbors[id]) || [])]);
      circles.attr('opacity', (d) => (nbrs.has(d) ? 1 : 0.15));
      d3.select(this).attr('r', 8);
      showPanel(id, clusterColors, nodeCluster);
    }).on('mouseout', function () {
      refresh();
      const panel = document.getElementById('semantic-panel');
      if (panel) panel.style.display = 'none';
    });

    zoomBehavior = d3.zoom().scaleExtent([0.2, 12]).on('zoom', (event) => {
      g.attr('transform', event.transform);
    });
    svg.call(zoomBehavior);
    svg.call(zoomBehavior.transform, prevTransform);

    refresh();
    renderLegend(clusters, clusterColors, refresh, ids);

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

  function renderLegend(clusters, clusterColors, refresh, ids) {
    const legend = document.getElementById('semantic-legend');
    if (!legend) return;
    legend.innerHTML = '';
    if (colorBy === 'type') {
      // Legend follows the color mode: list ontology types, click isolates one.
      const typeColor = {};
      ids.forEach((id) => {
        const nd = nodeById(id);
        if (nd && nd.type && !(nd.type in typeColor)) typeColor[nd.type] = nd.color || '#8a8a8a';
      });
      Object.keys(typeColor).sort().forEach((t) => {
        legend.appendChild(legendRow(typeColor[t], t, () => {
          isolatedType = isolatedType === t ? null : t;
          refresh();
        }));
      });
      return;
    }
    clusters.forEach((c) => {
      legend.appendChild(legendRow(clusterColors[c.id], c.label, () => {
        isolatedCluster = isolatedCluster === c.id ? null : c.id;
        refresh();
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
      colorBy = btn.dataset.colorby;
      isolatedCluster = null; isolatedType = null;  // isolation is per-mode
      render(true);
    });
  });

  // Layout toggle (Semantic / Structural).
  document.querySelectorAll('.sem-layout-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      setActive('.sem-layout-btn', btn);
      layoutMode = btn.dataset.layout;
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
    const queries = (SEARCH_EVENTS || []).filter(
      (e) => e && e.kind === 'search' && (e.node_ids || []).length,
    );
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
        recallSet = null;
        if (note) note.textContent = '';
      } else {
        const evt = queries[+idx];
        recallSet = new Set((evt.node_ids || []).map(String));
        const positions = window._semanticPositions || {};
        const onMap = (evt.node_ids || []).filter((id) => positions[String(id)]).length;
        if (note) note.textContent = onMap + ' of ' + (evt.node_ids || []).length + ' on map';
      }
      if (refreshStyles) refreshStyles();
    });
  })();

  // Deep link: #semantic opens the tab on load.
  if (window.location.hash === '#semantic') {
    const btn = document.querySelector('.tab-btn[data-view="semantic"]');
    if (btn) btn.click();
  }
})();
