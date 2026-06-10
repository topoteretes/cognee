// Memory-map view (STEP 2).
//
// A deterministic column map of cognee memory: Documents (with ordered chunk
// cells) → Entity-type groups (collapse-behind-count) → Summaries → Global
// context, plus a run/search timeline rail and a search-retrieval overlay.
//
// Layout is a pure function of the sorted __MEMORY_DATA__ payload — no force
// simulation, no randomness. Timeline scrubbing and the search overlay only
// toggle classes/opacity; positions are never touched (layout-once rule).
// Node/link details are read from window._vizNodeById / window._vizLinks,
// which story_view.js exposes (the slim payload carries structure only).
(function () {
  'use strict';
  const memoryMap = __MEMORY_DATA__;
  const searchEvents = __SEARCH_EVENTS__;

  const MM = memoryMap || {};
  const docsP = MM.documents || [];
  const orphansP = MM.orphan_chunks || [];
  const groupsP = MM.entity_groups || [];
  const ungroupedP = MM.ungrouped_entities || [];
  const sumsP = MM.summaries || [];
  const ctxP = MM.context || null;
  const edgesP = MM.edges || {};
  const timelineP = MM.timeline || [];

  // ── Helpers ──────────────────────────────────────────────────────
  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function trunc(s, n) {
    s = String(s == null ? '' : s);
    return s.length > n ? s.slice(0, n - 1) + '…' : s;
  }
  function fmtT(ms) {
    if (!ms) return '';
    try {
      return new Date(ms).toLocaleString(undefined, {
        month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
      });
    } catch (e) { return String(ms); }
  }
  function nodeOf(id) { return (window._vizNodeById && window._vizNodeById[id]) || { id: id }; }
  function nameOf(id) { return nodeOf(id).name || String(id); }
  function stageOf(id) { return nodeOf(id).stage || 'other'; }
  function linkAt(pos) { return (window._vizLinks && window._vizLinks[pos]) || null; }
  function endId(v) { return (typeof v === 'object' && v) ? (v.id || '') : v; }
  function bez(a, b) {
    const mx = (a.x + b.x) / 2;
    return 'M' + a.x + ',' + a.y + ' C' + mx + ',' + a.y + ' ' + mx + ',' + b.y + ' ' + b.x + ',' + b.y;
  }

  // ── Geometry constants (deterministic layout) ────────────────────
  const DOC_W = 250, DOC_HEAD = 30, CELL_H = 26, DOC_PAD = 8, DOC_GAP = 18;
  const ENT_W = 216, ROW_H = 19, GRP_HEAD = 24, PILL_H = 20, GRP_PAD = 6, GRP_GAP = 16, ENT_COL_GAP = 26;
  const SUM_W = 230, SUM_H = 48, SUM_GAP = 10;
  const CTX_W = 230, BAND_GAP = 120, TARGET_COL_H = 1700;

  // ── View state ───────────────────────────────────────────────────
  let built = false;
  let svgSel = null, rootSel = null, gOverlay = null, zoomBehavior = null;
  let L = null;                 // layout
  let unitEls = {};             // node id -> [unit <g> elements]
  let edgeEls = [];             // {kind, s, t, el}
  let edgeByObjId = {};         // edge_object_id -> [path elements]
  let groupByMember = {};       // collapsed entity id -> group layout (for promote)
  let entityGroupName = {};     // entity id -> type name
  let chunkEntities = {}, entityChunks = {}, entitySemantic = {}, summariesByChunk = {};
  let railItems = [], currentRunIdx = -1, visibleSet = null, activeSearch = null;
  let selection = null;

  // ── Layout (pure function of the payload arrays) ─────────────────
  function computeLayout() {
    const lay = {
      docs: [], cellRect: {}, groups: [], entityAnchor: {}, sums: [],
      ctxBoxes: [], ctxEmpty: null, bands: [], width: 0, height: 0,
    };

    // Documents column (+ pseudo-doc for orphan chunks).
    const docDefs = docsP.map(function (d) { return { id: d.id, name: d.name, chunks: d.chunks, pseudo: false }; });
    if (orphansP.length) {
      docDefs.push({
        id: '__orphans__', name: 'Unattributed chunks', pseudo: true,
        chunks: orphansP.map(function (cid) { return { id: cid, chunk_index: null }; }),
      });
    }
    let dy = 0;
    docDefs.forEach(function (d) {
      const h = DOC_HEAD + Math.max(1, d.chunks.length) * CELL_H + DOC_PAD;
      const doc = { id: d.id, name: d.name, pseudo: d.pseudo, x: 0, y: dy, w: DOC_W, h: h, cells: [] };
      d.chunks.forEach(function (c, i) {
        const cell = {
          id: c.id, index: c.chunk_index,
          x: DOC_PAD, y: dy + DOC_HEAD + i * CELL_H, w: DOC_W - 2 * DOC_PAD, h: CELL_H - 4,
        };
        doc.cells.push(cell);
        lay.cellRect[c.id] = cell;
      });
      lay.docs.push(doc);
      dy += h + DOC_GAP;
    });
    const docColH = dy;

    // Entity band: type groups in balanced sub-columns (sequential fill —
    // deterministic given the sorted payload). Empty groups are skipped.
    const groupDefs = groupsP
      .filter(function (g) { return g.members.length > 0; })
      .map(function (g) { return { type_id: g.type_id, type_name: g.type_name, members: g.members, pseudo: false }; });
    if (ungroupedP.length) {
      groupDefs.push({
        type_id: '__ungrouped__', type_name: 'Other entities', pseudo: true,
        members: ungroupedP.map(function (id, i) { return { id: id, important: i < 8 }; }),
      });
    }
    const heights = groupDefs.map(function (g) {
      const vis = g.members.filter(function (m) { return m.important; }).length;
      const hidden = g.members.length - vis;
      return GRP_HEAD + vis * ROW_H + (hidden ? PILL_H : 0) + GRP_PAD;
    });
    const totalH = heights.reduce(function (a, b) { return a + b + GRP_GAP; }, 0);
    const nCols = Math.min(4, Math.max(1, Math.ceil(totalH / TARGET_COL_H)));
    const colTarget = totalH / nCols;
    const ENT_X0 = DOC_W + BAND_GAP;
    let col = 0, cy = 0, entColH = 0;
    groupDefs.forEach(function (g, gi) {
      const h = heights[gi];
      if (cy > 0 && cy + h / 2 > colTarget && col < nCols - 1) { col++; cy = 0; }
      const gx = ENT_X0 + col * (ENT_W + ENT_COL_GAP);
      const grp = {
        type_id: g.type_id, type_name: g.type_name, pseudo: g.pseudo,
        x: gx, y: cy, w: ENT_W, h: h, rows: [], pill: null,
        memberIds: g.members.map(function (m) { return m.id; }), hiddenIds: [],
      };
      let ry = cy + GRP_HEAD;
      g.members.forEach(function (m) {
        if (m.important) {
          const row = { id: m.id, x: gx + 6, y: ry, w: ENT_W - 12, h: ROW_H - 2 };
          grp.rows.push(row);
          lay.entityAnchor[m.id] = { lx: gx, rx: gx + ENT_W, cx: gx + ENT_W / 2, y: ry + ROW_H / 2 };
          ry += ROW_H;
        } else {
          grp.hiddenIds.push(m.id);
        }
      });
      if (grp.hiddenIds.length) {
        grp.pill = { x: gx + 6, y: ry, w: ENT_W - 12, h: PILL_H - 4, count: grp.hiddenIds.length };
        grp.hiddenIds.forEach(function (id) {
          lay.entityAnchor[id] = { lx: gx, rx: gx + ENT_W, cx: gx + ENT_W / 2, y: ry + PILL_H / 2 };
          groupByMember[id] = grp;
        });
      }
      g.members.forEach(function (m) { entityGroupName[m.id] = g.type_name; });
      lay.groups.push(grp);
      cy += h + GRP_GAP;
      entColH = Math.max(entColH, cy);
    });
    const entBandW = groupDefs.length ? nCols * (ENT_W + ENT_COL_GAP) - ENT_COL_GAP : 0;

    // Summaries: y = mean of source chunk cell centers, collision-resolved
    // by a stable downward sweep in payload order.
    const SUM_X = ENT_X0 + (entBandW || ENT_W) + BAND_GAP;
    let prevBottom = -Infinity;
    sumsP.forEach(function (s) {
      const ys = s.chunk_ids
        .map(function (cid) { return lay.cellRect[cid]; })
        .filter(Boolean)
        .map(function (r) { return r.y + r.h / 2; });
      const desired = ys.length
        ? ys.reduce(function (a, b) { return a + b; }, 0) / ys.length - SUM_H / 2
        : (prevBottom === -Infinity ? 0 : prevBottom + SUM_GAP);
      const y = Math.max(desired, prevBottom + SUM_GAP);
      lay.sums.push({ id: s.id, chunk_ids: s.chunk_ids, bucket_id: s.bucket_id, x: SUM_X, y: y, w: SUM_W, h: SUM_H });
      prevBottom = y + SUM_H;
    });
    const sumColH = prevBottom === -Infinity ? 0 : prevBottom;

    // Global context band: bucket cards in payload order, or the mandatory
    // empty state when no GlobalContextSummary exists (the demo case).
    const CTX_X = SUM_X + SUM_W + BAND_GAP;
    let ctxColH = 0;
    if (ctxP && ctxP.buckets && ctxP.buckets.length) {
      let by = 0;
      ctxP.buckets.forEach(function (b) {
        const isRoot = ctxP.root_id != null && b.id === ctxP.root_id;
        lay.ctxBoxes.push({
          id: b.id, level: b.level, isRoot: isRoot,
          child_ids: b.child_ids || [], x: CTX_X, y: by, w: CTX_W, h: 44,
        });
        by += 44 + 10;
      });
      ctxColH = by;
    } else {
      lay.ctxEmpty = { x: CTX_X, y: 0, w: CTX_W, h: 110 };
      ctxColH = 110;
    }

    lay.bands = [
      { x: 0, w: DOC_W, label: 'Documents', sub: docsP.length + (orphansP.length ? ' + orphans' : '') },
      { x: ENT_X0, w: entBandW || ENT_W, label: 'Entities', sub: String(groupDefs.length) + ' types' },
      { x: SUM_X, w: SUM_W, label: 'Summaries', sub: String(sumsP.length) },
      { x: CTX_X, w: CTX_W, label: 'Global context', sub: ctxP ? String((ctxP.buckets || []).length) : 'none yet' },
    ];
    lay.width = CTX_X + CTX_W;
    lay.height = Math.max(docColH, entColH, sumColH, ctxColH);
    return lay;
  }

  // ── Adjacency (for detail panels), from the structural edge index ─
  function buildAdjacency() {
    (edgesP.contains || []).forEach(function (pos) {
      const l = linkAt(pos); if (!l) return;
      const s = endId(l.source), t = endId(l.target);
      const c = stageOf(s) === 'chunk' ? s : (stageOf(t) === 'chunk' ? t : null);
      const e = stageOf(s) === 'entity' ? s : (stageOf(t) === 'entity' ? t : null);
      if (!c || !e) return;
      (chunkEntities[c] = chunkEntities[c] || []).push(e);
      (entityChunks[e] = entityChunks[e] || []).push(c);
    });
    (edgesP.semantic || []).forEach(function (pos) {
      const l = linkAt(pos); if (!l) return;
      const s = endId(l.source), t = endId(l.target);
      (entitySemantic[s] = entitySemantic[s] || []).push({ rel: l.relation, other: t });
      (entitySemantic[t] = entitySemantic[t] || []).push({ rel: l.relation, other: s });
    });
    sumsP.forEach(function (s) {
      s.chunk_ids.forEach(function (cid) {
        (summariesByChunk[cid] = summariesByChunk[cid] || []).push(s.id);
      });
    });
  }

  // ── SVG build ─────────────────────────────────────────────────────
  function registerUnit(nid, el) { (unitEls[nid] = unitEls[nid] || []).push(el); }

  function makeUnit(parent, classes, nid, kind) {
    const g = parent.append('g').attr('class', classes);
    if (nid) {
      g.attr('data-nid', nid);
      registerUnit(nid, g.node());
    }
    if (kind) {
      g.on('click', function (event) {
        event.stopPropagation();
        select(kind, nid, this);
      });
    }
    return g;
  }

  function buildSvg() {
    const svg = d3.select('#memory-svg');
    svg.selectAll('*').remove();
    svgSel = svg;
    const root = svg.append('g');
    rootSel = root;
    const gSem = root.append('g');
    const gEdge = root.append('g');
    const gMain = root.append('g');
    gOverlay = root.append('g');

    // Band headers.
    L.bands.forEach(function (b) {
      root.append('text').attr('class', 'mm-band-header').attr('x', b.x).attr('y', -34).text(b.label);
      root.append('text').attr('class', 'mm-band-sub').attr('x', b.x).attr('y', -20).text(b.sub);
    });

    // Documents with ordered chunk cells.
    L.docs.forEach(function (doc) {
      const gDoc = gMain.append('g').attr('class', 'mm-doc');
      const head = makeUnit(gDoc, 'mm-unit mm-doc-head' + (doc.pseudo ? '' : ' mm-node'),
        doc.pseudo ? null : doc.id, doc.pseudo ? null : 'document');
      head.append('rect').attr('class', 'mm-frame')
        .attr('x', doc.x).attr('y', doc.y).attr('width', doc.w).attr('height', doc.h).attr('rx', 8);
      head.append('text').attr('class', 'mm-title')
        .attr('x', doc.x + 10).attr('y', doc.y + 19).text(trunc(doc.name, 26));
      head.append('text').attr('class', 'mm-count')
        .attr('x', doc.x + doc.w - 10).attr('y', doc.y + 19).attr('text-anchor', 'end')
        .text(doc.cells.length);
      doc.cells.forEach(function (cell) {
        const n = nodeOf(cell.id);
        const gc = makeUnit(gDoc, 'mm-unit mm-node mm-cell', cell.id, 'chunk');
        gc.append('rect').attr('class', 'mm-box')
          .attr('x', cell.x).attr('y', cell.y).attr('width', cell.w).attr('height', cell.h).attr('rx', 4);
        const label = (cell.index != null ? '#' + cell.index + ' ' : '') + trunc(n.text || n.name || cell.id, 30);
        gc.append('text').attr('x', cell.x + 7).attr('y', cell.y + cell.h / 2 + 4).text(label);
      });
    });

    // Entity-type groups with collapse-behind-count pills.
    L.groups.forEach(function (grp) {
      const gG = gMain.append('g').attr('class', 'mm-group');
      const head = makeUnit(gG, 'mm-unit mm-group-head' + (grp.pseudo ? '' : ' mm-node'),
        grp.pseudo ? null : grp.type_id, grp.pseudo ? null : 'group');
      head.append('rect').attr('class', 'mm-frame')
        .attr('x', grp.x).attr('y', grp.y).attr('width', grp.w).attr('height', grp.h).attr('rx', 7);
      head.append('text').attr('class', 'mm-title')
        .attr('x', grp.x + 9).attr('y', grp.y + 16).text(trunc(grp.type_name, 22));
      head.append('text').attr('class', 'mm-count')
        .attr('x', grp.x + grp.w - 9).attr('y', grp.y + 16).attr('text-anchor', 'end')
        .text(grp.memberIds.length);
      grp.rows.forEach(function (row) {
        const gr = makeUnit(gG, 'mm-unit mm-node mm-row', row.id, 'entity');
        gr.append('rect').attr('class', 'mm-box')
          .attr('x', row.x).attr('y', row.y).attr('width', row.w).attr('height', row.h).attr('rx', 3);
        gr.append('circle').attr('class', 'mm-dot')
          .attr('cx', row.x + 7).attr('cy', row.y + row.h / 2).attr('r', 2.5);
        gr.append('text').attr('x', row.x + 15).attr('y', row.y + row.h / 2 + 4)
          .text(trunc(nameOf(row.id), 26));
      });
      if (grp.pill) {
        const gp = gG.append('g').attr('class', 'mm-unit mm-node mm-pill');
        // Pseudo groups ('Other entities') have no graph node — never tie
        // them to timeline visibility.
        if (!grp.pseudo) {
          gp.attr('data-nid', grp.type_id);
          registerUnit(grp.type_id, gp.node());
        }
        gp.on('click', function (event) { event.stopPropagation(); openGroupPanel(grp); });
        gp.append('rect').attr('class', 'mm-box')
          .attr('x', grp.pill.x).attr('y', grp.pill.y).attr('width', grp.pill.w).attr('height', grp.pill.h).attr('rx', 8);
        gp.append('text').attr('x', grp.pill.x + grp.pill.w / 2).attr('y', grp.pill.y + grp.pill.h / 2 + 3.5)
          .attr('text-anchor', 'middle').text('+' + grp.pill.count + ' more');
      }
    });

    // Summaries.
    L.sums.forEach(function (s) {
      const n = nodeOf(s.id);
      const gs = makeUnit(gMain, 'mm-unit mm-node mm-sum', s.id, 'summary');
      gs.append('rect').attr('class', 'mm-box')
        .attr('x', s.x).attr('y', s.y).attr('width', s.w).attr('height', s.h).attr('rx', 7);
      gs.append('text').attr('class', 'mm-sub')
        .attr('x', s.x + 9).attr('y', s.y + 16).text('Summary');
      gs.append('text').attr('x', s.x + 9).attr('y', s.y + 33).text(trunc(n.text || n.name || s.id, 32));
    });

    // Global context band — bucket cards or the dashed empty state.
    if (L.ctxEmpty) {
      const ge = gMain.append('g').attr('class', 'mm-ctx-empty');
      ge.append('rect').attr('x', L.ctxEmpty.x).attr('y', L.ctxEmpty.y)
        .attr('width', L.ctxEmpty.w).attr('height', L.ctxEmpty.h).attr('rx', 8);
      ['No global context yet.', 'Global-context indexing runs', 'will appear here.'].forEach(function (line, i) {
        ge.append('text').attr('x', L.ctxEmpty.x + L.ctxEmpty.w / 2).attr('y', L.ctxEmpty.y + 40 + i * 16)
          .attr('text-anchor', 'middle').text(line);
      });
    }
    L.ctxBoxes.forEach(function (b) {
      const gb = makeUnit(gMain, 'mm-unit mm-node mm-sum', b.id, 'bucket');
      gb.append('rect').attr('class', 'mm-box')
        .attr('x', b.x).attr('y', b.y).attr('width', b.w).attr('height', b.h).attr('rx', 7);
      gb.append('text').attr('class', 'mm-sub')
        .attr('x', b.x + 9).attr('y', b.y + 16)
        .text(b.isRoot ? 'Context root' : 'Bucket' + (b.level != null ? ' · level ' + b.level : ''));
      gb.append('text').attr('x', b.x + 9).attr('y', b.y + 33).text(trunc(nameOf(b.id), 30));
    });

    buildEdges(gSem, gEdge);
  }

  function buildEdges(gSem, gEdge) {
    const cellRight = function (id) {
      const r = L.cellRect[id]; return r ? { x: r.x + r.w, y: r.y + r.h / 2 } : null;
    };
    const entLeft = function (id) {
      const a = L.entityAnchor[id]; return a ? { x: a.lx, y: a.y } : null;
    };
    const sumRect = {};
    L.sums.forEach(function (s) { sumRect[s.id] = s; });
    const ctxRect = {};
    L.ctxBoxes.forEach(function (b) { ctxRect[b.id] = b; });
    const leftOf = function (r) { return { x: r.x, y: r.y + r.h / 2 }; };
    const rightOf = function (r) { return { x: r.x + r.w, y: r.y + r.h / 2 }; };

    function addEdge(parent, kind, pos, a, b, d) {
      if (!a || !b) return;
      const l = linkAt(pos);
      const el = parent.append('path')
        .attr('class', 'mm-edge mm-edge-' + kind)
        .attr('d', d || bez(a, b))
        .node();
      const s = l ? endId(l.source) : null, t = l ? endId(l.target) : null;
      edgeEls.push({ kind: kind, s: s, t: t, el: el });
      const oid = l && l.edge_info && l.edge_info.edge_object_id;
      if (oid) (edgeByObjId[oid] = edgeByObjId[oid] || []).push(el);
    }

    (edgesP.contains || []).forEach(function (pos) {
      const l = linkAt(pos); if (!l) return;
      const s = endId(l.source), t = endId(l.target);
      const c = L.cellRect[s] ? s : (L.cellRect[t] ? t : null);
      const e = L.entityAnchor[s] ? s : (L.entityAnchor[t] ? t : null);
      if (c && e) addEdge(gEdge, 'contains', pos, cellRight(c), entLeft(e));
    });
    (edgesP.made_from || []).forEach(function (pos) {
      const l = linkAt(pos); if (!l) return;
      const s = endId(l.source), t = endId(l.target);
      const su = sumRect[s] ? s : (sumRect[t] ? t : null);
      const c = L.cellRect[s] ? s : (L.cellRect[t] ? t : null);
      if (su && c) addEdge(gEdge, 'made_from', pos, cellRight(c), leftOf(sumRect[su]));
    });
    (edgesP.summarized_in || []).forEach(function (pos) {
      const l = linkAt(pos); if (!l) return;
      const s = endId(l.source), t = endId(l.target);
      const child = sumRect[s] ? sumRect[s] : ctxRect[s];
      const parent = ctxRect[t];
      if (child && parent) addEdge(gEdge, 'summarized_in', pos, rightOf(child), leftOf(parent));
    });
    (edgesP.semantic || []).forEach(function (pos) {
      const l = linkAt(pos); if (!l) return;
      const a = L.entityAnchor[endId(l.source)], b = L.entityAnchor[endId(l.target)];
      if (!a || !b) return;
      let d;
      if (Math.abs(a.cx - b.cx) > 1) {
        const from = a.cx < b.cx ? { x: a.rx, y: a.y } : { x: a.lx, y: a.y };
        const to = a.cx < b.cx ? { x: b.lx, y: b.y } : { x: b.rx, y: b.y };
        d = bez(from, to);
      } else {
        const bow = Math.min(46, 14 + Math.abs(a.y - b.y) / 18);
        const x = a.lx - bow;
        d = 'M' + a.lx + ',' + a.y + ' C' + x + ',' + a.y + ' ' + x + ',' + b.y + ' ' + b.lx + ',' + b.y;
      }
      addEdge(gSem, 'semantic', pos, { x: 0, y: 0 }, { x: 0, y: 0 }, d);
    });
  }

  // ── Pan / zoom (schema_view pattern) ─────────────────────────────
  function panelOffset() {
    const panel = document.getElementById('memory-side-panel');
    return panel && panel.style.display === 'block' ? (panel.offsetWidth || 344) : 0;
  }

  function fitView(animate) {
    if (!svgSel || !rootSel || !zoomBehavior) return;
    const node = svgSel.node();
    const W = node.clientWidth || node.getBoundingClientRect().width || 1200;
    const H = node.clientHeight || node.getBoundingClientRect().height || 700;
    const avail = Math.max(240, W - panelOffset());
    let bbox;
    try { bbox = rootSel.node().getBBox(); } catch (e) { return; }
    const pad = 44;
    const bw = bbox.width || 1, bh = bbox.height || 1;
    const scale = Math.max(0.05, Math.min((avail - pad * 2) / bw, (H - pad * 2) / bh, 1.5));
    const tx = pad + (avail - pad * 2 - bw * scale) / 2 - bbox.x * scale;
    const ty = pad + (H - pad * 2 - bh * scale) / 2 - bbox.y * scale;
    const t = d3.zoomIdentity.translate(tx, ty).scale(scale);
    (animate ? svgSel.transition().duration(420) : svgSel).call(zoomBehavior.transform, t);
  }

  function wireZoom(savedTransform) {
    zoomBehavior = d3.zoom()
      .scaleExtent([0.04, 3])
      .on('start', function () { svgSel.classed('is-panning', true); })
      .on('zoom', function (event) { rootSel.attr('transform', event.transform); })
      .on('end', function () { svgSel.classed('is-panning', false); });
    svgSel.call(zoomBehavior).on('dblclick.zoom', null);
    if (savedTransform) svgSel.call(zoomBehavior.transform, savedTransform);
    else fitView(false);
    svgSel.on('click', function () { clearAll(); });
  }

  // ── Timeline rail ─────────────────────────────────────────────────
  function buildRail() {
    railItems = timelineP.map(function (e) { return { kind: 'run', t: e.t0 || 0, run: e }; })
      .concat((searchEvents || []).map(function (s) {
        return { kind: 'search', t: Date.parse(s.time) || 0, search: s };
      }))
      .sort(function (a, b) { return a.t - b.t || (a.kind === b.kind ? 0 : a.kind === 'run' ? -1 : 1); });

    const rail = document.getElementById('memory-timeline');
    rail.innerHTML = '';
    const lbl = document.createElement('span');
    lbl.className = 'mm-rail-label';
    lbl.textContent = 'Timeline';
    rail.appendChild(lbl);
    railItems.forEach(function (item, i) {
      const btn = document.createElement('button');
      btn.className = 'mm-rail-item' + (item.kind === 'search' ? ' mm-rail-search' : '');
      if (item.kind === 'run') {
        btn.innerHTML = '<span>' + esc(trunc(item.run.label, 20)) + '</span>' +
          '<span class="t">' + esc(fmtT(item.run.t0)) + ' · ' + item.run.node_count + ' nodes</span>';
        btn.title = 'Show memory state after this run';
        btn.addEventListener('click', function () { applyRun(item.run.index, i); openRunPanel(item.run); });
      } else {
        btn.innerHTML = '<span>⌕ ' + esc(trunc(item.search.question || 'search', 24)) + '</span>' +
          '<span class="t">' + esc(item.search.time || '') + '</span>';
        btn.title = item.search.question || '';
        btn.addEventListener('click', function () { applySearch(item.search, i); });
      }
      btn.dataset.rail = String(i);
      rail.appendChild(btn);
    });
  }

  function setActiveRail(i) {
    document.querySelectorAll('#memory-timeline .mm-rail-item').forEach(function (el) {
      el.classList.toggle('active', el.dataset.rail === String(i));
    });
  }

  function railIndexOfRun(runIdx) {
    for (let i = 0; i < railItems.length; i++) {
      if (railItems[i].kind === 'run' && railItems[i].run.index === runIdx) return i;
    }
    return -1;
  }

  // ── Visibility (timeline scrub): classes only, never positions ────
  function applyRun(runIdx, railIdx) {
    clearSearch();
    currentRunIdx = runIdx;
    if (!timelineP.length || runIdx >= timelineP.length - 1) {
      visibleSet = null;  // latest event = full current state
    } else {
      visibleSet = new Set();
      timelineP.forEach(function (e) {
        if (e.index <= runIdx) e.node_ids.forEach(function (id) { visibleSet.add(id); });
      });
    }
    applyVisibility();
    setActiveRail(railIdx != null ? railIdx : railIndexOfRun(runIdx));
    updateStatus();
  }

  function isVisible(id) { return !visibleSet || visibleSet.has(id); }

  function applyVisibility() {
    Object.keys(unitEls).forEach(function (nid) {
      const vis = isVisible(nid);
      unitEls[nid].forEach(function (el) { el.classList.toggle('mm-future', !vis); });
    });
    edgeEls.forEach(function (e) {
      const vis = (e.s == null || isVisible(e.s)) && (e.t == null || isVisible(e.t));
      e.el.classList.toggle('mm-future', !vis);
    });
  }

  function updateStatus() {
    const el = document.getElementById('memory-status');
    if (!el || !L) return;
    const entSet = new Set();
    L.groups.forEach(function (g) { g.memberIds.forEach(function (id) { entSet.add(id); }); });
    const nEnt = entSet.size;
    const nChunks = Object.keys(L.cellRect).length;
    const nCtx = L.ctxBoxes.length;
    const base = docsP.length + ' documents · ' + nChunks + ' chunks · ' +
      nEnt + ' entities · ' + sumsP.length + ' summaries' +
      (nCtx ? ' · ' + nCtx + ' context' : '');
    if (activeSearch) {
      el.textContent = base + ' — spotlighting retrieval: “' + trunc(activeSearch.question || '', 60) + '”';
    } else if (visibleSet) {
      const run = timelineP[currentRunIdx] || {};
      el.textContent = base + ' — state as of ' + (run.label || 'run') + ' (' + fmtT(run.t1) + '), ' +
        visibleSet.size + ' elements';
    } else {
      el.textContent = base + ' — current state';
    }
  }

  // ── Search overlay: dim + spotlight on the SAME positions ────────
  function applySearch(evt, railIdx) {
    clearSearch();
    visibleSet = null;
    applyVisibility();
    activeSearch = evt;
    svgSel.classed('mm-searching', true);
    const spot = new Set(evt.node_ids || []);
    spot.forEach(function (id) {
      (unitEls[id] || []).forEach(function (el) { el.classList.add('is-spotlit'); });
      if (!unitEls[id] && groupByMember[id]) promoteCollapsed(id);
    });
    // Retrieved edges: join edge_ids on edge_object_id; provenance trail =
    // structural edges whose both endpoints were retrieved.
    (evt.edge_ids || []).forEach(function (oid) {
      (edgeByObjId[oid] || []).forEach(function (el) { el.classList.add('is-spotlit-edge'); });
    });
    edgeEls.forEach(function (e) {
      if (e.s != null && e.t != null && spot.has(e.s) && spot.has(e.t) &&
          !e.el.classList.contains('is-spotlit-edge')) {
        e.el.classList.add('is-trail');
      }
    });
    setActiveRail(railIdx);
    openSearchPanel(evt);
    updateStatus();
  }

  // A collapsed retrieved entity is temporarily promoted out of its "+K"
  // pill into an overlay slot below its group — nothing else moves.
  let promotedCount = {};
  function promoteCollapsed(id) {
    const grp = groupByMember[id];
    const i = promotedCount[grp.type_id] = (promotedCount[grp.type_id] || 0) + 1;
    const y = grp.y + grp.h + 2 + (i - 1) * ROW_H;
    const gr = gOverlay.append('g').attr('class', 'mm-unit mm-node mm-row is-spotlit');
    gr.append('rect').attr('class', 'mm-box')
      .attr('x', grp.x + 6).attr('y', y).attr('width', grp.w - 12).attr('height', ROW_H - 2).attr('rx', 3);
    gr.append('circle').attr('class', 'mm-dot').attr('cx', grp.x + 13).attr('cy', y + ROW_H / 2 - 1).attr('r', 2.5);
    gr.append('text').attr('x', grp.x + 21).attr('y', y + ROW_H / 2 + 3).text(trunc(nameOf(id), 26));
    gr.on('click', function (event) { event.stopPropagation(); select('entity', id, this); });
  }

  function clearSearch() {
    activeSearch = null;
    if (svgSel) svgSel.classed('mm-searching', false);
    document.querySelectorAll('#memory-svg .is-spotlit').forEach(function (el) { el.classList.remove('is-spotlit'); });
    document.querySelectorAll('#memory-svg .is-spotlit-edge').forEach(function (el) { el.classList.remove('is-spotlit-edge'); });
    document.querySelectorAll('#memory-svg .is-trail').forEach(function (el) { el.classList.remove('is-trail'); });
    if (gOverlay) gOverlay.selectAll('*').remove();
    promotedCount = {};
  }

  function clearAll() {
    clearSearch();
    clearSelection();
    hidePanel();
    if (currentRunIdx >= 0) applyRun(currentRunIdx);
    updateStatus();
  }

  // ── Selection + detail panel ──────────────────────────────────────
  function clearSelection() {
    selection = null;
    document.querySelectorAll('#memory-svg .is-selected').forEach(function (el) { el.classList.remove('is-selected'); });
  }

  function select(kind, id, el) {
    clearSelection();
    selection = { kind: kind, id: id };
    if (el) el.classList.add('is-selected');
    else (unitEls[id] || []).forEach(function (e) { e.classList.add('is-selected'); });
    const fn = {
      document: openDocumentPanel, chunk: openChunkPanel, entity: openEntityPanel,
      summary: openSummaryPanel, bucket: openBucketPanel,
      group: function (gid) {
        const grp = L.groups.find(function (g) { return g.type_id === gid; });
        if (grp) openGroupPanel(grp);
      },
    }[kind];
    if (fn) fn(id);
  }

  function selectNodeById(id) {
    const kindByStage = { document: 'document', chunk: 'chunk', entity: 'entity', summary: 'summary', context: 'bucket', type: 'group' };
    const kind = kindByStage[stageOf(id)];
    if (kind) select(kind, id, (unitEls[id] || [])[0] || null);
  }

  const panelEl = function () { return document.getElementById('memory-side-panel'); };
  function showPanel(html) {
    const p = panelEl();
    p.innerHTML = '<div class="si-close" title="Close">×</div>' + html;
    p.style.display = 'block';
    p.querySelector('.si-close').addEventListener('click', function () { hidePanel(); clearSelection(); });
    p.querySelectorAll('[data-goto]').forEach(function (chip) {
      chip.addEventListener('click', function () { selectNodeById(chip.dataset.goto); });
    });
  }
  function hidePanel() { const p = panelEl(); if (p) p.style.display = 'none'; }

  function row(k, v) {
    if (v == null || v === '') return '';
    return '<div class="panel-row"><span class="k">' + esc(k) + '</span><span class="v">' + esc(v) + '</span></div>';
  }
  function chips(ids, limit) {
    const lim = limit || 24;
    let html = '<div class="si-chips">';
    ids.slice(0, lim).forEach(function (id) {
      html += '<span class="si-chip si-chip-link" data-goto="' + esc(id) + '">' + esc(trunc(nameOf(id), 28)) + '</span>';
    });
    if (ids.length > lim) html += '<span class="si-more">+' + (ids.length - lim) + ' more</span>';
    return html + '</div>';
  }
  function heading(t) { return '<div class="si-heading">' + esc(t) + '</div>'; }
  function title(badge, name) {
    return '<div class="si-count">' + esc(badge) + '</div><div class="si-title">' + esc(trunc(name, 60)) + '</div>';
  }

  function openDocumentPanel(id) {
    const n = nodeOf(id);
    const doc = L.docs.find(function (d) { return d.id === id; });
    showPanel(
      title('Document', n.name || id) +
      row('MIME type', n.mime_type) +
      row('Chunks', doc ? doc.cells.length : '') +
      row('Created', fmtT(n.t_created)) +
      row('Location', trunc(n.raw_data_location, 70))
    );
  }

  function openChunkPanel(id) {
    const n = nodeOf(id);
    let doc = null;
    L.docs.forEach(function (d) { if (d.cells.some(function (c) { return c.id === id; })) doc = d; });
    const ents = chunkEntities[id] || [];
    const sums = summariesByChunk[id] || [];
    showPanel(
      title('Chunk' + (n.chunk_index != null ? ' #' + n.chunk_index : ''), doc ? doc.name : (n.name || id)) +
      (n.text ? '<div class="mm-panel-text">' + esc(trunc(n.text, 400)) + '</div>' : '') +
      row('Size', n.chunk_size) +
      row('Cut type', n.cut_type) +
      row('Created', fmtT(n.t_created)) +
      (ents.length ? heading('Entities (' + ents.length + ')') + chips(ents) : '') +
      (sums.length ? heading('Summaries') + chips(sums) : '')
    );
  }

  function openEntityPanel(id) {
    const n = nodeOf(id);
    const rels = entitySemantic[id] || [];
    let relHtml = '';
    if (rels.length) {
      relHtml = heading('Relations (' + rels.length + ')');
      rels.slice(0, 15).forEach(function (r) {
        relHtml += '<div class="si-rel"><span class="si-rel-name">' + esc(trunc(r.rel, 26)) + '</span>' +
          '<span class="si-rel-to si-chip-link" data-goto="' + esc(r.other) + '">' + esc(trunc(nameOf(r.other), 24)) + '</span></div>';
      });
      if (rels.length > 15) relHtml += '<div class="si-more">+' + (rels.length - 15) + ' more</div>';
    }
    const srcChunks = entityChunks[id] || [];
    showPanel(
      title(entityGroupName[id] || 'Entity', n.name || id) +
      (n.description ? '<div class="mm-panel-text">' + esc(trunc(n.description, 320)) + '</div>' : '') +
      row('Degree', n.degree) +
      row('Created', fmtT(n.t_created)) +
      (srcChunks.length ? heading('Source chunks') + chips(srcChunks) : '') +
      relHtml
    );
  }

  function openGroupPanel(grp) {
    clearSelection();
    showPanel(
      title('Entity type', grp.type_name) +
      row('Members', grp.memberIds.length) +
      heading('Members') + chips(grp.memberIds, 40)
    );
  }

  function openSummaryPanel(id) {
    const n = nodeOf(id);
    const s = L.sums.find(function (x) { return x.id === id; });
    showPanel(
      title('Summary', n.name || 'TextSummary') +
      (n.text ? '<div class="mm-panel-text">' + esc(n.text) + '</div>' : '') +
      row('Created', fmtT(n.t_created)) +
      (s && s.chunk_ids.length ? heading('Source chunks') + chips(s.chunk_ids) : '') +
      (s && s.bucket_id ? heading('Context bucket') + chips([s.bucket_id]) : '')
    );
  }

  function openBucketPanel(id) {
    const n = nodeOf(id);
    const b = L.ctxBoxes.find(function (x) { return x.id === id; });
    showPanel(
      title(b && b.isRoot ? 'Context root' : 'Context bucket', n.name || id) +
      (n.text ? '<div class="mm-panel-text">' + esc(trunc(n.text, 400)) + '</div>' : '') +
      row('Level', b ? b.level : n.level) +
      row('Children', b ? b.child_ids.length : '') +
      (b && b.child_ids.length ? heading('Children') + chips(b.child_ids) : '')
    );
  }

  function openRunPanel(run) {
    const byStage = {};
    (run.node_ids || []).forEach(function (id) {
      const s = stageOf(id);
      byStage[s] = (byStage[s] || 0) + 1;
    });
    let breakdown = '';
    Object.keys(byStage).sort().forEach(function (s) { breakdown += row(s, byStage[s]); });
    showPanel(
      title('Run event', run.label) +
      row('Started', fmtT(run.t0)) +
      row('Finished', fmtT(run.t1)) +
      row('Nodes added', run.node_count) +
      heading('By stage') + breakdown
    );
  }

  function openSearchPanel(evt) {
    const byStage = {};
    (evt.node_ids || []).forEach(function (id) {
      const s = stageOf(id);
      byStage[s] = (byStage[s] || 0) + 1;
    });
    let counts = '';
    Object.keys(byStage).sort().forEach(function (s) { counts += row(s, byStage[s]); });
    const chunksRetrieved = (evt.node_ids || []).filter(function (id) { return stageOf(id) === 'chunk'; });
    showPanel(
      title('Search', evt.question || '') +
      (evt.answer ? heading('Answer') + '<div class="mm-panel-text">' + esc(trunc(evt.answer, 600)) + '</div>' : '') +
      row('Time', evt.time) +
      row('Retrieved elements', (evt.node_ids || []).length) +
      row('Retrieved edges', (evt.edge_ids || []).length) +
      (counts ? heading('Retrieved by stage') + counts : '') +
      (chunksRetrieved.length ? heading('Retrieved chunks') + chips(chunksRetrieved) : '')
    );
  }

  // ── Entry point (lazy, called by ui_chrome on tab click / theme) ──
  window._renderMemoryView = function (preserveTransform) {
    const view = document.getElementById('memory-view');
    if (!view) return;
    if (built) {
      // Theme changes are pure CSS (everything reads --mm-* vars); the
      // existing transform, selection and overlay all stay valid.
      if (!preserveTransform) updateStatus();
      return;
    }
    const isEmpty = !docsP.length && !orphansP.length && !groupsP.length &&
      !ungroupedP.length && !sumsP.length;
    const emptyEl = document.getElementById('memory-empty');
    if (isEmpty) {
      if (emptyEl) emptyEl.style.display = 'flex';
      document.getElementById('memory-diagram-section').style.display = 'none';
      document.getElementById('memory-zoom').style.display = 'none';
      document.getElementById('memory-timeline').style.display = 'none';
      built = true;
      return;
    }
    if (emptyEl) emptyEl.style.display = 'none';

    L = computeLayout();
    buildAdjacency();
    buildSvg();
    buildRail();
    wireZoom(null);

    // Zoom pill.
    const bind = function (id, fn) {
      const el = document.getElementById(id);
      if (el) el.addEventListener('click', fn);
    };
    bind('memory-zoom-in', function () { svgSel.transition().duration(180).call(zoomBehavior.scaleBy, 1.35); });
    bind('memory-zoom-out', function () { svgSel.transition().duration(180).call(zoomBehavior.scaleBy, 1 / 1.35); });
    bind('memory-zoom-fit', function () { fitView(true); });

    // Default selection: the latest run event (full current state).
    const lastRun = timelineP.length ? timelineP[timelineP.length - 1].index : -1;
    if (lastRun >= 0) applyRun(lastRun);
    else { visibleSet = null; updateStatus(); }

    built = true;
  };
})();
