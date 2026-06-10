// Schema view (Phase 2 redesign): explicit-position ontology diagram.
//
// Replaces the previous "type-cards above a force-graph hairball" layout
// with a single primary diagram that's structured, not floating:
//   - Node-type boxes positioned in canonical pipeline order
//     (Documents → Chunks → Entities → Types → Summaries → Context → Schema)
//   - Each box surfaces type name, instance count, modal pipeline+task,
//     and a short field inventory
//   - Edges between node types drawn as labelled curves carrying the
//     relation name + edge count
//   - Below the diagram: edge-type cards listing every relationship type
//     with its source→target pairs
//   - Click a node box or edge card to highlight related elements
//
// Reads ``schemaGraphData`` produced by preprocessor.extract_type_schema_graph_data:
//   - nodes of type GraphNodeType (one per node type, with source_pipeline/
//     source_task/instance_count surfaced at the top level)
//   - nodes of type GraphRelationshipType (one per source_type → target_type
//     pair, with source_type/target_type/relationship_label/edge_count)
//   - links connect type→rel ("from") and rel→type ("to")
(function(){
  const schemaData = __SCHEMA_DATA__;
  const schemaGraphData = __SCHEMA_GRAPH_DATA__;
  const emptyEl = document.getElementById('schema-empty');
  const svgEl = document.getElementById('schema-svg');

  function escapeHtml(s){
    return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  function stripCoverage(s){
    return String(s || '').replace(/\s*\d+%\s*$/, '');
  }
  function truncate(s, n){
    const text = String(s == null ? '' : s);
    return text.length > n ? text.slice(0, n) + '…' : text;
  }

  // Stage column position by node-type rank (matches preprocessor.node_type_rank).
  // Used to lay out type boxes in pipeline order.
  // Column titles keyed by rank. Negative ranks are the actor / ownership
  // layer that flows in before the document→memory pipeline.
  const STAGE_LABELS = {
    '-5': 'Organization',
    '-4': 'People',
    '-3': 'Agents',
    '-2': 'Sessions',
    '-1': 'Brain',
    '0': 'Documents',
    '1': 'Chunks',
    '2': 'Entities',
    '3': 'Types',
    '4': 'Summaries',
    '5': 'Context',
  };
  function stageLabel(rank){
    if (typeof rank !== 'number' || !isFinite(rank)) return '';
    return STAGE_LABELS[Math.round(rank)] || '';
  }

  // Per-type accent color — a calm, restrained Segment-style palette
  // (green-forward with supporting blue / violet / slate) for the airy
  // light schema surface.
  const ACCENT_BY_TYPE = {
    // Actor / ownership layer
    Tenant: '#0E7C66',
    User: '#2F80ED',
    Agent: '#F2994A',
    Dataset: '#7C5CFC',
    Session: '#EB5757',
    // Document → memory pipeline
    TextDocument: '#64748B',
    DocumentChunk: '#1F9E6E',
    TextSummary: '#B794F6',
    GlobalContextSummary: '#2D9CDB',
    Entity: '#1F9E6E',
    EntityType: '#B794F6',
    DatabaseSchema: '#2D9CDB',
    SchemaTable: '#7C5CFC',
    SchemaRelationship: '#8792A2',
    TableType: '#7C5CFC',
    TableRow: '#7C5CFC',
    ColumnValue: '#8792A2',
  };
  function accentFor(name){ return ACCENT_BY_TYPE[name] || '#1F9E6E'; }

  // ── Build the schema model from preprocessor data ────────────────
  function buildSchemaModel() {
    if (!schemaGraphData || !Array.isArray(schemaGraphData.nodes) || !schemaGraphData.nodes.length) {
      return { nodeTypes: [], edgeTypes: [] };
    }
    const nodeTypes = [];
    const edgeTypes = [];
    window._schemaTypeIndex = {};
    // Instance-level drill-down data (PR: instance-aware inspector).
    window._schemaInstancesByType = (schemaGraphData && schemaGraphData.instances_by_type) || {};
    window._schemaInstanceIndex = (schemaGraphData && schemaGraphData.instance_index) || {};
    // Transformation impact-layer (operations → schema types).
    window._schemaOperations = (schemaGraphData && schemaGraphData.operations) || [];
    window._schemaOperationLinks = (schemaGraphData && schemaGraphData.operation_links) || [];
    schemaGraphData.nodes.forEach(function(n) {
      if (n.type === 'GraphNodeType') {
        var t = {
          id: String(n.id),
          name: String(n.name || n.id),
          rank: Number.isFinite(Number(n.rank)) ? Number(n.rank) : 4,
          fields: Array.isArray(n.fields) ? n.fields : [],
          instance_count: n.instance_count != null ? n.instance_count : null,
          source_pipeline: n.source_pipeline || null,
          source_task: n.source_task || null,
          source_user: n.source_user || null,
          // PR2 contract: bounded instance display-names + sample size.
          samples: Array.isArray(n.samples) ? n.samples : [],
          sample_size: n.sample_size != null ? n.sample_size : (Array.isArray(n.samples) ? n.samples.length : 0),
          // PR2 contract: full outgoing relationship distribution per type.
          relationships: Array.isArray(n.relationships) ? n.relationships : [],
        };
        nodeTypes.push(t);
        // Index the full datum by name so the inspector click handler can
        // read samples / relationships without re-walking schemaGraphData.
        window._schemaTypeIndex[t.name] = t;
      } else if (n.type === 'GraphRelationshipType') {
        edgeTypes.push({
          id: String(n.id),
          source_type: n.source_type || '',
          target_type: n.target_type || '',
          // relationship_label looks like "is_a (4)" or "contains (4), made_from (2) +1 more"
          relationship_label: n.relationship_label || '',
          edge_count: n.edge_count || 0,
        });
      }
    });
    // Sort node types by rank, then name; gives stable Story order
    nodeTypes.sort(function(a, b) {
      if (a.rank !== b.rank) return a.rank - b.rank;
      return a.name.localeCompare(b.name);
    });
    return { nodeTypes: nodeTypes, edgeTypes: edgeTypes };
  }

  // Extract the primary relation name from a relationship_label string.
  // "contains (4), made_from (2) +1 more" → "contains"
  function primaryRelation(rel){
    const m = /^([^\s(,]+)/.exec(rel.relationship_label || '');
    return m ? m[1] : 'related';
  }

  // ── Layout: place node-type boxes in stage columns ───────────────
  // Returns a map of type-name → {x, y, w, h, type} and metadata for the
  // overall diagram bounds.
  // Layout sizing — exported so the renderer draws the same header/card metrics.
  const LAYOUT = {
    PAD_LEFT: 56, PAD_TOP: 64, BOX_W: 212, COL_GAP: 128, ROW_GAP: 44,
    HEADER_H: 42, CARD_H: 27, MORE_H: 20, BOTTOM_PAD: 12, MAX_CARDS: 5, LENS_H: 66,
  };

  function instancesForType(name) {
    return (window._schemaInstancesByType || {})[name] || [];
  }

  function isTypeExpanded(name) {
    return !!(window._schemaExpanded || {})[name];
  }

  // A type box grows to fit its instance mini-cards (capped at MAX_CARDS unless
  // expanded); a lensed single-instance box stays compact.
  function boxHeight(t) {
    if (t.instance_count == null) return LAYOUT.LENS_H;
    const insts = instancesForType(t.name);
    const total = t.instance_count || insts.length || 0;
    if (!total) return LAYOUT.LENS_H;
    const shown = isTypeExpanded(t.name)
      ? Math.min(insts.length, total)
      : Math.min(total, LAYOUT.MAX_CARDS);
    const more = total > LAYOUT.MAX_CARDS ? LAYOUT.MORE_H : 0;  // room for the expand/collapse toggle
    return LAYOUT.HEADER_H + shown * LAYOUT.CARD_H + more + LAYOUT.BOTTOM_PAD;
  }

  function layoutBoxes(nodeTypes, totalWidth) {
    const { PAD_LEFT, PAD_TOP, BOX_W, COL_GAP, ROW_GAP } = LAYOUT;
    const byRank = {};
    nodeTypes.forEach(function(t) {
      const r = Math.round(t.rank);
      if (!byRank[r]) byRank[r] = [];
      byRank[r].push(t);
    });
    const ranks = Object.keys(byRank).map(Number).sort(function(a, b) { return a - b; });
    const positions = {};
    let maxY = 0;
    ranks.forEach(function(r, colIdx) {
      const colX = PAD_LEFT + colIdx * (BOX_W + COL_GAP);
      let y = PAD_TOP;
      byRank[r].forEach(function(t) {
        const h = boxHeight(t);
        positions[t.name] = { x: colX, y: y, w: BOX_W, h: h, type: t, rank: r };
        if (y + h > maxY) maxY = y + h;
        y += h + ROW_GAP;
      });
    });
    const diagramW = Math.max(totalWidth, PAD_LEFT + ranks.length * (BOX_W + COL_GAP));
    const diagramH = maxY + 32;
    return {
      positions: positions,
      width: diagramW,
      height: diagramH,
      ranks: ranks,
      byRank: byRank,
      box_w: BOX_W,
      pad_left: PAD_LEFT,
      col_gap: COL_GAP,
      pad_top: PAD_TOP,
    };
  }

  // Pick a connection point on the box perimeter aimed at a target point.
  // Keeps arrows from emerging out of the middle of a box.
  function boxAnchor(box, towardX, towardY) {
    const cx = box.x + box.w / 2;
    const cy = box.y + box.h / 2;
    const dx = towardX - cx;
    const dy = towardY - cy;
    if (Math.abs(dx) < 1 && Math.abs(dy) < 1) return { x: cx, y: cy };
    const halfW = box.w / 2;
    const halfH = box.h / 2;
    const slope = Math.abs(dy / (dx || 0.001));
    if (slope * halfW < halfH) {
      // Exits left/right side
      const sign = dx > 0 ? 1 : -1;
      return { x: cx + sign * halfW, y: cy + sign * halfW * (dy / (dx || 0.001)) };
    }
    const sign = dy > 0 ? 1 : -1;
    return { x: cx + sign * halfH * (dx / (dy || 0.001)), y: cy + sign * halfH };
  }

  // ── Render the SVG diagram ───────────────────────────────────────
  function renderSchemaDiagram(model, savedTransform) {
    const svg = d3.select('#schema-svg');
    svg.selectAll('*').remove();
    // Follow the global theme toggle (ui_chrome re-renders the schema on
    // toggle). The light palette is the Segment-style airy surface; the dark
    // palette mirrors it on deep neutrals.
    const isLight = window._isLightMode !== false;
    const P = isLight
      ? {
          card: '#ffffff', cardBorder: '#E6E8EB',
          title: '#1A1F36', muted: '#8792A2',
          pill: '#EEF1F4', pillText: '#4A5568',
          mini: '#F6F8FA', miniBorder: '#EDF0F3', miniText: '#1A1F36',
          shadow: '#1A1F36', shadowOpacity: 0.10,
        }
      : {
          card: '#23272F', cardBorder: '#3A4049',
          title: '#ECEFF4', muted: '#9AA4B2',
          pill: '#2E333C', pillText: '#C3CAD4',
          mini: '#2A2F37', miniBorder: '#3A4049', miniText: '#E4E8EE',
          shadow: '#000000', shadowOpacity: 0.40,
        };
    // Absolute rects captured during render so the spotlight overlay can draw
    // instance-level connectors without re-laying-out the diagram.
    window._miniCardRects = {};
    window._boxRectByType = {};
    const modifiedTypes = modifiedTypeNames();

    if (!model.nodeTypes.length) {
      return;
    }

    const section = document.getElementById('schema-diagram-section');
    const containerW = (section && section.clientWidth) || 1200;
    const layout = layoutBoxes(model.nodeTypes, containerW);
    // The SVG fills the viewport; the diagram lives in a pan/zoomable root <g>
    // (so the content can extend past the viewport and under the side panel).
    svg.attr('width', (section && section.clientWidth) || containerW)
       .attr('height', (section && section.clientHeight) || 800);
    const root = svg.append('g');

    // Accent is keyed by TYPE. In the type view a box's name IS its type; in the
    // lensed instance view a box carries an explicit accentType (its type).
    const accentTypeByName = {};
    model.nodeTypes.forEach(function(t) { accentTypeByName[t.name] = t.accentType || t.name; });
    function accentForName(n) { return accentFor(accentTypeByName[n] || n); }

    // Stage-column headers
    layout.ranks.forEach(function(r, colIdx) {
      const colX = layout.pad_left + colIdx * (layout.box_w + layout.col_gap);
      const label = stageLabel(r);
      if (!label) return;
      root.append('text')
        .attr('class', 'sd-stage-header')
        .attr('x', colX + layout.box_w / 2)
        .attr('y', 28)
        .attr('text-anchor', 'middle')
        .text(label);
    });

    // ── Edges: render as Bezier curves between box anchors ─────────
    // For each unique source_type → target_type pair, group edges by
    // relation so multiple relations between the same pair stack
    // vertically next to the midpoint.
    const positions = layout.positions;
    const edgeGroups = {};
    model.edgeTypes.forEach(function(e) {
      if (!positions[e.source_type] || !positions[e.target_type]) return;
      const key = e.source_type + '|' + e.target_type;
      if (!edgeGroups[key]) edgeGroups[key] = [];
      edgeGroups[key].push(e);
    });

    const drawnEdges = [];
    Object.keys(edgeGroups).forEach(function(key) {
      const group = edgeGroups[key];
      group.forEach(function(e, idxInPair) {
        const src = positions[e.source_type];
        const tgt = positions[e.target_type];
        const srcCx = src.x + src.w / 2;
        const srcCy = src.y + src.h / 2;
        const tgtCx = tgt.x + tgt.w / 2;
        const tgtCy = tgt.y + tgt.h / 2;
        const isSelf = src === tgt;
        let pathD;
        let labelX, labelY;
        let sx, sy, tx, ty;
        if (isSelf) {
          // Self-loop on the right side of the box
          const lx = src.x + src.w;
          pathD = 'M ' + lx + ' ' + (src.y + 30) +
                  ' C ' + (lx + 60) + ' ' + (src.y + 10) +
                  ', ' + (lx + 60) + ' ' + (src.y + src.h - 10) +
                  ', ' + lx + ' ' + (src.y + src.h - 30);
          labelX = lx + 36;
          labelY = src.y + src.h / 2;
          sx = lx; sy = src.y + 30; tx = lx; ty = src.y + src.h - 30;
        } else if (Math.abs(src.x - tgt.x) < LAYOUT.BOX_W * 0.5) {
          // Same-column pair (vertically stacked cards — common in the tall
          // Entity column). The horizontal-ease curve degenerates into an
          // S-loop here, leaving arrowheads pointing backwards. Route a side
          // loop instead: exit the source's right edge, bulge outward, enter
          // the target's right edge — the head always points cleanly left
          // into the target card.
          const offset = (idxInPair - (group.length - 1) / 2) * 22;
          const sxp = src.x + src.w, syp = src.y + Math.min(34, src.h / 2);
          const txp = tgt.x + tgt.w, typ = tgt.y + Math.min(34, tgt.h / 2);
          const bulge = 56 + Math.min(150, Math.abs(typ - syp) * 0.14) + Math.abs(offset);
          // Straight run-in/run-out segments at both ends: the arrow marker
          // orients to the path tangent at its endpoint, so it must sit on a
          // visibly straight piece of line or the head looks detached.
          const RUN = 10;
          pathD = 'M ' + sxp + ' ' + syp +
                  ' L ' + (sxp + RUN) + ' ' + syp +
                  ' C ' + (sxp + bulge) + ' ' + syp +
                  ', ' + (txp + bulge) + ' ' + typ +
                  ', ' + (txp + RUN) + ' ' + typ +
                  ' L ' + txp + ' ' + typ;
          labelX = Math.max(sxp, txp) + bulge * 0.78;
          labelY = (syp + typ) / 2;
          sx = sxp; sy = syp; tx = txp; ty = typ;
        } else {
          const aSrc = boxAnchor(src, tgtCx, tgtCy);
          const aTgt = boxAnchor(tgt, srcCx, srcCy);
          // Offset multiple relations between the same pair so labels don't overlap
          const offset = (idxInPair - (group.length - 1) / 2) * 24;
          // Horizontal-ease cubic that lifts above the cards for multi-column
          // spans, so the edge arcs over intermediate cards (Change 2).
          const curve = flowCurve(aSrc.x, aSrc.y, aTgt.x, aTgt.y, offset);
          pathD = curve.d;
          labelX = curve.mx;
          labelY = curve.my;
          sx = aSrc.x; sy = aSrc.y; tx = aTgt.x; ty = aTgt.y;
        }
        drawnEdges.push({ edge: e, path: pathD, labelX: labelX, labelY: labelY,
                          sx: sx, sy: sy, tx: tx, ty: ty });
      });
    });

    const defs = svg.append('defs');

    // Soft drop shadow for the type cards.
    const shadow = defs.append('filter')
      .attr('id', 'sd-card-shadow')
      .attr('x', '-30%').attr('y', '-30%')
      .attr('width', '160%').attr('height', '180%');
    shadow.append('feDropShadow')
      .attr('dx', 0).attr('dy', 2)
      .attr('stdDeviation', 5)
      .attr('flood-color', P.shadow)
      .attr('flood-opacity', P.shadowOpacity);

    // Per-edge gradient: flows from the source type's accent to the target's.
    function gradId(e){ return 'sd-grad-' + String(e.id).replace(/[^a-zA-Z0-9_-]/g, ''); }
    drawnEdges.forEach(function(de) {
      const grad = defs.append('linearGradient')
        .attr('id', gradId(de.edge))
        .attr('gradientUnits', 'userSpaceOnUse')
        .attr('x1', de.sx).attr('y1', de.sy)
        .attr('x2', de.tx).attr('y2', de.ty);
      grad.append('stop').attr('offset', '0%')
        .attr('stop-color', accentForName(de.edge.source_type));
      grad.append('stop').attr('offset', '100%')
        .attr('stop-color', accentForName(de.edge.target_type));
    });

    // Arrow markers, tinted per target type so the head matches the flow.
    const markerTypes = {};
    drawnEdges.forEach(function(de) { markerTypes[accentTypeByName[de.edge.target_type] || de.edge.target_type] = true; });
    function arrowId(tname){ return 'sd-arrow-' + String(tname).replace(/[^a-zA-Z0-9_-]/g, ''); }
    Object.keys(markerTypes).forEach(function(tname) {
      // Tip at viewBox x=10 with refX=10: the tip sits exactly on the path
      // endpoint with the head body covering the line behind it — no empty
      // marker margin past the tip (which left the path's round linecap
      // poking out as a detached nub).
      defs.append('marker')
        .attr('id', arrowId(tname))
        .attr('viewBox', '0 -5 10 10')
        .attr('refX', 10).attr('refY', 0)
        .attr('markerWidth', 7).attr('markerHeight', 7)
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M0,-4L10,0L0,4')
        .attr('fill', accentFor(tname));
    });

    // Draw edge paths first so labels render on top
    const edgeGroup = root.append('g').attr('class', 'sd-edges');
    drawnEdges.forEach(function(de) {
      edgeGroup.append('path')
        .datum(de.edge)
        .attr('class', 'sd-edge-path')
        .attr('data-edge-id', de.edge.id)
        .attr('data-source-type', de.edge.source_type)
        .attr('data-target-type', de.edge.target_type)
        .attr('d', de.path)
        .attr('stroke', 'url(#' + gradId(de.edge) + ')')
        .attr('marker-end', 'url(#' + arrowId(accentTypeByName[de.edge.target_type] || de.edge.target_type) + ')');
    });

    // Edge labels
    const labelGroup = root.append('g').attr('class', 'sd-edge-labels');
    drawnEdges.forEach(function(de) {
      const rel = primaryRelation(de.edge);
      const cnt = de.edge.edge_count ? '×' + de.edge.edge_count : '';
      const labelText = rel + (cnt ? '  ' + cnt : '');
      // Background pill so the label reads cleanly over crossing arrows
      const tmp = labelGroup.append('text')
        .attr('class', 'sd-edge-label')
        .text(labelText)
        .style('visibility', 'hidden');
      const bbox = tmp.node().getBBox();
      tmp.remove();
      const pad = 6;
      labelGroup.append('rect')
        .attr('class', 'sd-edge-label-bg')
        .attr('data-edge-id', de.edge.id)
        .attr('x', de.labelX - bbox.width / 2 - pad)
        .attr('y', de.labelY - bbox.height / 2 - 1)
        .attr('width', bbox.width + pad * 2)
        .attr('height', bbox.height + 2)
        .attr('rx', 4);
      labelGroup.append('text')
        .attr('class', 'sd-edge-label')
        .attr('data-edge-id', de.edge.id)
        .attr('x', de.labelX)
        .attr('y', de.labelY)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'central')
        .text(labelText);
    });

    // ── Node-type boxes ────────────────────────────────────────────
    const nodeGroup = root.append('g').attr('class', 'sd-nodes');
    Object.keys(positions).forEach(function(name) {
      const p = positions[name];
      const t = p.type;
      const accent = accentForName(name);
      // Record the box rect for the spotlight overlay (fallback anchor).
      window._boxRectByType[name] = { x: p.x, y: p.y, w: p.w, h: p.h };

      const g = nodeGroup.append('g')
        .attr('class', 'sd-node-box')
        .attr('data-node-type', name)
        .attr('data-node-id', t.id || '')
        .attr('transform', 'translate(' + p.x + ',' + p.y + ')');

      // Card body — soft themed card with a gentle drop shadow.
      g.append('rect')
        .attr('width', p.w)
        .attr('height', p.h)
        .attr('rx', 14)
        .attr('fill', P.card)
        .attr('stroke', P.cardBorder)
        .attr('stroke-width', 1)
        .attr('filter', 'url(#sd-card-shadow)');

      if (t.instance_count == null) {
        // Single-instance card: the instance name is the headline.
        g.append('circle').attr('cx', 21).attr('cy', 25).attr('r', 5).attr('fill', accent);
        g.append('text').attr('x', 34).attr('y', 30)
          .attr('font-size', 14.5).attr('font-weight', 700).attr('fill', P.title)
          .text(truncate(name, 15));
        return;
      }

      // Type card: the column header above already names the category, so the
      // type label is small + muted — the instance mini-cards are the focus.
      g.append('circle').attr('cx', 18).attr('cy', 22).attr('r', 4).attr('fill', accent);
      g.append('text').attr('x', 30).attr('y', 25.5)
        .attr('font-size', 10.5).attr('font-weight', 700).attr('letter-spacing', '0.04em')
        .attr('fill', P.muted)
        .text(truncate(String(name).toUpperCase(), 18));

      // Count pill (top-right).
      const cstr = String(t.instance_count);
      const pw = 16 + cstr.length * 8;
      g.append('rect')
        .attr('x', p.w - 14 - pw).attr('y', 13)
        .attr('width', pw).attr('height', 19).attr('rx', 9.5)
        .attr('fill', P.pill);
      g.append('text')
        .attr('x', p.w - 14 - pw / 2).attr('y', 26.5)
        .attr('text-anchor', 'middle')
        .attr('font-size', 11).attr('font-weight', 600).attr('fill', P.pillText)
        .text(cstr);

      // "Modified by operations" badge (amber dot) — see the Transformations layer.
      if (modifiedTypes.has(name)) {
        const dot = g.append('circle')
          .attr('cx', p.w - 14 - pw - 11).attr('cy', 22.5).attr('r', 3.5)
          .attr('fill', '#F2994A');
        dot.append('title').text('Modified by transformation operations');
      }

      // Instance mini-cards — capped at MAX_CARDS unless this type is expanded.
      const insts = instancesForType(name);
      const expanded = isTypeExpanded(name);
      const limit = expanded ? insts.length : LAYOUT.MAX_CARDS;
      insts.slice(0, limit).forEach(function(inst, i) {
        const cy = LAYOUT.HEADER_H + i * LAYOUT.CARD_H;
        window._miniCardRects[inst.id] =
          { x: p.x + 13, y: p.y + cy, w: p.w - 26, h: LAYOUT.CARD_H - 6 };
        const card = g.append('g')
          .attr('class', 'sd-mini-card')
          .attr('data-iid', inst.id);
        card.append('rect')
          .attr('x', 13).attr('y', cy)
          .attr('width', p.w - 26).attr('height', LAYOUT.CARD_H - 6)
          .attr('rx', 7)
          .attr('fill', P.mini)
          .attr('stroke', P.miniBorder)
          .attr('stroke-width', 1);
        card.append('text')
          .attr('x', 25).attr('y', cy + (LAYOUT.CARD_H - 6) / 2 + 4)
          .attr('font-size', 11.5)
          .attr('fill', P.miniText)
          .text(truncate(inst.name || inst.id, 24));
      });
      const total = t.instance_count || insts.length;
      if (total > LAYOUT.MAX_CARDS) {
        const shownCount = Math.min(limit, insts.length);
        const my = LAYOUT.HEADER_H + shownCount * LAYOUT.CARD_H;
        const toggle = g.append('g')
          .attr('class', 'sd-expand-toggle')
          .attr('data-type', name);
        // Invisible hit area so the whole toggle row is clickable.
        toggle.append('rect')
          .attr('x', 13).attr('y', my).attr('width', p.w - 26).attr('height', 16)
          .attr('fill', 'transparent');
        toggle.append('text')
          .attr('x', 21).attr('y', my + 12)
          .attr('font-size', 11).attr('font-weight', 600).attr('fill', '#1F9E6E')
          .text(expanded ? 'Show less' : ('+ ' + (total - LAYOUT.MAX_CARDS) + ' more'));
      }
    });

    // ── Transformation operation rail (impact-layer) ────────────────
    window._opRects = {};
    const ops = window._schemaOperations || [];
    if (window._schemaShowOps && ops.length) {
      const xs = Object.keys(positions).map(function(k) { return positions[k].x; });
      const minX = xs.length ? Math.min.apply(null, xs) : LAYOUT.PAD_LEFT;
      const maxRight = (xs.length ? Math.max.apply(null, xs) : 800) + LAYOUT.BOX_W;
      const railY = -120, chipW = 150, chipH = 30;
      const availW = Math.max(maxRight - minX, ops.length * (chipW + 16));
      const opGroup = root.append('g').attr('class', 'sd-op-rail');
      ops.forEach(function(op, i) {
        const x = minX + i * (availW / ops.length);
        window._opRects[op.id] = { x: x, y: railY, w: chipW, h: chipH };
        const kindColor = OP_KIND_COLORS[op.op_kind] || '#8792A2';
        const g = opGroup.append('g').attr('class', 'sd-op-chip').attr('data-op', op.id);
        g.append('rect').attr('x', x).attr('y', railY).attr('width', chipW).attr('height', chipH)
          .attr('rx', 8).attr('fill', P.card).attr('stroke', kindColor).attr('stroke-width', 1.5)
          .attr('filter', 'url(#sd-card-shadow)');
        g.append('circle').attr('cx', x + 13).attr('cy', railY + chipH / 2).attr('r', 4).attr('fill', kindColor);
        g.append('text').attr('x', x + 24).attr('y', railY + chipH / 2 + 4)
          .attr('font-size', 11.5).attr('font-weight', 600).attr('fill', P.title)
          .text(truncate(op.name, 16));
      });
    }

    // ── Selection wiring ───────────────────────────────────────────
    let selected = null;
    const selectionEl = document.getElementById('schema-selection');
    function applySelection() {
      svg.selectAll('.sd-node-box').classed('is-dim', false);
      svg.selectAll('.sd-edge-path').classed('is-hot', false).classed('is-dim', false);
      svg.selectAll('.sd-edge-label-bg').classed('is-hot', false).classed('is-dim', false);
      svg.selectAll('.sd-edge-label').classed('is-dim', false);
      if (selectionEl) selectionEl.textContent = '';
      if (!selected) return;

      if (selected.kind === 'node') {
        const name = selected.name;
        svg.selectAll('.sd-node-box').classed('is-dim', function() { return this.dataset.nodeType !== name; });
        svg.selectAll('.sd-edge-path').classed('is-hot', function() {
          return this.dataset.sourceType === name || this.dataset.targetType === name;
        }).classed('is-dim', function() {
          return this.dataset.sourceType !== name && this.dataset.targetType !== name;
        });
        svg.selectAll('.sd-edge-label-bg').each(function() {
          const id = this.getAttribute('data-edge-id');
          const path = svg.select('.sd-edge-path[data-edge-id="' + id + '"]').node();
          if (path) {
            this.classList.toggle('is-hot', path.classList.contains('is-hot'));
            this.classList.toggle('is-dim', path.classList.contains('is-dim'));
          }
        });
        svg.selectAll('.sd-edge-label').each(function() {
          const id = this.getAttribute('data-edge-id');
          const path = svg.select('.sd-edge-path[data-edge-id="' + id + '"]').node();
          if (path) this.classList.toggle('is-dim', path.classList.contains('is-dim'));
        });
        if (selectionEl) selectionEl.textContent = 'Selected node type: ' + name + ' — click again to clear';
      } else if (selected.kind === 'edge') {
        const id = selected.id;
        const edgePath = svg.select('.sd-edge-path[data-edge-id="' + id + '"]');
        const sourceType = edgePath.attr('data-source-type');
        const targetType = edgePath.attr('data-target-type');
        svg.selectAll('.sd-node-box').classed('is-dim', function() {
          return this.dataset.nodeType !== sourceType && this.dataset.nodeType !== targetType;
        });
        svg.selectAll('.sd-edge-path').classed('is-dim', function() { return this.getAttribute('data-edge-id') !== id; });
        edgePath.classed('is-dim', false).classed('is-hot', true);
        svg.select('.sd-edge-label-bg[data-edge-id="' + id + '"]').classed('is-hot', true);
        if (selectionEl) selectionEl.textContent = 'Selected edge: ' + sourceType + ' → ' + targetType + ' — click again to clear';
      }
    }
    svg.selectAll('.sd-node-box').on('click', function() {
      const name = this.dataset.nodeType;
      // A type-box click is a type-level selection; clear any instance spotlight.
      if (window._schemaSpotlight) { window._schemaSpotlight = null; clearSpotlight(); }
      if (selected && selected.kind === 'node' && selected.name === name) {
        selected = null;
        if (window._hideSchemaInspector) window._hideSchemaInspector();
        updateSchemaStatus(null);
      } else {
        selected = { kind: 'node', name: name };
        // Open the type inspector side panel for the clicked type.
        if (window._showSchemaInspector) window._showSchemaInspector(name);
      }
      applySelection();
    });
    // Instance mini-cards: click focuses the diagram on that instance and opens
    // its inspector (stop propagation so the parent type box doesn't also fire).
    svg.selectAll('.sd-mini-card').on('click', function(event) {
      if (event) event.stopPropagation();
      const iid = this.dataset.iid;
      if (!iid) return;
      if (window._showSchemaInstanceInspector) window._showSchemaInstanceInspector(iid);
      if (window._focusSchemaInstance) window._focusSchemaInstance(iid);
    });
    // Click an operation chip → spotlight the types it transforms.
    svg.selectAll('.sd-op-chip').on('click', function(event) {
      if (event) event.stopPropagation();
      const opId = this.dataset.op;
      if (opId) applyOpSpotlight(opId);
    });
    // Expand / collapse the instance list inside a type box.
    svg.selectAll('.sd-expand-toggle').on('click', function(event) {
      if (event) event.stopPropagation();
      const tp = this.dataset.type;
      window._schemaExpanded = window._schemaExpanded || {};
      window._schemaExpanded[tp] = !window._schemaExpanded[tp];
      window._renderSchemaGraph(true);  // re-layout, preserving the current pan/zoom
    });
    svg.selectAll('.sd-edge-path, .sd-edge-label-bg, .sd-edge-label').on('click', function() {
      const id = this.getAttribute('data-edge-id');
      if (!id) return;
      if (selected && selected.kind === 'edge' && selected.id === id) { selected = null; }
      else { selected = { kind: 'edge', id: id }; }
      applySelection();
    });

    // ── Pan / zoom on the schema canvas ─────────────────────────────
    schemaSvgSel = svg;
    schemaRootSel = root;
    schemaZoomBehavior = d3.zoom()
      .scaleExtent([0.2, 3])
      .on('start', function() { svg.classed('is-panning', true); })
      .on('zoom', function(event) { root.attr('transform', event.transform); })
      .on('end', function() { svg.classed('is-panning', false); });
    svg.call(schemaZoomBehavior).on('dblclick.zoom', null);
    if (savedTransform) {
      // Preserve the current pan/zoom across re-renders (e.g. expand/collapse).
      svg.call(schemaZoomBehavior.transform, savedTransform);
    } else {
      // Initial framing: fit the whole diagram into the viewport.
      fitSchema(null, false);
    }
  }


  // ── Empty / non-empty state plumbing ─────────────────────────────
  function setSchemaEmpty(message) {
    emptyEl.style.display = 'flex';
    emptyEl.querySelector('div').textContent = message || 'No schema configured for this dataset.';
    if (svgEl) svgEl.style.display = 'none';
    const hdr = document.getElementById('schema-header');
    const dia = document.getElementById('schema-diagram-section');
    if (hdr) hdr.style.display = 'none';
    if (dia) dia.style.display = 'none';
  }
  function hideSchemaEmpty() {
    emptyEl.style.display = 'none';
    if (svgEl) svgEl.style.display = '';
    const hdr = document.getElementById('schema-header');
    const dia = document.getElementById('schema-diagram-section');
    if (hdr) hdr.style.display = '';
    if (dia) dia.style.display = '';
  }

  // ── Pan / zoom ───────────────────────────────────────────────────
  // d3.zoom on the SVG, applied to the root <g>, so the schema can be panned
  // and zoomed like the graph view. Fit accounts for the side panel so the
  // spotlighted connections land in the visible area, not under the panel.
  let schemaSvgSel = null, schemaRootSel = null, schemaZoomBehavior = null;

  function panelOffset() {
    const panel = document.getElementById('schema-side-panel');
    return panel && panel.style.display !== 'none' ? (panel.offsetWidth || 344) : 0;
  }

  function fitSchema(bbox, animate) {
    if (!schemaSvgSel || !schemaRootSel || !schemaZoomBehavior) return;
    const node = schemaSvgSel.node();
    const W = node.clientWidth || node.getBoundingClientRect().width || 1200;
    const H = node.clientHeight || node.getBoundingClientRect().height || 700;
    const avail = Math.max(240, W - panelOffset());
    if (!bbox) {
      try { bbox = schemaRootSel.node().getBBox(); } catch (e) { return; }
    }
    const pad = 44;
    const bw = bbox.width || 1, bh = bbox.height || 1;
    const scale = Math.max(0.2, Math.min((avail - pad * 2) / bw, (H - pad * 2) / bh, 1.5));
    const tx = pad + (avail - pad * 2 - bw * scale) / 2 - bbox.x * scale;
    const ty = pad + (H - pad * 2 - bh * scale) / 2 - bbox.y * scale;
    const t = d3.zoomIdentity.translate(tx, ty).scale(scale);
    const target = animate ? schemaSvgSel.transition().duration(420) : schemaSvgSel;
    target.call(schemaZoomBehavior.transform, t);
  }

  // ── Instance spotlight ───────────────────────────────────────────
  // Clicking an instance keeps the full diagram and spotlights that instance's
  // connections: its card + linked cards pop, everything else dims, and glowing
  // connector edges are drawn from it to its neighbours. window._schemaSpotlight
  // holds the focused instance id (or null).
  window._schemaSpotlight = null;

  // ── Transformation impact-layer ──────────────────────────────────
  // Operations (cognify, memify, improve, …) shown as a rail of chips with
  // color-coded edges to the schema types they affect. Off by default.
  window._schemaShowOps = window._schemaShowOps || false;
  window._opRects = window._opRects || {};
  const EFFECT_COLORS = {
    produces: '#1F9E6E', enriches: '#2F80ED', modifies: '#F2994A', removes: '#EB5757',
  };
  const EFFECT_DASH = { modifies: '6 4', removes: '6 4' };
  const OP_KIND_COLORS = {
    pipeline: '#2F80ED', self_improve: '#F2994A', lifecycle: '#EB5757',
  };

  function modifiedTypeNames() {
    const names = new Set();
    (window._schemaOperationLinks || []).forEach(function(l) {
      if (l.effect === 'modifies') names.add(String(l.target).replace(/^type:/, ''));
    });
    return names;
  }

  // Anchor two rects on their facing sides (left→right flow).
  function anchorBetween(s, t) {
    const sCx = s.x + s.w / 2, tCx = t.x + t.w / 2;
    if (tCx >= sCx) {
      return { sx: s.x + s.w, sy: s.y + s.h / 2, tx: t.x, ty: t.y + t.h / 2 };
    }
    return { sx: s.x, sy: s.y + s.h / 2, tx: t.x + t.w, ty: t.y + t.h / 2 };
  }

  // Horizontal-ease cubic; for multi-column spans it drops the curve below the
  // cards so it arcs under the intermediate cards instead of cutting through them.
  function flowCurve(sx, sy, tx, ty, offset) {
    offset = offset || 0;
    const dx = tx - sx;
    const dy = Math.abs(ty - sy);
    const span = Math.abs(dx) / (LAYOUT.BOX_W + LAYOUT.COL_GAP);
    const handle = Math.max(60, Math.abs(dx) * 0.5);
    const dir = dx >= 0 ? 1 : -1;
    let c1y = sy + offset, c2y = ty + offset;
    // Drop the control points below the card band so multi-column edges arc
    // clearly under the intermediate cards rather than grazing them. Only
    // when the endpoints sit at similar heights: with a large vertical span
    // (tall columns) the drop overshot below BOTH cards, hooking the curve
    // back up so the arrowhead pointed the wrong way ("inverted" arrows).
    // Vertically distant cards get the plain horizontal-ease instead, which
    // already clears the band diagonally.
    if (span >= 1.25 && dy < 140) {
      const drop = Math.max(sy, ty) + (64 + span * 40);
      c1y = drop + offset;
      c2y = drop + offset;
    } else if (span >= 0.4 && dy < 90) {
      // Adjacent columns: a gentle downward bow so the edge clears the card edges.
      const drop = Math.max(sy, ty) + 34;
      c1y = drop + offset;
      c2y = drop + offset;
    }
    const c1x = sx + dir * handle, c2x = tx - dir * handle;
    // Straight run-in/run-out segments: the arrow marker orients to the path
    // tangent at the endpoint, so it must sit on a visibly straight piece of
    // line or the head appears detached/rotated off the curve.
    const RUN = 10;
    const sx2 = sx + dir * RUN, tx2 = tx - dir * RUN;
    return {
      d: 'M ' + sx + ' ' + sy +
         ' L ' + sx2 + ' ' + sy +
         ' C ' + c1x + ' ' + c1y + ', ' + c2x + ' ' + c2y + ', ' + tx2 + ' ' + ty +
         ' L ' + tx + ' ' + ty,
      mx: (c1x + c2x) / 2,
      my: (c1y + c2y) / 2,
    };
  }

  function updateSchemaStatus(focusId) {
    const selEl = document.getElementById('schema-selection');
    if (!selEl) return;
    selEl.textContent = '';
    if (focusId) {
      const name = ((window._schemaInstanceIndex || {})[focusId] || {}).name || focusId;
      const label = document.createElement('span');
      label.textContent = 'Focused on ' + name + '   ';
      label.style.cssText = 'color:#1A1F36;font-weight:600;';
      const clear = document.createElement('span');
      clear.textContent = 'Clear focus';
      clear.style.cssText = 'color:#1F9E6E;cursor:pointer;font-weight:600;';
      clear.addEventListener('click', window._clearSchemaLens);
      selEl.appendChild(label);
      selEl.appendChild(clear);
    } else {
      selEl.style.color = '#8792A2';
      selEl.textContent = 'Click any instance card to trace its connections.';
    }
  }

  function clearSpotlight() {
    const svg = d3.select('#schema-svg');
    svg.select('g.sd-spotlight').remove();
    svg.selectAll('.sd-node-box, .sd-mini-card')
      .classed('is-dim', false).classed('is-spotlit', false).classed('is-linked', false);
    svg.selectAll('.sd-edge-path, .sd-edge-label-bg, .sd-edge-label').classed('is-dim', false);
    svg.selectAll('.sd-op-chip').classed('is-dim', false).classed('is-spotlit', false);
  }

  // Spotlight a transformation operation: highlight the types it affects and
  // draw color-coded impact edges (produces/enriches/modifies/removes).
  function applyOpSpotlight(opId) {
    const svg = d3.select('#schema-svg');
    clearSpotlight();
    const links = (window._schemaOperationLinks || []).filter(function(l) { return l.source === opId; });
    if (!links.length) { updateSchemaStatus(null); return; }
    const targetNames = new Set(links.map(function(l) { return String(l.target).replace(/^type:/, ''); }));

    svg.selectAll('.sd-node-box').classed('is-dim', function() { return !targetNames.has(this.dataset.nodeType); });
    svg.selectAll('.sd-mini-card').classed('is-dim', true);
    svg.selectAll('.sd-edge-path, .sd-edge-label-bg, .sd-edge-label').classed('is-dim', true);
    svg.selectAll('.sd-op-chip').classed('is-dim', function() { return this.dataset.op !== opId; })
      .classed('is-spotlit', function() { return this.dataset.op === opId; });

    const opRect = (window._opRects || {})[opId];
    const boxRects = window._boxRectByType || {};
    const fitRects = [];
    if (opRect) {
      fitRects.push(opRect);
      const overlay = (schemaRootSel || svg).append('g').attr('class', 'sd-spotlight');
      links.forEach(function(l) {
        const tname = String(l.target).replace(/^type:/, '');
        const r = boxRects[tname];
        if (!r) return;
        fitRects.push(r);
        const color = EFFECT_COLORS[l.effect] || '#8792A2';
        // Vertical-ease cubic: operation chip (top) feeds down into the type card.
        const sx = opRect.x + opRect.w / 2, sy = opRect.y + opRect.h;
        const tx = r.x + r.w / 2, ty = r.y;
        const midy = (sy + ty) / 2;
        const d = 'M ' + sx + ' ' + sy + ' C ' + sx + ' ' + midy + ', ' + tx + ' ' + midy + ', ' + tx + ' ' + ty;
        overlay.append('path').attr('class', 'sd-spotlight-edge-halo').attr('fill', 'none').attr('stroke', color).attr('d', d);
        const path = overlay.append('path').attr('class', 'sd-op-edge').attr('fill', 'none').attr('stroke', color).attr('d', d);
        if (EFFECT_DASH[l.effect]) path.attr('stroke-dasharray', EFFECT_DASH[l.effect]);
      });
    }
    // Frame the operation + its targets in view.
    if (fitRects.length) {
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      fitRects.forEach(function(r) {
        x0 = Math.min(x0, r.x); y0 = Math.min(y0, r.y);
        x1 = Math.max(x1, r.x + r.w); y1 = Math.max(y1, r.y + r.h);
      });
      fitSchema({ x: x0 - 40, y: y0 - 40, width: (x1 - x0) + 80, height: (y1 - y0) + 80 }, true);
    }
    const op = (window._schemaOperations || []).find(function(o) { return o.id === opId; });
    updateOpStatus(op);
  }

  function updateOpStatus(op) {
    const selEl = document.getElementById('schema-selection');
    if (!selEl) return;
    selEl.textContent = '';
    if (!op) { updateSchemaStatus(null); return; }
    const label = document.createElement('span');
    label.textContent = (op.summary || op.name) + '   ';
    label.style.cssText = 'color:#1A1F36;font-weight:600;';
    const clear = document.createElement('span');
    clear.textContent = 'Clear';
    clear.style.cssText = 'color:#1F9E6E;cursor:pointer;font-weight:600;';
    clear.addEventListener('click', window._clearSchemaLens);
    selEl.appendChild(label);
    selEl.appendChild(clear);
  }

  function applySpotlight(focusId) {
    const index = window._schemaInstanceIndex || {};
    const focus = index[focusId];
    const svg = d3.select('#schema-svg');
    clearSpotlight();
    if (!focus) { updateSchemaStatus(null); return; }

    const links = [];
    (focus.out || []).forEach(function(e) { links.push({ id: e.id, relation: e.relation, dir: 'out' }); });
    (focus.in || []).forEach(function(e) { links.push({ id: e.id, relation: e.relation, dir: 'in' }); });
    const linkedIds = {}; linkedIds[focusId] = true;
    links.forEach(function(l) { linkedIds[l.id] = true; });
    const activeTypes = {};
    Object.keys(linkedIds).forEach(function(id) { const n = index[id]; if (n) activeTypes[n.type] = true; });

    svg.selectAll('.sd-mini-card')
      .classed('is-dim', function() { return !linkedIds[this.dataset.iid]; })
      .classed('is-spotlit', function() { return this.dataset.iid === focusId; })
      .classed('is-linked', function() { return this.dataset.iid !== focusId && !!linkedIds[this.dataset.iid]; });
    svg.selectAll('.sd-node-box').classed('is-dim', function() { return !activeTypes[this.dataset.nodeType]; });
    svg.selectAll('.sd-edge-path, .sd-edge-label-bg, .sd-edge-label').classed('is-dim', true);

    const rects = window._miniCardRects || {};
    const boxRects = window._boxRectByType || {};
    const focusRect = rects[focusId] || boxRects[focus.type];
    if (focusRect) {
      // Append inside the zoomable root so the overlay pans/zooms with the cards.
      const overlay = (schemaRootSel || svg).append('g').attr('class', 'sd-spotlight');
      const accent = accentFor(focus.type);
      const marker = overlay.append('defs').append('marker')
        .attr('id', 'sd-spot-arrow').attr('viewBox', '0 -5 10 10')
        .attr('refX', 10).attr('refY', 0).attr('markerWidth', 7).attr('markerHeight', 7)
        .attr('orient', 'auto');
      marker.append('path').attr('d', 'M0,-4L10,0L0,4').attr('fill', accent);
      let sameCardIdx = 0;
      let crossCardIdx = 0;
      // Greedy vertical declutter: connector labels for links to nearby
      // instances land on nearly the same midpoint and overprint into
      // garbage. Push each new label below any already-placed neighbor.
      const placedLabelYs = [];
      function declutterY(y) {
        let moved = true;
        while (moved) {
          moved = false;
          for (let i = 0; i < placedLabelYs.length; i++) {
            if (Math.abs(y - placedLabelYs[i]) < 13) { y = placedLabelYs[i] + 13; moved = true; }
          }
        }
        placedLabelYs.push(y);
        return y;
      }
      links.forEach(function(l) {
        const n = index[l.id];
        if (!n) return;
        const tgtRect = rects[l.id] || boxRects[n.type];
        if (!tgtRect) return;
        let s = focusRect, t = tgtRect;
        if (l.dir === 'in') { s = tgtRect; t = focusRect; }
        let d, lx, ly, anchor = 'middle';
        if (n.type === focus.type) {
          // Connection between two instances of the SAME type card. The
          // facing-side anchors degenerate here: the connector cut straight
          // across the card with its label overprinting the others. Route a
          // side loop out of the card's LEFT edge instead (the right side
          // already hosts the type-pair loops), nesting loops and labels by
          // index so multiple relations stay individually readable.
          sameCardIdx += 1;
          const sx = s.x, sy = s.y + s.h / 2;
          const tx = t.x, ty = t.y + t.h / 2;
          const bulge = 46 + Math.min(140, Math.abs(ty - sy) * 0.18) + sameCardIdx * 16;
          const RUN = 8;
          d = 'M ' + sx + ' ' + sy +
              ' L ' + (sx - RUN) + ' ' + sy +
              ' C ' + (sx - bulge) + ' ' + sy +
              ', ' + (tx - bulge) + ' ' + ty +
              ', ' + (tx - RUN) + ' ' + ty +
              ' L ' + tx + ' ' + ty;
          // Right-align the label just left of the loop apex so the text
          // extends away from the nested loop strokes.
          lx = Math.min(sx, tx) - bulge - 4;
          ly = declutterY((sy + ty) / 2);
          anchor = 'end';
        } else {
          // Cross-card connector: spread vertically so labels between the
          // same pair of cards don't land on one coincident midpoint.
          crossCardIdx += 1;
          const spread = ((crossCardIdx % 5) - 2) * 16;
          const a = anchorBetween(s, t);
          const curve = flowCurve(a.sx, a.sy, a.tx, a.ty, spread);
          d = curve.d;
          lx = curve.mx;
          ly = declutterY(curve.my);
        }
        overlay.append('path').attr('class', 'sd-spotlight-edge-halo').attr('fill', 'none').attr('stroke', accent).attr('d', d);
        overlay.append('path').attr('class', 'sd-spotlight-edge').attr('fill', 'none').attr('stroke', accent)
          .attr('marker-end', 'url(#sd-spot-arrow)').attr('d', d);
        overlay.append('text').attr('class', 'sd-spotlight-label')
          .attr('x', lx).attr('y', ly).attr('text-anchor', anchor)
          .attr('fill', accent).text(l.relation);
      });
    }

    // Frame the spotlighted connections in the visible area (left of the panel)
    // so they aren't hidden behind it.
    const fitRects = [];
    if (focusRect) fitRects.push(focusRect);
    links.forEach(function(l) {
      const n = index[l.id];
      const r = rects[l.id] || (n && boxRects[n.type]);
      if (r) fitRects.push(r);
    });
    if (fitRects.length) {
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      fitRects.forEach(function(r) {
        x0 = Math.min(x0, r.x); y0 = Math.min(y0, r.y);
        x1 = Math.max(x1, r.x + r.w); y1 = Math.max(y1, r.y + r.h);
      });
      // Extra bottom padding for connector edges that arc below the cards.
      fitSchema({ x: x0 - 40, y: y0 - 30, width: (x1 - x0) + 80, height: (y1 - y0) + 210 }, true);
    }
    updateSchemaStatus(focusId);
  }

  window._focusSchemaInstance = function(instanceId) {
    if (!(window._schemaInstanceIndex || {})[instanceId]) return;
    window._schemaSpotlight = instanceId;
    applySpotlight(instanceId);
  };
  window._clearSchemaLens = function() {
    window._schemaSpotlight = null;
    clearSpotlight();
    updateSchemaStatus(null);
    fitSchema(null, true);  // zoom back out to the whole diagram
  };

  window._renderSchemaGraph = function(preserveView) {
    // Preserve the current pan/zoom when re-rendering in place (expand/collapse).
    let saved = null;
    if (preserveView && schemaSvgSel) {
      try { saved = d3.zoomTransform(schemaSvgSel.node()); } catch (e) { saved = null; }
    }
    const model = buildSchemaModel();
    if (!model.nodeTypes.length) {
      setSchemaEmpty('No schema configured for this dataset.');
      updateSchemaStatus(null);
      return;
    }
    hideSchemaEmpty();
    renderSchemaDiagram(model, saved);
    // Re-apply an active spotlight after a fresh render (e.g. theme toggle).
    if (window._schemaSpotlight && (window._schemaInstanceIndex || {})[window._schemaSpotlight]) {
      applySpotlight(window._schemaSpotlight);
    } else {
      window._schemaSpotlight = null;
      updateSchemaStatus(null);
    }
  };

  // Zoom control buttons (wired once; act on the active zoom behavior).
  (function () {
    function bind(id, fn) {
      const el = document.getElementById(id);
      if (el) el.addEventListener('click', fn);
    }
    bind('schema-zoom-in', function () {
      if (schemaSvgSel && schemaZoomBehavior) schemaSvgSel.transition().duration(180).call(schemaZoomBehavior.scaleBy, 1.4);
    });
    bind('schema-zoom-out', function () {
      if (schemaSvgSel && schemaZoomBehavior) schemaSvgSel.transition().duration(180).call(schemaZoomBehavior.scaleBy, 1 / 1.4);
    });
    bind('schema-zoom-fit', function () { fitSchema(null, true); });
    bind('schema-ops-toggle', function () {
      window._schemaShowOps = !window._schemaShowOps;
      const btn = document.getElementById('schema-ops-toggle');
      const legend = document.getElementById('schema-legend');
      if (btn) btn.classList.toggle('active', window._schemaShowOps);
      if (legend) legend.style.display = window._schemaShowOps ? 'block' : 'none';
      window._schemaSpotlight = null;
      window._renderSchemaGraph();  // re-render + reframe to include/exclude the rail
    });
  })();

  if (!schemaData && (!schemaGraphData || !schemaGraphData.nodes || !schemaGraphData.nodes.length)) {
    setSchemaEmpty('No schema configured for this dataset.');
  }
})();
