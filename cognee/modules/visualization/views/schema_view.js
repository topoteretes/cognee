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
  const STAGE_LABELS = ['Documents', 'Chunks', 'Entities', 'Types', 'Summaries', 'Context'];
  function stageLabel(rank){
    if (typeof rank !== 'number' || !isFinite(rank)) return '';
    const idx = Math.round(rank);
    return STAGE_LABELS[idx] || '';
  }

  // Per-type accent color — keeps the schema palette consistent with the
  // graph view (Entity = purple, Chunk = green, etc.)
  const ACCENT_BY_TYPE = {
    TextDocument: '#A550FF',
    DocumentChunk: '#0DFF00',
    TextSummary: '#6510F4',
    GlobalContextSummary: '#00C2FF',
    Entity: '#6510F4',
    EntityType: '#D5C2FF',
    DatabaseSchema: '#6510F4',
    SchemaTable: '#A550FF',
    SchemaRelationship: '#747470',
    TableType: '#A550FF',
    TableRow: '#A550FF',
    ColumnValue: '#747470',
  };
  function accentFor(name){ return ACCENT_BY_TYPE[name] || '#A550FF'; }

  // ── Build the schema model from preprocessor data ────────────────
  function buildSchemaModel() {
    if (!schemaGraphData || !Array.isArray(schemaGraphData.nodes) || !schemaGraphData.nodes.length) {
      return { nodeTypes: [], edgeTypes: [] };
    }
    const nodeTypes = [];
    const edgeTypes = [];
    schemaGraphData.nodes.forEach(function(n) {
      if (n.type === 'GraphNodeType') {
        nodeTypes.push({
          id: String(n.id),
          name: String(n.name || n.id),
          rank: Number.isFinite(Number(n.rank)) ? Number(n.rank) : 4,
          fields: Array.isArray(n.fields) ? n.fields : [],
          instance_count: n.instance_count != null ? n.instance_count : null,
          source_pipeline: n.source_pipeline || null,
          source_task: n.source_task || null,
          source_user: n.source_user || null,
        });
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
  function layoutBoxes(nodeTypes, totalWidth) {
    const PAD_LEFT = 40;
    const PAD_TOP = 56;       // room for stage header labels
    const BOX_W = 220;
    const BOX_H = 130;
    const COL_GAP = 80;
    const ROW_GAP = 26;
    // Bin by rank
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
      const col = byRank[r];
      const colX = PAD_LEFT + colIdx * (BOX_W + COL_GAP);
      col.forEach(function(t, rowIdx) {
        const y = PAD_TOP + rowIdx * (BOX_H + ROW_GAP);
        positions[t.name] = { x: colX, y: y, w: BOX_W, h: BOX_H, type: t, rank: r };
        if (y + BOX_H > maxY) maxY = y + BOX_H;
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
      box_h: BOX_H,
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
  function renderSchemaDiagram(model) {
    const svg = d3.select('#schema-svg');
    svg.selectAll('*').remove();
    const isLight = document.documentElement.classList.contains('light');

    if (!model.nodeTypes.length) {
      return;
    }

    const containerW = document.getElementById('schema-diagram-section').clientWidth || 1200;
    const layout = layoutBoxes(model.nodeTypes, containerW);
    svg.attr('width', layout.width).attr('height', layout.height);
    const root = svg.append('g');

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
        if (isSelf) {
          // Self-loop on the right side of the box
          const lx = src.x + src.w;
          const ly = src.y + 20;
          pathD = 'M ' + lx + ' ' + (src.y + 30) +
                  ' C ' + (lx + 60) + ' ' + (src.y + 10) +
                  ', ' + (lx + 60) + ' ' + (src.y + src.h - 10) +
                  ', ' + lx + ' ' + (src.y + src.h - 30);
          labelX = lx + 36;
          labelY = src.y + src.h / 2;
        } else {
          const aSrc = boxAnchor(src, tgtCx, tgtCy);
          const aTgt = boxAnchor(tgt, srcCx, srcCy);
          // Offset multiple relations between the same pair so labels don't overlap
          const offset = (idxInPair - (group.length - 1) / 2) * 22;
          const midX = (aSrc.x + aTgt.x) / 2;
          const midY = (aSrc.y + aTgt.y) / 2;
          const dx = aTgt.x - aSrc.x;
          const dy = aTgt.y - aSrc.y;
          const len = Math.sqrt(dx * dx + dy * dy) || 1;
          // Perpendicular for curvature
          const px = -dy / len;
          const py = dx / len;
          const curveStrength = 30 + Math.abs(offset);
          const cpx = midX + px * curveStrength + (px * offset);
          const cpy = midY + py * curveStrength + (py * offset);
          pathD = 'M ' + aSrc.x + ' ' + aSrc.y +
                  ' Q ' + cpx + ' ' + cpy + ' ' + aTgt.x + ' ' + aTgt.y;
          labelX = cpx;
          labelY = cpy;
        }
        drawnEdges.push({ edge: e, path: pathD, labelX: labelX, labelY: labelY });
      });
    });

    // Arrow marker (one shared definition)
    const defs = svg.append('defs');
    defs.append('marker')
      .attr('id', 'sd-arrow')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 8)
      .attr('refY', 0)
      .attr('markerWidth', 7)
      .attr('markerHeight', 7)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-4L8,0L0,4')
      .attr('fill', isLight ? '#666' : '#aaa');

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
        .attr('marker-end', 'url(#sd-arrow)');
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
      const accent = accentFor(name);
      const g = nodeGroup.append('g')
        .attr('class', 'sd-node-box')
        .attr('data-node-type', name)
        .attr('transform', 'translate(' + p.x + ',' + p.y + ')');

      // Card body
      g.append('rect')
        .attr('width', p.w)
        .attr('height', p.h)
        .attr('rx', 10)
        .attr('fill', isLight ? '#ffffff' : '#1a1a22')
        .attr('stroke', isLight ? 'rgba(30,30,30,0.18)' : 'rgba(244,244,244,0.16)')
        .attr('stroke-width', 1);
      // Accent stripe on the left
      g.append('rect')
        .attr('width', 4)
        .attr('height', p.h)
        .attr('rx', 1)
        .attr('fill', accent);

      // Header: type name + count
      g.append('text')
        .attr('x', 16)
        .attr('y', 22)
        .attr('font-size', 14)
        .attr('font-weight', 700)
        .attr('fill', isLight ? '#1a1a1a' : '#F4F4F4')
        .text(truncate(name, 22));
      if (t.instance_count != null) {
        const txt = t.instance_count + ' instance' + (t.instance_count === 1 ? '' : 's');
        g.append('text')
          .attr('x', p.w - 14)
          .attr('y', 22)
          .attr('text-anchor', 'end')
          .attr('font-size', 10.5)
          .attr('font-weight', 500)
          .attr('fill', isLight ? '#666' : '#A8A8B3')
          .text(txt);
      }

      // Pipeline / task pills
      let pillY = 46;
      if (t.source_pipeline) {
        const v = stripCoverage(t.source_pipeline);
        g.append('text')
          .attr('x', 16).attr('y', pillY)
          .attr('font-size', 9)
          .attr('font-weight', 600)
          .attr('letter-spacing', '0.05em')
          .attr('text-transform', 'uppercase')
          .attr('fill', isLight ? '#666' : '#A8A8B3')
          .text('PIPELINE');
        g.append('text')
          .attr('x', 76).attr('y', pillY)
          .attr('font-size', 10.5)
          .attr('font-family', "'SF Mono', ui-monospace, monospace")
          .attr('fill', isLight ? '#1a1a1a' : '#E0E0E0')
          .text(truncate(v, 22));
        pillY += 16;
      }
      if (t.source_task) {
        const v = stripCoverage(t.source_task);
        g.append('text')
          .attr('x', 16).attr('y', pillY)
          .attr('font-size', 9)
          .attr('font-weight', 600)
          .attr('letter-spacing', '0.05em')
          .attr('text-transform', 'uppercase')
          .attr('fill', isLight ? '#666' : '#A8A8B3')
          .text('TASK');
        g.append('text')
          .attr('x', 76).attr('y', pillY)
          .attr('font-size', 10.5)
          .attr('font-family', "'SF Mono', ui-monospace, monospace")
          .attr('fill', isLight ? '#1a1a1a' : '#E0E0E0')
          .text(truncate(v, 22));
        pillY += 16;
      }

      // Top fields (skip count/source_* which are surfaced above)
      const bodyFields = (t.fields || []).filter(function(f) {
        return ['count', 'source_pipeline', 'source_task', 'source_user'].indexOf(f.name) === -1;
      }).slice(0, 3);
      if (bodyFields.length) {
        const fieldsY = Math.max(pillY + 8, 92);
        bodyFields.forEach(function(f, i) {
          g.append('text')
            .attr('x', 16).attr('y', fieldsY + i * 12)
            .attr('font-size', 10)
            .attr('fill', isLight ? '#666' : '#999')
            .text(truncate(f.name + ': ' + (f.type || ''), 32));
        });
      }
    });

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
      if (selected && selected.kind === 'node' && selected.name === name) { selected = null; }
      else { selected = { kind: 'node', name: name }; }
      applySelection();
    });
    svg.selectAll('.sd-edge-path, .sd-edge-label-bg, .sd-edge-label').on('click', function() {
      const id = this.getAttribute('data-edge-id');
      if (!id) return;
      if (selected && selected.kind === 'edge' && selected.id === id) { selected = null; }
      else { selected = { kind: 'edge', id: id }; }
      applySelection();
    });
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

  window._renderSchemaGraph = function() {
    const model = buildSchemaModel();
    if (!model.nodeTypes.length) {
      setSchemaEmpty('No schema configured for this dataset.');
      const statsEl = document.getElementById('schema-stats');
      if (statsEl) statsEl.textContent = '';
      return;
    }
    hideSchemaEmpty();

    const totalInstances = model.nodeTypes.reduce(function(s, t) {
      return s + (t.instance_count || 0);
    }, 0);
    // Edge-type count = unique relation names (collapses multiple
    // source→target pairs that share a relation).
    const uniqueRels = new Set(model.edgeTypes.map(primaryRelation));
    const statsEl = document.getElementById('schema-stats');
    if (statsEl) {
      statsEl.textContent =
        model.nodeTypes.length + ' node types · ' +
        uniqueRels.size + ' edge types · ' +
        totalInstances + ' total instances';
    }

    renderSchemaDiagram(model);
  };

  if (!schemaData && (!schemaGraphData || !schemaGraphData.nodes || !schemaGraphData.nodes.length)) {
    setSchemaEmpty('No schema configured for this dataset.');
  }
})();
