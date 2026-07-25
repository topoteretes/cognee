// Business view ("LOOM") — the default tab: the story of a business becoming
// one connected model.
//
// Layout is three layers in one scene:
//   HTML rails   — source cards (left) and agent cards (right), pinned in
//                  screen space so they never scale with the camera;
//   Canvas       — the living entity graph (force-settled, breathing,
//                  colored by the customer's own source colors) plus the
//                  filaments that tie rails to the model;
//   HTML chrome  — lower-third narration bar, moment reel with auto-play,
//                  and a word altimeter (Business / Players / Connections /
//                  Records) driving semantic zoom.
//
// Semantic zoom levels (one d3.zoom camera, thresholds with hysteresis):
//   L0 Business    — top entities only, region captions, zero plumbing
//   L1 Players     — every entity, second label tier, hover relations
//   L2 Connections — documents unfold from their entities as provenance
//   L3 Records     — explicit toggle; remaining stages fade in dimmed
//
// Reads the payload through story_view's globals (window._vizNodeById /
// window._vizLinks) plus its own small tokens. Live events arrive through
// window._bvLiveEvent (fan-out from the template poller).
(function () {
  'use strict';

  const view = document.getElementById('business-view');
  if (!view) return;

  // ── Palette (LOOM) ────────────────────────────────────────────────
  const C = {
    deepfield: '#0E1526',
    deeplift: '#141D33',
    card: '#1A2438',
    cardBorder: '#2A3652',
    bone: '#E9EEF6',
    haze: '#7E8CA6',
    inflow: '#43D9E8',
    amber: '#F5A83C',
  };

  const nodeSetColors = __NODESET_COLORS__ || {};
  const bakedSearchEvents = __SEARCH_EVENTS__ || [];

  // ── Payload adapter ───────────────────────────────────────────────
  // Sources are node_sets when they exist (the true source dimension —
  // Slack stamps "slack", the demo stamps crm/marketing/…), falling back
  // to Dataset actor nodes for graphs ingested without node_sets.
  const byId = window._vizNodeById || {};
  const allNodes = Object.values(byId);
  const allLinks = window._vizLinks || [];
  // Link endpoints may already be node OBJECTS (the classic Graph view's d3
  // simulation mutates the shared link array in place) — normalize to ids.
  function endId(v) { return v && typeof v === 'object' ? v.id : v; }
  allLinks.forEach(l => { l._sid = endId(l.source); l._tid = endId(l.target); });

  // Own copies of entity nodes: story_view pins the shared objects with
  // fx/fy for its column layout, which would freeze this view's force
  // simulation. Copies keep the LOOM layout independent.
  const entities = allNodes.filter(n => n.stage === 'entity')
    .map(n => Object.assign({}, n, { x: undefined, y: undefined, fx: null, fy: null, vx: 0, vy: 0 }));
  const E = {};
  entities.forEach(n => { E[n.id] = n; });
  const documents = allNodes.filter(n => n.stage === 'document');
  const agents = allNodes.filter(n => n.type === 'Agent');
  const datasets = allNodes.filter(n => n.type === 'Dataset');
  const tenantNodes = allNodes.filter(n => n.type === 'Tenant');
  const userNodes = allNodes.filter(n => n.type === 'User')
    .sort((a, b) => (b.is_current ? 1 : 0) - (a.is_current ? 1 : 0));
  const userNode = userNodes.find(n => n.is_current) || userNodes[0] || null;

  // Governance indexes from the overlay edges: who owns / can touch what,
  // which user operates which agent, which layers a dataset carries.
  const access = {};          // userId -> {datasetId -> {owns, perms:Set}}
  const agentOwner = {};      // agentId -> userId
  const datasetLayers = {};   // datasetId -> [node_set names]
  const dsById = {};
  datasets.forEach(d => { dsById[d.id] = d; });
  allLinks.forEach(l => {
    const s = byId[l._sid], t = byId[l._tid];
    if (!s || !t) return;
    if (s.type === 'User' && t.type === 'Agent' && l.relation === 'operates') {
      agentOwner[t.id] = s.id;
    }
    if (s.type === 'User' && t.type === 'Dataset' &&
        (l.relation === 'owns' || String(l.relation).indexOf('can_') === 0)) {
      const slot = ((access[s.id] = access[s.id] || {})[t.id] =
        access[s.id][t.id] || { owns: false, perms: new Set() });
      if (l.relation === 'owns') slot.owns = true;
      else slot.perms.add(String(l.relation).slice(4));
    }
    if (s.type === 'Dataset' && t.type === 'NodeSet' && l.relation === 'has_layer') {
      (datasetLayers[s.id] = datasetLayers[s.id] || []).push(t.name);
    }
  });
  const nodeSets = allNodes.filter(n => n.type === 'NodeSet' && n.name &&
    !['session_learnings', 'user_sessions_from_cache', 'agent_trace_feedbacks'].includes(n.name));

  function setsOf(n) {
    if (Array.isArray(n.belongs_to_set) && n.belongs_to_set.length) return n.belongs_to_set;
    if (n.source_node_set) return String(n.source_node_set).split(',').map(s => s.trim()).filter(Boolean);
    return [];
  }

  // Source list: node_sets preferred, datasets as fallback cards.
  const sourceNames = nodeSets.length ? nodeSets.map(n => n.name) : datasets.map(n => n.name);
  const usingSets = nodeSets.length > 0;

  // Guard the answer-amber encoding: nudge any source color too close to
  // amber's hue so "amber = live signal" stays unambiguous.
  function hueDistance(a, b) {
    const d = Math.abs(a - b) % 360;
    return d > 180 ? 360 - d : d;
  }
  function colorForSet(name, i) {
    let hex = nodeSetColors[name];
    if (!hex) {
      const hue = (i * 137.508 + 200) % 360;
      return d3.hsl(hue, 0.55, 0.62).formatHex();
    }
    const hsl = d3.hsl(hex);
    if (!isNaN(hsl.h) && hueDistance(hsl.h, 35) < 15) {
      hsl.h = (hsl.h + 40) % 360;
      hex = hsl.formatHex();
    }
    return hex;
  }
  const setColor = {};
  sourceNames.forEach((s, i) => { setColor[s] = colorForSet(s, i); });

  // Per-source membership counts (entities + documents credited to it).
  const setEntityCount = {};
  const setDocCount = {};
  entities.forEach(n => setsOf(n).forEach(s => { setEntityCount[s] = (setEntityCount[s] || 0) + 1; }));
  documents.forEach(n => setsOf(n).forEach(s => { setDocCount[s] = (setDocCount[s] || 0) + 1; }));

  // Entity adjacency (semantic links only — the business relationships).
  const semanticLinks = allLinks.filter(l =>
    l.edge_class === 'semantic' && byId[l._sid] && byId[l._tid] &&
    byId[l._sid].stage === 'entity' && byId[l._tid].stage === 'entity');

  // Cross-source bridges are first-class: the "surprising connections".
  semanticLinks.forEach(l => {
    const a = setsOf(byId[l._sid]), b = setsOf(byId[l._tid]);
    l._bridge = a.length && b.length && !a.some(s => b.includes(s));
  });

  // Document → entities index (for L2 provenance unfolding).
  const docLinks = allLinks.filter(l => {
    const s = byId[l._sid], t = byId[l._tid];
    return s && t && ((s.stage === 'document' && t.stage === 'entity') ||
      (s.stage === 'entity' && t.stage === 'document') ||
      (s.stage === 'chunk' && t.stage === 'entity'));
  });

  // Agent → sources captions from actor edges.
  const agentReads = {};
  allLinks.forEach(l => {
    const s = byId[l._sid], t = byId[l._tid];
    if (s && t && s.type === 'Agent' && (l.relation === 'reads' || l.relation === 'writes')) {
      (agentReads[s.id] = agentReads[s.id] || new Set()).add(t.name || 'dataset');
    }
  });

  // ── DOM scaffold (built once, inside #business-view) ──────────────
  view.innerHTML = `
    <div id="bv-canvas-wrap"><canvas id="bv-canvas"></canvas></div>
    <div id="bv-left" class="bv-rail"><div class="bv-rail-title">sources</div></div>
    <div id="bv-right" class="bv-rail"><div class="bv-rail-title">operators</div><div id="bv-org"></div></div>
    <div id="bv-chip"></div>
    <div id="bv-answer"></div>
    <div id="bv-dock">
      <div id="bv-narration-row"><span id="bv-narration-text"></span></div>
      <div id="bv-dock-row">
        <button id="bv-play" title="Play the story">▶</button>
        <div id="bv-reel"></div>
        <div id="bv-altimeter">
          <span data-l="0" class="on">Business</span><span data-l="1">Players</span><span data-l="2">Connections</span><span data-l="3">Records</span>
        </div>
      </div>
    </div>`;

  const css = document.createElement('style');
  css.textContent = `
  #business-view{position:fixed;inset:0;background:radial-gradient(1200px 700px at 50% 42%, ${C.deeplift}, ${C.deepfield});
    font-family:-apple-system,'Segoe UI',system-ui,sans-serif;color:${C.bone};overflow:hidden;}
  #bv-canvas-wrap{position:absolute;inset:0;}
  #bv-canvas{position:absolute;inset:0;cursor:grab;}
  .bv-rail{position:absolute;top:56px;bottom:96px;width:196px;display:flex;flex-direction:column;gap:10px;
    padding:10px;overflow-y:auto;scrollbar-width:none;z-index:5;transition:transform .3s ease,opacity .3s ease;}
  .bv-rail::-webkit-scrollbar{display:none}
  #bv-left{left:0;} #bv-right{right:0;}
  .bv-rail-title{font-size:10px;letter-spacing:.18em;text-transform:uppercase;color:${C.haze};padding:2px 4px;}
  .bv-card{background:${C.card};border:1px solid ${C.cardBorder};border-radius:10px;padding:10px 12px;position:relative;
    transition:transform .25s ease,border-color .4s ease;}
  .bv-card .t{font-size:13px;font-weight:600;letter-spacing:-.01em;}
  .bv-card .s{font-size:11px;color:${C.haze};margin-top:2px;font-variant-numeric:tabular-nums;}
  .bv-card .spine{position:absolute;left:0;top:8px;bottom:8px;width:3px;border-radius:2px;}
  .bv-card.ghost{border-style:dashed;color:${C.haze};background:transparent;}
  .bv-card.flash{border-color:${C.inflow};}
  .bv-card.lift{transform:translateY(-2px);border-color:${C.amber};}
  #bv-org{display:flex;flex-direction:column;}
  .bv-org-node{position:relative;padding:8px 10px 8px 14px;background:${C.card};border:1px solid ${C.cardBorder};
    border-radius:10px;margin-bottom:2px;}
  .bv-org-node .t{font-size:12.5px;font-weight:600;display:flex;align-items:center;gap:6px;}
  .bv-org-node .s{font-size:10.5px;color:${C.haze};margin-top:1px;}
  .bv-org-node.tenant{background:transparent;border-style:dashed;}
  .bv-org-node .badge{font-size:9px;letter-spacing:.06em;text-transform:uppercase;color:${C.deepfield};
    background:${C.amber};border-radius:4px;padding:1px 5px;font-weight:700;}
  .bv-org-child{margin-left:14px;position:relative;}
  .bv-org-child::before{content:'';position:absolute;left:-8px;top:-2px;bottom:14px;width:1px;background:${C.cardBorder};}
  .bv-org-child::after{content:'';position:absolute;left:-8px;top:20px;width:8px;height:1px;background:${C.cardBorder};}
  .bv-org-node .dot{width:7px;height:7px;border-radius:50%;flex:none;}
  .bv-org-node.asking{border-color:${C.amber};box-shadow:0 0 12px rgba(245,168,60,.25);}
  .bv-qcount{font-size:10.5px;color:${C.haze};margin-left:auto;}
  .bv-org-node .acc{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px;}
  .bv-acc{font-size:9.5px;color:${C.haze};border:1px solid ${C.cardBorder};border-radius:5px;
    padding:1px 6px;white-space:nowrap;}
  .bv-acc b{color:${C.bone};font-weight:700;margin-left:3px;}
  .bv-acc.own{border-color:rgba(67,217,232,.45);color:${C.inflow};}
  .bv-acc.own b{color:${C.inflow};}
  .bv-card.external{opacity:.72;border-style:dashed;}
  .bv-card.knowledge{cursor:pointer;}
  .bv-card.knowledge.focused{border-color:${C.inflow};box-shadow:0 0 12px rgba(67,217,232,.2);}
  .bv-card .brain{font-size:8.5px;letter-spacing:.05em;text-transform:uppercase;border:1px solid ${C.cardBorder};
    color:${C.haze};border-radius:4px;padding:1px 5px;margin-left:6px;font-weight:600;vertical-align:1px;}
  .bv-card .brain.team{border-color:rgba(67,217,232,.45);color:${C.inflow};}
  .bv-card .live-dot{color:${C.inflow};font-size:8px;vertical-align:2px;}
  .bv-card .share{position:absolute;right:10px;top:9px;font-size:10px;color:${C.haze};cursor:pointer;
    opacity:0;transition:opacity .15s;}
  .bv-card:hover .share{opacity:1;}
  .bv-dim{opacity:.25 !important;transition:opacity .2s;}
  .bv-rail.compressed{transform:translateX(var(--bv-hide,-160px));opacity:.75;}
  #bv-right.compressed{--bv-hide:160px;}
  #bv-dock{position:absolute;left:0;right:0;bottom:0;z-index:6;
    background:linear-gradient(transparent, rgba(14,21,38,.96) 55%);padding:6px 16px 10px;}
  #bv-narration-row{height:24px;display:flex;align-items:center;justify-content:center;}
  #bv-narration-text{font-size:12.5px;color:${C.haze};letter-spacing:.01em;transition:opacity .4s;}
  #bv-dock-row{display:flex;align-items:center;gap:10px;height:36px;}
  #bv-play{width:28px;height:28px;border-radius:50%;border:1px solid ${C.cardBorder};background:${C.card};
    color:${C.bone};cursor:pointer;font-size:11px;flex:none;}
  #bv-play.playing{border-color:${C.amber};color:${C.amber};}
  #bv-reel{flex:1;display:flex;gap:6px;overflow-x:auto;scrollbar-width:none;align-items:center;}
  #bv-reel::-webkit-scrollbar{display:none}
  .bv-moment{flex:none;height:26px;padding:0 10px;border-radius:13px;border:1px solid ${C.cardBorder};
    background:${C.card};color:${C.haze};font-size:11px;line-height:24px;cursor:pointer;white-space:nowrap;}
  .bv-moment.q{color:${C.amber};border-color:rgba(245,168,60,.4);}
  .bv-moment.on{color:${C.bone};border-color:${C.bone};}
  #bv-altimeter{flex:none;display:flex;gap:2px;background:${C.card};border:1px solid ${C.cardBorder};
    border-radius:8px;padding:3px;}
  #bv-altimeter span{font-size:11px;padding:3px 10px;border-radius:6px;color:${C.haze};cursor:pointer;}
  #bv-altimeter span.on{background:${C.deeplift};color:${C.bone};}
  #bv-answer{position:absolute;left:50%;transform:translateX(-50%);bottom:84px;max-width:560px;z-index:7;
    display:none;background:rgba(26,36,56,.96);border:1px solid rgba(245,168,60,.5);border-radius:12px;
    padding:12px 16px;backdrop-filter:blur(4px);}
  #bv-answer .q{font-size:13px;font-weight:600;color:${C.amber};}
  #bv-answer .a{font-size:12px;color:${C.bone};margin-top:6px;line-height:1.5;max-height:120px;overflow-y:auto;}
  #bv-answer .x{position:absolute;top:6px;right:10px;color:${C.haze};cursor:pointer;font-size:14px;}
  #bv-chip{position:absolute;right:216px;top:64px;z-index:7;display:none;background:${C.card};
    border:1px solid ${C.amber};color:${C.amber};font-size:11px;border-radius:14px;padding:4px 12px;cursor:pointer;}
  /* The template's LIVE badge sits at the bottom-right corner, which the
     dock now owns — lift it above the dock while the Business tab is up. */
  #live-events-badge{bottom:56px !important;}
  #business-view .bv-hovercard{position:absolute;z-index:8;background:${C.card};border:1px solid ${C.cardBorder};
    border-radius:8px;padding:8px 10px;font-size:11px;pointer-events:none;display:none;max-width:240px;}
  `;
  document.head.appendChild(css);

  const canvas = document.getElementById('bv-canvas');
  const ctx = canvas.getContext('2d');
  const railL = document.getElementById('bv-left');
  const railR = document.getElementById('bv-right');
  const narration = document.getElementById('bv-narration-text');
  const chipEl = document.getElementById('bv-chip');
  const hover = document.createElement('div');
  hover.className = 'bv-hovercard';
  view.appendChild(hover);

  // ── Rails ─────────────────────────────────────────────────────────
  const sourceCardEls = {};
  sourceNames.forEach(name => {
    const el = document.createElement('div');
    el.className = 'bv-card';
    const docs = setDocCount[name] || 0, ents = setEntityCount[name] || 0;
    el.innerHTML = `<div class="spine" style="background:${setColor[name]}"></div>
      <div class="t">${esc(name)}</div>
      <div class="s">${ents ? ents + ' entities · ' + docs + ' item' + (docs === 1 ? '' : 's') : 'weaving…'}</div>`;
    railL.appendChild(el);
    sourceCardEls[name] = el;
  });
  const ghost = document.createElement('div');
  ghost.className = 'bv-card ghost';
  ghost.innerHTML = '<div class="t">+ connect a source</div><div class="s">slack · files · crm · anything</div>';
  railL.appendChild(ghost);

  // Operators: the governance tree — tenant → users → their agents — with
  // each user's ACCESS spelled out (owner / R / W / share per dataset).
  // Hovering a user highlights what they can touch; everything else dims.
  const org = document.getElementById('bv-org');
  const questionCount = bakedSearchEvents.filter(e => (e.kind || 'search') === 'search').length;

  function orgNode(cls, html) {
    const el = document.createElement('div');
    el.className = 'bv-org-node' + (cls ? ' ' + cls : '');
    el.innerHTML = html;
    return el;
  }
  function permCode(slot) {
    if (slot.owns) return 'owner';
    const p = slot.perms;
    const code = (p.has('read') ? 'R' : '') + (p.has('write') ? 'W' : '') +
      (p.has('share') ? 'S' : '') + (p.has('delete') ? 'D' : '');
    return code || 'R';
  }
  function accessChips(uid) {
    const slots = access[uid] || {};
    return Object.keys(slots).map(did => {
      const d = dsById[did];
      if (!d) return '';
      const slot = slots[did];
      return `<span class="bv-acc${slot.owns ? ' own' : ''}" data-ds="${did}">${esc(d.name)}
        <b>${permCode(slot)}</b></span>`;
    }).join('');
  }
  function accessibleDatasets(uid) {
    return new Set(Object.keys(access[uid] || {}));
  }
  function wireUserHover(el, uid) {
    el.addEventListener('mouseenter', () => {
      const ok = accessibleDatasets(uid);
      document.querySelectorAll('[data-dsrow]').forEach(row => {
        row.classList.toggle('bv-dim', !ok.has(row.dataset.dsrow));
      });
      // The rendered dataset's sources dim too when this user can't read it.
      const renderedOk = datasets.some(d => !d.external && ok.has(d.id));
      document.querySelectorAll('#bv-left .bv-card:not(.ghost):not([data-dsrow])')
        .forEach(c => c.classList.toggle('bv-dim', !renderedOk));
    });
    el.addEventListener('mouseleave', () => {
      document.querySelectorAll('.bv-dim').forEach(x => x.classList.remove('bv-dim'));
    });
  }

  let orgParent = org;
  if (tenantNodes.length) {
    orgParent.appendChild(orgNode('tenant',
      `<div class="t" style="color:${C.haze}">⌂ ${esc(tenantNodes[0].name || 'organization')}</div>
       <div class="s">${userNodes.length} member${userNodes.length === 1 ? '' : 's'}</div>`));
    const wrap = document.createElement('div');
    wrap.className = 'bv-org-child';
    org.appendChild(wrap);
    orgParent = wrap;
  }

  const agentCardEls = {};
  const agentsByUser = {};
  agents.forEach(a => {
    const uid = agentOwner[a.id] || (userNode && userNode.id);
    (agentsByUser[uid] = agentsByUser[uid] || []).push(a);
  });

  userNodes.forEach(u => {
    const you = !!u.is_current;
    const userEl = orgNode('', `<div class="t"><span class="dot" style="background:${you ? C.bone : C.haze}"></span>
        ${you ? 'you' : esc((u.name || '').split('@')[0])}
        <span class="bv-qcount">${you && questionCount ? questionCount + ' questions' : ''}</span></div>
      <div class="s">${esc(u.name || '')}</div>
      <div class="acc">${accessChips(u.id)}</div>`);
    userEl.dataset.uid = u.id;
    wireUserHover(userEl, u.id);
    orgParent.appendChild(userEl);

    const ownAgents = agentsByUser[u.id] || [];
    if (ownAgents.length) {
      const agentsWrap = document.createElement('div');
      agentsWrap.className = 'bv-org-child';
      orgParent.appendChild(agentsWrap);
      ownAgents.forEach(a => {
        const reads = agentReads[a.id] ? [...agentReads[a.id]].join(', ') : null;
        const el = orgNode('', `<div class="t"><span class="dot" style="background:${C.amber}"></span>${esc(a.name)}
            <span class="badge">agent</span></div>
          <div class="s">${reads ? 'reads ' + esc(reads) : 'connected'}</div>`);
        el.dataset.uid = u.id;
        wireUserHover(el, u.id);
        agentsWrap.appendChild(el);
        agentCardEls[a.id] = el;
      });
    }
  });
  if (!agents.length) {
    orgParent.appendChild(orgNode('tenant', `<div class="t" style="color:${C.haze}">+ plug in your agent</div>
      <div class="s">claude code · mcp · sdk</div>`));
  }

  // Knowledge estate on the left rail: BRAINS, not rows. Each dataset is a
  // team brain (multiple people can touch it) or a personal brain (owner
  // only), it names who it belongs to, and CLICKING it shows what it applies
  // to — the people with access light up in the Operators tree and, for the
  // rendered brain, the camera fits its territories.
  const knowledgeTitle = document.createElement('div');
  knowledgeTitle.className = 'bv-rail-title';
  knowledgeTitle.style.marginTop = '12px';
  knowledgeTitle.textContent = 'knowledge';
  railL.appendChild(knowledgeTitle);

  const userLabel = uid => {
    const u = userNodes.find(x => x.id === uid);
    if (!u) return null;
    return u.is_current ? 'you' : (u.name || '').split('@')[0];
  };
  // Reverse access index: who can touch each dataset (owner first).
  const dsUsers = {};
  Object.keys(access).forEach(uid => {
    Object.keys(access[uid]).forEach(did => {
      (dsUsers[did] = dsUsers[did] || []).push({
        uid, name: userLabel(uid), owns: access[uid][did].owns,
        perms: access[uid][did].perms,
      });
    });
  });
  Object.values(dsUsers).forEach(list => list.sort((a, b) => (b.owns ? 1 : 0) - (a.owns ? 1 : 0)));

  let knowledgeFocus = null;   // dataset id currently focused
  function clearKnowledgeFocus() {
    knowledgeFocus = null;
    document.querySelectorAll('.bv-dim').forEach(x => x.classList.remove('bv-dim'));
    document.querySelectorAll('[data-dsrow].focused').forEach(x => x.classList.remove('focused'));
  }
  function focusKnowledge(d) {
    if (knowledgeFocus === d.id) { clearKnowledgeFocus(); return; }
    clearKnowledgeFocus();
    knowledgeFocus = d.id;
    const holders = dsUsers[d.id] || [];
    const holderIds = new Set(holders.map(h => h.uid));
    // Light up who this brain applies to; dim everyone else.
    document.querySelectorAll('#bv-org .bv-org-node').forEach(el => {
      const uid = el.dataset.uid;
      el.classList.toggle('bv-dim', !!uid && !holderIds.has(uid));
    });
    document.querySelectorAll('[data-dsrow]').forEach(row => {
      row.classList.toggle('bv-dim', row.dataset.dsrow !== d.id);
      row.classList.toggle('focused', row.dataset.dsrow === d.id);
    });
    const who = holders.map(h => h.name + (h.owns ? ' (owner)' : '')).join(', ');
    const layers = (datasetLayers[d.id] || []);
    if (d.external) {
      narrate(`${d.name} — ${holders.length > 1 ? 'team' : 'personal'} brain of ${who || 'this workspace'} · lives in its own graph, not shown here`, C.inflow);
    } else {
      narrate(`${d.name} — ${holders.length > 1 ? 'team' : 'personal'} brain shared by ${who} · ${layers.length} layers · ${entities.length} entities below`, C.inflow);
      fit(true);
    }
  }
  canvas.addEventListener('click', clearKnowledgeFocus);

  datasets.forEach(d => {
    const el = document.createElement('div');
    el.className = 'bv-card knowledge' + (d.external ? ' external' : '');
    el.setAttribute('data-dsrow', d.id);
    const holders = dsUsers[d.id] || [];
    const team = holders.length > 1;
    const layers = (datasetLayers[d.id] || []).join(' · ');
    const who = holders.map(h => h.name).filter(Boolean).join(' + ') || 'this workspace';
    el.innerHTML = `<div class="spine" style="background:${d.external ? C.haze : C.inflow}"></div>
      <div class="t">${esc(d.name)}${d.external ? '' : ' <span class="live-dot">●</span>'}
        <span class="brain ${team ? 'team' : ''}">${team ? 'team brain' : 'personal brain'}</span></div>
      <div class="s">${esc(who)}${layers ? ' · ' + esc(layers) : ''}${d.external ? ' · own graph' : ''}</div>
      <div class="share" title="Grant read/write on this dataset — cognee permissions API (give_permission_on_dataset)">+ share</div>`;
    el.addEventListener('click', ev => {
      if (ev.target.classList.contains('share')) return;
      focusKnowledge(d);
    });
    railL.appendChild(el);
  });

  // ── Simulation ────────────────────────────────────────────────────
  let W = 0, H = 0, dpr = 1;
  function resize() {
    W = view.clientWidth; H = view.clientHeight;
    dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr; canvas.height = H * dpr;
    canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
  }
  resize();
  window.addEventListener('resize', () => { resize(); draw(); });

  // Region anchors: each source gets an angle slice around center.
  const anchors = {};
  sourceNames.forEach((s, i) => {
    const angle = (i / Math.max(sourceNames.length, 1)) * Math.PI * 2 - Math.PI / 2;
    const r = sourceNames.length > 1 ? 300 : 0;
    anchors[s] = { x: Math.cos(angle) * r * 1.25, y: Math.sin(angle) * r * 0.6 };
  });
  function anchorOf(n) {
    const sets = setsOf(n);
    if (!sets.length) return { x: 0, y: 0 };
    let x = 0, y = 0;
    sets.forEach(s => { const a = anchors[s] || { x: 0, y: 0 }; x += a.x; y += a.y; });
    return { x: x / sets.length, y: y / sets.length };
  }

  const importanceMax = Math.max(...entities.map(n => n.importance || 0), 1);
  entities.forEach(n => {
    n._r = 5 + 11 * ((n.importance || 0) / importanceMax);
    const a = anchorOf(n);
    n.x = a.x + (Math.sin(hash(n.id)) * 150);
    n.y = a.y + (Math.cos(hash(n.id) * 2) * 110);
  });

  // Restore cached positions so growth reloads read as growth, not reshuffle.
  const storeKey = 'bv-state:' + location.pathname;
  let prevIds = null;
  try {
    const saved = JSON.parse(localStorage.getItem(storeKey) || 'null');
    if (saved && saved.ids) {
      prevIds = new Set(saved.ids);
      entities.forEach(n => {
        const p = saved.pos && saved.pos[n.id];
        if (p) { n.x = p[0]; n.y = p[1]; }
      });
    }
  } catch (e) { /* storage unavailable */ }

  // New-since-last-render entities materialize from their source's card.
  const newborn = [];
  if (prevIds) entities.forEach(n => { if (!prevIds.has(n.id)) newborn.push(n); });

  const sim = d3.forceSimulation(entities)
    .force('link', d3.forceLink(semanticLinks.map(l => ({ source: l._sid, target: l._tid }))).id(d => d.id).distance(110).strength(0.2))
    .force('charge', d3.forceManyBody().strength(-220))
    // Collide radius reserves label room, so names rarely overlap.
    .force('collide', d3.forceCollide().radius(d => d._r + 26))
    .force('ax', d3.forceX(d => anchorOf(d).x).strength(0.07))
    .force('ay', d3.forceY(d => anchorOf(d).y).strength(0.07))
    .alpha(prevIds ? 0.08 : 0.9)
    .on('tick', () => requestDraw());

  function persist() {
    try {
      const pos = {};
      entities.forEach(n => { pos[n.id] = [Math.round(n.x), Math.round(n.y)]; });
      localStorage.setItem(storeKey, JSON.stringify({ ids: entities.map(n => n.id), pos }));
    } catch (e) { /* best effort */ }
  }
  sim.on('end', persist);
  window.addEventListener('beforeunload', persist);

  // ── Camera + semantic zoom ────────────────────────────────────────
  let transform = d3.zoomIdentity;
  let level = 0;               // 0..2 by zoom; 3 by toggle
  let plumbing = false;        // L3 flag
  let lastInteraction = 0;

  const zoom = d3.zoom().scaleExtent([0.25, 8]).on('zoom', ev => {
    transform = ev.transform;
    lastInteraction = performance.now();
    updateLevel();
    requestDraw();
  });
  d3.select(canvas).call(zoom);

  function updateLevel() {
    const k = transform.k;
    let next = level;
    if (level === 0 && k > 1.7) next = 1;
    else if (level === 1 && k < 1.4) next = 0;
    if (level <= 1 && k > 3.2) next = 2;
    else if (level === 2 && k < 2.7) next = 1;
    if (next !== level) {
      level = next;
      railL.classList.toggle('compressed', level >= 1);
      railR.classList.toggle('compressed', level >= 1);
      syncAltimeter();
    }
  }
  function syncAltimeter() {
    document.querySelectorAll('#bv-altimeter span').forEach(el => {
      const l = +el.dataset.l;
      el.classList.toggle('on', l === (plumbing ? 3 : level));
    });
  }
  document.querySelectorAll('#bv-altimeter span').forEach(el => {
    el.addEventListener('click', () => {
      const l = +el.dataset.l;
      lastInteraction = performance.now();
      if (l === 3) { plumbing = !plumbing; syncAltimeter(); requestDraw(); return; }
      plumbing = false;
      const k = l === 0 ? 1 : l === 1 ? 2.1 : 3.8;
      d3.select(canvas).transition().duration(600)
        .call(zoom.transform, d3.zoomIdentity.translate(W / 2, H / 2).scale(k).translate(-cx(), -cy()));
    });
  });
  function cx() { return d3.mean(entities, n => n.x) || 0; }
  function cy() { return d3.mean(entities, n => n.y) || 0; }

  function fit(animate) {
    if (!entities.length) return;
    const xs = entities.map(n => n.x), ys = entities.map(n => n.y);
    const minX = Math.min(...xs) - 60, maxX = Math.max(...xs) + 60;
    const minY = Math.min(...ys) - 60, maxY = Math.max(...ys) + 60;
    const k = Math.min(1.5, 0.85 * Math.min((W - 420) / (maxX - minX || 1), (H - 160) / (maxY - minY || 1)));
    const t = d3.zoomIdentity.translate(W / 2, H / 2 - 20).scale(k)
      .translate(-(minX + maxX) / 2, -(minY + maxY) / 2);
    (animate ? d3.select(canvas).transition().duration(700) : d3.select(canvas)).call(zoom.transform, t);
  }

  // ── Spotlight (agent question) with idle gate ─────────────────────
  let spotlight = null;   // {ids:Set, edgeKeys:Set, until, question}
  let pendingEvent = null;

  function playSearchEvent(evt, agentId) {
    const ids = new Set(evt.node_ids || []);
    if (!ids.size) return;
    spotlight = { ids, until: performance.now() + 9000, question: evt.question };
    const card = agentCardEls[agentId] || Object.values(agentCardEls)[0];
    if (card) { card.classList.add('asking'); setTimeout(() => card.classList.remove('asking'), 9000); }
    narrate('agent asked: “' + trunc(evt.question, 70) + '” — ' + ids.size + ' connected facts produced the answer', C.amber);
    // Camera to the retrieved subgraph's bbox (entities only).
    const hit = entities.filter(n => ids.has(n.id));
    if (hit.length) {
      const minX = Math.min(...hit.map(n => n.x)) - 80, maxX = Math.max(...hit.map(n => n.x)) + 80;
      const minY = Math.min(...hit.map(n => n.y)) - 80, maxY = Math.max(...hit.map(n => n.y)) + 80;
      const k = Math.min(3, 0.8 * Math.min(W / (maxX - minX || 1), H / (maxY - minY || 1)));
      d3.select(canvas).transition().duration(900)
        .call(zoom.transform, d3.zoomIdentity.translate(W / 2, H / 2).scale(k)
          .translate(-(minX + maxX) / 2, -(minY + maxY) / 2));
    }
    showAnswer(evt);
    requestDraw();
    setTimeout(requestDraw, 9500);
  }

  // One floating answer card above the dock — the agent "speaking" — instead
  // of a stack of cards competing with the operators panel.
  const answerEl = document.getElementById('bv-answer');
  let answerTimer = null;
  function showAnswer(evt) {
    answerEl.innerHTML = `<div class="x">✕</div>
      <div class="q">${esc(trunc(evt.question || '', 120))}</div>
      <div class="a">${esc(evt.answer || '')}</div>`;
    answerEl.style.display = 'block';
    answerEl.querySelector('.x').addEventListener('click', () => { answerEl.style.display = 'none'; });
    clearTimeout(answerTimer);
    answerTimer = setTimeout(() => { answerEl.style.display = 'none'; }, 14000);
  }

  // Live events: never steal the camera mid-interaction. If the presenter
  // touched anything in the last 8s, dock a quiet chip instead.
  window._bvLiveEvent = function (evt) {
    if (!evt || evt.kind === 'improve') return;
    addMoment(evt, true);
    const idleFor = performance.now() - lastInteraction;
    if (lastInteraction === 0 || idleFor > 8000) playSearchEvent(evt);
    else {
      pendingEvent = evt;
      chipEl.textContent = '▶ new answer — “' + trunc(evt.question || '', 40) + '”';
      chipEl.style.display = 'block';
    }
  };
  chipEl.addEventListener('click', () => {
    chipEl.style.display = 'none';
    if (pendingEvent) { lastInteraction = 0; playSearchEvent(pendingEvent); pendingEvent = null; }
  });

  // ── Narration bar ─────────────────────────────────────────────────
  let narrationTimer = null;
  function narrate(text, color) {
    narration.style.opacity = 0;
    clearTimeout(narrationTimer);
    setTimeout(() => {
      narration.textContent = text;
      narration.style.color = color || C.haze;
      narration.style.opacity = 1;
    }, 250);
    narrationTimer = setTimeout(() => { narration.style.opacity = 0.55; }, 9000);
  }

  // ── Moment reel + auto-play ───────────────────────────────────────
  const reel = document.getElementById('bv-reel');
  const moments = [];
  function addMoment(m, isQuestion) {
    const el = document.createElement('div');
    el.className = 'bv-moment' + (isQuestion ? ' q' : '');
    el.textContent = isQuestion ? '? ' + trunc(m.question || 'question', 26) : m.label;
    const entry = { data: m, el, isQuestion };
    el.addEventListener('click', () => { lastInteraction = 0; playMoment(entry); });
    moments.push(entry);
    reel.appendChild(el);
    reel.scrollLeft = reel.scrollWidth;
  }
  function playMoment(entry) {
    moments.forEach(m => m.el.classList.remove('on'));
    entry.el.classList.add('on');
    if (entry.isQuestion) playSearchEvent(entry.data);
    else {
      narrate(entry.data.label + ' — ' + entry.data.detail, C.inflow);
      fit(true);
    }
  }
  // Seed: cognify runs (via memory view's timeline when exposed) + baked Q&A.
  (window._mmTimeline || []).forEach(run => {
    addMoment({
      label: (run.label || 'data added'),
      detail: (run.node_count || '?') + ' elements joined the model',
    }, false);
  });
  bakedSearchEvents.filter(e => (e.kind || 'search') === 'search' && (e.node_ids || []).length)
    .forEach(e => addMoment(e, true));

  const playBtn = document.getElementById('bv-play');
  let playing = null;
  playBtn.addEventListener('click', () => {
    if (playing) { clearInterval(playing); playing = null; playBtn.classList.remove('playing'); return; }
    if (!moments.length) return;
    playBtn.classList.add('playing');
    let i = 0;
    lastInteraction = 0;
    playMoment(moments[0]);
    playing = setInterval(() => {
      i += 1;
      if (i >= moments.length) { clearInterval(playing); playing = null; playBtn.classList.remove('playing'); fit(true); return; }
      lastInteraction = 0;
      playMoment(moments[i]);
    }, 8000);
  });

  // ── Hover ─────────────────────────────────────────────────────────
  let hovered = null;
  canvas.addEventListener('mousemove', ev => {
    const [mx, my] = worldPoint(ev);
    let best = null, bestD = 18 / transform.k;
    entities.forEach(n => {
      const d = Math.hypot(n.x - mx, n.y - my);
      if (d < n._r + bestD && d < (best ? bestD : 1e9)) { best = n; bestD = d; }
    });
    if (best !== hovered) { hovered = best; requestDraw(); }
    if (best) {
      const sets = setsOf(best);
      const docs = docLinks.filter(l => l._sid === best.id || l._tid === best.id).length;
      hover.innerHTML = `<b>${esc(best.name)}</b><br>
        <span style="color:${C.haze}">${esc(best.type || '')}${sets.length ? ' · from ' + esc(sets.join(', ')) : ''}${docs ? ' · seen in ' + docs + ' places' : ''}</span>`;
      hover.style.display = 'block';
      hover.style.left = Math.min(ev.offsetX + 14, W - 260) + 'px';
      hover.style.top = (ev.offsetY + 14) + 'px';
    } else hover.style.display = 'none';
  });
  canvas.addEventListener('mouseleave', () => { hovered = null; hover.style.display = 'none'; requestDraw(); });
  function worldPoint(ev) {
    return [(ev.offsetX - transform.x) / transform.k, (ev.offsetY - transform.y) / transform.k];
  }

  // ── Draw loop ─────────────────────────────────────────────────────
  let drawQueued = false;
  function requestDraw() {
    if (drawQueued) return;
    drawQueued = true;
    requestAnimationFrame(() => { drawQueued = false; draw(); });
  }

  const born = performance.now();
  const newbornAt = {};
  newborn.forEach((n, i) => { newbornAt[n.id] = born + 400 + i * 40; });
  if (newborn.length) {
    const bySet = {};
    newborn.forEach(n => { (setsOf(n)[0] ? (bySet[setsOf(n)[0]] = (bySet[setsOf(n)[0]] || 0) + 1) : 0); });
    const src = Object.keys(bySet).sort((a, b) => bySet[b] - bySet[a])[0];
    narrate('cognify complete — ' + newborn.length + ' new entities joined the model' + (src ? ' from ' + src : ''), C.inflow);
    const card = sourceCardEls[src];
    if (card) { card.classList.add('flash'); setTimeout(() => card.classList.remove('flash'), 2000); }
  } else if (moments.length) {
    narrate('this is your business — ' + entities.length + ' entities from ' + sourceNames.length +
      ' source' + (sourceNames.length === 1 ? '' : 's') + ', one connected model');
  }

  function draw() {
    const now = performance.now();
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    // Filaments: source cards → their region centroid (screen space mix).
    sourceNames.forEach(name => {
      const card = sourceCardEls[name];
      if (!card) return;
      const r = card.getBoundingClientRect(), vr = view.getBoundingClientRect();
      const x0 = r.right - vr.left, y0 = r.top - vr.top + r.height / 2;
      const a = anchors[name] || { x: 0, y: 0 };
      const cxs = transform.applyX(a.x), cys = transform.applyY(a.y);
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.bezierCurveTo(x0 + 90, y0, cxs - 120, cys, cxs, cys);
      ctx.strokeStyle = 'rgba(126,140,166,0.18)';
      ctx.lineWidth = 1;
      ctx.stroke();
      // drifting inflow particles
      for (let p = 0; p < 2; p++) {
        const t = ((now / 2600) + p * 0.5 + hash(name) % 1) % 1;
        const px = bez(x0, x0 + 90, cxs - 120, cxs, t), py = bez(y0, y0, cys, cys, t);
        ctx.beginPath();
        ctx.arc(px, py, 1.6, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(67,217,232,' + (0.5 * (1 - t)) + ')';
        ctx.fill();
      }
    });

    // The rendered brain's card ties into the whole model.
    document.querySelectorAll('[data-dsrow].knowledge:not(.external)').forEach(card => {
      const r = card.getBoundingClientRect(), vr = view.getBoundingClientRect();
      const x0 = r.right - vr.left, y0 = r.top - vr.top + r.height / 2;
      const cxs = transform.applyX(cx()), cys = transform.applyY(cy());
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.bezierCurveTo(x0 + 110, y0, cxs - 140, cys, cxs, cys);
      ctx.strokeStyle = knowledgeFocus === card.dataset.dsrow
        ? 'rgba(67,217,232,0.5)' : 'rgba(67,217,232,0.12)';
      ctx.lineWidth = knowledgeFocus === card.dataset.dsrow ? 1.6 : 1;
      ctx.stroke();
    });

    ctx.translate(transform.x, transform.y);
    ctx.scale(transform.k, transform.k);

    const spot = spotlight && now < spotlight.until ? spotlight : (spotlight = null);
    const dimmed = spot ? 0.16 : 1;

    // Source territories: soft convex hulls shaded in the source's color,
    // with the caption riding the hull's crown (L0/L1).
    if (level <= 1) {
      ctx.textAlign = 'center';
      sourceNames.forEach(name => {
        const members = entities.filter(n => setsOf(n).includes(name));
        if (members.length < 2) return;
        const pad = 34;
        const pts = [];
        members.forEach(n => {
          for (let a = 0; a < 8; a++) {
            pts.push([n.x + Math.cos(a * Math.PI / 4) * (n._r + pad),
                      n.y + Math.sin(a * Math.PI / 4) * (n._r + pad)]);
          }
        });
        const hull = d3.polygonHull(pts);
        if (!hull) return;
        const col = d3.color(setColor[name] || '#888');
        ctx.beginPath();
        // Smooth the hull with quadratic midpoint curves.
        for (let i = 0; i < hull.length; i++) {
          const p = hull[i], q = hull[(i + 1) % hull.length];
          const mx = (p[0] + q[0]) / 2, my = (p[1] + q[1]) / 2;
          if (i === 0) ctx.moveTo(mx, my);
          else ctx.quadraticCurveTo(p[0], p[1], mx, my);
        }
        const p0 = hull[0], p1 = hull[1 % hull.length];
        ctx.quadraticCurveTo(p0[0], p0[1], (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2);
        ctx.closePath();
        ctx.fillStyle = `rgba(${col.r},${col.g},${col.b},${0.06 * dimmed})`;
        ctx.fill();
        ctx.strokeStyle = `rgba(${col.r},${col.g},${col.b},${0.22 * dimmed})`;
        ctx.lineWidth = 1 / transform.k;
        ctx.stroke();
        const cxm = d3.mean(members, n => n.x);
        const topY = Math.min(...hull.map(p => p[1])) - 12;
        ctx.font = '600 13px -apple-system, sans-serif';
        ctx.fillStyle = `rgba(${col.r},${col.g},${col.b},${0.85 * dimmed})`;
        ctx.fillText(name, cxm, topY);
      });
    }

    // Edges: gentle arcs (perpendicular bow) instead of straight wires.
    semanticLinks.forEach(l => {
      const s = E[l._sid], t = E[l._tid];
      if (!s || !t || s.x == null || t.x == null) return;
      const inSpot = spot && spot.ids.has(s.id) && spot.ids.has(t.id);
      const dx = t.x - s.x, dy = t.y - s.y;
      const mx = (s.x + t.x) / 2 - dy * 0.12, my = (s.y + t.y) / 2 + dx * 0.12;
      ctx.beginPath();
      ctx.moveTo(s.x, s.y);
      ctx.quadraticCurveTo(mx, my, t.x, t.y);
      if (inSpot) { ctx.strokeStyle = C.amber; ctx.lineWidth = 1.8 / transform.k; }
      else if (l._bridge) { ctx.strokeStyle = 'rgba(233,238,246,' + 0.42 * dimmed + ')'; ctx.lineWidth = 1.4 / transform.k; }
      else { ctx.strokeStyle = 'rgba(126,140,166,' + 0.4 * dimmed + ')'; ctx.lineWidth = 1.1 / transform.k; }
      ctx.stroke();
    });

    // L2: documents unfold near their entities
    if (level >= 2 || plumbing) {
      docLinks.forEach(l => {
        const d = byId[l._sid].stage !== 'entity' ? byId[l._sid] : byId[l._tid];
        const e = E[byId[l._sid].stage === 'entity' ? l._sid : l._tid];
        if (!e || e.x == null) return;
        const dx = e.x + 26 + (hash(d.id) % 20), dy = e.y + 18 + (hash(d.id) * 3 % 16);
        ctx.setLineDash([3 / transform.k, 3 / transform.k]);
        ctx.beginPath(); ctx.moveTo(e.x, e.y); ctx.lineTo(dx, dy);
        ctx.strokeStyle = 'rgba(126,140,166,0.3)'; ctx.lineWidth = 0.8 / transform.k; ctx.stroke();
        ctx.setLineDash([]);
        const sets = setsOf(d);
        ctx.fillStyle = sets.length ? (setColor[sets[0]] || C.haze) : C.haze;
        ctx.globalAlpha = 0.75;
        roundRect(ctx, dx - 5, dy - 6, 10, 12, 2);
        ctx.fill();
        ctx.globalAlpha = 1;
      });
    }

    // Entities (breathing radius = the "alive" signature, subtle)
    const showAll = level >= 1;
    const importanceCut = topImportanceCut();
    ctx.textAlign = 'center';
    entities.forEach(n => {
      if (!showAll && (n.importance || 0) < importanceCut && !(spot && spot.ids.has(n.id))) return;
      const bornAt = newbornAt[n.id];
      let scale = 1;
      if (bornAt) {
        if (now < bornAt) return;
        scale = Math.min(1, (now - bornAt) / 700);
      }
      const breathe = 1 + 0.045 * Math.sin(now / 1400 + hash(n.id));
      const r = n._r * breathe * scale;
      const inSpot = spot && spot.ids.has(n.id);
      const alpha = spot ? (inSpot ? 1 : 0.16) : 1;

      if (inSpot) {
        ctx.beginPath();
        ctx.arc(n.x, n.y, r + 6 / transform.k, 0, Math.PI * 2);
        ctx.strokeStyle = C.amber; ctx.lineWidth = 2 / transform.k; ctx.stroke();
      }
      if (bornAt && now - bornAt < 1200) {
        ctx.beginPath();
        ctx.arc(n.x, n.y, r + 4 + (now - bornAt) / 90, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(67,217,232,' + (1 - (now - bornAt) / 1200) + ')';
        ctx.lineWidth = 1.2 / transform.k; ctx.stroke();
      }
      const sets = setsOf(n);
      ctx.globalAlpha = alpha;
      ctx.beginPath();
      ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
      ctx.fillStyle = sets.length ? (setColor[sets[0]] || '#6510F4') : '#8A7BD8';
      ctx.fill();
      if (sets.length > 1) {
        ctx.beginPath();
        ctx.arc(n.x, n.y, r + 2.2 / transform.k, 0, Math.PI * 2);
        ctx.strokeStyle = setColor[sets[1]] || C.bone;
        ctx.lineWidth = 1.6 / transform.k;
        ctx.stroke();
      }
      if (hovered === n) {
        ctx.beginPath(); ctx.arc(n.x, n.y, r + 3 / transform.k, 0, Math.PI * 2);
        ctx.strokeStyle = C.bone; ctx.lineWidth = 1 / transform.k; ctx.stroke();
      }
      // Labels: L0 = priority tier only; L1+ = all named entities.
      const labeled = level >= 1 ? !n.is_unnamed : (n.label_priority || (n.importance || 0) >= importanceCut * 1.4);
      if (labeled && !n.is_unnamed) {
        const fs = Math.max(10, Math.min(16, 10 + 6 * ((n.importance || 0) / importanceMax))) / Math.sqrt(transform.k);
        ctx.font = '600 ' + fs + 'px -apple-system, sans-serif';
        ctx.fillStyle = 'rgba(233,238,246,' + 0.92 * alpha + ')';
        ctx.fillText(n.name, n.x, n.y + r + fs + 2);
      }
      ctx.globalAlpha = 1;
    });

    // L3: plumbing — everything else, dimmed gray
    if (plumbing) {
      ctx.globalAlpha = 0.35;
      allNodes.forEach(n => {
        if (n.stage === 'entity' || n.x != null) return;
        // place unpositioned plumbing on a ring
        n._px = n._px || (Math.sin(hash(n.id)) * 380);
        n._py = n._py || (Math.cos(hash(n.id) * 7) * 300);
        ctx.beginPath();
        ctx.arc(n._px, n._py, 3, 0, Math.PI * 2);
        ctx.fillStyle = '#5B6880';
        ctx.fill();
      });
      ctx.globalAlpha = 1;
    }

    ctx.restore();
    // Keep breathing / particles alive at a gentle cadence.
    if (!document.hidden && view.style.display !== 'none') {
      setTimeout(requestDraw, spot || newborn.length ? 33 : 90);
    }
  }

  function topImportanceCut() {
    if (entities.length <= 60) return 0;
    const sorted = entities.map(n => n.importance || 0).sort((a, b) => b - a);
    return sorted[49] || 0;
  }

  // ── Helpers ───────────────────────────────────────────────────────
  function esc(s) {
    return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  function trunc(s, n) { s = String(s || ''); return s.length > n ? s.slice(0, n - 1) + '…' : s; }
  function hash(s) {
    let h = 0;
    for (let i = 0; i < String(s).length; i++) h = (h * 31 + String(s).charCodeAt(i)) | 0;
    return Math.abs(h) / 2147483647 * 6.28;
  }
  function bez(p0, p1, p2, p3, t) {
    const u = 1 - t;
    return u * u * u * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t * t * t * p3;
  }
  function roundRect(c, x, y, w, h, r) {
    c.beginPath();
    c.moveTo(x + r, y);
    c.arcTo(x + w, y, x + w, y + h, r);
    c.arcTo(x + w, y + h, x, y + h, r);
    c.arcTo(x, y + h, x, y, r);
    c.arcTo(x, y, x + w, y, r);
    c.closePath();
  }

  // ── Entry point ───────────────────────────────────────────────────
  let started = false;
  window._renderBusinessView = function () {
    if (view.style.display === 'none') return;
    if (!started) {
      started = true;
      resize();
      fit(false);
      syncAltimeter();
    }
    requestDraw();
  };
  // Business is the default tab — render immediately if visible.
  if (view.style.display !== 'none') window._renderBusinessView();
})();
