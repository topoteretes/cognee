const state = {
  sessionId: "demo_session",
  datasetName: "session_feedback_lifecycle_demo",
  graph: { nodes: [], edges: [] },
  changedNodeIds: new Set(),
  changedEdgeIds: new Set(),
  latestQaId: null,
  simulation: null,
};

const loadingOverlay = document.getElementById("loadingOverlay");
const loadingSteps = document.getElementById("loadingSteps");
const loadingTitle = document.getElementById("loadingTitle");
const loadingSubtitle = document.getElementById("loadingSubtitle");
const loadingPhase = document.getElementById("loadingPhase");
const graphStats = document.getElementById("graphStats");
const sessionMeta = document.getElementById("sessionMeta");
const messages = document.getElementById("messages");
const selectedElement = document.getElementById("selectedElement");
const sessionIdBadge = document.getElementById("sessionIdBadge");
const sessionEntries = document.getElementById("sessionEntries");
const appShell = document.getElementById("appShell");
const mainResizer = document.getElementById("mainResizer");
const graphBody = document.getElementById("graphBody");
const graphDetailsResizer = document.getElementById("graphDetailsResizer");
const chatPanel = document.querySelector(".chat-panel");
const chatResizer = document.getElementById("chatResizer");
const inlineLoading = document.getElementById("inlineLoading");

const runDemoBtn = document.getElementById("runDemoBtn");
const runMemifyBtn = document.getElementById("runMemifyBtn");
const sendForm = document.getElementById("sendForm");
const questionInput = document.getElementById("questionInput");
const feedbackQaId = document.getElementById("feedbackQaId");
const feedbackScore = document.getElementById("feedbackScore");
const feedbackText = document.getElementById("feedbackText");
const submitFeedbackBtn = document.getElementById("submitFeedbackBtn");
let activeNodeSelection = null;
let activeEdgeSelection = null;
let loadingPhaseTimer = null;
let loadingPhaseIndex = 0;

function setBusy(isBusy) {
  [runDemoBtn, runMemifyBtn, submitFeedbackBtn, questionInput].forEach((el) => {
    el.disabled = isBusy;
  });
}

function renderLoadingSteps(steps) {
  loadingSteps.innerHTML = "";
  steps.forEach((step) => {
    const li = document.createElement("li");
    li.textContent = `${step.name}: ${step.detail}`;
    loadingSteps.appendChild(li);
  });
}

function addMessage(type, text, qaId = null) {
  const node = document.createElement("div");
  node.className = `message ${type}`;
  node.textContent = text;

  if (qaId) {
    const meta = document.createElement("div");
    meta.className = "qa-meta";
    meta.textContent = `qa_id: ${qaId}`;
    node.appendChild(meta);
  }

  messages.appendChild(node);
  messages.scrollTop = messages.scrollHeight;
}

function renderActivity(logItems) {
  const activityLog = document.getElementById("activityLog");
  if (!activityLog) {
    return;
  }
  activityLog.innerHTML = "";
  (logItems || []).slice(-50).reverse().forEach((item) => {
    const row = document.createElement("div");
    row.className = "activity-item";
    row.innerHTML = `<div><strong>${item.event}</strong>: ${item.detail}</div><div class="stamp">${item.time}</div>`;
    activityLog.appendChild(row);
  });
}

function renderSelection(type, item) {
  if (!item) {
    selectedElement.textContent = "Click a node or edge to inspect properties.";
    return;
  }

  const safe = (value) =>
    String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");

  const weight = Number(item.feedback_weight ?? 0.5);
  const weightColor = colorByWeight(weight);

  const payload = {
    type,
    id: item.id,
    label: item.display_label || item.label || item.relation || item.id,
    feedback_weight: item.feedback_weight,
    properties: item.properties || {},
  };
  selectedElement.innerHTML = `
    <div class="sel-head">
      <span class="sel-type">${safe(type)}</span>
      <span class="weight-chip" style="background:${safe(weightColor)}">${safe(weight.toFixed(3))}</span>
    </div>
    <div class="weight-row">
      <span class="weight-label">Feedback Weight</span>
      <span class="weight-value" style="color:${safe(weightColor)}">${safe(weight.toFixed(3))}</span>
    </div>
    <div><strong>${safe(payload.label)}</strong></div>
    <div class="sel-id">${safe(payload.id)}</div>
    <pre class="sel-props">${safe(JSON.stringify(payload.properties, null, 2))}</pre>
  `;
}

function nodeRadius(weight) {
  const w = Math.max(0, Math.min(1, Number(weight ?? 0.5)));
  return 10 + w * 18;
}

function edgeWidth(weight) {
  const w = Math.max(0, Math.min(1, Number(weight ?? 0.5)));
  return 1.6 + w * 4.4;
}

function colorByWeight(weight) {
  const w = Math.max(0, Math.min(1, Number(weight ?? 0.5)));
  if (w <= 0.5) {
    return d3.interpolateRgb("#e65555", "#f2d85f")(w / 0.5);
  }
  return d3.interpolateRgb("#f2d85f", "#66e08a")((w - 0.5) / 0.5);
}

function splitLabelLines(value, maxChars = 22) {
  const text = String(value || "");
  if (!text) return [""];

  const words = text.split(/\s+/).filter(Boolean);
  if (!words.length) return [text];

  const lines = [];
  let current = "";

  words.forEach((word) => {
    if (!current) {
      current = word;
      return;
    }
    if ((current.length + 1 + word.length) <= maxChars) {
      current += ` ${word}`;
    } else {
      lines.push(current);
      current = word;
    }
  });

  if (current) lines.push(current);
  return lines;
}

function shortLabel(value, maxLen = 34) {
  const text = String(value || "").trim();
  if (text.length <= maxLen) return text;
  return `${text.slice(0, maxLen - 3)}...`;
}

function fitTransformForNodes(width, height, nodes, padding = 56) {
  if (!nodes.length) return d3.zoomIdentity;

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  nodes.forEach((node) => {
    const r = nodeRadius(node.feedback_weight);
    minX = Math.min(minX, node.x - r);
    minY = Math.min(minY, node.y - r);
    maxX = Math.max(maxX, node.x + r);
    maxY = Math.max(maxY, node.y + r);
  });

  const graphWidth = Math.max(1, maxX - minX);
  const graphHeight = Math.max(1, maxY - minY);
  const scale = Math.max(
    0.42,
    Math.min(
      1.6,
      Math.min(
        (width - padding * 2) / graphWidth,
        (height - padding * 2) / graphHeight
      )
    )
  );

  const offsetX = (width - graphWidth * scale) / 2 - minX * scale;
  const offsetY = (height - graphHeight * scale) / 2 - minY * scale;
  return d3.zoomIdentity.translate(offsetX, offsetY).scale(scale);
}

function stableHash(text) {
  const value = String(text || "");
  let hash = 0;
  for (let i = 0; i < value.length; i += 1) {
    hash = (hash * 31 + value.charCodeAt(i)) >>> 0;
  }
  return hash;
}

function looksLikeId(value) {
  const text = String(value || "");
  if (!text) return false;
  if (/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(text)) return true;
  if (/^[0-9a-f]{20,}$/i.test(text)) return true;
  if (/^text_[0-9a-f]{12,}$/i.test(text)) return true;
  return false;
}

function nodeDisplayLabel(node) {
  const props = node?.properties || {};
  const textCandidate = String(props.text || "").trim();
  if (textCandidate) return textCandidate;

  const nameCandidate = String(props.name || "").trim();
  if (nameCandidate) return nameCandidate;

  const labelCandidate = String(node?.label || "").trim();
  if (labelCandidate && !looksLikeId(labelCandidate)) return labelCandidate;

  return String(node?.type || "node");
}

function edgeDisplayLabel(edge) {
  const relation = String(edge?.relation || "").trim();
  if (relation) return relation;
  const label = String(edge?.label || "").trim();
  if (label && !looksLikeId(label)) return label;
  return "relation";
}

function showOverlay(title, subtitle, steps = []) {
  if (loadingTitle) loadingTitle.textContent = title;
  if (loadingSubtitle) loadingSubtitle.textContent = subtitle;
  if (loadingPhase) {
    loadingPhase.textContent = "";
  }
  if (loadingSteps) {
    renderLoadingSteps(steps.length ? steps : []);
  }
  loadingOverlay.classList.add("visible");
}

function hideOverlay(delayMs = 180) {
  setTimeout(() => {
    loadingOverlay.classList.remove("visible");
  }, delayMs);
}

function showInlineLoading(text) {
  if (!inlineLoading) return;
  inlineLoading.textContent = text || "Working...";
  inlineLoading.classList.add("visible");
}

function hideInlineLoading() {
  if (!inlineLoading) return;
  inlineLoading.classList.remove("visible");
  inlineLoading.textContent = "";
}

function startInitLoadingPhases() {
  if (!loadingPhase) return;
  loadingPhase.textContent = "We are preparing the demo for you...";
  if (loadingPhaseTimer) {
    clearInterval(loadingPhaseTimer);
    loadingPhaseTimer = null;
  }
}

function stopInitLoadingPhases(finalText = "Ready") {
  if (loadingPhaseTimer) {
    clearInterval(loadingPhaseTimer);
    loadingPhaseTimer = null;
  }
  if (loadingPhase) {
    loadingPhase.textContent = finalText;
  }
}

function renderSessionEntriesList(entries = []) {
  sessionEntries.innerHTML = "";
  if (!entries.length) {
    sessionEntries.textContent = "No session turns yet.";
    return;
  }
  const normalized = [...entries];
  normalized.sort((a, b) => {
    const ta = Date.parse(a?.time || "");
    const tb = Date.parse(b?.time || "");
    if (Number.isFinite(ta) && Number.isFinite(tb)) {
      return ta - tb;
    }
    return 0;
  });

  normalized.forEach((entry) => {
    const card = document.createElement("div");
    card.className = "session-entry";
    card.innerHTML = `
      <div><strong>${entry.qa_id || "qa"}</strong> ${entry.feedback_score != null ? `(score=${entry.feedback_score})` : ""}</div>
      <div class="q">Q: ${entry.question || ""}</div>
      <div class="a">A: ${entry.answer || ""}</div>
    `;
    sessionEntries.appendChild(card);
  });
}

async function refreshSessionEntries() {
  const payload = await api(
    `/demo/session?session_id=${encodeURIComponent(state.sessionId)}&last_n=5000`
  );
  renderSessionEntriesList(payload.entries || []);
}

function renderGraph(graph, changed = { nodes: [], edges: [] }) {
  state.graph = graph;

  graphStats.textContent = `${graph.stats.node_count} nodes | ${graph.stats.edge_count} edges`;
  sessionMeta.textContent = `session: ${state.sessionId} | dataset: ${state.datasetName}`;
  sessionIdBadge.textContent = state.sessionId;

  const svg = d3.select("#graphSvg");
  svg.selectAll("*").remove();

  const width = svg.node().clientWidth;
  const height = svg.node().clientHeight;

  const defs = svg.append("defs");
  const marker = defs
    .append("marker")
    .attr("id", "arrow")
    .attr("viewBox", "0 -3 6 6")
    .attr("refX", 10)
    .attr("refY", 0)
    .attr("markerWidth", 4)
    .attr("markerHeight", 4)
    .attr("markerUnits", "userSpaceOnUse")
    .attr("orient", "auto");
  marker.append("path").attr("d", "M0,-3L6,0L0,3").attr("fill", "#A550FF");

  const container = svg.append("g");

  const zoom = d3.zoom().scaleExtent([0.3, 4]).on("zoom", (event) => {
    container.attr("transform", event.transform);
  });
  svg.call(zoom);

  const nodes = graph.nodes.map((node) => ({
    ...node,
    display_label: nodeDisplayLabel(node),
  }));
  const edges = graph.edges.map((edge) => ({
    ...edge,
    display_label: edgeDisplayLabel(edge),
  }));
  const nodeIds = new Set(nodes.map((node) => node.id));

  const links = edges.map((edge) => ({
    ...edge,
    source: edge.source,
    target: edge.target,
  })).filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target));

  const radiusById = new Map(nodes.map((node) => [node.id, nodeRadius(node.feedback_weight)]));
  const getRadius = (nodeRef) => {
    if (nodeRef && typeof nodeRef === "object" && nodeRef.id) {
      return radiusById.get(nodeRef.id) || nodeRadius(nodeRef.feedback_weight);
    }
    return radiusById.get(nodeRef) || 12;
  };

  // Deterministic spiral anchors to avoid dense central lines/clusters.
  const centerX = width / 2;
  const centerY = height / 2;
  const spreadRadiusX = Math.min(width, height) * 0.43;
  const spreadRadiusY = Math.min(width, height) * 0.34;
  const targetById = new Map();
  const orderedNodes = [...nodes].sort((a, b) => stableHash(a.id) - stableHash(b.id));
  orderedNodes.forEach((node, index) => {
    const hash = stableHash(node.id);
    const angle = index * 2.399963229728653;
    const spiral = Math.min(spreadRadiusX, 62 + Math.sqrt(index + 1) * 60);
    const jitterX = (((hash >> 5) % 13) - 6) * 7;
    const jitterY = (((hash >> 3) % 11) - 5) * 9;
    targetById.set(node.id, {
      x: centerX + Math.cos(angle) * Math.min(spreadRadiusX, spiral * 1.12) + jitterX,
      y: centerY + Math.sin(angle) * Math.min(spreadRadiusY, spiral * 0.92) + jitterY,
    });
  });

  const simulation = d3
    .forceSimulation(nodes)
    .force(
      "link",
      d3
        .forceLink(links)
        .id((d) => d.id)
        .distance((d) => {
          const sourceR = getRadius(d.source);
          const targetR = getRadius(d.target);
          const basePadding = 96;
          const weightPadding = edgeWidth(d.feedback_weight) * 3.1;
          return sourceR + targetR + basePadding + weightPadding;
        })
    )
    .force("charge", d3.forceManyBody().strength((node) => -(380 + getRadius(node) * 28)))
    .force("collide", d3.forceCollide().radius((node) => getRadius(node) + 34).iterations(4))
    .force("x", d3.forceX((d) => targetById.get(d.id)?.x ?? centerX).strength(0.044))
    .force("y", d3.forceY((d) => targetById.get(d.id)?.y ?? centerY).strength(0.052))
    .force("center", d3.forceCenter(centerX, centerY))
    .alpha(1.0)
    .alphaDecay(0.017)
    .velocityDecay(0.34);

  state.simulation = simulation;

  const edgeGroup = container
    .append("g")
    .selectAll("line")
    .data(links)
    .join("line")
    .attr("stroke", (d) => colorByWeight(d.feedback_weight))
    .attr("stroke-opacity", (d) => 0.62 + Math.max(0, Math.min(1, Number(d.feedback_weight ?? 0.5))) * 0.28)
    .attr("stroke-width", (d) => edgeWidth(d.feedback_weight))
    .attr("stroke-linecap", "round")
    .attr("marker-end", "url(#arrow)")
    .classed("selected-edge", (d) => activeEdgeSelection === d.id);

  const edgeHitArea = container
    .append("g")
    .selectAll("line")
    .data(links)
    .join("line")
    .attr("stroke", "transparent")
    .attr("stroke-width", 18)
    .style("cursor", "pointer")
    .on("click", (event, d) => {
      event.stopPropagation();
      activeEdgeSelection = d.id;
      activeNodeSelection = null;
      edgeGroup.classed("selected-edge", (edge) => edge.id === activeEdgeSelection);
      nodeGroup.classed("selected-node", (node) => node.id === activeNodeSelection);
      renderSelection("edge", d);
      updateLabelVisibility();
    });

  const edgeLabel = container
    .append("g")
    .selectAll("text")
    .data(links)
    .join("text")
    .attr("font-size", "10px")
    .attr("fill", "#97abc0")
    .attr("paint-order", "stroke")
    .attr("stroke", "rgba(8, 18, 30, 0.9)")
    .attr("stroke-width", 2.8)
    .attr("opacity", 0)
    .attr("pointer-events", "none")
    .text((d) => shortLabel(d.display_label, 26));

  const nodeGroup = container
    .append("g")
    .selectAll("circle")
    .data(nodes)
    .join("circle")
    .attr("r", (d) => nodeRadius(d.feedback_weight))
    .attr("fill", (d) => colorByWeight(d.feedback_weight))
    .attr("stroke", "rgba(165, 80, 255, 0.48)")
    .attr("stroke-width", 1.35)
    .classed("selected-node", (d) => activeNodeSelection === d.id)
    .style("cursor", "pointer")
    .on("click", (event, d) => {
      event.stopPropagation();
      activeNodeSelection = d.id;
      activeEdgeSelection = null;
      nodeGroup.classed("selected-node", (node) => node.id === activeNodeSelection);
      edgeGroup.classed("selected-edge", (edge) => edge.id === activeEdgeSelection);
      renderSelection("node", d);
      updateLabelVisibility();
    })
    .call(
      d3
        .drag()
        .on("start", (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on("drag", (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on("end", (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        })
    );

  nodeGroup.append("title").text((d) => `${d.display_label}\nweight=${d.feedback_weight}`);

  const nodeLabelMetrics = new Map();
  nodes.forEach((node) => {
    const text = shortLabel(String(node.display_label || "node"), 30);
    nodeLabelMetrics.set(node.id, {
      w: Math.max(26, text.length * 6.1 + 8),
      h: 13,
    });
  });

  const nodeLabel = container
    .append("g")
    .selectAll("text")
    .data(nodes)
    .join("text")
    .attr("font-size", "12px")
    .attr("fill", "#e6eef8")
    .attr("text-anchor", "middle")
    .attr("paint-order", "stroke")
    .attr("stroke", "rgba(8, 18, 30, 0.88)")
    .attr("stroke-width", 3.5)
    .attr("opacity", 0)
    .attr("pointer-events", "none")
    .text((d) => shortLabel(String(d.display_label || "node"), 30));

  const updateLabelVisibility = () => {
    const placed = [];
    const visibleNodeIds = new Set();

    const nodeCandidates = nodes
      .map((node) => {
        const side = stableHash(node.id) % 2 === 0 ? 1 : -1;
        const y = node.y + side * (nodeRadius(node.feedback_weight) + 16);
        const m = nodeLabelMetrics.get(node.id) || { w: 30, h: 13 };
        const priority = (activeNodeSelection === node.id ? 1000 : 0) + nodeRadius(node.feedback_weight);
        return { id: node.id, x: node.x, y, w: m.w, h: m.h, priority };
      })
      .sort((a, b) => b.priority - a.priority);

    nodeCandidates.forEach((candidate) => {
      const left = candidate.x - candidate.w / 2;
      const right = candidate.x + candidate.w / 2;
      const top = candidate.y - candidate.h / 2;
      const bottom = candidate.y + candidate.h / 2;
      const inBounds = left >= 8 && right <= width - 8 && top >= 8 && bottom <= height - 8;

      if (!inBounds && candidate.id !== activeNodeSelection) {
        return;
      }

      const overlaps = placed.some((p) => {
        const overlapX = candidate.w / 2 + p.w / 2 - Math.abs(candidate.x - p.x);
        const overlapY = candidate.h / 2 + p.h / 2 - Math.abs(candidate.y - p.y);
        return overlapX > 0 && overlapY > 0;
      });

      if (!overlaps || candidate.id === activeNodeSelection) {
        visibleNodeIds.add(candidate.id);
        placed.push(candidate);
      }
    });

    nodeLabel
      .attr("x", (d) => d.x)
      .attr("y", (d) => {
        const side = stableHash(d.id) % 2 === 0 ? 1 : -1;
        return d.y + side * (nodeRadius(d.feedback_weight) + 16);
      })
      .attr("opacity", (d) => (visibleNodeIds.has(d.id) ? 0.96 : 0));

    edgeLabel
      .attr("x", (d) => {
        const midX = (d.source.x + d.target.x) / 2;
        const dx = d.target.x - d.source.x;
        const dy = d.target.y - d.source.y;
        const len = Math.max(1, Math.hypot(dx, dy));
        const nx = -dy / len;
        return midX + nx * 14;
      })
      .attr("y", (d) => {
        const midY = (d.source.y + d.target.y) / 2;
        const dx = d.target.x - d.source.x;
        const dy = d.target.y - d.source.y;
        const len = Math.max(1, Math.hypot(dx, dy));
        const ny = dx / len;
        return midY + ny * 14;
      })
      .attr("opacity", (d) => {
        if (activeEdgeSelection && d.id === activeEdgeSelection) return 0.92;
        if (
          activeNodeSelection &&
          (d.source.id === activeNodeSelection || d.target.id === activeNodeSelection)
        ) {
          return 0.72;
        }
        return 0;
      });
  };

  simulation.on("tick", () => {
    nodes.forEach((node) => {
      const radius = nodeRadius(node.feedback_weight);
      // Soft boundary: nudge back toward the center instead of hard clipping to walls.
      const minX = radius + 18;
      const maxX = width - radius - 18;
      const minY = radius + 18;
      const maxY = height - radius - 18;

      if (node.x < minX) node.x += (minX - node.x) * 0.12;
      if (node.x > maxX) node.x -= (node.x - maxX) * 0.12;
      if (node.y < minY) node.y += (minY - node.y) * 0.12;
      if (node.y > maxY) node.y -= (node.y - maxY) * 0.12;
    });

    edgeGroup
      .attr("x1", (d) => d.source.x)
      .attr("y1", (d) => d.source.y)
      .attr("x2", (d) => d.target.x)
      .attr("y2", (d) => d.target.y);

    edgeHitArea
      .attr("x1", (d) => d.source.x)
      .attr("y1", (d) => d.source.y)
      .attr("x2", (d) => d.target.x)
      .attr("y2", (d) => d.target.y);

    nodeGroup.attr("cx", (d) => d.x).attr("cy", (d) => d.y);

    updateLabelVisibility();
  });

  svg.on("click", () => {
    activeNodeSelection = null;
    activeEdgeSelection = null;
    nodeGroup.classed("selected-node", false);
    edgeGroup.classed("selected-edge", false);
    renderSelection(null, null);
    updateLabelVisibility();
  });

  // Auto-fit once layout stabilizes so all elements remain visible and centered.
  const fitOnce = () => {
    const transform = fitTransformForNodes(width, height, nodes, 56);
    svg.transition().duration(380).call(zoom.transform, transform);
    updateLabelVisibility();
  };
  setTimeout(fitOnce, 650);
}

async function api(path, method = "GET", body = null) {
  const response = await fetch(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : null,
  });

  if (!response.ok) {
    const payload = await response.text();
    throw new Error(payload || `Request failed with ${response.status}`);
  }

  return response.json();
}

async function initializeDemo() {
  showOverlay(
    "Loading",
    "Preparing demo..."
  );
  startInitLoadingPhases();

  try {
    const result = await api("/demo/init", "POST");
    state.sessionId = result.session_id;
    state.datasetName = result.dataset_name;

    renderGraph(result.graph);
    renderActivity(result.activity_log || []);
    await refreshSessionEntries();
    stopInitLoadingPhases("Ready");
    hideOverlay(360);
  } catch (error) {
    stopInitLoadingPhases("Initialization failed");
    throw error;
  }
}

async function refreshStateAndGraph(changed = null) {
  const [statePayload, graphPayload] = await Promise.all([api("/demo/state"), api("/demo/graph")]);

  state.sessionId = statePayload.session_id;
  state.datasetName = statePayload.dataset_name;

  const changedPayload = changed || { nodes: [], edges: [] };
  renderGraph(graphPayload, changedPayload);
  renderActivity(statePayload.activity_log || []);
  await refreshSessionEntries();
}

function setupMainResizer() {
  if (!appShell || !mainResizer) return;
  let isDragging = false;

  const onMove = (event) => {
    if (!isDragging) return;
    const rect = appShell.getBoundingClientRect();
    const leftPx = event.clientX - rect.left;
    const clampedPx = Math.max(320, Math.min(rect.width - 360, leftPx));
    const percent = (clampedPx / rect.width) * 100;
    document.documentElement.style.setProperty("--left-pane-width", `${percent}%`);
  };

  const onUp = () => {
    isDragging = false;
    document.body.classList.remove("is-resizing");
    window.removeEventListener("pointermove", onMove);
    window.removeEventListener("pointerup", onUp);
  };

  mainResizer.addEventListener("pointerdown", () => {
    isDragging = true;
    document.body.classList.add("is-resizing");
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  });
}

function setupGraphDetailsResizer() {
  if (!graphBody || !graphDetailsResizer) return;
  let isDragging = false;

  const onMove = (event) => {
    if (!isDragging) return;
    const rect = graphBody.getBoundingClientRect();
    const topPx = event.clientY - rect.top;
    const svgPx = Math.max(220, Math.min(rect.height - 160, topPx));
    const detailsPx = Math.max(120, rect.height - svgPx - 8);
    document.documentElement.style.setProperty(
      "--graph-body-rows",
      `${svgPx}px 8px ${detailsPx}px`
    );
  };

  const onUp = () => {
    isDragging = false;
    document.body.classList.remove("is-resizing");
    window.removeEventListener("pointermove", onMove);
    window.removeEventListener("pointerup", onUp);
  };

  graphDetailsResizer.addEventListener("pointerdown", () => {
    isDragging = true;
    document.body.classList.add("is-resizing");
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  });
}

function setupChatResizer() {
  if (!chatPanel || !chatResizer) return;
  let isDragging = false;

  const onMove = (event) => {
    if (!isDragging) return;
    const rect = chatPanel.getBoundingClientRect();
    const topPx = event.clientY - rect.top;
    const topHeight = Math.max(280, Math.min(rect.height - 180, topPx));
    const bottomHeight = Math.max(160, rect.height - topHeight - 8);
    document.documentElement.style.setProperty(
      "--chat-rows",
      `${topHeight}px 8px ${bottomHeight}px`
    );
  };

  const onUp = () => {
    isDragging = false;
    document.body.classList.remove("is-resizing");
    window.removeEventListener("pointermove", onMove);
    window.removeEventListener("pointerup", onUp);
  };

  chatResizer.addEventListener("pointerdown", () => {
    isDragging = true;
    document.body.classList.add("is-resizing");
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  });
}

runDemoBtn.addEventListener("click", async () => {
  try {
    setBusy(true);
    showInlineLoading("Running demo flow...");
    addMessage("user", "Running scripted demo flow...");
    const result = await api("/demo/run_demo", "POST");
    if (result.session_id) {
      state.sessionId = result.session_id;
      sessionIdBadge.textContent = state.sessionId;
    }

    (result.turns || []).forEach((turn, index) => {
      addMessage("user", `Q${index + 1}: ${turn.question}`);
      addMessage("system", turn.answer, turn.qa_id || null);
    });

    const changed = {
      nodes: result.deltas.changed_nodes,
      edges: result.deltas.changed_edges,
    };
    renderGraph(result.after, changed);
    renderActivity(result.activity_log || []);
    await refreshSessionEntries();
  } catch (error) {
    addMessage("system", `Run_demo failed: ${error.message}`);
  } finally {
    setBusy(false);
    hideInlineLoading();
  }
});

runMemifyBtn.addEventListener("click", async () => {
  try {
    setBusy(true);
    showInlineLoading("Applying memify...");
    const result = await api("/demo/run_memify_pipeline", "POST", {
      session_id: state.sessionId,
    });

    const changed = {
      nodes: result.deltas.changed_nodes,
      edges: result.deltas.changed_edges,
    };

    renderGraph(result.after, changed);
    renderActivity(result.activity_log || []);
    await refreshSessionEntries();
    addMessage(
      "system",
      `Memify complete: nodes changed=${result.deltas.summary.changed_node_count}, edges changed=${result.deltas.summary.changed_edge_count}`
    );
  } catch (error) {
    addMessage("system", `Memify failed: ${error.message}`);
  } finally {
    setBusy(false);
    hideInlineLoading();
  }
});

sendForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const question = questionInput.value.trim();
  if (!question) return;

  try {
    setBusy(true);
    showInlineLoading("Searching...");
    addMessage("user", question);

    const result = await api("/demo/send", "POST", {
      question,
      session_id: state.sessionId,
    });
    if (result.session_id) {
      state.sessionId = result.session_id;
      sessionIdBadge.textContent = state.sessionId;
    }

    if (result.qa_id) {
      state.latestQaId = result.qa_id;
      feedbackQaId.value = result.qa_id;
    }

    addMessage("system", result.answer, result.qa_id || null);

    if (result.auto_feedback && (result.auto_feedback.feedback_score !== null || result.auto_feedback.feedback_text)) {
      addMessage(
        "system",
        `Auto feedback detected: score=${result.auto_feedback.feedback_score ?? "n/a"}, text=${result.auto_feedback.feedback_text ?? ""}`
      );
      const memifyResult = await api("/demo/run_memify_pipeline", "POST", {
        session_id: state.sessionId,
      });
      const changed = {
        nodes: memifyResult.deltas.changed_nodes,
        edges: memifyResult.deltas.changed_edges,
      };
      await refreshStateAndGraph(changed);
      addMessage(
        "system",
        `Auto memify applied. Nodes changed=${memifyResult.deltas.summary.changed_node_count}, edges changed=${memifyResult.deltas.summary.changed_edge_count}`
      );
    } else {
      await refreshStateAndGraph();
    }

    questionInput.value = "";
  } catch (error) {
    addMessage("system", `Send failed: ${error.message}`);
  } finally {
    setBusy(false);
    hideInlineLoading();
  }
});

submitFeedbackBtn.addEventListener("click", async () => {
  const qaId = feedbackQaId.value.trim();
  const score = Number(feedbackScore.value);
  const text = feedbackText.value.trim();

  if (!qaId) {
    addMessage("system", "Feedback failed: missing QA ID");
    return;
  }

  try {
    setBusy(true);
    showInlineLoading("Saving feedback...");
    await api("/demo/feedback", "POST", {
      session_id: state.sessionId,
      qa_id: qaId,
      feedback_score: score,
      feedback_text: text || null,
    });

    const memifyResult = await api("/demo/run_memify_pipeline", "POST", {
      session_id: state.sessionId,
    });
    if (memifyResult.session_id) {
      state.sessionId = memifyResult.session_id;
      sessionIdBadge.textContent = state.sessionId;
    }

    const changed = {
      nodes: memifyResult.deltas.changed_nodes,
      edges: memifyResult.deltas.changed_edges,
    };

    renderGraph(memifyResult.after, changed);
    renderActivity(memifyResult.activity_log || []);
    await refreshSessionEntries();
    addMessage(
      "system",
      `Feedback saved and applied. Nodes changed=${memifyResult.deltas.summary.changed_node_count}, edges changed=${memifyResult.deltas.summary.changed_edge_count}`
    );
    feedbackText.value = "";
  } catch (error) {
    addMessage("system", `Feedback failed: ${error.message}`);
  } finally {
    setBusy(false);
    hideInlineLoading();
  }
});

(async function boot() {
  setupMainResizer();
  setupGraphDetailsResizer();
  setupChatResizer();
  try {
    await initializeDemo();
  } catch (error) {
    loadingSteps.innerHTML = `<li>Initialization failed: ${error.message}</li>`;
    addMessage("system", `Initialization failed: ${error.message}`);
  }
})();
