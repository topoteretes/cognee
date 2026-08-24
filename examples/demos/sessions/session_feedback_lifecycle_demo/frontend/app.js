const state = {
  sessionId: "demo_session",
  datasetName: "session_feedback_weights_demo",
  graph: { nodes: [], edges: [] },
  graphTopologyKey: null,
  changedNodeIds: new Set(),
  changedEdgeIds: new Set(),
  latestQaId: null,
  simulation: null,
  graphViz: null,
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
const resetLayoutBtn = document.getElementById("resetLayoutBtn");
const showDocsBtn = document.getElementById("showDocsBtn");
const guideBtn = document.getElementById("guideBtn");
const guidePanel = document.getElementById("guidePanel");
const guideBackdrop = document.getElementById("guideBackdrop");
const closeGuideBtn = document.getElementById("closeGuideBtn");
const guideStepStatus = document.getElementById("guideStepStatus");
const guideStepButtons = Array.from(document.querySelectorAll(".guide-step-btn"));
const docsModal = document.getElementById("docsModal");
const docsBackdrop = document.getElementById("docsBackdrop");
const closeDocsBtn = document.getElementById("closeDocsBtn");
const docsMeta = document.getElementById("docsMeta");
const docsList = document.getElementById("docsList");

const runDemoBtn = document.getElementById("runDemoBtn");
const runMemifyBtn = document.getElementById("runMemifyBtn");
const sendForm = document.getElementById("sendForm");
const questionInput = document.getElementById("questionInput");
const topKSlider = document.getElementById("topKSlider");
const topKValue = document.getElementById("topKValue");
let activeNodeSelection = null;
let activeEdgeSelection = null;
let loadingPhaseTimer = null;
let loadingPhaseIndex = 0;
const DEFAULT_LEFT_PANE_WIDTH = "62%";
const DEFAULT_GRAPH_ROWS = "minmax(0, 1fr) 8px minmax(0, 1fr)";
const DEFAULT_CHAT_ROWS = "minmax(0, 1fr) 8px minmax(0, 1fr)";
let guideCurrentStep = 1;
let guideFocusedElement = null;
let guideBeacon = null;
let messageQueue = Promise.resolve();

function setBusy(isBusy) {
  [runDemoBtn, runMemifyBtn, showDocsBtn, questionInput, topKSlider]
    .filter(Boolean)
    .forEach((el) => {
      el.disabled = isBusy;
    });
}

function getTopK() {
  const raw = Number(topKSlider?.value ?? 5);
  const value = Number.isFinite(raw) ? Math.round(raw) : 5;
  return Math.max(1, Math.min(10, value));
}

function renderLoadingSteps(steps) {
  if (!loadingSteps) {
    return;
  }
  loadingSteps.innerHTML = "";
  (steps || []).forEach((step) => {
    const li = document.createElement("li");
    li.textContent = `${step.name}: ${step.detail}`;
    loadingSteps.appendChild(li);
  });
}

function addMessage(type, text, qaId = null) {
  messageQueue = messageQueue.then(() => addMessageTyped(type, text, qaId, 12));
  return messageQueue;
}

async function addFeedbackChatTurn({ score, text, qaId }) {
  const feedbackText = (text || "").trim();
  const line = feedbackText
    ? `Feedback (score ${score}/5): ${feedbackText}`
    : `Feedback (score ${score}/5)`;
  await addMessage("user", line, qaId || null);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function addMessageTyped(type, text, qaId = null, charDelayMs = 12) {
  const node = document.createElement("div");
  node.className = `message ${type}`;
  node.textContent = "";
  messages.appendChild(node);

  const content = String(text || "");
  for (let i = 0; i < content.length; i += 1) {
    node.textContent += content[i];
    if (i % 2 === 0) {
      messages.scrollTop = messages.scrollHeight;
    }
    // Slower cadence for punctuation to feel more natural.
    const char = content[i];
    if (char === "." || char === "," || char === ":" || char === ";" || char === "!") {
      await sleep(charDelayMs + 18);
    } else if (char === "\n") {
      await sleep(charDelayMs + 10);
    } else {
      await sleep(charDelayMs);
    }
  }

  if (qaId) {
    const meta = document.createElement("div");
    meta.className = "qa-meta";
    meta.textContent = `qa_id: ${qaId}`;
    node.appendChild(meta);
  }

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

function fitTransformForNodes(width, height, nodes, padding = 120) {
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

function topologyKey(graph) {
  const nodeIds = (graph?.nodes || []).map((n) => String(n.id)).sort().join("|");
  const edgeIds = (graph?.edges || []).map((e) => String(e.id)).sort().join("|");
  return `${nodeIds}::${edgeIds}`;
}

function updateGraphVisualsInPlace(graph, changed = { nodes: [], edges: [] }) {
  const viz = state.graphViz;
  if (!viz) return;

  state.graph = graph;
  graphStats.textContent = `${graph.stats.node_count} nodes | ${graph.stats.edge_count} edges`;
  sessionMeta.textContent = `session: ${state.sessionId} | dataset: ${state.datasetName}`;
  sessionIdBadge.textContent = state.sessionId;

  const changedNodeIds = new Set((changed?.nodes || []).map((item) => String(item.id)));
  const changedEdgeIds = new Set((changed?.edges || []).map((item) => String(item.id)));

  const nodeById = new Map(
    (graph.nodes || []).map((node) => [
      node.id,
      {
        ...node,
        display_label: nodeDisplayLabel(node),
      },
    ])
  );
  const edgeById = new Map(
    (graph.edges || []).map((edge) => [
      edge.id,
      {
        ...edge,
        display_label: edgeDisplayLabel(edge),
      },
    ])
  );

  viz.nodes.forEach((node) => {
    const latest = nodeById.get(node.id);
    if (!latest) return;
    node.feedback_weight = latest.feedback_weight;
    node.properties = latest.properties || {};
    node.display_label = latest.display_label;
    node.label = latest.label;
    node.type = latest.type;
    viz.radiusById.set(node.id, nodeRadius(node.feedback_weight));
  });

  viz.links.forEach((edge) => {
    const latest = edgeById.get(edge.id);
    if (!latest) return;
    edge.feedback_weight = latest.feedback_weight;
    edge.properties = latest.properties || {};
    edge.display_label = latest.display_label;
    edge.relation = latest.relation;
  });

  viz.nodeGroup
    .interrupt("weights")
    .attr("r", (d) => nodeRadius(d.feedback_weight))
    .transition("weights")
    .duration(620)
    .ease(d3.easeCubicOut)
    .attr("r", (d) => nodeRadius(d.feedback_weight))
    .attr("fill", (d) => colorByWeight(d.feedback_weight));

  viz.edgeGroup
    .interrupt("weights")
    .attr("stroke", (d) => colorByWeight(d.feedback_weight))
    .attr(
      "stroke-opacity",
      (d) => 0.62 + Math.max(0, Math.min(1, Number(d.feedback_weight ?? 0.5))) * 0.28
    )
    .transition("weights")
    .duration(620)
    .ease(d3.easeCubicOut)
    .attr("stroke", (d) => colorByWeight(d.feedback_weight))
    .attr(
      "stroke-opacity",
      (d) => 0.62 + Math.max(0, Math.min(1, Number(d.feedback_weight ?? 0.5))) * 0.28
    )
    .attr("stroke-width", (d) => edgeWidth(d.feedback_weight));

  // Emphasize changed elements with a short bump animation.
  if (changedNodeIds.size) {
    viz.nodeGroup
      .filter((d) => changedNodeIds.has(String(d.id)))
      .interrupt("bump")
      .transition("bump")
      .duration(220)
      .ease(d3.easeBackOut.overshoot(2.6))
      .attr("r", (d) => nodeRadius(d.feedback_weight) + 12)
      .attr("stroke-width", 4.6)
      .attr("stroke", "#ffffff")
      .attr("fill", (d) => d3.interpolateRgb(colorByWeight(d.feedback_weight), "#ffffff")(0.38))
      .attr("filter", "drop-shadow(0 0 14px rgba(255,255,255,0.88))")
      .transition("bump")
      .duration(620)
      .ease(d3.easeCubicOut)
      .attr("r", (d) => nodeRadius(d.feedback_weight))
      .attr("stroke-width", 1.35)
      .attr("stroke", "rgba(165, 80, 255, 0.48)")
      .attr("fill", (d) => colorByWeight(d.feedback_weight))
      .attr("filter", null);
  }

  if (changedEdgeIds.size) {
    viz.edgeGroup
      .filter((d) => changedEdgeIds.has(String(d.id)))
      .interrupt("bump")
      .transition("bump")
      .duration(220)
      .ease(d3.easeBackOut.overshoot(2.2))
      .attr("stroke-width", (d) => edgeWidth(d.feedback_weight) + 4.4)
      .attr("stroke-opacity", 1)
      .attr("stroke", "#ffffff")
      .attr("filter", "drop-shadow(0 0 10px rgba(255,255,255,0.9))")
      .transition("bump")
      .duration(620)
      .ease(d3.easeCubicOut)
      .attr("stroke-width", (d) => edgeWidth(d.feedback_weight))
      .attr(
        "stroke-opacity",
        (d) => 0.62 + Math.max(0, Math.min(1, Number(d.feedback_weight ?? 0.5))) * 0.28
      );
    viz.edgeGroup
      .filter((d) => changedEdgeIds.has(String(d.id)))
      .transition("bump")
      .duration(620)
      .ease(d3.easeCubicOut)
      .attr("stroke", (d) => colorByWeight(d.feedback_weight))
      .attr("filter", null);
  }

  viz.nodeGroup.select("title").text((d) => `${d.display_label}\nweight=${d.feedback_weight}`);
  viz.nodeLabel.text((d) => shortLabel(String(d.display_label || "node"), 30));
  viz.edgeLabel.text((d) => shortLabel(d.display_label, 26));
  viz.updateLabelVisibility();

  if (activeNodeSelection) {
    const selected = viz.nodes.find((node) => node.id === activeNodeSelection);
    if (selected) renderSelection("node", selected);
  } else if (activeEdgeSelection) {
    const selected = viz.links.find((edge) => edge.id === activeEdgeSelection);
    if (selected) renderSelection("edge", selected);
  }
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

function setGuideStep(step) {
  const maxSteps = Math.max(1, guideStepButtons.length);
  guideCurrentStep = Math.max(1, Math.min(maxSteps, Number(step) || 1));
  if (guideStepStatus) {
    guideStepStatus.textContent = `Current step: ${guideCurrentStep}/${maxSteps}`;
  }
  guideStepButtons.forEach((btn) => {
    const btnStep = Number(btn.dataset.step || "0");
    btn.classList.toggle("active", btnStep === guideCurrentStep);
  });
}

function openGuidePanel() {
  if (!guidePanel) return;
  guidePanel.hidden = false;
  document.body.classList.add("guide-open");
}

function closeGuidePanel() {
  if (!guidePanel) return;
  guidePanel.hidden = true;
  document.body.classList.remove("guide-open");
  clearGuideFocus();
}

function clearGuideFocus() {
  if (guideFocusedElement) {
    guideFocusedElement.classList.remove("guide-focus");
    guideFocusedElement = null;
  }
  if (guideBeacon) {
    guideBeacon.style.display = "none";
  }
}

function ensureGuideBeacon() {
  if (guideBeacon) return guideBeacon;
  guideBeacon = document.createElement("div");
  guideBeacon.className = "guide-beacon";
  guideBeacon.style.display = "none";
  document.body.appendChild(guideBeacon);
  return guideBeacon;
}

function positionGuideBeacon(target) {
  const beacon = ensureGuideBeacon();
  const rect = target.getBoundingClientRect();
  const pad = 8;
  beacon.style.left = `${Math.max(4, rect.left - pad)}px`;
  beacon.style.top = `${Math.max(4, rect.top - pad)}px`;
  beacon.style.width = `${Math.max(16, rect.width + pad * 2)}px`;
  beacon.style.height = `${Math.max(16, rect.height + pad * 2)}px`;
  beacon.style.display = "block";
}

function focusGuideTarget(targetId) {
  clearGuideFocus();
  if (!targetId) return;
  const target = document.getElementById(targetId);
  if (!target) return;
  target.classList.add("guide-focus");
  guideFocusedElement = target;
  target.scrollIntoView({ behavior: "smooth", block: "center", inline: "center" });
  window.setTimeout(() => {
    if (guideFocusedElement === target) {
      positionGuideBeacon(target);
    }
  }, 120);
}

function openDocsModal() {
  if (!docsModal) return;
  docsModal.hidden = false;
}

function closeDocsModal() {
  if (!docsModal) return;
  docsModal.hidden = true;
}

function renderIngestedDocs(payload) {
  if (!docsMeta || !docsList) return;
  const docs = payload?.documents || [];
  docsMeta.textContent = `dataset: ${payload?.dataset_name || state.datasetName} | documents: ${docs.length}`;
  docsList.innerHTML = "";
  if (!docs.length) {
    docsList.textContent = "No ingested documents available.";
    return;
  }

  const escapeHtml = (value) =>
    String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");

  docs.forEach((doc) => {
    const item = document.createElement("article");
    item.className = "docs-item";
    item.innerHTML = `
      <div class="docs-item-head">
        <span>Document ${doc.id}</span>
        <span>${doc.char_count} chars</span>
      </div>
      <pre>${escapeHtml(doc.text || "")}</pre>
    `;
    docsList.appendChild(item);
  });
}

function startInitLoadingPhases() {
  if (!loadingPhase) return;
  loadingPhase.textContent = "Preparing your demo...";
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

function clearLoadingGateDetails() {
  const node = document.getElementById("loadingGateDetails");
  if (node) {
    node.remove();
  }
}

function showLoadingGateDetails(text) {
  const loadingCard = document.querySelector(".loading-card");
  if (!loadingCard) return;
  clearLoadingGateDetails();
  const details = document.createElement("div");
  details.id = "loadingGateDetails";
  details.className = "loading-gate-details";
  details.textContent = text;
  loadingCard.appendChild(details);
}

async function validateRequiredDemoSettings() {
  const gate = await api("/demo/config_gate");
  if (gate?.ok) {
    clearLoadingGateDetails();
    return;
  }

  const mismatchLines = (gate?.mismatches || []).map(
    (entry) => `- ${entry.name}: current=${entry.current || "(empty)"} expected=${entry.expected}`
  );
  const details = [
    "Demo blocked: set these env vars and restart the server:",
    "CACHING=True",
    "AUTO_FEEDBACK=True",
    "CACHE_BACKEND=fs",
    "",
    ...(mismatchLines.length ? ["Current mismatches:", ...mismatchLines] : []),
  ].join("\n");
  stopInitLoadingPhases("Configuration required");
  showLoadingGateDetails(details);
  throw new Error("Required environment settings are not configured.");
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
  const nextTopologyKey = topologyKey(graph);
  if (state.graphViz && state.graphTopologyKey === nextTopologyKey) {
    updateGraphVisualsInPlace(graph, changed);
    return;
  }

  state.graph = graph;
  state.graphTopologyKey = nextTopologyKey;

  graphStats.textContent = `${graph.stats.node_count} nodes | ${graph.stats.edge_count} edges`;
  sessionMeta.textContent = `session: ${state.sessionId} | dataset: ${state.datasetName}`;
  sessionIdBadge.textContent = state.sessionId;

  if (state.simulation) {
    state.simulation.stop();
  }

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
  const spreadRadiusX = Math.min(width, height) * 0.46;
  const spreadRadiusY = Math.min(width, height) * 0.38;
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
          const basePadding = 108;
          const weightPadding = edgeWidth(d.feedback_weight) * 3.5;
          return sourceR + targetR + basePadding + weightPadding;
        })
    )
    .force("charge", d3.forceManyBody().strength((node) => -(460 + getRadius(node) * 30)))
    .force("collide", d3.forceCollide().radius((node) => getRadius(node) + 34).iterations(2))
    .force("x", d3.forceX((d) => targetById.get(d.id)?.x ?? centerX).strength(0.04))
    .force("y", d3.forceY((d) => targetById.get(d.id)?.y ?? centerY).strength(0.046))
    .force("center", d3.forceCenter(centerX, centerY))
    .alpha(1.0)
    .alphaDecay(0.04)
    .velocityDecay(0.44);

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
    .attr("stroke-width", 2.2)
    .attr("opacity", 0.94)
    .attr("pointer-events", "none")
    .text((d) => shortLabel(String(d.display_label || "node"), 30));

  const updateLabelVisibility = () => {
    nodeLabel
      .attr("x", (d) => d.x)
      .attr("y", (d) => {
        const side = stableHash(d.id) % 2 === 0 ? 1 : -1;
        return d.y + side * (nodeRadius(d.feedback_weight) + 16);
      })
      .attr("opacity", 0.94);

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

  simulation.on("end", () => {
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
    const transform = fitTransformForNodes(width, height, nodes, 128);
    svg.transition().duration(380).call(zoom.transform, transform);
    updateLabelVisibility();
  };
  setTimeout(fitOnce, 650);

  state.graphViz = {
    svg,
    zoom,
    nodes,
    links,
    radiusById,
    nodeGroup,
    edgeGroup,
    edgeLabel,
    nodeLabel,
    updateLabelVisibility,
    fitGraphView: () => {
      const transform = fitTransformForNodes(width, height, nodes, 128);
      svg.transition().duration(420).call(zoom.transform, transform);
      updateLabelVisibility();
    },
  };
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
  await validateRequiredDemoSettings();

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
    const total = rect.height - 8;
    if (total <= 0) return;
    const topPx = event.clientY - rect.top;
    const topRatio = Math.max(0.2, Math.min(0.8, topPx / total));
    const bottomRatio = 1 - topRatio;
    document.documentElement.style.setProperty(
      "--graph-body-rows",
      `minmax(0, ${topRatio}fr) 8px minmax(0, ${bottomRatio}fr)`
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
    const total = rect.height - 8;
    if (total <= 0) return;
    const topPx = event.clientY - rect.top;
    const topRatio = Math.max(0.2, Math.min(0.8, topPx / total));
    const bottomRatio = 1 - topRatio;
    document.documentElement.style.setProperty(
      "--chat-rows",
      `minmax(0, ${topRatio}fr) 8px minmax(0, ${bottomRatio}fr)`
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
    setGuideStep(6);
    setBusy(true);
    showInlineLoading("Running scripted demo flow...");
    addMessage("user", "Running scripted demo flow...");

    const flow = await api("/demo/scripted_flow");
    if (flow.session_id) {
      state.sessionId = flow.session_id;
      sessionIdBadge.textContent = state.sessionId;
    }

    const turns = flow.questions || [];
    for (const [index, turn] of turns.entries()) {
      const question = String(turn.question || "");
      if (!question) continue;

      await addMessage("user", `Q${index + 1}: ${question}`);
      showInlineLoading(`Searching answer ${index + 1}/${turns.length}...`);
      const sendResult = await api("/demo/send", "POST", {
        question,
        session_id: state.sessionId,
        top_k: getTopK(),
      });
      if (sendResult.session_id) {
        state.sessionId = sendResult.session_id;
        sessionIdBadge.textContent = state.sessionId;
      }

      await addMessage("system", sendResult.answer, sendResult.qa_id || null);

      if (sendResult.qa_id) {
        await addFeedbackChatTurn({
          score: Number(turn.feedback_score ?? 3),
          text: String(turn.feedback_text || "Scripted demo feedback"),
          qaId: sendResult.qa_id,
        });

        showInlineLoading(`Applying feedback ${index + 1}/${turns.length}...`);
        await api("/demo/feedback", "POST", {
          session_id: state.sessionId,
          qa_id: sendResult.qa_id,
          feedback_score: Number(turn.feedback_score ?? 3),
          feedback_text: String(turn.feedback_text || "Scripted demo feedback"),
        });

        showInlineLoading(`Memify ${index + 1}/${turns.length}...`);
        const perTurnMemify = await api("/demo/run_memify_pipeline", "POST", {
          session_id: state.sessionId,
        });
        if (perTurnMemify.session_id) {
          state.sessionId = perTurnMemify.session_id;
          sessionIdBadge.textContent = state.sessionId;
        }
        await refreshStateAndGraph({
          nodes: perTurnMemify.deltas.changed_nodes,
          edges: perTurnMemify.deltas.changed_edges,
        });
        addMessage(
          "system",
          `Feedback applied for ${sendResult.qa_id}. Memify updated nodes=${perTurnMemify.deltas.summary.changed_node_count}, edges=${perTurnMemify.deltas.summary.changed_edge_count}`
        );
      } else {
        await refreshStateAndGraph();
      }
      await sleep(180);
    }

    addMessage(
      "system",
      "Scripted demo complete. Feedback and memify were applied after each QA turn."
    );
  } catch (error) {
    addMessage("system", `Run_demo failed: ${error.message}`);
  } finally {
    setBusy(false);
    hideInlineLoading();
  }
});

runMemifyBtn.addEventListener("click", async () => {
  try {
    setGuideStep(7);
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
    if (guideCurrentStep < 2) {
      setGuideStep(2);
    }
    setBusy(true);
    showInlineLoading("Searching...");
    addMessage("user", question);

    const result = await api("/demo/send", "POST", {
      question,
      session_id: state.sessionId,
      top_k: getTopK(),
    });
    if (result.session_id) {
      state.sessionId = result.session_id;
      sessionIdBadge.textContent = state.sessionId;
    }

    if (result.qa_id) {
      state.latestQaId = result.qa_id;
    }

    addMessage("system", result.answer, result.qa_id || null);

    if (result.auto_feedback && (result.auto_feedback.feedback_score !== null || result.auto_feedback.feedback_text)) {
      await addFeedbackChatTurn({
        score: result.auto_feedback.feedback_score ?? "n/a",
        text: result.auto_feedback.feedback_text ?? "",
        qaId: result.qa_id || null,
      });
    }

    const memifyResult = await api("/demo/run_memify_pipeline", "POST", {
      session_id: state.sessionId,
    });
    const changed = {
      nodes: memifyResult.deltas.changed_nodes,
      edges: memifyResult.deltas.changed_edges,
    };
    await refreshStateAndGraph(changed);
    if (
      memifyResult?.deltas?.summary?.changed_node_count > 0 ||
      memifyResult?.deltas?.summary?.changed_edge_count > 0
    ) {
      addMessage(
        "system",
        `Memify updated nodes=${memifyResult.deltas.summary.changed_node_count}, edges=${memifyResult.deltas.summary.changed_edge_count}`
      );
    }

    questionInput.value = "";
  } catch (error) {
    addMessage("system", `Send failed: ${error.message}`);
  } finally {
    setBusy(false);
    hideInlineLoading();
  }
});

resetLayoutBtn?.addEventListener("click", () => {
  setGuideStep(7);
  document.documentElement.style.setProperty("--left-pane-width", DEFAULT_LEFT_PANE_WIDTH);
  document.documentElement.style.setProperty("--graph-body-rows", DEFAULT_GRAPH_ROWS);
  document.documentElement.style.setProperty("--chat-rows", DEFAULT_CHAT_ROWS);

  if (state.graphViz?.fitGraphView) {
    state.graphViz.fitGraphView();
  }

  if (state.simulation) {
    state.simulation.alpha(0.22).restart();
  }
});

guideBtn?.addEventListener("click", () => {
  setGuideStep(guideCurrentStep);
  openGuidePanel();
  const activeBtn = guideStepButtons.find((btn) => Number(btn.dataset.step || "0") === guideCurrentStep);
  focusGuideTarget(activeBtn?.dataset.target || "showDocsBtn");
});

closeGuideBtn?.addEventListener("click", closeGuidePanel);
guideBackdrop?.addEventListener("click", closeGuidePanel);

window.addEventListener("resize", () => {
  if (guideFocusedElement) {
    positionGuideBeacon(guideFocusedElement);
  }
});

window.addEventListener(
  "scroll",
  () => {
    if (guideFocusedElement) {
      positionGuideBeacon(guideFocusedElement);
    }
  },
  true
);

guideStepButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    const step = Number(btn.dataset.step || "1");
    setGuideStep(step);
    focusGuideTarget(btn.dataset.target || "");
  });
});

topKSlider?.addEventListener("input", () => {
  const value = getTopK();
  if (topKValue) {
    topKValue.textContent = String(value);
  }
});

showDocsBtn?.addEventListener("click", async () => {
  try {
    showInlineLoading("Loading ingested docs...");
    const payload = await api("/demo/ingested_documents");
    renderIngestedDocs(payload);
    openDocsModal();
  } catch (error) {
    addMessage("system", `Failed to load ingested docs: ${error.message}`);
  } finally {
    hideInlineLoading();
  }
});

closeDocsBtn?.addEventListener("click", closeDocsModal);
docsBackdrop?.addEventListener("click", closeDocsModal);

(async function boot() {
  if (docsModal) {
    docsModal.hidden = true;
  }
  if (guidePanel) {
    guidePanel.hidden = true;
  }
  setGuideStep(1);
  if (topKValue) {
    topKValue.textContent = String(getTopK());
  }
  setupMainResizer();
  setupGraphDetailsResizer();
  setupChatResizer();
  try {
    await initializeDemo();
    openGuidePanel();
    const firstStepBtn = guideStepButtons.find((btn) => Number(btn.dataset.step || "0") === 1);
    focusGuideTarget(firstStepBtn?.dataset.target || "showDocsBtn");
  } catch (error) {
    if (loadingSteps) {
      loadingSteps.innerHTML = `<li>Initialization failed: ${error.message}</li>`;
    }
    addMessage("system", `Initialization failed: ${error.message}`);
  }
})();
