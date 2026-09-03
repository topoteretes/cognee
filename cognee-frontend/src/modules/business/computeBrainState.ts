import { hsl } from "d3-color";
import type { BusinessGraphNode, BusinessGraphLink } from "./types";
import type { BusinessEntity, SemanticLink, TypeNode, TypeLink, Anchor, BrainState } from "./sceneTypes";
import { topImportanceCut } from "./canvas/businessEntityLayer";
import { computeConnectedIds } from "./canvas/businessGraphFocus";

// A synthetic source for content ingested with no node_set at all — the
// product's own manual-upload path (rememberData.ts) never sends node_set,
// so a directly-uploaded file's entities/documents used to have NO source
// at all: no Sources tile, no hull, no anchor (they defaulted to world
// origin, per anchorOf), invisible to a focus lens. Attributing them here
// instead of returning [] means every downstream consumer of setsOf —
// coloring, hulls, filaments, plumbing, the hub metric, the focus lens —
// picks this up for free, without each needing its own "no source" branch
// (COG-6233).
export const UNCATEGORIZED_SOURCE = "uncategorized";

// A node's source(s) — the true source dimension when node_sets exist
// (Slack stamps "slack", the demo stamps crm/marketing/…), falling back to
// UNCATEGORIZED_SOURCE for content ingested without one.
export function setsOf(node: BusinessGraphNode): string[] {
  const belongs = node.belongs_to_set;
  if (Array.isArray(belongs) && belongs.length) return belongs;
  const raw = node.source_node_set;
  if (raw) {
    const parsed = String(raw)
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    if (parsed.length) return parsed;
  }
  return [UNCATEGORIZED_SOURCE];
}

const SESSION_SET_RE = /^(session_learnings|user_sessions_from_cache|agent_trace_feedbacks)/;
export function isSessionSetName(name: string): boolean {
  return SESSION_SET_RE.test(name);
}

const SESSION_SET_BASE_LABELS: Record<string, string> = {
  session_learnings: "session learnings",
  user_sessions_from_cache: "user sessions",
  agent_trace_feedbacks: "agent feedback",
};

// A session set's raw name is machine-assembled — base, agent, then a full
// UUID ("session_learnings:claude_d868a78b-8dff-…") — and reads as noise in
// every UI spot that shows sources. This derives the human version ("session
// learnings · claude · d868a7"): base label, agent, and just enough of the
// id to tell two of the same agent's sessions apart. The RAW name stays the
// key everywhere (focus lens, anchors, colors all key by it) — this is
// display only. Non-session sources pass through untouched.
export function sourceLabel(name: string): string {
  const base = Object.keys(SESSION_SET_BASE_LABELS).find((k) => name.startsWith(k));
  if (!base) return name;
  const rest = name.slice(base.length).replace(/^[:_\-\s]+/, "");
  if (!rest) return SESSION_SET_BASE_LABELS[base];
  const agent = rest.split(/[:_]/)[0];
  const id = rest.slice(agent.length).replace(/^[:_\-\s]+/, "").replace(/-/g, "");
  const shortId = id.slice(0, 6);
  return shortId
    ? `${SESSION_SET_BASE_LABELS[base]} · ${agent} · ${shortId}`
    : `${SESSION_SET_BASE_LABELS[base]} · ${agent}`;
}

// What a source's tooltip should say. sourceLabel truncates a session set's
// uuid to six characters to fit the rail, so using it for BOTH the visible
// text and the tooltip made hovering repeat what was already on screen and
// left the full identifier unreachable anywhere in the UI — the readable
// label leads, the raw name follows it when the two differ.
export function sourceTooltipLabel(name: string): string {
  const label = sourceLabel(name);
  return label === name ? name : `${label} · ${name}`;
}

// Session-memory sets wear the AGENT color family (amber): this memory
// exists because agents talked. Shades keep the three layers apart.
const AGENT_SET_SHADES: Record<string, string> = {
  session_learnings: "#F5A83C",
  user_sessions_from_cache: "#E08E1B",
  agent_trace_feedbacks: "#FFC46B",
};

function hueDistance(a: number, b: number): number {
  const d = Math.abs(a - b) % 360;
  return d > 180 ? 360 - d : d;
}

// Guards the answer-amber encoding: nudges any source color too close to
// amber's hue so "amber = live signal" stays unambiguous.
export function colorForSet(name: string, i: number, colorsMap?: Record<string, string>): string {
  let hex = (colorsMap || {})[name];
  if (!hex) {
    const hue = (i * 137.508 + 200) % 360;
    return hsl(hue, 0.55, 0.62).formatHex();
  }
  const color = hsl(hex);
  if (!isNaN(color.h) && hueDistance(color.h, 35) < 15) {
    color.h = (color.h + 40) % 360;
    hex = color.formatHex();
  }
  return hex;
}

interface SplitSets {
  sets: BusinessGraphNode[];
  sessionSetNodes: BusinessGraphNode[];
}

function splitNodeSets(nodesArr: BusinessGraphNode[]): SplitSets {
  const nodeSets = nodesArr.filter((n) => n.type === "NodeSet" && n.name);
  return {
    sets: nodeSets.filter((n) => !isSessionSetName(n.name as string)),
    sessionSetNodes: nodeSets.filter((n) => isSessionSetName(n.name as string)),
  };
}

// Union, not either/or: a node_set with no explicit NodeSet node (backend
// didn't create one for it) must still surface via its entities' own tags —
// otherwise it silently disappears the moment ANY other node_set in the same
// graph does have a NodeSet node (COG-6233).
export function deriveSourceNames(nodesArr: BusinessGraphNode[], entities: BusinessGraphNode[]): string[] {
  const { sets } = splitNodeSets(nodesArr);
  return [...new Set([...sets.map((n) => n.name as string), ...entities.flatMap(setsOf)])];
}

function buildSourceColors(
  srcNames: string[],
  sessionSetNodes: BusinessGraphNode[],
  colorsMap?: Record<string, string>,
): Record<string, string> {
  const colors: Record<string, string> = {};
  srcNames.forEach((s, i) => {
    colors[s] = colorForSet(s, i, colorsMap);
  });
  sessionSetNodes.forEach((n) => {
    const name = n.name as string;
    const base = Object.keys(AGENT_SET_SHADES).find((k) => name.indexOf(k) === 0);
    colors[name] = (base && AGENT_SET_SHADES[base]) || "#F5A83C";
  });
  return colors;
}

function countBySet(nodes: BusinessGraphNode[]): Record<string, number> {
  const counts: Record<string, number> = {};
  nodes.forEach((n) => setsOf(n).forEach((s) => { counts[s] = (counts[s] || 0) + 1; }));
  return counts;
}

// Circular layout for sources; session-memory sets get their own arc BELOW
// the sources — without an anchor those entities all pull to (0,0) and bury
// the center.
function buildAnchors(srcNames: string[], sessionNames: string[]): Record<string, Anchor> {
  const anchors: Record<string, Anchor> = {};
  srcNames.forEach((s, i) => {
    const angle = (i / Math.max(srcNames.length, 1)) * Math.PI * 2 - Math.PI / 2;
    const r = srcNames.length > 1 ? 300 : 0;
    anchors[s] = { x: Math.cos(angle) * r * 1.25, y: Math.sin(angle) * r * 0.6 };
  });
  sessionNames.forEach((name, i) => {
    const spread = (i - (sessionNames.length - 1) / 2) * 0.55;
    anchors[name] = { x: Math.sin(spread) * 480, y: 430 + Math.abs(spread) * 60 };
  });
  return anchors;
}

function endId(v: unknown): string {
  return typeof v === "object" && v !== null ? String((v as { id: string }).id) : String(v);
}

function withEndpointIds<T extends BusinessGraphLink>(links: T[]): (T & { _sid: string; _tid: string })[] {
  return links.map((l) => ({ ...l, _sid: endId(l.source), _tid: endId(l.target) }));
}

// The business-model (L0) layer: entity TYPES and how they relate —
// "campaign generates customer", not individual campaigns. Types float at
// the centroid of their members, computed each draw frame by the canvas.
function buildTypeLayer(
  entities: BusinessEntity[],
  links: (BusinessGraphLink & { _sid: string; _tid: string })[],
  byId: Record<string, BusinessGraphNode>,
  semanticLinks: SemanticLink[],
): { typeNodes: TypeNode[]; typeLinks: TypeLink[] } {
  const typeName: Record<string, string> = {};
  links.forEach((l) => {
    const s = byId[l._sid], t = byId[l._tid];
    if (s && t && s.stage === "entity" && t.stage === "type" &&
      (l.relation === "is_a" || l.relation === "instance_of")) {
      typeName[s.id] = t.name as string;
    }
  });

  const buckets: Record<string, TypeNode> = {};
  entities.forEach((n) => {
    const tn = typeName[n.id] || "other";
    const bucket = (buckets[tn] = buckets[tn] || { name: tn, members: [], sets: {} });
    bucket.members.push(n);
    setsOf(n).forEach((s) => { bucket.sets[s] = (bucket.sets[s] || 0) + 1; });
  });

  const links2: Record<string, TypeLink> = {};
  semanticLinks.forEach((l) => {
    const a = typeName[l._sid] || "other", b = typeName[l._tid] || "other";
    if (a === b) return;
    const relation = l.relation || "related";
    const key = `${a}→${b}|${relation}`;
    const slot = (links2[key] = links2[key] || { a, b, relation, count: 0 });
    slot.count += 1;
  });

  return { typeNodes: Object.values(buckets), typeLinks: Object.values(links2) };
}

const HUB_MIN_DEGREE = 2;

// The graph-dashboard "single point of failure" callout: whichever entity
// carries the most connections, and how many distinct sources those
// connections span. Below HUB_MIN_DEGREE the number is noise (any entity
// with one link "wins" by default on a sparse graph), so this returns null
// rather than surface a meaningless hub.
function computeHubInsight(
  semanticLinks: SemanticLink[],
  byId: Record<string, BusinessGraphNode>,
): BrainState["hub"] {
  const degree: Record<string, number> = {};
  const touchedSets: Record<string, Set<string>> = {};
  semanticLinks.forEach((l) => {
    const sSets = setsOf(byId[l._sid] || ({} as BusinessGraphNode));
    const tSets = setsOf(byId[l._tid] || ({} as BusinessGraphNode));
    [l._sid, l._tid].forEach((id) => {
      degree[id] = (degree[id] || 0) + 1;
      const bucket = (touchedSets[id] = touchedSets[id] || new Set());
      sSets.forEach((s) => bucket.add(s));
      tSets.forEach((s) => bucket.add(s));
    });
  });
  let bestId: string | null = null;
  let bestDegree = 0;
  Object.entries(degree).forEach(([id, d]) => {
    if (d > bestDegree) { bestDegree = d; bestId = id; }
  });
  if (!bestId || bestDegree < HUB_MIN_DEGREE) return null;
  const node = byId[bestId];
  if (!node?.name) return null;
  return {
    entityId: bestId,
    name: String(node.name),
    degree: bestDegree,
    sourceCount: touchedSets[bestId]?.size || 0,
  };
}

// Everything the canvas needs about ONE brain, computed from that brain's own
// node/link arrays — switching brains (or toggling a layer) is just calling
// this again with a different merged input.
export function computeBrainState(
  nodesArr: BusinessGraphNode[],
  linksArr: BusinessGraphLink[],
  colorsMap?: Record<string, string>,
): BrainState {
  const byId: Record<string, BusinessGraphNode> = {};
  nodesArr.forEach((n) => { byId[n.id] = n; });
  const links = withEndpointIds(linksArr);

  const entities: BusinessEntity[] = nodesArr
    .filter((n) => n.stage === "entity")
    .map((n) => ({ ...n, x: undefined, y: undefined, fx: null, fy: null, vx: 0, vy: 0 }));
  const entityById: Record<string, BusinessEntity> = {};
  entities.forEach((n) => { entityById[n.id] = n; });
  const docs = nodesArr.filter((n) => n.stage === "document");

  const { sessionSetNodes } = splitNodeSets(nodesArr);
  const sourceNames = deriveSourceNames(nodesArr, entities);
  const setColor = buildSourceColors(sourceNames, sessionSetNodes, colorsMap);

  const semanticLinks = links.filter((l) =>
    l.edge_class === "semantic" && byId[l._sid] && byId[l._tid] &&
    byId[l._sid].stage === "entity" && byId[l._tid].stage === "entity") as SemanticLink[];
  semanticLinks.forEach((l) => {
    const a = setsOf(byId[l._sid]), b = setsOf(byId[l._tid]);
    l._bridge = a.length > 0 && b.length > 0 && !a.some((s) => b.includes(s));
  });

  const docLinks = links.filter((l) => {
    const s = byId[l._sid], t = byId[l._tid];
    return !!s && !!t && (
      (s.stage === "document" && t.stage === "entity") ||
      (s.stage === "entity" && t.stage === "document") ||
      (s.stage === "chunk" && t.stage === "entity")
    );
  }) as SemanticLink[];

  const anchors = buildAnchors(sourceNames, sessionSetNodes.map((n) => n.name as string));
  const { typeNodes, typeLinks } = buildTypeLayer(entities, links, byId, semanticLinks);

  // Records layer inputs: NodeSet and type-stage nodes are meta (the source
  // chips and the L0 schema respectively), not records — what's left is the
  // pipeline's raw material: chunks, documents, summaries, context. First
  // linked entity wins as each record's anchor; a record linking multiple
  // entities near-always links them from one extraction pass, so any of
  // them places it in the right neighborhood.
  const plumbingNodes = nodesArr.filter(
    (n) => n.stage !== "entity" && n.stage !== "type" && n.type !== "NodeSet",
  );
  const plumbingEntityId: Record<string, string> = {};
  links.forEach((l) => {
    const s = byId[l._sid], t = byId[l._tid];
    if (!s || !t) return;
    if (s.stage === "entity" && t.stage !== "entity" && !plumbingEntityId[t.id]) {
      plumbingEntityId[t.id] = s.id;
    } else if (t.stage === "entity" && s.stage !== "entity" && !plumbingEntityId[s.id]) {
      plumbingEntityId[s.id] = t.id;
    }
  });

  return {
    byId,
    entities,
    entityById,
    sourceNames,
    setColor,
    isSessionSet: isSessionSetName,
    setEntityCount: countBySet(entities),
    setDocCount: countBySet(docs),
    setMemberCount: countBySet(nodesArr.filter((n) => n.type !== "NodeSet")),
    semanticLinks,
    docLinks,
    anchors,
    typeNodes,
    typeLinks,
    importanceMax: entities.reduce((max, n) => Math.max(max, n.importance || 0), 1),
    plumbingNodes,
    plumbingEntityId,
    importanceCut: topImportanceCut(entities),
    connectedIds: computeConnectedIds(semanticLinks),
    hub: computeHubInsight(semanticLinks, byId),
  };
}
