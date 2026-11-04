import { hsl } from "d3-color";
import type { BusinessGraphNode, BusinessGraphLink } from "./types";
import type { BusinessEntity, SemanticLink, TypeNode, TypeLink, BrainState } from "./sceneTypes";
import { topImportanceCut } from "./canvas/businessEntityLayer";
import { computeConnectedIds } from "./canvas/businessGraphFocus";
import { isCodeNode, normalizeCodeNodes } from "./codeGraph";
import { endId } from "./linkEndpoint";
import { normalizeStructuredNodes, structuredTypeName } from "./structuredGraph";
import { groupStructuredSources } from "./structuredSources";
import { assignTypeColors, createSourceEncoding, createTypeEncoding } from "./colorEncoding";
import { resolveLayoutGroups } from "./layoutGroups";
import { buildLayoutAnchors } from "./layoutAnchors";

// A synthetic source for content ingested with no node_set at all — the
// product's own manual-upload path (rememberData.ts) never sends node_set,
// so a directly-uploaded file's entities/documents used to have NO source
// at all: no Sources tile, no hull, no anchor (they defaulted to world
// origin, per buildLayoutAnchors), invisible to a focus lens. Attributing them here
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
  // A session set no entity is tagged with yet still needs a color of its own.
  // It gets one like any other source: amber is kept for what is live now.
  sessionSetNodes.forEach((n, i) => {
    const name = n.name as string;
    if (!colors[name]) colors[name] = colorForSet(name, srcNames.length + i, colorsMap);
  });
  return colors;
}

function countBySet(nodes: BusinessGraphNode[]): Record<string, number> {
  const counts: Record<string, number> = {};
  nodes.forEach((n) => setsOf(n).forEach((s) => { counts[s] = (counts[s] || 0) + 1; }));
  return counts;
}

// How many of `links` touch the entity, for the panel and tooltip counts.
export function countLinksTouching(links: SemanticLink[], id: string): number {
  return links.filter((l) => l._sid === id || l._tid === id).length;
}

function withEndpointIds<T extends BusinessGraphLink>(links: T[]): (T & { _sid: string; _tid: string })[] {
  return links.map((l) => ({ ...l, _sid: endId(l.source), _tid: endId(l.target) }));
}

// The business-model (L0) layer: entity TYPES and how they relate —
// "campaign generates customer", not individual campaigns. Types float at
// the centroid of their members, computed each draw frame by the canvas.
// `typeEdges` is the drawn graph (graphLinks), not insight links: for a
// structured dataset the schema (rows belong to a table) is the model, so
// those edges are kept here on purpose even though insights exclude them.
function buildTypeLayer(
  entities: BusinessEntity[],
  links: (BusinessGraphLink & { _sid: string; _tid: string })[],
  byId: Record<string, BusinessGraphNode>,
  typeEdges: SemanticLink[],
): { typeNodes: TypeNode[]; typeLinks: TypeLink[]; entityType: Record<string, string> } {
  const typeName: Record<string, string> = {};
  links.forEach((l) => {
    const s = byId[l._sid], t = byId[l._tid];
    if (s && t && s.stage === "entity" && t.stage === "type" &&
      (l.relation === "is_a" || l.relation === "instance_of")) {
      typeName[s.id] = t.name as string;
    }
  });

  // entity_type is present on EVERY node since cognee 1.6.1 (falling back to
  // the node's own class name — "Entity" — when no is_a edge resolved a real
  // type), not only on code nodes. Trusting it for an ordinary entity would
  // surface that literal "Entity" as its own Types-rail tile on every graph;
  // a code node has no such ambiguity, entity_type there always carries its
  // actual class (CodeSymbol, CodeModule, …), so this only reads it there.
  function typeOf(id: string): string {
    if (typeName[id]) return typeName[id];
    const node = byId[id];
    const structuredType = node && structuredTypeName(node);
    if (structuredType) return structuredType;
    if (node && isCodeNode(node) && node.entity_type) return node.entity_type;
    return "other";
  }

  const buckets: Record<string, TypeNode> = {};
  const entityType: Record<string, string> = {};
  entities.forEach((n) => {
    const tn = typeOf(n.id);
    entityType[n.id] = tn;
    const bucket = (buckets[tn] = buckets[tn] || { name: tn, members: [], sets: {} });
    bucket.members.push(n);
    setsOf(n).forEach((s) => { bucket.sets[s] = (bucket.sets[s] || 0) + 1; });
  });

  const links2: Record<string, TypeLink> = {};
  typeEdges.forEach((l) => {
    const a = typeOf(l._sid), b = typeOf(l._tid);
    if (a === b) return;
    const relation = l.relation || "related";
    const key = `${a}→${b}|${relation}`;
    const slot = (links2[key] = links2[key] || { a, b, relation, count: 0 });
    slot.count += 1;
  });

  return { typeNodes: Object.values(buckets), typeLinks: Object.values(links2), entityType };
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
  // Code and structured data have no LLM entity stage. Normalize before
  // deriving the scene so the canvas, filters and empty state agree.
  const nodes = normalizeStructuredNodes(normalizeCodeNodes(groupStructuredSources(nodesArr, linksArr)));
  const byId: Record<string, BusinessGraphNode> = {};
  nodes.forEach((n) => { byId[n.id] = n; });
  const links = withEndpointIds(linksArr);

  const entities: BusinessEntity[] = nodes
    .filter((n) => n.stage === "entity")
    .map((n) => ({ ...n, x: undefined, y: undefined, fx: null, fy: null, vx: 0, vy: 0 }));
  const entityById: Record<string, BusinessEntity> = {};
  entities.forEach((n) => { entityById[n.id] = n; });
  const docs = nodes.filter((n) => n.stage === "document");

  const { sessionSetNodes } = splitNodeSets(nodes);
  const sourceNames = deriveSourceNames(nodes, entities);
  const setColor = buildSourceColors(sourceNames, sessionSetNodes, colorsMap);

  // graphLinks is what the canvas draws and navigates: semantic edges plus
  // any edge touching a structured node, because their wiring is mostly
  // structural (is_row_of, is_part_of) and would otherwise drop out, leaving
  // a structured node's neighbours unreachable. semanticLinks, which feeds
  // insights, drops every structured endpoint: a table is linked to each of
  // its rows, and that is schema, not a business relationship.
  const graphLinks = links.filter((l) =>
    byId[l._sid]?.stage === "entity" && byId[l._tid]?.stage === "entity" &&
    (l.edge_class === "semantic" ||
      structuredTypeName(byId[l._sid]) || structuredTypeName(byId[l._tid]))) as SemanticLink[];
  const semanticLinks = graphLinks.filter((l) =>
    l.edge_class === "semantic" &&
    !structuredTypeName(byId[l._sid]) && !structuredTypeName(byId[l._tid]));
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

  const { typeNodes, typeLinks, entityType } = buildTypeLayer(entities, links, byId, graphLinks);
  const { groupOf, groupOrder } = resolveLayoutGroups(entities, entityType);
  const { anchors, entityAnchors } = buildLayoutAnchors(
    entities.map((n) => ({ id: n.id, sets: setsOf(n), group: groupOf[n.id] })),
    groupOrder,
    sourceNames,
    sessionSetNodes.map((n) => n.name as string),
    isSessionSetName,
  );

  // Records layer inputs: NodeSet and type-stage nodes are meta (the source
  // chips and the L0 schema respectively), not records — what's left is the
  // pipeline's raw material: chunks, documents, summaries, context. First
  // linked entity wins as each record's anchor; a record linking multiple
  // entities near-always links them from one extraction pass, so any of
  // them places it in the right neighborhood.
  const plumbingNodes = nodes.filter(
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
    sessionSetNames: Object.keys(setColor).filter(isSessionSetName),
    encodings: {
      source: createSourceEncoding(setColor, isSessionSetName),
      type: createTypeEncoding(
        assignTypeColors(typeNodes.map((t) => ({ name: t.name, count: t.members.length }))),
        isSessionSetName,
      ),
    },
    setEntityCount: countBySet(entities),
    setDocCount: countBySet(docs),
    setMemberCount: countBySet(nodes.filter((n) => n.type !== "NodeSet")),
    graphLinks,
    semanticLinks,
    docLinks,
    anchors,
    entityAnchors,
    typeNodes,
    typeLinks,
    entityType,
    importanceMax: entities.reduce((max, n) => Math.max(max, n.importance || 0), 1),
    plumbingNodes,
    plumbingEntityId,
    importanceCut: topImportanceCut(entities),
    connectedIds: computeConnectedIds(graphLinks),
    hub: computeHubInsight(semanticLinks, byId),
  };
}
