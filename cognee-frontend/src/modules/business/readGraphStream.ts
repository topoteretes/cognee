import { errorFrameToHttpError, parseFrameData, readSseFrames } from "@/services/http/sse";
import type { BusinessGraphLink, BusinessGraphNode, VisualizationPayload } from "./types";

// The streamed GET /v1/visualize/json (cognee 1.6.1, SDK-787), assembled into
// the same VisualizationPayload the JSON response carries. Events arrive in order:
// `meta`, one `chunk` per store chunk (compact nodes and links, every link's
// endpoints already sent), one or more `summary` events carrying the
// graph-wide fields, then `done`. The summary is what sizes and labels the
// nodes (importance, label_priority) and corrects an `entity_type` a chunk
// sent as "Entity" before its is_a link arrived, so the graph is handed on
// only once `done` has arrived and its totals match what was assembled: a
// stream that stops short, a chunk that did not parse, or a node sent twice
// throws instead of passing a partial or doubled graph off as the whole one.
//
// Where the two transports differ: the compact node shape has no
// `symbol_kind`, so codeGraph.ts tells a code file apart from a symbol by its
// extension alone on this path (a fallback that already existed, because the
// JSON payload set the field only on some files); `color_maps` carries only
// `node_set` (no task, pipeline or user maps), the one useBusinessScene reads;
// and none of the JSON-only side payloads such as `search_events` are sent,
// which nothing reads.

// How long the pod may go without sending a byte before the read is given up.
// getBrainGraph's timeout bounds only the wait for the headers, so without
// this a pod that stalls mid-stream is held to its own 300 s drop. It is an
// idle limit, not a total one, because the transfer grows with the node cap
// (a 20k-node stream is several MB) and a slow link must not fail a read that
// is still arriving. The pod sends a keepalive comment every 15 s while it has
// nothing else to send (KEEPALIVE_SECONDS in cognee's graph_stream.py), so
// twice that tolerates one lost keepalive and still catches a real stall.
export const STREAM_IDLE_TIMEOUT_MS = 30_000;

type SummaryFields = Partial<Pick<BusinessGraphNode, "importance" | "label_priority" | "entity_type">>;

interface GraphStreamState {
  chunks: number;
  nodes: BusinessGraphNode[];
  links: BusinessGraphLink[];
  summaries: Map<string, SummaryFields>;
  colorMaps: VisualizationPayload["color_maps"];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isNode(value: unknown): value is BusinessGraphNode {
  return isRecord(value) && typeof value.id === "string";
}

function isLink(value: unknown): value is BusinessGraphLink {
  return isRecord(value) && typeof value.source === "string" && typeof value.target === "string";
}

function isColorMaps(value: unknown): value is VisualizationPayload["color_maps"] {
  return isRecord(value) && Object.values(value).every(isRecord);
}

function summaryFieldsOf(entry: Record<string, unknown>): SummaryFields {
  const fields: SummaryFields = {};
  if (typeof entry.importance === "number") fields.importance = entry.importance;
  if (typeof entry.label_priority === "boolean") fields.label_priority = entry.label_priority;
  if (typeof entry.entity_type === "string") fields.entity_type = entry.entity_type;
  return fields;
}

// Every chunk the server sends carries both arrays, even when one is empty, so
// a frame without them did not parse (parseFrameData answers `{}` for that).
// Counting it would let the `chunks` total vouch for a chunk nobody read.
function addChunk(state: GraphStreamState, payload: Record<string, unknown>): void {
  if (!Array.isArray(payload.nodes) || !Array.isArray(payload.links)) {
    throw new Error("The graph stream sent a chunk that could not be read, so the graph cannot be trusted as complete.");
  }
  state.chunks += 1;
  state.nodes.push(...payload.nodes.filter(isNode));
  state.links.push(...payload.links.filter(isLink));
}

// The summary may arrive split across several events, and only the first
// carries the color maps. The server splits the entries by node, so a node
// appears in one part only today; merging rather than replacing keeps what an
// earlier part set if that ever changes, instead of dropping it silently.
function addSummary(state: GraphStreamState, payload: Record<string, unknown>): void {
  if (isRecord(payload.nodes)) {
    for (const [id, entry] of Object.entries(payload.nodes)) {
      if (isRecord(entry)) state.summaries.set(id, { ...state.summaries.get(id), ...summaryFieldsOf(entry) });
    }
  }
  if (isColorMaps(payload.color_maps)) state.colorMaps = payload.color_maps;
}

// The server sends all three totals on every `done`, so one that is missing
// or not a number means the frame itself was cut or mangled, and nothing is
// left to vouch for the rest. `done.nodes` is the server's count of distinct
// node ids and `done.links` its count of links sent; the graph store's
// chunking contract says a node is delivered once, but each adapter
// implements that contract itself, so a repeat is checked here rather than
// assumed away. Comparing against every node received makes a repeat throw: the
// canvas would draw it twice, and an error screen is the less misleading
// failure than a graph that looks complete. The server sends a summary entry
// for every node before `done`, so a node without one means a summary part
// was lost or did not parse, and the graph would render with no sizes, labels
// or colors while looking complete.
function checkTotals(state: GraphStreamState, done: Record<string, unknown>): void {
  if (typeof done.nodes !== "number" || typeof done.links !== "number" || typeof done.chunks !== "number") {
    throw new Error("The graph stream ended with unreadable totals, so the graph cannot be trusted as complete.");
  }
  const nodes = state.nodes.length;
  if (nodes !== done.nodes || state.links.length !== done.links || state.chunks !== done.chunks) {
    throw new Error(
      `The graph stream lost part of the graph: ${nodes} of ${done.nodes} nodes, ` +
        `${state.links.length} of ${done.links} links, ${state.chunks} of ${done.chunks} chunks.`,
    );
  }
  const summarised = state.nodes.filter((node) => state.summaries.has(node.id)).length;
  if (summarised !== nodes) {
    throw new Error(`The graph stream's summary covered ${summarised} of ${nodes} nodes.`);
  }
}

function assemble(state: GraphStreamState, done: Record<string, unknown>): VisualizationPayload {
  checkTotals(state, done);
  return {
    nodes: state.nodes.map((node) => ({ ...node, ...state.summaries.get(node.id) })),
    links: state.links,
    color_maps: state.colorMaps,
  };
}

export default async function readGraphStream(body: ReadableStream<Uint8Array>): Promise<VisualizationPayload> {
  const state: GraphStreamState = { chunks: 0, nodes: [], links: [], summaries: new Map(), colorMaps: {} };
  for await (const frame of readSseFrames(body, STREAM_IDLE_TIMEOUT_MS)) {
    const payload = parseFrameData(frame.data);
    switch (frame.event) {
      case "chunk":
        addChunk(state, payload);
        break;
      case "summary":
        addSummary(state, payload);
        break;
      case "error":
        throw errorFrameToHttpError(payload, "Graph read failed");
      case "done":
        return assemble(state, payload);
      default:
        // `meta` describes the read (seeds, bounds); nothing here needs it.
        break;
    }
  }
  throw new Error("The graph stream ended before the whole graph arrived.");
}
