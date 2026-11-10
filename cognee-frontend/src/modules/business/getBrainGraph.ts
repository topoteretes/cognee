import { SLOW_READ_TIMEOUT_MS } from "@/services/http";
import { HttpError } from "@/services/http/errors";
import { isEventStream, SSE_ACCEPT } from "@/services/http/sse";
import type { CogneeInstance } from "@/modules/instances/types";
import readGraphStream from "./readGraphStream";
import type { BusinessGraphNode, VisualizationPayload } from "./types";
import { hasRowDetail, isStructuredRow } from "./structuredRow";

// Twice the density /brains gave a dataset entry, because the server turned
// out not to care: measured on a 3.7k-node dataset, 500 nodes cost 0.95 s and
// 2151 edges, 1000 cost 0.69 s and 3969 — flat time, and edges growing
// sub-linearly per node (4.3 → 4.0). The cost was payload size (3.97 MB →
// 6.67 MB as JSON with every node's properties) and canvas legibility, not
// backend work. Streamed, the same 1000 nodes are 0.29 MB (SDK-787), so what
// holds this cap now is the canvas alone.
//
// Note how this sits against ACTIVE_CAP_ENTITY_THRESHOLD (1500) in
// simulation/useViewportActiveIds.ts: this cap stays deliberately below that
// threshold, so entities being a subset of these nodes, ONE dataset can never
// contain enough of them to trigger viewport-capping however dense it is.
// Capping therefore stays what it was written for, a scene merging several
// datasets (CLO-578). Raising this past 1500 is not a free knob: capping the
// ordinary single-dataset scene was measured, and the capping machinery
// itself rather than d3-force was what made the layout unstable (that file
// carries the numbers).
const MAX_NODES = 1000;

const JSON_ACCEPT = "application/json";
// What the pod answers when it already holds its per-process maximum of open
// graph streams (GraphStreamCapacityError, 8 in cognee 1.6.1). That permit is
// load-shedding for streams only: the JSON read of the same 1000 nodes never
// takes one, so it is asked for instead of failing the view, at the cost of
// the bigger payload.
const STREAM_CAPACITY_STATUS = 503;

function requestGraph(
  instance: CogneeInstance,
  datasetId: string,
  accept: string,
  signal?: AbortSignal,
): Promise<Response> {
  const params = new URLSearchParams({ dataset_id: datasetId, max_nodes: String(MAX_NODES) });
  // Not the default GET budget: this is the read the graph_grew push
  // triggers, so it lands exactly while the pod is finishing the cognify
  // run that caused it (see SLOW_READ_TIMEOUT_MS). On a stream it bounds the
  // wait for the headers only; readGraphStream puts an idle limit on the body
  // (STREAM_IDLE_TIMEOUT_MS), so a pod that stalls mid-stream is not held to
  // its own 300 s drop. The JSON fallback's body has no such limit.
  return instance.fetch(`/v1/visualize/json?${params.toString()}`, {
    timeoutMs: SLOW_READ_TIMEOUT_MS,
    // React Query's signal, so a read it cancels (invalidation, dataset
    // switch, unmount) releases its browser connection instead of holding it
    // to the timeout beside the read that replaced it. Whether the pod stops
    // computing it is the server's call, which is why graphRefreshThrottle
    // still keeps pushed refreshes from overlapping.
    signal,
    headers: { Accept: accept },
  });
}

async function requestPreferringStream(
  instance: CogneeInstance,
  datasetId: string,
  signal?: AbortSignal,
): Promise<Response> {
  try {
    return await requestGraph(instance, datasetId, SSE_ACCEPT, signal);
  } catch (e) {
    if (!(e instanceof HttpError) || e.status !== STREAM_CAPACITY_STATUS) throw e;
    return requestGraph(instance, datasetId, JSON_ACCEPT, signal);
  }
}

const ROW_DETAIL_FIELDS = ["text", "properties", "document_id", "document_name", "chunk_index"] as const;

// Copies only the fields `source` actually has, so a carry never overwrites
// a value the stream sent with undefined. A value the stream did send wins
// over the carried one.
function definedFields(
  source: BusinessGraphNode,
  fields: readonly string[],
): Record<string, unknown> {
  return Object.fromEntries(fields.filter((field) => source[field] !== undefined).map((field) => [field, source[field]]));
}

function withDetails(graph: VisualizationPayload, details: Map<string, BusinessGraphNode>): VisualizationPayload {
  return {
    ...graph,
    nodes: graph.nodes.map((node) => {
      const detail = isStructuredRow(node) ? details.get(node.id) : undefined;
      return detail ? { ...node, ...definedFields(detail, ROW_DETAIL_FIELDS), ...definedFields(node, ROW_DETAIL_FIELDS) } : node;
    }),
  };
}

function detailsById(graph: VisualizationPayload | undefined): Map<string, BusinessGraphNode> {
  return new Map((graph?.nodes ?? []).filter((node) => isStructuredRow(node) && hasRowDetail(node)).map((node) => [node.id, node]));
}

// A row still without values after the JSON read, or after it failed, gets an
// empty `text`, which counts as "asked" in hasRowDetail. Without it that row
// would put the multi-MB read back on every poll. structuredRowContent and
// structuredRowValues ignore blank text, so labels and the inspector are
// unaffected. The cost: a read that failed transiently is not retried until a
// new row arrives or the page reloads.
function markAsked(graph: VisualizationPayload): VisualizationPayload {
  return {
    ...graph,
    nodes: graph.nodes.map((node) => isStructuredRow(node) && !hasRowDetail(node) ? { ...node, text: "" } : node),
  };
}

// Compact row nodes carry a truncated schema header, not their values, and
// only the bounded JSON read (several MB) sends those. So the poll stays on
// the stream and carries the values already read from `previous`; the JSON
// read happens only when a row appears that no earlier payload described.
// If that read fails the streamed graph is still complete and drawable, just
// with weaker row labels, so it is returned instead of failing the view.
async function withRowDetails(
  instance: CogneeInstance,
  datasetId: string,
  graph: VisualizationPayload,
  previous: VisualizationPayload | undefined,
  signal?: AbortSignal,
): Promise<VisualizationPayload> {
  if (!graph.nodes.some(isStructuredRow)) return graph;
  const carried = withDetails(graph, detailsById(previous));
  if (carried.nodes.every((node) => !isStructuredRow(node) || hasRowDetail(node))) return carried;
  try {
    const detailed: VisualizationPayload = await (await requestGraph(instance, datasetId, JSON_ACCEPT, signal)).json();
    // The streamed node and link set is kept; only the row values are taken
    // from the JSON read, which may sample a different set while ingesting.
    // JSON values go on the streamed nodes first and the previous payload only
    // fills the rest, so an "asked" marker carried from an earlier failed or
    // partial read cannot beat the values this read just returned.
    return markAsked(withDetails(withDetails(graph, detailsById(detailed)), detailsById(previous)));
  } catch (e) {
    if (signal?.aborted) throw e;
    return markAsked(carried);
  }
}

// One focused dataset's content graph. Goes through the same cognee-core
// `preprocess()` as a /brains entry, so nodes/links/color_maps carry the
// exact fields computeBrainState and the canvas read — but for ONE dataset,
// instead of /brains' full graph for every readable dataset (CLO-597).
//
// Asked for as a stream: a pod on cognee 1.6.1 or later sends compact nodes
// instead of every node's property bag, and one that cannot stream (1.5.x,
// which ignores the Accept header) answers with the JSON payload, so the
// response's content type decides, not the pod's version.
export default async function getBrainGraph(
  instance: CogneeInstance,
  datasetId: string,
  signal?: AbortSignal,
  previous?: VisualizationPayload,
): Promise<VisualizationPayload> {
  const response = await requestPreferringStream(instance, datasetId, signal);
  if (!isEventStream(response)) return response.json();
  if (!response.body) throw new Error("The pod announced a graph stream but sent no body.");
  return withRowDetails(instance, datasetId, await readGraphStream(response.body), previous, signal);
}
