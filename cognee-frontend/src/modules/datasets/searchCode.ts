import { CogneeInstance } from "../instances/types";

// No model is called, but the first query against a dataset loads the whole
// code graph into the pod's snapshot cache, and on a large repository that
// read is what the wait is — so this keeps the headroom search and recall have.
const CODE_SEARCH_TIMEOUT_MS = 120_000;

/** Node kinds the pod's code graph has carried since code search first shipped. */
export type CodeFactKind =
  | "module"
  | "route"
  | "service"
  | "storage"
  | "symbol"
  | "dependency"
  // The node standing for a source file itself.
  | "file_ref";

/**
 * The structured half of a CODE search. The pod's CodeRetriever "runs one exact
 * code-graph operation without model inference": it never reads a question, it
 * runs the operation named here. Left out, the operation defaults to "explore"
 * and the query text is resolved as a symbol name — which is how a sentence
 * typed into the search box came back as "could not resolve explore seed name".
 */
export type CodeQuery =
  | { operation: "architecture" }
  | {
      operation: "query_facts";
      /** Case-insensitive substring of a node's name. */
      name?: string;
      /** Exact path of the file a node is defined in. */
      file?: string;
      kinds?: CodeFactKind[];
      limit?: number;
    };

export interface CodeFact {
  id: string;
  kind: string;
  type?: string;
  name?: string | null;
  file?: string;
  line?: number;
  end_line?: number;
  repo?: string;
  path?: string;
  description?: string;
  doc?: string;
  symbol_kind?: string;
  properties?: Record<string, unknown>;
  /** Outgoing edges: what this node calls, imports, declares as a method, ... */
  relations?: { type: string; target?: string | null }[];
}

export interface CodeEdge {
  source?: string | null;
  target?: string | null;
  type: string;
  count?: number;
}

export interface CodeFactsResult {
  operation: "query_facts";
  facts: CodeFact[];
  total: number;
  has_more?: boolean;
}

export interface CodeArchitectureResult {
  operation: "architecture";
  repos?: string[];
  nodes: CodeFact[];
  edges?: CodeEdge[];
  stats?: { nodes_total?: number; nodes_shown?: number; truncated?: boolean };
}

export type CodeSearchResult = CodeFactsResult | CodeArchitectureResult;

export interface CodeNeighbor {
  relation: string;
  direction: "in" | "out";
  node: CodeFact & { type: string };
}

export interface CodeNeighborsResult {
  operation: "neighbors";
  anchor: CodeFact & { type: string };
  neighbors: CodeNeighbor[];
  total: number;
  offset: number;
  limit: number;
  has_more: boolean;
}

interface CodeNeighborsQuery {
  operation: "neighbors";
  id: string;
  direction: "both";
  offset: number;
}

export interface CodeSearchRequest {
  /** What the user typed. Sent for the pod's own logs; neither operation reads it. */
  query: string;
  codeQuery: CodeQuery;
  datasetIds: string[];
}

function isCodeSearchResult(value: unknown): value is CodeSearchResult {
  if (typeof value !== "object" || value === null) return false;
  const operation = (value as { operation?: unknown }).operation;
  return operation === "query_facts" || operation === "architecture";
}

/**
 * The pod answers with one `{dataset_id, dataset_name, search_result}` entry
 * per dataset searched, `search_result` being the operation's own payload.
 * Read defensively rather than cast: the route is typed `list[SearchResult] |
 * list`, so a list of payloads or a bare one are both within its contract.
 */
function unwrapCodeResults<T>(data: unknown, isResult: (value: unknown) => value is T): T[] {
  const entries = Array.isArray(data) ? data : [data];
  const results: T[] = [];
  for (const entry of entries) {
    if (isResult(entry)) {
      results.push(entry);
      continue;
    }
    if (typeof entry !== "object" || entry === null) continue;
    const payload = (entry as { search_result?: unknown }).search_result;
    for (const candidate of Array.isArray(payload) ? payload : [payload]) {
      if (isResult(candidate)) results.push(candidate);
    }
  }
  return results;
}

/**
 * Runs one code-graph operation on POST /v1/search.
 *
 * /v1/search rather than /v1/recall, where the rest of the Search tab goes,
 * for two reasons that both come down to what is deployed. `code_query` has
 * been on this route since code search first shipped, whereas recall only
 * honours it inside a "code" scope added five weeks later — so this is the one
 * call every pod that can search code at all will accept. And recall's code
 * lane turns a failed lookup into an empty list, which leaves nothing to tell
 * the user; this route returns the error.
 *
 * The cost is that the pod records nothing: CODE makes no completion, so there
 * is no turn to write to the session, and a Code answer does not come back
 * with the conversation after a reload.
 */
export default async function searchCode(
  instance: CogneeInstance,
  request: CodeSearchRequest,
): Promise<CodeSearchResult[]> {
  return unwrapCodeResults(await runCodeQuery(instance, request), isCodeSearchResult);
}

async function runCodeQuery(
  instance: CogneeInstance,
  request: Omit<CodeSearchRequest, "codeQuery"> & { codeQuery: CodeQuery | CodeNeighborsQuery },
  signal?: AbortSignal,
): Promise<unknown> {
  const response = await instance.fetch("/v1/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      searchType: "CODE",
      query: request.query,
      codeQuery: request.codeQuery,
      datasetIds: request.datasetIds,
    }),
    timeoutMs: CODE_SEARCH_TIMEOUT_MS,
    ...(signal ? { signal } : {}),
  });
  return response.json();
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isDetailedFact(value: unknown): value is CodeFact & { type: string } {
  if (!isRecord(value) || typeof value.id !== "string" || !value.id ||
      typeof value.kind !== "string" || typeof value.type !== "string") return false;
  const strings = ["name", "file", "path", "repo", "description", "doc", "symbol_kind"];
  if (strings.some((key) => value[key] != null && typeof value[key] !== "string")) return false;
  return ["line", "end_line"].every((key) => value[key] == null ||
    Number.isInteger(value[key]));
}

function isNeighborsResult(value: unknown): value is CodeNeighborsResult {
  return isRecord(value) && value.operation === "neighbors" && isDetailedFact(value.anchor) &&
    Array.isArray(value.neighbors) && value.neighbors.every((entry) =>
      isRecord(entry) && typeof entry.relation === "string" &&
      (entry.direction === "in" || entry.direction === "out") && isDetailedFact(entry.node)) &&
    ["total", "offset", "limit"].every((key) => Number.isInteger(value[key]) && Number(value[key]) >= 0) &&
    typeof value.has_more === "boolean";
}

/** Reads one page using the server's existing page size, including the anchor's details. */
export async function getCodeNeighbors(
  instance: CogneeInstance,
  datasetId: string,
  nodeId: string,
  offset: number,
  signal?: AbortSignal,
): Promise<CodeNeighborsResult> {
  const data = await runCodeQuery(instance, {
    query: "",
    codeQuery: { operation: "neighbors", id: nodeId, direction: "both", offset },
    datasetIds: [datasetId],
  }, signal);
  const results = unwrapCodeResults(data, isNeighborsResult);
  if (results.length !== 1) throw new Error("The server did not return readable code details.");
  const result = results[0];
  if (result.anchor.id !== nodeId || result.offset !== offset ||
      result.offset + result.neighbors.length > result.total ||
      result.has_more !== (result.offset + result.neighbors.length < result.total) ||
      (result.has_more && !result.neighbors.length)) {
    throw new Error("The server returned inconsistent code relationships.");
  }
  return result;
}
