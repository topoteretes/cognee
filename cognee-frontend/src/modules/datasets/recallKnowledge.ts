import { CogneeInstance } from "../instances/types";

export type RecallScope =
  | "all"
  | "graph"
  | "session"
  | "trace"
  | "graph_context";

export interface RecallRequest {
  query: string;
  scope?: RecallScope | RecallScope[];
  sessionId?: string;
  datasets?: string[];
  datasetIds?: string[];
  topK?: number;
  searchType?: string | null;
}

/**
 * Unified recall call. Hits POST /v1/recall (not /v1/search), so the
 * server's scope-aware fan-out applies: graph + session + trace +
 * graph_context, tagged with _source.
 *
 * Pass ``searchType: null`` to opt into the server's auto-router.
 * Default keeps ``GRAPH_COMPLETION`` for backward compat.
 */
export default function recallKnowledge(
  instance: CogneeInstance,
  req: RecallRequest,
): Promise<unknown[]> {
  const body: Record<string, unknown> = { query: req.query };
  if (req.scope !== undefined) body.scope = req.scope;
  if (req.sessionId) body.session_id = req.sessionId;
  if (req.datasets) body.datasets = req.datasets;
  if (req.datasetIds) body.dataset_ids = req.datasetIds;
  if (req.topK !== undefined) body.top_k = req.topK;
  // Explicit null asks the server to auto-route; omitting keeps
  // the current HTTP default (GRAPH_COMPLETION).
  if (req.searchType !== undefined) body.search_type = req.searchType;

  return instance
    .fetch("/v1/recall", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
    .then((r) => (r.ok ? r.json() : []))
    .catch(() => []);
}
