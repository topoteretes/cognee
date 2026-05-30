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
 * Unified recall call. Hits POST /v2/recall (not /v1/search), so the
 * server's scope-aware fan-out applies: graph + session + trace +
 * graph_context, tagged with _source.
 *
 * Default keeps ``GRAPH_COMPLETION`` for backward compatibility with
 * the current backend recall DTO.
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
  if (req.searchType != null) body.search_type = req.searchType;

  return instance
    .fetch("/v2/recall", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
    .then(async (r) => {
      if (r.ok) return r.json();
      // Extract error detail from response body
      let errorMsg = `Recall failed (${r.status})`;
      try {
        const body = await r.json();
        if (body.error) errorMsg = body.error;
        if (body.hint) errorMsg += ` — ${body.hint}`;
      } catch { /* no JSON body */ }
      throw new Error(errorMsg);
    });
}
