import { CogneeInstance } from "../instances/types";
import { getPipelineSettingsFromStorage } from "../configuration/pipelineSettings";

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
 * Default is ``HYBRID_COMPLETION``.
 */
export default function recallKnowledge(
  instance: CogneeInstance,
  req: RecallRequest,
): Promise<unknown[]> {
  const pipelineSettings = getPipelineSettingsFromStorage();
  const body: Record<string, unknown> = { query: req.query };
  if (req.scope !== undefined) body.scope = req.scope;
  if (req.sessionId) body.session_id = req.sessionId;
  if (req.datasets) body.datasets = req.datasets;
  if (req.datasetIds) body.dataset_ids = req.datasetIds;
  body.top_k = req.topK ?? pipelineSettings.topK;
  // Server default is false; we default on so completions ship with citations.
  body.include_references = pipelineSettings.includeReferences;
  // Explicit null asks the server to auto-route; undefined falls back to HYBRID_COMPLETION.
  body.search_type = req.searchType !== undefined ? req.searchType : "HYBRID_COMPLETION";

  return instance
    .fetch("/v1/recall", {
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
