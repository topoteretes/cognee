/**
 * Wire types for the Cognee HTTP API.
 *
 * These mirror the server contracts exactly:
 * - `POST /api/v1/recall`   → `RecallPayloadDTO` / `list[RecallResponse]`
 * - `POST /api/v1/remember` → multipart form / `RememberResult.to_dict()`
 * - `POST /api/v1/forget`   → `ForgetPayloadDTO`
 * - `POST /api/v1/improve`  → `{ dataset_name | dataset_id, ... }`
 *
 * The recall endpoint accepts both camelCase and snake_case keys; this client
 * sends snake_case to match the Python DTO field names precisely.
 */

/** Recall strategy. `null` asks the server to auto-route to the best strategy. */
export type SearchType =
  | "GRAPH_COMPLETION"
  | "GRAPH_COMPLETION_COT"
  | "RAG_COMPLETION"
  | "CHUNKS"
  | "SUMMARIES"
  | (string & {});

/**
 * One normalized search hit (`SearchResultItem` on the server). `text` is
 * always populated and renderable; for completion strategies with
 * `include_references` enabled it carries the answer plus an `Evidence:` block.
 */
export interface SearchResultItem {
  kind: string;
  search_type: string;
  text: string;
  score?: number | null;
  dataset_id?: string | null;
  dataset_name?: string | null;
  metadata?: Record<string, unknown>;
  raw?: Record<string, unknown>;
  structured?: unknown;
}

/**
 * A single recall result. The server returns a discriminated union keyed by
 * `source`; graph results carry a renderable `text`, while the context sources
 * carry `content`.
 */
export type RecallResponseItem =
  | ({ source: "graph" } & SearchResultItem)
  | { source: "graph_context"; content: string }
  | { source: "session_context"; content: string; context_profile: string }
  | ({ source: "session"; question?: string; answer?: string; context?: string } & Record<
      string,
      unknown
    >)
  | ({ source: "trace" } & Record<string, unknown>);

/** Result envelope returned by `POST /api/v1/remember`. */
export interface RememberResult {
  /** e.g. "completed", "running" (background), "errored". */
  status?: string;
  pipeline_run_id?: string;
  dataset_id?: string;
  [key: string]: unknown;
}
