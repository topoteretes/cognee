import type { HttpError } from "@/services/http/errors";
import {
  errorFrameToHttpError,
  isEventStream,
  parseFrameData,
  readSseFrames,
  SSE_ACCEPT,
} from "@/services/http/sse";
import { notifyInsufficientCredits } from "@/services/credits/insufficientCreditsBridge";
import { CogneeInstance } from "../instances/types";
import { getPipelineSettingsFromStorage } from "../configuration/pipelineSettings";

// HYBRID_COMPLETION over a large graph routinely exceeds the client's 30s
// default — this is the endpoint CLO-333 was filed against. On a streamed
// request this bounds the wait for the first frame only: the pod holds the
// response until retrieval is done, and reading the body is not timed.
const RECALL_TIMEOUT_MS = 120_000;

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
 * Callbacks for a streamed recall (CLO-845). Every one is a preview: the
 * tokens are a strict prefix of the answer at best (citations are appended
 * after generation, a retried LLM call starts over), and a recall that takes
 * a path the pod cannot stream produces no deltas at all. The resolved
 * payload is authoritative and identical to the JSON transport's.
 */
export interface RecallStreamHandlers {
  /** `retrieving` while the graph is searched, `generating` once tokens flow. */
  onStage?: (stage: string) => void;
  onDelta?: (text: string) => void;
  /** Discard everything rendered so far — the answer is being regenerated. */
  onReset?: () => void;
}

/**
 * Unified recall call. Hits POST /v1/recall (not /v1/search), so the
 * server's scope-aware fan-out applies: graph + session + trace +
 * graph_context, tagged with _source.
 *
 * Pass ``searchType: null`` to opt into the server's auto-router.
 * Default is ``HYBRID_COMPLETION``.
 *
 * With `stream` handlers the answer is requested as server-sent events and
 * relayed to them as it is generated; the promise still resolves with the
 * full payload. A JSON response is accepted either way, so the handlers are
 * safe against a pod that does not stream.
 */
export default async function recallKnowledge(
  instance: CogneeInstance,
  req: RecallRequest,
  stream?: RecallStreamHandlers,
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
  // The body flag decides outright on the pod; the Accept header is what a
  // proxy in between reads to know not to buffer the response.
  if (stream) body.stream = true;

  const r = await instance.fetch("/v1/recall", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(stream ? { Accept: SSE_ACCEPT } : {}) },
    body: JSON.stringify(body),
    timeoutMs: RECALL_TIMEOUT_MS,
  });
  if (!r.ok) throw await recallError(r);
  if (stream && isEventStream(r) && r.body) {
    return readStreamedRecall(r.body, stream, instance.instanceId);
  }
  return r.json();
}

async function recallError(r: Response): Promise<Error> {
  // Extract error detail from response body
  let errorMsg = `Recall failed (${r.status})`;
  try {
    const body = await r.json();
    if (body.error) errorMsg = body.error;
    if (body.hint) errorMsg += ` — ${body.hint}`;
  } catch { /* no JSON body */ }
  return new Error(errorMsg);
}

/**
 * A 402 is announced to the credits modal here because the pod client's error
 * interceptor never sees a response that succeeded at the HTTP level.
 */
function streamError(payload: Record<string, unknown>, tenantId: string): HttpError {
  const error = errorFrameToHttpError(payload, "Recall failed");
  if (error.status === 402) {
    notifyInsufficientCredits({ operation: "recall", message: error.message, at: Date.now(), tenantId });
  }
  return error;
}

async function readStreamedRecall(
  body: ReadableStream<Uint8Array>,
  handlers: RecallStreamHandlers,
  tenantId: string,
): Promise<unknown[]> {
  for await (const frame of readSseFrames(body)) {
    const payload = parseFrameData(frame.data);
    switch (frame.event) {
      case "stage":
        if (typeof payload.stage === "string") handlers.onStage?.(payload.stage);
        break;
      case "delta":
        if (typeof payload.text === "string" && payload.text) handlers.onDelta?.(payload.text);
        break;
      case "reset":
        handlers.onReset?.();
        break;
      case "final":
        return Array.isArray(payload.results) ? payload.results : [];
      case "error":
        throw streamError(payload, tenantId);
      default:
        // `answer_done` marks the end of generation; persistence follows and
        // the payload arrives as `final`, so there is nothing to do yet.
        break;
    }
  }
  throw new Error("The connection dropped before the answer was delivered.");
}
