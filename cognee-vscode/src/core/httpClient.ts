import type {
  CogneeClient,
  ForgetOptions,
  ImproveOptions,
  RecallOptions,
  RememberOptions,
} from "./client";
import { CogneeApiError, CogneeNetworkError } from "./errors";
import type { RecallResponseItem, RememberResult } from "./types";

export interface HttpCogneeClientOptions {
  /** Base URL of the backend (already normalized, no trailing slash). */
  endpoint: string;
  /** Optional API key, sent as `X-Api-Key`. */
  apiKey?: string;
  /** Default request timeout in milliseconds. */
  timeoutMs?: number;
  /**
   * Injectable fetch implementation. Defaults to the global `fetch` (Node 18+ /
   * VS Code extension host). Tests pass a mock to run without a live backend.
   */
  fetch?: typeof fetch;
}

const DEFAULT_TIMEOUT_MS = 300_000;

/**
 * HTTP implementation of {@link CogneeClient}.
 *
 * Endpoints and payload shapes match the Cognee server routers and the official
 * Cloud client (`X-Api-Key` auth). Every request is bounded by a timeout and can
 * be cancelled via an external `AbortSignal`.
 */
export class HttpCogneeClient implements CogneeClient {
  private readonly endpoint: string;
  private readonly apiKey: string | undefined;
  private readonly timeoutMs: number;
  private readonly fetchImpl: typeof fetch;

  constructor(options: HttpCogneeClientOptions) {
    this.endpoint = options.endpoint.replace(/\/+$/, "");
    this.apiKey = options.apiKey?.trim() || undefined;
    this.timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    const fetchImpl = options.fetch ?? globalThis.fetch;
    if (typeof fetchImpl !== "function") {
      throw new Error("global fetch is unavailable; pass a fetch implementation.");
    }
    this.fetchImpl = fetchImpl;
  }

  async health(signal?: AbortSignal): Promise<boolean> {
    try {
      const response = await this.send("GET", "/health", undefined, signal);
      return response.ok;
    } catch {
      return false;
    }
  }

  async recall(query: string, options: RecallOptions = {}): Promise<RecallResponseItem[]> {
    const body: Record<string, unknown> = { query };

    // `search_type` is only sent when specified: `null` opts into auto-routing,
    // a string forces a strategy, and omitting it uses the server default.
    if (options.searchType !== undefined) {
      body.search_type = options.searchType;
    }
    if (options.datasets && options.datasets.length > 0) {
      body.datasets = options.datasets;
    }
    if (options.topK !== undefined) {
      body.top_k = options.topK;
    }
    if (options.includeReferences !== undefined) {
      body.include_references = options.includeReferences;
    }
    if (options.sessionId) {
      body.session_id = options.sessionId;
    }
    if (options.scope !== undefined) {
      body.scope = options.scope;
    }
    if (options.systemPrompt) {
      body.system_prompt = options.systemPrompt;
    }

    const response = await this.send("POST", "/api/v1/recall", { json: body }, options.signal);
    const data = await this.parse(response, "recall");
    return Array.isArray(data) ? (data as RecallResponseItem[]) : [];
  }

  async remember(data: string, options: RememberOptions): Promise<RememberResult> {
    const form = new FormData();
    form.append("datasetName", options.datasetName);
    if (options.sessionId) {
      form.append("session_id", options.sessionId);
    }
    if (options.runInBackground) {
      form.append("run_in_background", "true");
    }
    for (const tag of options.nodeSet ?? []) {
      if (tag) {
        form.append("node_set", tag);
      }
    }
    const filename = options.filename?.trim() || "memory.txt";
    form.append("data", new Blob([data], { type: "text/plain" }), filename);

    const response = await this.send(
      "POST",
      "/api/v1/remember",
      { form },
      options.signal,
    );
    return (await this.parse(response, "remember")) as RememberResult;
  }

  async forget(options: ForgetOptions): Promise<Record<string, unknown>> {
    const body: Record<string, unknown> = {};
    if (options.everything) {
      body.everything = true;
    }
    if (options.dataset) {
      body.dataset = options.dataset;
    }
    if (options.datasetId) {
      body.dataset_id = options.datasetId;
    }
    if (options.dataId) {
      body.data_id = options.dataId;
    }
    if (options.memoryOnly !== undefined) {
      body.memory_only = options.memoryOnly;
    }

    const response = await this.send("POST", "/api/v1/forget", { json: body }, options.signal);
    return (await this.parse(response, "forget")) as Record<string, unknown>;
  }

  async improve(options: ImproveOptions = {}): Promise<Record<string, unknown>> {
    const body: Record<string, unknown> = {};
    if (options.datasetId) {
      body.dataset_id = options.datasetId;
    } else if (options.datasetName) {
      body.dataset_name = options.datasetName;
    }
    if (options.runInBackground) {
      body.run_in_background = true;
    }

    const response = await this.send("POST", "/api/v1/improve", { json: body }, options.signal);
    return (await this.parse(response, "improve")) as Record<string, unknown>;
  }

  // --- internals -----------------------------------------------------------

  private buildHeaders(extra?: Record<string, string>): Record<string, string> {
    const headers: Record<string, string> = { Accept: "application/json", ...extra };
    if (this.apiKey) {
      headers["X-Api-Key"] = this.apiKey;
    }
    return headers;
  }

  private async send(
    method: string,
    path: string,
    payload?: { json?: unknown; form?: FormData },
    externalSignal?: AbortSignal,
  ): Promise<Response> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    if (externalSignal) {
      if (externalSignal.aborted) {
        controller.abort();
      } else {
        externalSignal.addEventListener("abort", () => controller.abort(), { once: true });
      }
    }

    const init: RequestInit = { method, signal: controller.signal };
    if (payload?.json !== undefined) {
      init.body = JSON.stringify(payload.json);
      init.headers = this.buildHeaders({ "Content-Type": "application/json" });
    } else if (payload?.form) {
      // Let fetch set the multipart boundary; do not set Content-Type manually.
      init.body = payload.form;
      init.headers = this.buildHeaders();
    } else {
      init.headers = this.buildHeaders();
    }

    try {
      return await this.fetchImpl(`${this.endpoint}${path}`, init);
    } catch (error) {
      if (controller.signal.aborted) {
        throw new CogneeNetworkError(`Request to ${path} timed out or was cancelled.`, error);
      }
      throw new CogneeNetworkError(`Could not reach Cognee at ${this.endpoint}.`, error);
    } finally {
      clearTimeout(timer);
    }
  }

  private async parse(response: Response, operation: string): Promise<unknown> {
    const rawBody = await response.text();
    if (!response.ok) {
      throw new CogneeApiError(
        `${operation} failed (${response.status}).`,
        response.status,
        rawBody,
      );
    }
    if (!rawBody) {
      return null;
    }
    try {
      return JSON.parse(rawBody);
    } catch {
      throw new CogneeApiError(
        `${operation} returned a non-JSON response.`,
        response.status,
        rawBody,
      );
    }
  }
}
