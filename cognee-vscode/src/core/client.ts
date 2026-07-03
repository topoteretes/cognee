import type { RecallResponseItem, RememberResult } from "./types";

/**
 * Transport-agnostic contract for talking to a Cognee backend.
 *
 * The VS Code layer depends only on this interface, so tests inject a mock and
 * a future JetBrains sidecar can share the same core with a different concrete
 * client. `HttpCogneeClient` is the shipped implementation.
 */
export interface CogneeClient {
  /** Liveness probe against `GET /health`. Never throws; resolves to false on failure. */
  health(signal?: AbortSignal): Promise<boolean>;

  /** Query project memory. Maps to `POST /api/v1/recall`. */
  recall(query: string, options?: RecallOptions): Promise<RecallResponseItem[]>;

  /** Ingest text and build the graph in one call. Maps to `POST /api/v1/remember`. */
  remember(data: string, options: RememberOptions): Promise<RememberResult>;

  /** Delete project memory. Maps to `POST /api/v1/forget`. */
  forget(options: ForgetOptions): Promise<Record<string, unknown>>;

  /** Enrich the graph (prune/reinforce). Maps to `POST /api/v1/improve`. */
  improve(options?: ImproveOptions): Promise<Record<string, unknown>>;
}

export interface RecallOptions {
  /** Dataset names to search within (the workspace's derived dataset). */
  datasets?: string[];
  /** Explicit strategy, or `null` to auto-route. Omit to use the server default. */
  searchType?: string | null;
  topK?: number;
  includeReferences?: boolean;
  sessionId?: string;
  scope?: string | string[];
  systemPrompt?: string;
  signal?: AbortSignal;
}

export interface RememberOptions {
  /** Target dataset (required by the server). */
  datasetName: string;
  sessionId?: string;
  runInBackground?: boolean;
  /** Filename to attach to the uploaded text (helps document naming/citations). */
  filename?: string;
  /** Node sets to tag the ingested data with. */
  nodeSet?: string[];
  signal?: AbortSignal;
}

export interface ForgetOptions {
  dataset?: string;
  datasetId?: string;
  dataId?: string;
  everything?: boolean;
  /** Clear graph + vectors but keep raw files, so the dataset can be re-cognified. */
  memoryOnly?: boolean;
  signal?: AbortSignal;
}

export interface ImproveOptions {
  datasetName?: string;
  datasetId?: string;
  runInBackground?: boolean;
  signal?: AbortSignal;
}
