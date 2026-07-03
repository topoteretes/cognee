/**
 * Editor-agnostic configuration for the Cognee client.
 *
 * The VS Code layer reads these values from workspace settings; the core layer
 * only depends on this plain object so it can be reused by other editors (e.g. a
 * JetBrains sidecar) without change.
 */
export interface CogneeConfig {
  /** Base URL of the Cognee backend, e.g. http://localhost:8011 or a Cloud tenant URL. */
  endpoint: string;
  /** Optional API key, sent as the `X-Api-Key` header. Required for Cognee Cloud. */
  apiKey?: string;
  /** Fixed dataset name for the workspace; when empty, the dataset is derived by hash. */
  datasetOverride?: string;
  /** Recall strategy; "auto" (or empty) enables server-side auto-routing. */
  searchType: string;
  /** Maximum recall results (1–100). */
  topK: number;
  /** Ask the backend to attach source citations to answers. */
  includeReferences: boolean;
  /** Skip .gitignore/.cogneeignore entries when indexing the workspace. */
  respectGitignore: boolean;
  /** Skip files larger than this (KB) during workspace indexing. */
  maxFileSizeKb: number;
  /** HTTP request timeout in milliseconds. */
  requestTimeoutMs: number;
}

export interface ConfigValidation {
  ok: boolean;
  errors: string[];
}

/** Validate a config, returning a flat list of human-readable problems. */
export function validateConfig(config: CogneeConfig): ConfigValidation {
  const errors: string[] = [];

  const endpoint = config.endpoint?.trim() ?? "";
  if (!endpoint) {
    errors.push("Cognee endpoint is not set.");
  } else if (!/^https?:\/\//i.test(endpoint)) {
    errors.push("Cognee endpoint must start with http:// or https://.");
  }

  if (!Number.isFinite(config.topK) || config.topK < 1 || config.topK > 100) {
    errors.push("topK must be between 1 and 100.");
  }

  if (!Number.isFinite(config.requestTimeoutMs) || config.requestTimeoutMs <= 0) {
    errors.push("requestTimeoutMs must be a positive number.");
  }

  if (!Number.isFinite(config.maxFileSizeKb) || config.maxFileSizeKb <= 0) {
    errors.push("maxFileSizeKb must be a positive number.");
  }

  return { ok: errors.length === 0, errors };
}

/**
 * Resolve the configured search type to what the API expects: `null` for
 * auto-routing, or the explicit strategy name otherwise.
 */
export function resolveSearchType(searchType: string): string | null {
  const value = searchType?.trim().toLowerCase();
  return !value || value === "auto" ? null : searchType.trim();
}

/** Normalize the endpoint URL by trimming trailing slashes. */
export function normalizeEndpoint(endpoint: string): string {
  return endpoint.trim().replace(/\/+$/, "");
}
