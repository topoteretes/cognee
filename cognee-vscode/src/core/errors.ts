/** Raised when the Cognee backend returns a non-2xx HTTP response. */
export class CogneeApiError extends Error {
  readonly status: number;
  readonly body: string;

  constructor(message: string, status: number, body: string) {
    super(message);
    this.name = "CogneeApiError";
    this.status = status;
    this.body = body;
  }
}

/** Raised when the request never reaches the backend (DNS, connection, timeout). */
export class CogneeNetworkError extends Error {
  readonly cause?: unknown;

  constructor(message: string, cause?: unknown) {
    super(message);
    this.name = "CogneeNetworkError";
    this.cause = cause;
  }
}

/**
 * Produce a short, human-readable reason for a failed Cognee call, suitable for
 * surfacing in an editor notification.
 */
export function describeError(error: unknown): string {
  if (error instanceof CogneeApiError) {
    if (error.status === 401 || error.status === 403) {
      return "Authentication failed — check the endpoint and API key (Cognee: Set Up).";
    }
    if (error.status === 422) {
      return "No memory to recall yet — remember or index something first.";
    }
    return `Cognee returned ${error.status}. ${error.message}`;
  }
  if (error instanceof CogneeNetworkError) {
    return `${error.message} Is the Cognee server running and reachable?`;
  }
  return error instanceof Error ? error.message : String(error);
}
