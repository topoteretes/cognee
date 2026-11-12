// Shared retry policy for one-shot (non-polling) background queries against
// the tenant pod — e.g. a dataset list or graph summary fetched once and
// refetched only when its query key changes. Polled queries (refetchInterval
// set) should use `retry: false` instead: the next tick is the retry, and a
// circuit breaker (see useCircuitBreaker.ts) tracks consecutive tick
// failures — an extra per-fetch retry there only delays that signal.
export const BACKGROUND_QUERY_RETRY_COUNT = 3;

export function backgroundQueryRetryDelay(attempt: number): number {
  return Math.min(1000 * 2 ** attempt, 8000);
}

/**
 * Same budget, but never spent on an answer that cannot change.
 *
 * A 4xx is the pod saying no, not the pod being unreachable: a route the
 * running pod image has not allowlisted answers 404 every time, so retrying
 * costs four requests and ~7s of backoff per page load and reports a pod
 * failure to the log sink on each attempt. 429 is the exception — that is a
 * "not now", not a "no". 5xx and network errors keep the full budget.
 */
export function backgroundQueryRetry(failureCount: number, error: unknown): boolean {
  const status = (error as { status?: unknown } | null)?.status;
  if (typeof status === "number" && status >= 400 && status < 500 && status !== 429) {
    return false;
  }
  return failureCount < BACKGROUND_QUERY_RETRY_COUNT;
}
