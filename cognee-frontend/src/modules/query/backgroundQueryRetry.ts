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
