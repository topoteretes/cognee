import { useRef, useState } from "react";

// Circuit breaker for polling queries: after this many consecutive *tick*
// failures, stop hammering a struggling backend on the normal cadence and
// back off instead. Recovers automatically the moment a poll succeeds.
export const CIRCUIT_BREAKER_THRESHOLD = 3;
export const CIRCUIT_BREAKER_BACKOFF_MS = 60_000;

interface QueryOutcome {
  isError: boolean;
  isSuccess: boolean;
}

// Deliberately NOT built on React Query's own `fetchFailureCount` — that
// counter resets to 0 at the start of every fetch (it counts retries within
// one tick, not consecutive failed ticks), so it never reaches a useful
// threshold under a normal `retry: 1` config. This tracks real consecutive
// tick failures ourselves instead.
export function useCircuitBreaker(normalIntervalMs: number) {
  const consecutiveFailuresRef = useRef(0);
  const [isOpen, setIsOpen] = useState(false);

  // Frozen on first render: both closed-over values (the ref and normalIntervalMs)
  // are stable for the lifetime of the component, so recreating this per
  // render would add nothing.
  const refetchInterval = useRef(() =>
    consecutiveFailuresRef.current >= CIRCUIT_BREAKER_THRESHOLD ? CIRCUIT_BREAKER_BACKOFF_MS : normalIntervalMs,
  ).current;

  // Call from a useEffect keyed on the query's outcome fields (isError,
  // isSuccess, dataUpdatedAt, errorUpdatedAt) after each tick settles.
  const report = useRef((outcome: QueryOutcome) => {
    if (outcome.isError) consecutiveFailuresRef.current += 1;
    else if (outcome.isSuccess) consecutiveFailuresRef.current = 0;
    setIsOpen(consecutiveFailuresRef.current >= CIRCUIT_BREAKER_THRESHOLD);
  }).current;

  return { refetchInterval, isOpen, report };
}
