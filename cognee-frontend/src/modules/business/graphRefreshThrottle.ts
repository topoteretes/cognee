import { SLOW_READ_TIMEOUT_MS } from "@/services/http";

// A GitHub-sized ingest emits `graph_grew` far faster than a 13k-node graph
// read completes, and every push used to invalidate the graph query on the
// spot. `invalidateQueries` cancels an in-flight fetch by default, and at the
// time getBrainGraph did not pass React Query's abort signal on, so the old
// read kept both its browser connection and its pod-side work while the next
// one started beside it, until the pod's connection pool ran dry and every
// read failed with a 409 or a timeout. Forwarding the signal now frees the
// browser side, but nothing guarantees an aborted request stops the pod, so
// refresh requests are still collapsed: none starts while the previous read
// is still in flight, consecutive reads are spaced by a gap derived from what
// the last one cost, and requests that arrive in between fold into one
// trailing refresh, so the last growth of a run is always fetched.
//
// The gap scales with the measured read, because the pool pressure is
// proportional to how long a read holds connections, not to wall time. It is
// measured rather than derived from max_nodes on purpose: on cognee 1.6.1 the
// pod reads the whole bounded-depth neighbourhood and caps it in Python, so
// what a read costs follows the dataset's size, not the 1000 nodes it returns.
// A small dataset's read stays well under a second and keeps near-live growth
// at the 1s floor; any read of 7.5 s or more, which the 13959-node GitHub
// dataset that exhausted a pod was well past, reaches the 15s ceiling. A
// failed read is spaced at the ceiling whatever it cost: a pod answering 409s
// fast is one out of connections, and a short gap would only hammer it. The
// ceiling bounds how stale the Mindmap can get while the socket is healthy,
// since this throttle is then its only growth signal (useBrainGraph turns the
// poll off).
export const MIN_REFRESH_GAP_MS = 1_000;
export const MAX_REFRESH_GAP_MS = 15_000;
const READ_COST_MULTIPLIER = 2;
// How soon to look again when a refresh is due but the previous read has not
// come back yet.
export const IN_FLIGHT_RECHECK_MS = 1_000;
// How long past SLOW_READ_TIMEOUT_MS a read may stay in flight before it is
// reported as stuck. SLOW_READ_TIMEOUT_MS bounds the wait for the response
// headers only. A streamed body is held by an idle limit (see readGraphStream)
// and the JSON fallback's body by nothing at all, so this is a heuristic for
// "far longer than a healthy read takes", not a derived bound. It only reports:
// a read that is still arriving is left alone. A stuck gate means the graph has
// silently stopped growing, the one failure nobody would otherwise notice.
const STUCK_MARGIN_MS = 15_000;
export const STUCK_AFTER_MS = SLOW_READ_TIMEOUT_MS + STUCK_MARGIN_MS;

export interface GraphRefreshThrottleDeps {
  // Resolves when the refetch it started has settled, which is what the next
  // gap is measured from.
  refresh: () => Promise<unknown>;
  isFetching: () => boolean;
  // Read right after `refresh` settles. invalidateQueries resolves even when
  // the refetch failed, so the outcome has to come from the query itself.
  lastReadFailed: () => boolean;
  onStuck: (waitedMs: number) => void;
  // Monotonic by default: a wall clock stepped back (NTP after a laptop wakes)
  // would otherwise push the trailing refresh out by the size of the step.
  now?: () => number;
}

export interface GraphRefreshThrottle {
  request: () => void;
  dispose: () => void;
}

export function refreshGapFor(lastReadMs: number | null): number {
  if (lastReadMs === null) return MIN_REFRESH_GAP_MS;
  return Math.min(MAX_REFRESH_GAP_MS, Math.max(MIN_REFRESH_GAP_MS, lastReadMs * READ_COST_MULTIPLIER));
}

export function createGraphRefreshThrottle({
  refresh,
  isFetching,
  lastReadFailed,
  onStuck,
  now = () => performance.now(),
}: GraphRefreshThrottleDeps): GraphRefreshThrottle {
  let lastRefreshAt: number | null = null;
  let lastReadMs: number | null = null;
  let waitingSince: number | null = null;
  // Set while a read this throttle started is out, so a stuck read is timed
  // from when it began rather than from when a request first queued behind it.
  let ownReadStartedAt: number | null = null;
  // When the gate first saw a read it did not start (the mount fetch, the
  // poll). Its end has to count as a refresh, or the next one starts the moment
  // it lands: two full reads back to back, exactly when a large ingest is on.
  let foreignReadSeenAt: number | null = null;
  let reportedStuck = false;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let disposed = false;

  function schedule(delayMs: number): void {
    timer = setTimeout(run, delayMs);
  }

  function waitForInFlightRead(): void {
    const t = now();
    waitingSince ??= ownReadStartedAt ?? t;
    if (ownReadStartedAt === null) foreignReadSeenAt ??= t;
    if (!reportedStuck && t - waitingSince >= STUCK_AFTER_MS) {
      reportedStuck = true;
      onStuck(t - waitingSince);
    }
    schedule(IN_FLIGHT_RECHECK_MS);
  }

  function startRefresh(): void {
    const startedAt = now();
    lastRefreshAt = startedAt;
    ownReadStartedAt = startedAt;
    const settle = (): void => {
      if (disposed) return;
      ownReadStartedAt = null;
      lastReadMs = lastReadFailed() ? MAX_REFRESH_GAP_MS : now() - startedAt;
    };
    void refresh().then(settle, settle);
  }

  // Seen from its first sighting, not its real start, so the cost is a lower
  // bound: enough to space by, and never shorter than one this throttle
  // measured itself.
  function countForeignRead(seenAt: number): void {
    foreignReadSeenAt = null;
    lastRefreshAt = Math.max(lastRefreshAt ?? seenAt, seenAt);
    const cost = lastReadFailed() ? MAX_REFRESH_GAP_MS : now() - seenAt;
    lastReadMs = Math.max(lastReadMs ?? 0, cost);
  }

  function run(): void {
    timer = null;
    if (disposed) return;
    if (isFetching()) {
      waitForInFlightRead();
      return;
    }
    waitingSince = null;
    if (foreignReadSeenAt !== null) countForeignRead(foreignReadSeenAt);
    // Each stuck episode is its own failure: one reported earlier in this
    // mount must not silence the next.
    reportedStuck = false;
    // Re-derived here rather than at request time: a request that lands while
    // the first read is still out has no cost to go on yet, and waiting for it
    // above is what supplies one.
    const dueAt = lastRefreshAt === null ? 0 : lastRefreshAt + refreshGapFor(lastReadMs);
    const wait = dueAt - now();
    if (wait > 0) {
      schedule(wait);
      return;
    }
    startRefresh();
  }

  function request(): void {
    // A pending timer already covers this request: it refreshes after it,
    // and the refresh reads the graph as it stands then.
    if (disposed || timer !== null) return;
    run();
  }

  function dispose(): void {
    disposed = true;
    if (timer !== null) clearTimeout(timer);
    timer = null;
  }

  return { request, dispose };
}
