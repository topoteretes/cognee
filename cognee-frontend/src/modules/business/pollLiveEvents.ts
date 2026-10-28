import type { CogneeInstance } from "@/modules/instances/types";
import { HttpError } from "@/services/http/errors";
import getLiveEvents from "./getLiveEvents";
import type { SessionEvent } from "./types";

const POLL_MS = 1500;
const MAX_BACKOFF_MS = 10000;

export interface PollLiveEventsCallbacks {
  // Raw batch straight off the wire — the caller owns dedupe/append so both
  // transports can share one path (see useBusinessLiveUpdates).
  onFresh: (events: SessionEvent[], cursor: string | null) => void;
  onLiveChange: (live: boolean) => void;
  // Passed straight to getLiveEvents, so the poll reports dropped events with
  // the caller's context instead of this module reaching for Sentry itself.
  onDroppedEvents?: (dropped: number, total: number) => void;
}

// A ref-shaped cursor holder rather than an internal variable: this is the
// poll fallback's half of a transport handover, so the caller (which also
// owns the socket) must be able to seed it with the socket's last cursor and
// read where the poll left off.
export interface CursorRef {
  current: string | null;
}

// Ports the live-events poller (customer_tutorial.html:7086-7194) minus its
// location.reload() growth path, moved out of useBusinessLiveUpdates verbatim
// (CLO-604) so it can run as the WebSocket's give-up fallback instead of the
// only transport.
export function pollLiveEvents(
  datasetId: string,
  cogniInstance: CogneeInstance,
  cursorRef: CursorRef,
  callbacks: PollLiveEventsCallbacks,
): () => void {
  let cancelled = false;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let failures = 0;

  async function poll(): Promise<void> {
    try {
      const { events: fresh, cursor } = await getLiveEvents(
        datasetId,
        cursorRef.current,
        cogniInstance,
        callbacks.onDroppedEvents,
      );
      if (cancelled) return;
      failures = 0;
      // One cursor policy, shared with the socket (see liveSocket's
      // parseFrame): an unusable cursor does not advance the resume point and
      // does not reset it. Assigning null here made the next request drop
      // `since` and replay the entire history every 1.5s, which is the
      // opposite of what a missing cursor should cost. The 409 branch below
      // still resets deliberately, which is a different thing.
      if (cursor !== null) cursorRef.current = cursor;
      callbacks.onLiveChange(true);
      callbacks.onFresh(fresh, cursor);
    } catch (error) {
      if (!cancelled) {
        failures++;
        callbacks.onLiveChange(false);
        // A 409 means the backend rejected this exact cursor (e.g. it aged
        // out) — retrying with the same `since` value only produces the
        // same 409 forever, which is exactly what left the indicator
        // stuck on "reconnecting" indefinitely (COG-6412: traced via a
        // live network capture, same stale cursor repeating with no
        // backoff-driven change). Dropping the cursor makes the next poll
        // ask for a fresh snapshot instead of repeating a rejected delta.
        if (error instanceof HttpError && error.status === 409) {
          cursorRef.current = null;
        }
      }
    } finally {
      // Back off (max 10s) while the backend is unreachable; recover
      // instantly once a poll succeeds (customer_tutorial.html ~7170).
      if (!cancelled) {
        const delay = failures ? Math.min(POLL_MS * (failures + 1), MAX_BACKOFF_MS) : POLL_MS;
        timer = setTimeout(poll, delay);
      }
    }
  }

  poll();
  return () => {
    cancelled = true;
    if (timer) clearTimeout(timer);
  };
}
