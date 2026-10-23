import { addBreadcrumb } from "@/utils/monitoring";
import type { SessionEvent } from "./types";

// Shared by BOTH live-events transports (liveSocket's `live_events` frame and
// getLiveEvents' HTTP poll). It lives here rather than inside either one
// because the pod serves both endpoints off the same event store: a guard on
// only one of them means the same session renders a different history
// depending on which transport happens to be up, which is worse than no guard
// at all — it makes a contract drift look like a UI bug.
function isOptionalString(value: unknown): boolean {
  return value === undefined || typeof value === "string";
}

// SessionEvent types every field as optional, which is honest about the wire
// shape but useless as an acceptance test: an object with nothing in it
// satisfies it, and useBusinessLiveUpdates keys its dedupe on
// kind/qa_id/time/question. Two such empty events collapse onto the same key
// and the second is dropped as a repeat of the first. So a usable event is
// one whose known fields have the right types AND that carries at least one
// of the two identity fields the key is really built from.
export function isSessionEvent(value: unknown): value is SessionEvent {
  if (typeof value !== "object" || value === null) return false;
  const event = value as Record<string, unknown>;
  if (!isOptionalString(event.kind) || !isOptionalString(event.time)) return false;
  if (!isOptionalString(event.qa_id) || !isOptionalString(event.question)) return false;
  if (event.node_ids !== undefined && !(Array.isArray(event.node_ids) && event.node_ids.every((id) => typeof id === "string"))) {
    return false;
  }
  return typeof event.qa_id === "string" || typeof event.time === "string";
}

// Keeps the usable events and reports how many it had to drop. Silent drops
// were the whole problem: a pod that starts emitting a shape this guard
// rejects would eat events with no signal anywhere, which is exactly the
// failure mode the socket's Sentry alerting exists to prevent.
export function filterSessionEvents(
  events: readonly unknown[],
  onDropped?: (dropped: number, total: number) => void,
): SessionEvent[] {
  const kept = events.filter(isSessionEvent);
  if (kept.length !== events.length) onDropped?.(events.length - kept.length, events.length);
  return kept;
}

// Rate limit per transport+dataset. Without it a pod-wide shape change emits
// one warning breadcrumb per poll tick — every 1.5s (pollLiveEvents) — and
// Sentry's default 100-breadcrumb buffer is full of nothing else within a
// couple of minutes, evicting the HTTP and slow-request trail pod.ts puts
// there. The signal is "this is happening", not "it happened 900 times".
const DROP_REPORT_COOLDOWN_MS = 5 * 60_000;
const lastDropReportAt = new Map<string, number>();

// Exported for tests: module-level state otherwise leaks the cooldown between
// cases and makes them order-dependent.
export function resetDropReportCooldown(): void {
  lastDropReportAt.clear();
}

// One breadcrumb shape for both transports, so a drift shows up in Sentry as
// the same signal regardless of which transport saw it first. Breadcrumb, not
// captureMessage: a drop costs one event, it is not on its own worth waking
// someone for — but it has to be on the trail when something else is.
export function reportDroppedSessionEvents(transport: "socket" | "poll", datasetId: string) {
  return (dropped: number, total: number): void => {
    const key = `${transport}|${datasetId}`;
    const now = Date.now();
    const last = lastDropReportAt.get(key) ?? 0;
    if (now - last < DROP_REPORT_COOLDOWN_MS) return;
    lastDropReportAt.set(key, now);
    addBreadcrumb({
      category: "mindmap.live-events",
      level: "warning",
      message: `dropped ${dropped}/${total} unusable live events over ${transport}`,
      data: { datasetId, transport, dropped, total },
    });
  };
}
