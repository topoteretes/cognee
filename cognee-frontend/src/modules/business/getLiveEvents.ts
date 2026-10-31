import type { CogneeInstance } from "@/modules/instances/types";
import type { LiveEventsPayload } from "./types";
import { filterSessionEvents } from "./sessionEvent";

// Same acceptance test the socket applies to its `live_events` frame. Both
// endpoints are served off one event store at the pod, so validating only the
// socket meant an event shape this guard rejects rendered over the poll and
// vanished over the socket — the same session showing a different history
// depending on which transport happened to be up.
function normalizeLiveEvents(
  payload: unknown,
  onDroppedEvents?: (dropped: number, total: number) => void,
): LiveEventsPayload {
  if (typeof payload !== "object" || payload === null) {
    throw new Error("live-events response was not an object");
  }
  const { events, cursor } = payload as { events?: unknown; cursor?: unknown };
  // Thrown, not defaulted to []: an unreadable payload is a failed tick, and
  // pollLiveEvents' catch already backs off and flips the live indicator.
  // Swallowing it would read to the caller as "this dataset has no activity".
  if (!Array.isArray(events)) throw new Error("live-events response carried no events array");
  return {
    events: filterSessionEvents(events, onDroppedEvents),
    // One cursor policy, shared with the socket (see liveSocket's parseFrame):
    // null means "do not advance the resume point", and pollLiveEvents is what
    // honours that by leaving the previous value in place. It must never be
    // read as "start from the beginning".
    cursor: typeof cursor === "string" ? cursor : null,
  };
}

// Delta of search/improve events since a cursor — polled by
// useBusinessLiveUpdates instead of re-fetching the whole graph payload.
// `datasetId` only gates authorization; the events themselves are the
// caller's own, not filtered to that dataset (see get_live_events docstring).
// Verified server-side: the backend route runs the same
// get_authorized_existing_datasets read-permission check as every other
// visualize endpoint and returns 403 on a dataset the caller can't read, so
// a guessed dataset_id cannot pull another tenant's session events.
export default function getLiveEvents(
  datasetId: string,
  since: string | null,
  instance: CogneeInstance,
  // Injected rather than built here, so this stays a fetcher and the caller
  // owns the reporting context — the same shape liveSocket uses for its half
  // of the signal.
  onDroppedEvents?: (dropped: number, total: number) => void,
): Promise<LiveEventsPayload> {
  const params = new URLSearchParams({ dataset_id: datasetId });
  if (since) params.set("since", since);
  return instance
    .fetch(`/v1/visualize/live-events?${params.toString()}`)
    .then((response) => response.json())
    .then((payload) => normalizeLiveEvents(payload, onDroppedEvents));
}
