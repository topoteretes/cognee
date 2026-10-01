import type { CogneeInstance } from "@/modules/instances/types";
import type { LiveEventsPayload } from "./types";

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
): Promise<LiveEventsPayload> {
  const params = new URLSearchParams({ dataset_id: datasetId });
  if (since) params.set("since", since);
  return instance
    .fetch(`/v1/visualize/live-events?${params.toString()}`)
    .then((response) => response.json());
}
