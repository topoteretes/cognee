import { SLOW_READ_TIMEOUT_MS } from "@/services/http";
import type { CogneeInstance } from "@/modules/instances/types";
import type { BrainsSummaryPayload } from "./types";

// Every dataset the caller may read, described without reading a single
// graph. Replaces GET /visualize/brains, which answered the same question by
// building one bounded graph per dataset: measured on an 11-dataset
// workspace that ran 10405 ms and was cut off by the client's 10s GET
// budget, transferring nothing. This one is served from relational metadata
// and the per-cognify-run count cache, so it costs nothing per dataset on a
// warm cache (COG-6258).
//
// Already group-aware (CLO-399's get_all_user_permission_datasets union), so
// a dataset shared with a group appears here with no extra frontend logic.
export default function getBrainsSummary(instance: CogneeInstance): Promise<BrainsSummaryPayload> {
  // Cheap is not the same as fast under load. This read shares the pod with
  // whatever cognify run is in flight, and measured there it went from 421 ms
  // idle to past 10 s twice in three minutes — the gateway logged both as
  // `http2.remote_reset` at 9995/9999 ms, i.e. the client giving up, not the
  // server failing. The default GET budget therefore reintroduces exactly the
  // cut-off this endpoint was created to escape (see SLOW_READ_TIMEOUT_MS).
  return instance
    .fetch("/v1/visualize/brains-summary", { timeoutMs: SLOW_READ_TIMEOUT_MS })
    .then((response) => response.json());
}
