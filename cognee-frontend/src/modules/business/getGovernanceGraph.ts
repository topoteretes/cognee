import { SLOW_READ_TIMEOUT_MS } from "@/services/http";
import type { CogneeInstance } from "@/modules/instances/types";
import type { VisualizationPayload } from "./types";

// The "who may do what" side of Business: tenant/user/role membership, dataset
// ownership and ACL grants, projected as the same payload shape a content
// graph comes back in (see types.ts) — so the two layers merge with no
// adapter beyond what computeBrainState already does.
export default function getGovernanceGraph(
  instance: CogneeInstance,
): Promise<VisualizationPayload> {
  // Not the default GET budget: this builds a whole-tenant provenance graph
  // from the relational store, and pod latency climbs with uptime (COG-5722
  // measured 5.9s -> 10.9s over an hour at zero concurrency), so the 10s
  // default failed it on a healthy-but-warm pod and took the rails down with
  // it.
  //
  // The canvas no longer waits on it (the dataset seed comes from
  // FilterContext), but the view's empty states still do: useBusinessScene
  // folds this into `isLoading`, which "no datasets here" and "couldn't load
  // this workspace" both gate on. So on a dataset-less workspace whose pod is
  // hanging, those messages now arrive at 45s rather than 10s. That is the
  // deliberate trade: a spurious failure on a warm pod is the common case and
  // takes the rails down with it, while the delayed message needs an empty
  // workspace AND a hanging pod at once.
  return instance
    .fetch("/v1/schema/provenance/json", { timeoutMs: SLOW_READ_TIMEOUT_MS })
    .then((response) => response.json());
}
