"use client";

import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import type { CogneeInstance } from "@/modules/instances/types";
import getBrainsSummary from "./getBrainsSummary";
import type { BrainsSummaryPayload } from "./types";

// Scoped by instanceId — without it, switching tenants kept serving the previous
// tenant's cached response under a shared literal key (COG-6233: a stale
// node_set from another tenant rendered as a Source tile on a dataset that
// had zero real data).
export function businessBrainsSummaryQueryKey(instanceId: string): readonly ["business-brains-summary", string] {
  return ["business-brains-summary", instanceId] as const;
}

// Fetched once per session (plus normal staleness), NOT polled. Nothing on
// the critical path waits for it: the dataset list itself comes from the
// governance layer (see BusinessView's auto-select), and live growth of the
// FOCUSED dataset is useBrainGraph's job. This only enriches the switcher's
// per-dataset source-name preview, so it refreshes on staleness plus window
// focus — a deliberate lag for what are just labels.
export function useBrainsSummary(cogniInstance: CogneeInstance): UseQueryResult<BrainsSummaryPayload> {
  return useQuery({
    queryKey: businessBrainsSummaryQueryKey(cogniInstance.instanceId),
    queryFn: () => getBrainsSummary(cogniInstance),
    staleTime: 60_000,
  });
}
