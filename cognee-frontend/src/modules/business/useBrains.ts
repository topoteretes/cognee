"use client";

import { useQuery } from "@tanstack/react-query";
import type { CogneeInstance } from "@/modules/instances/types";
import getBrains from "./getBrains";

// Scoped by instanceId — without it, switching tenants kept serving the previous
// tenant's cached /visualize/brains response under a shared literal key
// (COG-6233: a stale node_set from another tenant rendered as a Source tile
// on a dataset that had zero real data).
export function businessBrainsQueryKey(instanceId: string): readonly ["business-brains", string] {
  return ["business-brains", instanceId] as const;
}

// Fetched once per session (plus normal staleness), NOT polled: this payload
// carries the full graph of every readable dataset, so an 8s poll here cost
// O(#datasets) graph builds per tick per viewer — hundreds of requests per
// open tab (CLO-597). Live growth of the FOCUSED dataset is useBrainGraph's
// job (its own bounded poll, WebSocket after CLO-598); this list only feeds
// the switcher, auto-select, and the empty state. The switcher's per-dataset
// source-name previews therefore refresh only on staleness + window focus —
// a deliberate lag for what are just labels; anything live (counts, the
// graph itself) reads from the content layer instead.
export function useBrains(cogniInstance: CogneeInstance) {
  return useQuery({
    queryKey: businessBrainsQueryKey(cogniInstance.instanceId),
    queryFn: () => getBrains(cogniInstance),
    staleTime: 60_000,
  });
}
