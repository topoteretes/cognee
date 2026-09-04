"use client";

import { useEffect } from "react";
import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import type { CogneeInstance } from "@/modules/instances/types";
import { useCircuitBreaker } from "@/modules/query/useCircuitBreaker";
import getBrainGraph from "./getBrainGraph";
import type { VisualizationPayload } from "./types";

// Scoped by instanceId for the same reason as businessBrainsQueryKey (COG-6233:
// an unscoped key served another tenant's cached graph after switching).
function brainGraphQueryKey(
  instanceId: string,
  datasetId: string,
): readonly ["business-brain-graph", string, string] {
  return ["business-brain-graph", instanceId, datasetId] as const;
}

// Interim growth signal until the dataset WebSocket lands (CLO-598): poll the
// FOCUSED dataset's graph only, replacing /brains' every-dataset 8s poll
// (CLO-597). React Query's structural sharing keeps node/link references
// stable across same-data refetches — useBusinessSimulation relies on that
// to avoid tearing down the force simulation on every poll.
const GRAPH_REFETCH_INTERVAL_MS = 8_000;

export function useBrainGraph(
  cogniInstance: CogneeInstance,
  datasetId: string | null,
): UseQueryResult<VisualizationPayload> {
  // Polled-query discipline (see modules/query/backgroundQueryRetry.ts): the
  // next tick is the retry, so a struggling pod isn't hit three extra times
  // per tick, and the breaker backs the cadence off to a minute once ticks
  // keep failing — matching useDashboardTelemetry, the other polled pod query.
  const breaker = useCircuitBreaker(GRAPH_REFETCH_INTERVAL_MS);

  const query = useQuery({
    queryKey: brainGraphQueryKey(cogniInstance.instanceId, datasetId ?? ""),
    queryFn: () => getBrainGraph(cogniInstance, datasetId ?? ""),
    enabled: datasetId !== null,
    staleTime: 5_000,
    refetchInterval: breaker.refetchInterval,
    // Pausing the poll in background tabs delivered one big catch-up refetch
    // on window focus — useBusinessSimulation reads that batched jump as a
    // real membership change, reheating and staggering every "newborn" in,
    // so the graph visibly re-settled for seconds right as the user returned
    // (COG-6233). Polling through the background keeps growth arriving in
    // small increments; it is one dataset's payload now, not every dataset's.
    refetchIntervalInBackground: true,
    retry: false,
    // The view renders the failure itself (BusinessView's error state reads
    // this through the content layer's `error`), so a failed tick must not
    // reach the nearest error boundary and take the whole page down.
    throwOnError: false,
  });

  useEffect(() => {
    breaker.report(query);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query.isError, query.isSuccess, query.dataUpdatedAt, query.errorUpdatedAt]);

  return query;
}
