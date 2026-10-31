"use client";

import { useEffect } from "react";
import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import type { CogneeInstance } from "@/modules/instances/types";
import { useCircuitBreaker } from "@/modules/query/useCircuitBreaker";
import getBrainGraph from "./getBrainGraph";
import type { VisualizationPayload } from "./types";

// Scoped by instanceId for the same reason as businessBrainsSummaryQueryKey (COG-6233:
// an unscoped key served another tenant's cached graph after switching).
// Exported so useBusinessLiveUpdates can invalidate exactly this dataset's
// entry when the socket reports `graph_grew`, instead of polling it.
export function brainGraphQueryKey(
  instanceId: string,
  datasetId: string,
): readonly ["business-brain-graph", string, string] {
  return ["business-brain-graph", instanceId, datasetId] as const;
}

// Growth signal for the FOCUSED dataset's graph only, replacing /brains'
// every-dataset 8s poll (CLO-597). React Query's structural sharing keeps
// node/link references stable across same-data refetches — useBusinessSimulation
// relies on that to avoid tearing down the force simulation on every refresh.
const GRAPH_REFETCH_INTERVAL_MS = 8_000;

export function useBrainGraph(
  cogniInstance: CogneeInstance,
  datasetId: string | null,
  // Whether the live socket is currently delivering `graph_grew` pushes for
  // this dataset (useBusinessLiveUpdates reports it). Defaults to false so
  // every caller that does not know polls, which is the safe direction: an
  // unnecessary poll costs one request, a missing one costs every growth
  // update until the page is remounted.
  graphPushActive = false,
): UseQueryResult<VisualizationPayload> {
  // Polled-query discipline (see modules/query/backgroundQueryRetry.ts): the
  // next tick is the retry, so a struggling pod isn't hit three extra times
  // per tick, and the breaker backs the cadence off to a minute once ticks
  // keep failing — matching useDashboardTelemetry, the other polled pod query.
  const breaker = useCircuitBreaker(GRAPH_REFETCH_INTERVAL_MS);

  const query = useQuery({
    queryKey: brainGraphQueryKey(cogniInstance.instanceId, datasetId ?? ""),
    queryFn: ({ signal, client, queryKey }) => getBrainGraph(
      cogniInstance, datasetId ?? "", signal,
      // The previous payload of this dataset (the key is dataset-scoped) lets
      // a refresh reuse row values already read instead of re-fetching them.
      client.getQueryData<VisualizationPayload>(queryKey),
    ),
    enabled: datasetId !== null,
    // Matches the poll's BASE interval: this payload is multi-MB and
    // uncompressed (getBrainGraph carries the measurements), and a staleTime
    // shorter than the poll meant every remount and every window focus pulled
    // it again on top of the tick that was already due. Under breaker backoff
    // the poll stretches to a minute while this stays at 8s, so the two are
    // not locked together - only the common case is. Governs
    // mount/focus/reconnect refetches only; refetchInterval runs on its own
    // timer regardless.
    staleTime: GRAPH_REFETCH_INTERVAL_MS,
    // Off only while pushes are actually arriving: `graph_grew` invalidates
    // this exact query key (see useBusinessLiveUpdates), so polling on top of
    // a healthy socket would be redundant traffic. Gating this on the
    // instance merely BEING socket-capable was a silent trap - createPodClient
    // sets socketTarget unconditionally, so a pod that does not serve the
    // endpoint (or any outage after it does) left this query with no growth
    // signal at all, for the rest of the session, with nothing on screen to
    // say so.
    refetchInterval: graphPushActive ? false : breaker.refetchInterval,
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
