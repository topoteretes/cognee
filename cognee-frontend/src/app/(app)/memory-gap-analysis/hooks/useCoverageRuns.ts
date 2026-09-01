"use client";

import { useCallback, useMemo } from "react";
import { useQueries, useQueryClient } from "@tanstack/react-query";
import { useCogniInstance, useTenant } from "@/modules/tenant/TenantProvider";
import { useFilter } from "@/ui/layout/FilterContext";
import type { Brain, CoverageResult } from "@/app/(app)/memory-gap-analysis/types";

interface CoverageRuns {
  /** The workspace's brains, each with its latest run (null while unscored). */
  brains: Brain[];
  loading: boolean;
  /** True when brains exist but no run data can be fetched for any of them. */
  unavailable: boolean;
  /** True while any brain has a run that has not finished yet. */
  isRunning: boolean;
  /** Re-fetches the latest run per brain. */
  refresh: () => void;
  /** Starts a replay-and-judge run for one brain and begins polling it. */
  startRun: (brainId: string) => Promise<void>;
}

// A run replays every distinct question the brain was ever asked, so it takes
// minutes. The backend answers the POST immediately with a pending row and
// does the work in the background; this is how often we ask whether it landed.
const RUN_POLL_INTERVAL_MS = 4_000;
// A cold or loaded pod can be slow to answer (see COG-5722); a background poll
// should not report failure just because the pod took its time.
const COVERAGE_FETCH_TIMEOUT_MS = 25_000;

function isInFlight(result: CoverageResult | null | undefined): boolean {
  return result?.run.status === "pending" || result?.run.status === "running";
}

/**
 * Latest coverage run per brain, for the Memory Gap Analysis page.
 *
 * Both halves are real: the brain list is the tenant's datasets, and each
 * brain's result comes from `GET /v1/datasets/{id}/coverage` on the tenant
 * pod. A brain that has never been scored returns null rather than an error,
 * which is what the page's empty state reads.
 *
 * While a run is pending or running the query polls, so the page fills in on
 * its own once the backend finishes — the user does not have to reload.
 */
export function useCoverageRuns(): CoverageRuns {
  const { datasets, loading } = useFilter();
  const { cogniInstance, isInitializing } = useCogniInstance();
  const { tenantReady } = useTenant();
  const queryClient = useQueryClient();

  const ready = !!cogniInstance && !isInitializing && tenantReady;

  const results = useQueries({
    queries: datasets.map((dataset) => ({
      queryKey: ["coverage-run", dataset.id],
      queryFn: async ({ signal }: { signal: AbortSignal }): Promise<CoverageResult | null> => {
        if (!cogniInstance) throw new Error("cogniInstance unavailable");
        const init: RequestInit & { timeoutMs?: number } = { signal, timeoutMs: COVERAGE_FETCH_TIMEOUT_MS };
        const response = await cogniInstance.fetch(`/v1/datasets/${dataset.id}/coverage`, init);
        if (!response.ok) throw new Error(`Coverage fetch failed: ${response.status}`);
        // The endpoint returns null for a brain that has never been scored,
        // which is a valid answer rather than a missing one.
        return (await response.json()) as CoverageResult | null;
      },
      enabled: ready,
      // Poll only while something is actually in flight, so an idle page
      // costs one request per brain rather than one every few seconds.
      refetchInterval: (query: { state: { data?: CoverageResult | null } }) =>
        isInFlight(query.state.data) ? RUN_POLL_INTERVAL_MS : false,
      staleTime: 15_000,
      retry: false,
    })),
  });

  const brains: Brain[] = useMemo(
    () =>
      datasets.map((dataset, index) => ({
        id: dataset.id,
        name: dataset.name,
        result: results[index]?.data ?? null,
      })),
    [datasets, results],
  );

  const refresh = useCallback((): void => {
    void queryClient.invalidateQueries({ queryKey: ["coverage-run"] });
  }, [queryClient]);

  const startRun = useCallback(
    async (brainId: string): Promise<void> => {
      if (!cogniInstance) return;
      const response = await cogniInstance.fetch(`/v1/datasets/${brainId}/coverage/runs`, {
        method: "POST",
      });
      if (!response.ok) throw new Error(`Could not start the run: ${response.status}`);
      // Refetch immediately so the pending run appears, which also switches
      // this brain's query into polling until it completes.
      await queryClient.invalidateQueries({ queryKey: ["coverage-run", brainId] });
    },
    [cogniInstance, queryClient],
  );

  return {
    brains,
    loading: loading || (ready && results.some((result) => result.isPending)),
    // Only a real fetch failure counts as unavailable. A brain that simply has
    // no run yet is not an outage, and must keep the "not scored" empty state.
    unavailable: brains.length > 0 && results.length > 0 && results.every((result) => result.isError),
    isRunning: results.some((result) => isInFlight(result.data)),
    refresh,
    startRun,
  };
}
