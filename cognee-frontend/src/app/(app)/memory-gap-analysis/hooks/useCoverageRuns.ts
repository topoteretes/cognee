"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQueries, useQueryClient, type UseQueryResult } from "@tanstack/react-query";
import { trackEvent } from "@/modules/analytics";
import { HttpError } from "@/services/http/errors";
import { useCogniInstance, useTenant } from "@/modules/tenant/TenantProvider";
import { useFilter } from "@/ui/layout/FilterContext";
import isCloudEnvironment from "@/utils/isCloudEnvironment";
import type { Brain, BrainLoadState, CoverageResult } from "@/app/(app)/memory-gap-analysis/types";

/** Why a "Run analysis" click never became a run, and which brain it was for. */
export interface RunStartFailure {
  brainId: string;
  message: string;
}

export interface CoverageRunsOptions {
  /**
   * Caps the fetch to the first `limit` datasets, in the order `useFilter`
   * returns them, and confines `brains` to the same set. Omitted, the hook
   * fetches and tracks every dataset. That is what Memory Gap Analysis needs:
   * the full fan-out, one query per dataset, and it has to keep getting it.
   * The dashboard passes this because its coverage panel only ever renders a
   * handful of rows, so the rest of the workspace's datasets were requests it
   * paid for and never showed.
   */
  limit?: number;
}

interface CoverageRuns {
  /** The workspace's brains, each with its latest run and its own fetch state. */
  brains: Brain[];
  /** True while the workspace's dataset list itself is still loading. */
  loading: boolean;
  /** True when the dataset list could not be fetched at all. */
  unavailable: boolean;
  /** Re-fetches the latest run per brain — the retry behind every error state. */
  refresh: () => void;
  /** Starts a replay-and-judge run for one brain and begins polling it. */
  startRun: (brainId: string) => Promise<void>;
  /** The brain whose POST is in flight right now, if any. */
  startingBrainId: string | null;
  /** Set when the last start attempt was rejected; cleared once it is stale. */
  runError: RunStartFailure | null;
}

// A run replays every distinct question the brain was ever asked, so it takes
// minutes. The backend answers the POST immediately with a pending row and
// does the work in the background; this is how often we ask whether it landed.
const RUN_POLL_INTERVAL_MS = 4_000;
// A cold or loaded pod can be slow to answer (see COG-5722); a background poll
// should not report failure just because the pod took its time.
const COVERAGE_FETCH_TIMEOUT_MS = 25_000;

/**
 * What to tell the user per status, instead of the server's own words.
 *
 * A 401 is not a failed run, it is a session that ended, and rendering the
 * pod's "Not authenticated" under "Couldn't start the run" sends the user
 * looking for a problem with their dataset.
 */
const START_FAILURE_MESSAGE: Record<number, string> = {
  401: "Your session has expired. Reload the page and sign in again.",
  403: "You do not have permission to run analysis on this dataset.",
  404: "This dataset no longer exists.",
  409: "A run is already in progress for this dataset.",
};
// A gateway answers with an HTML page, not a sentence; its markup is never the
// reason we show, and even a plain-text body has to stay one line.
const MAX_SERVER_MESSAGE_LENGTH = 160;

/** Only this page starts runs, so its name is the one on the events. */
const RUN_PAGE_NAME = "MemoryGapAnalysis";

type CoverageQuery = UseQueryResult<CoverageResult | null, Error>;

/** One brain's query, reduced to what the page reads off it. */
interface BrainState {
  result: CoverageResult | null;
  state: BrainLoadState;
  isRunning: boolean;
}

function isInFlight(result: CoverageResult | null | undefined): boolean {
  return result?.run.status === "pending" || result?.run.status === "running";
}

/**
 * One brain's own fetch state — never the page's. A retry after a failure keeps
 * react-query's status on "error" for as long as the refetch runs, so a failed
 * query that is fetching again reads as loading: that is what makes a Retry
 * click visibly do something.
 */
function loadStateFor(query: CoverageQuery | undefined): BrainLoadState {
  if (query === undefined || query.isPending) return "loading";
  if (query.isError) return query.isFetching ? "loading" : "error";
  return "loaded";
}

function startFailureMessage(error: unknown): string {
  if (error instanceof HttpError) {
    const mapped = START_FAILURE_MESSAGE[error.status];
    if (mapped !== undefined) return mapped;
    const body = error.message.trim();
    const usable = body !== "" && !body.startsWith("<") && body.length <= MAX_SERVER_MESSAGE_LENGTH;
    return usable ? body : `The workspace could not start the run (${error.status}).`;
  }
  if (error instanceof Error && error.message.length <= MAX_SERVER_MESSAGE_LENGTH) return error.message;
  return "The workspace could not start the run.";
}

/**
 * Latest coverage run per brain, for the Memory Gap Analysis page and the
 * dashboard's Memory Coverage panel. The dashboard passes `limit` to bound it
 * to the rows it actually shows, so opening it does not fetch one query per
 * dataset in the whole workspace.
 *
 * Both halves are real: the brain list is the tenant's datasets, and each
 * brain's result comes from `GET /v1/datasets/{id}/coverage` on the tenant
 * pod. A brain that has never been scored returns null rather than an error,
 * which is what the page's empty state reads.
 *
 * While a run is pending or running the query polls, so the page fills in on
 * its own once the backend finishes — the user does not have to reload.
 *
 * Everything per-brain stays per-brain: one dataset's failed fetch or running
 * run says nothing about the others, and reporting either page-wide is what
 * let a partial outage read as healthy (CLO-726).
 */
export function useCoverageRuns(options: CoverageRunsOptions = {}): CoverageRuns {
  const { limit } = options;
  const { datasets, loading, datasetsUnavailable, refreshDatasets } = useFilter();
  const { cogniInstance, isInitializing } = useCogniInstance();
  const { tenantReady } = useTenant();
  const queryClient = useQueryClient();
  const [runError, setRunError] = useState<RunStartFailure | null>(null);
  const [startingBrainId, setStartingBrainId] = useState<string | null>(null);
  // State lands a render late, so a second click inside the same tick would
  // still see `startingBrainId === null`. The ref is what actually guards the
  // POST, and a duplicate run is minutes of replay against the LLM.
  const startingRef = useRef<string | null>(null);

  // The local SDK does not serve the Cloud coverage endpoints.
  const ready = isCloudEnvironment() && !!cogniInstance && !isInitializing && tenantReady;

  // Sliced once and reused below for both the query list and `brains`, rather
  // than calling `.slice(0, limit)` again at each site: two calls in the same
  // render return equal but distinct arrays, which would give `brains`'s own
  // memo below a new dependency every render for no reason. Equal to
  // `datasets` itself, not a copy of it, when `limit` is omitted, so the
  // unbounded call this hook always supported stays on the exact array
  // identity it always had.
  const activeDatasets = useMemo(
    () => (limit === undefined ? datasets : datasets.slice(0, limit)),
    [datasets, limit],
  );

  const brainStates = useQueries({
    queries: activeDatasets.map((dataset) => ({
      queryKey: ["coverage-run", dataset.id],
      queryFn: async ({ signal }: { signal: AbortSignal }): Promise<CoverageResult | null> => {
        if (!cogniInstance) throw new Error("cogniInstance unavailable");
        const init: RequestInit & { timeoutMs?: number } = { signal, timeoutMs: COVERAGE_FETCH_TIMEOUT_MS };
        // The pod client rejects on any non-2xx (services/http/client), so a
        // failed fetch is already this query's error — there is no non-ok
        // Response to check for here.
        const response = await cogniInstance.fetch(`/v1/datasets/${dataset.id}/coverage`, init);
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
    // `useQueries` hands back a new array on every render, which defeated every
    // memo downstream. The combined value is structurally shared, so the
    // derivations below only recompute when a query actually changed.
    combine: (results): BrainState[] =>
      results.map((query) => ({
        result: query.data ?? null,
        state: loadStateFor(query),
        isRunning: isInFlight(query.data),
      })),
  });

  const brains: Brain[] = useMemo(
    () =>
      activeDatasets.map((dataset, index) => ({
        id: dataset.id,
        name: dataset.name,
        result: brainStates[index]?.result ?? null,
        state: brainStates[index]?.state ?? "loading",
        isRunning: brainStates[index]?.isRunning ?? false,
      })),
    [activeDatasets, brainStates],
  );

  // A start failure describes one click on one brain. Once that brain reports a
  // run in flight the POST evidently landed (a slow gateway can reject the
  // response after the pod queued the work), and once the brain is gone the
  // message belongs to nothing at all — a workspace switch used to bring a
  // minutes-old failure back on screen.
  useEffect(() => {
    if (runError === null) return;
    // Existence is asked of the workspace's dataset list, not of `brains`: with
    // `limit` set, `brains` is only the fetched window, and a failure for a
    // dataset outside it would look like a dataset that no longer exists and be
    // cleared in the same flush, before the caller ever saw it. `isRunning`
    // still comes from `brains`, since only a fetched brain has a query that
    // can report a run in flight.
    const stillExists = datasets.some((dataset) => dataset.id === runError.brainId);
    const brain = brains.find((candidate) => candidate.id === runError.brainId);
    if (!stillExists || brain?.isRunning === true) setRunError(null);
  }, [brains, datasets, runError]);

  const refresh = useCallback((): void => {
    if (!isCloudEnvironment()) return;
    // The dataset list too: when it is the list that failed, the page has no
    // brains at all, and a retry that only refetched coverage would look
    // broken because there was nothing to refetch it for.
    refreshDatasets();
    // Prefix-matched, so it marks every cached brain stale, not just the
    // fetched window. Harmless while the two consumers live on separate routes,
    // because invalidateQueries only refetches queries that have an observer:
    // the day coverage renders over the dashboard, this refetches the whole
    // workspace again and the bound is gone.
    void queryClient.invalidateQueries({ queryKey: ["coverage-run"] });
  }, [queryClient, refreshDatasets]);

  const startRun = useCallback(
    async (brainId: string): Promise<void> => {
      if (!isCloudEnvironment()) {
        setRunError({ brainId, message: "Memory Coverage is a Cognee Cloud feature." });
        return;
      }
      if (startingRef.current !== null) return;
      if (!cogniInstance) {
        // Not a silent return: this window lasts as long as the workspace
        // takes to come online, and the Run button is live throughout it.
        setRunError({ brainId, message: "The workspace is still coming online. Try again in a moment." });
        return;
      }
      startingRef.current = brainId;
      setStartingBrainId(brainId);
      setRunError(null);
      trackEvent({
        pageName: RUN_PAGE_NAME,
        eventName: "coverage_run_started",
        additionalProperties: { dataset_id: brainId },
      });
      try {
        await cogniInstance.fetch(`/v1/datasets/${brainId}/coverage/runs`, { method: "POST" });
      } catch (e) {
        // A rejected POST used to be an unhandled rejection: the user clicked
        // Run and the page did not move. The reason has to reach the page.
        const message = startFailureMessage(e);
        console.error(`Coverage run could not be started for ${brainId}: ${message}`);
        setRunError({ brainId, message });
        // A failure the user can see and we cannot is how a broken Run button
        // stays broken for a release; `dataset_upload_failed` is the precedent.
        trackEvent({
          pageName: RUN_PAGE_NAME,
          eventName: "coverage_run_start_failed",
          additionalProperties: {
            dataset_id: brainId,
            error_status: e instanceof HttpError ? String(e.status) : "none",
            error_message: message,
          },
        });
        return;
      } finally {
        startingRef.current = null;
        setStartingBrainId(null);
      }
      // Refetch immediately so the pending run appears, which also switches
      // this brain's query into polling until it completes. Not awaited: the
      // run has already started, and this promise must not be able to reject
      // the call and become the unhandled rejection this fixes.
      void queryClient.invalidateQueries({ queryKey: ["coverage-run", brainId] });
    },
    [cogniInstance, queryClient],
  );

  return {
    brains,
    loading,
    unavailable: datasetsUnavailable,
    refresh,
    startRun,
    startingBrainId,
    runError,
  };
}
