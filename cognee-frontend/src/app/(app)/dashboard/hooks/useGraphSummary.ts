"use client";

import { useState, useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { useCogniInstance, useTenant } from "@/modules/tenant/TenantProvider";
import type { Dataset } from "@/ui/layout/FilterContext";
import type { PipelineRun } from "@/ui/elements/AgentActivityTerminal";
import getDatasetGraphSummary from "@/modules/datasets/getDatasetGraphSummary";
import { getCachedGraphNodes, setCachedGraphNodes } from "@/utils/browserStorage";
import { BACKGROUND_QUERY_RETRY_COUNT, backgroundQueryRetryDelay } from "@/modules/query/backgroundQueryRetry";

// Extra headroom for background fetches — slow pods (COG-5722) shouldn't surface
// as false errors when they're still processing.
const BACKGROUND_POLL_TIMEOUT_MS = 25_000;
// GraphMetrics is computed asynchronously on the backend after cognify
// finishes — a refetch fired the instant a run shows "completed" can still
// read a stale/degraded row (see DatasetGraphSummary.computedAt), so a
// second check shortly after gives that computation time to land.
const RECHECK_AFTER_COMPLETION_MS = 15_000;
// Safety net for pipeline runs this tab's telemetry poll never saw transition
// (started elsewhere — another tab, an agent, a direct API call) — without
// this, the count would only catch up whenever the dataset selection happens
// to change.
const IDLE_FALLBACK_POLL_MS = 2 * 60 * 1000;

export interface GraphSummary {
  graphNodes: number | null;
  graphEdges: number | null;
}

/**
 * Fetches node/edge counts for the current dataset scope via the precomputed
 * /graph-summary endpoint (COG-5726). Hydrates from cache so the last-known
 * count renders immediately before the first fetch completes.
 */
export function useGraphSummary(
  selectedDataset: Dataset | null,
  datasets: Dataset[],
  runs: PipelineRun[] = [],
): GraphSummary {
  const { cogniInstance } = useCogniInstance();
  const { tenantReady } = useTenant();
  const [graphNodes, setGraphNodes] = useState<number | null>(getCachedGraphNodes);
  const [graphEdges, setGraphEdges] = useState<number | null>(null);

  const datasetsToFetch = selectedDataset ? [selectedDataset] : datasets;
  // Keyed on sorted ids (not the array reference) so this doesn't refire on
  // every 15s datasets poll unless the set of ids actually changed.
  const datasetIdsKey = datasetsToFetch.map((d) => d.id).sort().join(",");

  const graphQuery = useQuery({
    queryKey: ["dataset-graph-summary", datasetIdsKey],
    queryFn: async ({ signal }) => {
      if (!cogniInstance) throw new Error("cogniInstance unavailable");
      if (datasetsToFetch.length === 0) return { totalNodes: 0, totalEdges: 0 };
      const ids = selectedDataset ? [selectedDataset.id] : undefined;
      const summaries = await getDatasetGraphSummary(
        cogniInstance,
        ids,
        signal,
        BACKGROUND_POLL_TIMEOUT_MS,
      );
      let totalNodes = 0;
      let totalEdges = 0;
      for (const s of summaries) {
        totalNodes += s.numNodes;
        totalEdges += s.numEdges;
      }
      return { totalNodes, totalEdges };
    },
    // tenantReady, not just cogniInstance: see useDashboardTelemetry.ts for
    // why cogniInstance alone isn't enough — a fresh workspace's pod can
    // still be unreachable while cogniInstance already exists.
    enabled: !!cogniInstance && tenantReady,
    // Not a one-shot query anymore: refetchOnPipelineCompletion below handles
    // the real trigger (a run just finished), this interval is only the idle
    // fallback for runs this tab's telemetry poll never saw transition.
    refetchInterval: IDLE_FALLBACK_POLL_MS,
    retry: BACKGROUND_QUERY_RETRY_COUNT,
    retryDelay: backgroundQueryRetryDelay,
  });

  useEffect(() => {
    if (!graphQuery.data) return;
    setGraphNodes(graphQuery.data.totalNodes);
    setGraphEdges(graphQuery.data.totalEdges);
    setCachedGraphNodes(graphQuery.data.totalNodes);
  }, [graphQuery.data]);

  // Detects a run transitioning to "completed" against the previous poll's
  // snapshot (same technique as AgentConnectionSection's session baseline)
  // and refetches — once immediately, once again after GraphMetrics has had
  // time to catch up. The first poll only ever populates the baseline: every
  // run would otherwise look "newly completed" on mount.
  const prevRunStatusRef = useRef<Map<string, string> | null>(null);
  useEffect(() => {
    const prev = prevRunStatusRef.current;
    const next = new Map(runs.map((r) => [r.pipeline_run_id || r.id, r.status]));
    prevRunStatusRef.current = next;
    if (prev === null) return;

    const justCompleted = runs.some((r) => {
      const key = r.pipeline_run_id || r.id;
      return r.status.includes("COMPLETED") && !prev.get(key)?.includes("COMPLETED");
    });
    if (!justCompleted) return;

    graphQuery.refetch();
    const timer = setTimeout(() => graphQuery.refetch(), RECHECK_AFTER_COMPLETION_MS);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runs]);

  return { graphNodes, graphEdges };
}
