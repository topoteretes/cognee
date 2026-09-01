"use client";

import { useQuery } from "@tanstack/react-query";
import { useCogniInstance, useTenant } from "@/modules/tenant/TenantProvider";
import { listSessions, type SessionRow, type TimeRange } from "@/modules/sessions/getSessions";
import type { PipelineRun } from "@/ui/elements/AgentActivityTerminal";

// A browsable table can afford a slower, larger fetch than the dashboard's
// live-glance panel (useDashboardTelemetry: limit 50, 15s poll) — this loads
// once per range change instead of polling.
const ACTIVITY_FETCH_TIMEOUT_MS = 25_000;
const SESSIONS_LIMIT = 200;

export interface ActivityFeed {
  runs: PipelineRun[];
  sessions: SessionRow[];
  loading: boolean;
  error: boolean;
  refetch: () => void;
}

export function useActivityFeed(range: TimeRange): ActivityFeed {
  const { cogniInstance, isInitializing } = useCogniInstance();
  const { tenant, tenantReady } = useTenant();

  const query = useQuery({
    queryKey: ["activity-feed", tenant?.tenant_id ?? null, range],
    queryFn: async ({ signal }) => {
      if (!cogniInstance) throw new Error("cogniInstance unavailable");
      const init: RequestInit & { timeoutMs?: number } = { signal, timeoutMs: ACTIVITY_FETCH_TIMEOUT_MS };
      const [runsRes, sessionsPage] = await Promise.all([
        cogniInstance.fetch("/v1/activity/pipeline-runs", init).then((r) => (r.ok ? r.json() : [])),
        listSessions(cogniInstance, { range, limit: SESSIONS_LIMIT }, { signal, timeoutMs: ACTIVITY_FETCH_TIMEOUT_MS }),
      ]);
      return {
        runs: (Array.isArray(runsRes) ? runsRes : []) as PipelineRun[],
        sessions: sessionsPage?.sessions ?? [],
      };
    },
    enabled: !!cogniInstance && !isInitializing && tenantReady,
    staleTime: 15_000,
    retry: false,
  });

  return {
    runs: query.data?.runs ?? [],
    sessions: query.data?.sessions ?? [],
    loading: query.isPending,
    error: query.isError,
    refetch: () => { void query.refetch(); },
  };
}
