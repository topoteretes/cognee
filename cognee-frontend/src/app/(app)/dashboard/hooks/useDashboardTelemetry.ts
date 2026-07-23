"use client";

import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { notifications } from "@mantine/notifications";
import { useCogniInstance, useTenant } from "@/modules/tenant/TenantProvider";
import { useFilter } from "@/ui/layout/FilterContext";
import { listSessions } from "@/modules/sessions/getSessions";
import type { SessionRow } from "@/modules/sessions/getSessions";
import { useCircuitBreaker } from "@/modules/query/useCircuitBreaker";
import type { PipelineRun, Range } from "@/ui/elements/AgentActivityTerminal";

const TELEMETRY_POLL_INTERVAL_MS = 15_000;
// Background polls get extra headroom because local/loaded pods can take several
// seconds per request (COG-5722) — we don't want a slow pod surfacing as a false
// error while it's still working.
const BACKGROUND_POLL_TIMEOUT_MS = 25_000;

export interface DashboardTelemetry {
  runs: PipelineRun[];
  sessions: SessionRow[];
  loading: boolean;
}

export function useDashboardTelemetry(range: Range): DashboardTelemetry {
  const { cogniInstance, isInitializing } = useCogniInstance();
  const { tenant, tenantReady } = useTenant();
  const { loading: filterLoading } = useFilter();

  const [runs, setRuns] = useState<PipelineRun[]>([]);
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [loading, setLoading] = useState(true);

  const telemetryBreaker = useCircuitBreaker(TELEMETRY_POLL_INTERVAL_MS);

  const telemetryQuery = useQuery({
    queryKey: ["dashboard-telemetry", tenant?.tenant_id ?? null, range],
    queryFn: async ({ signal }) => {
      if (!cogniInstance) throw new Error("cogniInstance unavailable");
      const init: RequestInit & { timeoutMs?: number } = { signal, timeoutMs: BACKGROUND_POLL_TIMEOUT_MS };
      // Let genuine failures (timeout/network/5xx) reject the query so the circuit
      // breaker sees them. listSessions still swallows internally (shared with pages
      // that want that leniency), but a dead pod fails pipeline-runs first anyway.
      const [runData, sessionsPage] = await Promise.all([
        cogniInstance.fetch("/v1/activity/pipeline-runs", init).then((r) => r.json()),
        listSessions(cogniInstance, { range, limit: 50 }, { signal, timeoutMs: BACKGROUND_POLL_TIMEOUT_MS }),
      ]);
      return {
        runs: (Array.isArray(runData) ? runData : []) as PipelineRun[],
        sessions: sessionsPage?.sessions ?? [],
      };
    },
    // tenantReady, not just cogniInstance: a freshly-created workspace gets a
    // cogniInstance immediately (optimistic), but the pod itself can take up
    // to a minute to actually answer — firing this against it early just
    // produces a burst of failed requests (dashboard now mounts before the
    // pod is ready, since it's excluded from TenantProvider's isInitializing
    // wait; see TenantProvider.tsx).
    enabled: !!cogniInstance && !isInitializing && !filterLoading && tenantReady,
    refetchInterval: telemetryBreaker.refetchInterval,
    refetchIntervalInBackground: false,
    // No per-fetch retry: this is a poll, so the next tick IS the retry, and
    // the circuit breaker above already tracks consecutive tick failures —
    // an extra retry here would just delay that signal.
    retry: false,
  });

  useEffect(() => {
    telemetryBreaker.report(telemetryQuery);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [telemetryQuery.isError, telemetryQuery.isSuccess, telemetryQuery.dataUpdatedAt, telemetryQuery.errorUpdatedAt]);

  // Surface the breaker opening once per episode — not on every failed tick.
  useEffect(() => {
    if (!telemetryBreaker.isOpen) return;
    notifications.show({
      title: "Workspace is having trouble responding",
      message: "We'll keep trying in the background — some data may be out of date.",
      color: "orange",
      autoClose: 8000,
    });
  }, [telemetryBreaker.isOpen]);

  useEffect(() => {
    if (!telemetryQuery.data) return;
    setRuns(telemetryQuery.data.runs);
    setSessions(telemetryQuery.data.sessions);
  }, [telemetryQuery.data]);

  // isPending (not isLoading) so `loading` stays true while the query is disabled
  // (no cogniInstance yet) — matching previous behavior where the fetch effect
  // simply never ran without an instance.
  useEffect(() => {
    if (!telemetryQuery.isPending) setLoading(false);
  }, [telemetryQuery.isPending]);

  return { runs, sessions, loading };
}
