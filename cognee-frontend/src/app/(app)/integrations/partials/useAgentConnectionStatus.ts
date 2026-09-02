"use client";

import { useEffect, useState } from "react";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import { useTenant } from "@/modules/tenant/TenantContext";
import { listSessions, type SessionRow } from "@/modules/sessions/getSessions";
import {
  useConnectedIntegrations,
  INTEGRATION_SESSION_PREFIX,
} from "@/app/(app)/dashboard/hooks/useConnectedIntegrations";

const UNCONNECTED: Record<string, boolean> = Object.fromEntries(
  Object.keys(INTEGRATION_SESSION_PREFIX).map((key) => [key, false]),
);

/**
 * One-shot (not polled) session lookup for the Agents card badges — this page
 * just needs "did this agent ever check in", not the dashboard's live feed.
 * Delegates the actual session_id → integration matching to the existing
 * useConnectedIntegrations hook so both pages agree on what "Connected" means.
 * That hook only ever reports `true` (sticky, never flips back to false), so
 * we seed every known key at `false` first and let real detections override it —
 * otherwise cards that were never connected would render no badge at all.
 */
export function useAgentConnectionStatus(): Record<string, boolean> {
  const { cogniInstance, isInitializing } = useCogniInstance();
  const { tenant, tenantReady } = useTenant();
  const [sessions, setSessions] = useState<SessionRow[]>([]);

  useEffect(() => {
    if (!cogniInstance || isInitializing || !tenantReady) return;
    let cancelled = false;
    listSessions(cogniInstance, { range: "24h", limit: 50 })
      .then((page) => { if (!cancelled) setSessions(page.sessions); })
      .catch(() => { if (!cancelled) setSessions([]); });
    return () => { cancelled = true; };
  }, [cogniInstance, isInitializing, tenantReady]);

  const detected = useConnectedIntegrations(sessions, tenant?.tenant_id ?? null);
  return { ...UNCONNECTED, ...detected };
}
