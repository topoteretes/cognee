"use client";

import { useTenant } from "@/modules/tenant/TenantContext";
import { lastSeenByAgent, runningAgents } from "@/modules/agents/getAgentConnections";
import type { AgentIntegrationKey } from "@/modules/agents/getAgentConnections";
import {
  useConnectedIntegrations,
  INTEGRATION_KEYS,
} from "@/app/(app)/dashboard/hooks/useConnectedIntegrations";
import { useDashboardTelemetry } from "@/app/(app)/dashboard/hooks/useDashboardTelemetry";

const UNCONNECTED: Record<string, boolean> = Object.fromEntries(
  INTEGRATION_KEYS.map((key) => [key, false]),
);

export interface AgentConnectionStatus {
  connected: Record<string, boolean>;
  /** Newest known activity per plugin-agent key (registration or session),
   *  absent when only the sticky registry says "connected" with no row to
   *  date it. See lastSeenByAgent. */
  lastSeenAt: Partial<Record<AgentIntegrationKey, string>>;
  /** Plugin agents with a running session right now, see runningAgents. */
  running: ReadonlySet<AgentIntegrationKey>;
  /** Telemetry's first poll hasn't answered yet — cards should show
   *  "Checking…" instead of a confident "Not connected yet" while this is
   *  true, unless the sticky registry already says otherwise. */
  loading: boolean;
}

/**
 * Agent card badges, read off the same polled feed the dashboard reads
 * (useDashboardTelemetry: one react-query cache entry per tenant, 15s poll,
 * paused while the tab isn't visible, circuit breaker on repeated failures).
 * This page used to run its own one-shot sessions-plus-connections fetch,
 * which meant a card here could go stale until the page was reloaded while
 * the same connection on the dashboard kept updating — two sources of truth
 * for one fact. There are now exactly two: this shared poll for everything
 * outside a tutorial, and the tutorial's own faster poll while a wizard is
 * open (useAgentConnectionState), which needs to react in seconds rather
 * than the 15s this page is content with.
 *
 * Delegates the actual matching to the existing useConnectedIntegrations
 * hook so this page and the dashboard agree on what "connected" means. That
 * hook only ever reports `true` (sticky, never flips back to false), so we
 * seed every known key at `false` first and let real detections override
 * it, otherwise cards that were never connected would render no badge at
 * all.
 */
export function useAgentConnectionStatus(): AgentConnectionStatus {
  const { tenant } = useTenant();
  const { sessions, connections, loading } = useDashboardTelemetry();
  const detected = useConnectedIntegrations(sessions, tenant?.tenant_id ?? null, connections);
  return {
    connected: { ...UNCONNECTED, ...detected },
    lastSeenAt: lastSeenByAgent(connections, sessions),
    running: runningAgents(sessions),
    loading,
  };
}
