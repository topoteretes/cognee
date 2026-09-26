"use client";

import { useEffect, useState } from "react";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import { useTenant } from "@/modules/tenant/TenantContext";
import { listSessions, type SessionRow } from "@/modules/sessions/getSessions";
import { CogneeInstance } from "@/modules/instances/types";
import {
  useConnectedIntegrations,
  INTEGRATION_SESSION_PREFIX,
} from "@/app/(app)/dashboard/hooks/useConnectedIntegrations";

const UNCONNECTED: Record<string, boolean> = Object.fromEntries(
  Object.keys(INTEGRATION_SESSION_PREFIX).map((key) => [key, false]),
);

// Backend `AgentConnection.type` values (see KNOWN_AGENT_CONNECTION_TYPES in
// cognee/modules/agents/models.py) → the integration card keys used here and
// in INTEGRATION_SESSION_PREFIX. Only the types with a card need an entry.
const CONNECTION_TYPE_TO_INTEGRATION_KEY: Record<string, string> = {
  claude_code: "claude-code",
  codex: "codex",
};

interface AgentConnectionRow {
  type?: string;
  status?: string;
}

interface AgentConnectionsPage {
  // AgentsListResponse.agents (cognee/modules/agents/models.py) — the field
  // is named `agents`, not `connections`, despite the endpoint's own path.
  agents?: AgentConnectionRow[];
}

/**
 * A registered agent counts as connected as soon as GET /agents/connections
 * reports it `active`, even if it has not produced a session row yet (a fresh
 * registration can precede its first session/Q&A by a noticeable gap). This
 * is intentionally independent of the session-derived detection in
 * useConnectedIntegrations — that one stays as a fallback for agents that
 * never call the register endpoint but do show up in session traffic.
 */
export function deriveConnectedFromAgentConnections(
  connections: AgentConnectionRow[],
): Record<string, boolean> {
  const result: Record<string, boolean> = {};
  for (const connection of connections) {
    if (connection.status !== "active") continue;
    const key = connection.type && CONNECTION_TYPE_TO_INTEGRATION_KEY[connection.type];
    if (key) result[key] = true;
  }
  return result;
}

async function listActiveAgentConnections(
  instance: CogneeInstance,
): Promise<AgentConnectionRow[]> {
  try {
    const response = await instance.fetch("/v1/agents/connections?active_only=true&limit=500");
    if (!response.ok) return [];
    const page = (await response.json()) as AgentConnectionsPage;
    return Array.isArray(page.agents) ? page.agents : [];
  } catch (err) {
    console.warn(
      "[useAgentConnectionStatus] listActiveAgentConnections failed:",
      err instanceof Error ? err.message : err,
    );
    return [];
  }
}

/**
 * One-shot (not polled) session lookup for the Agents card badges — this page
 * just needs "did this agent ever check in", not the dashboard's live feed.
 * Delegates the actual session_id → integration matching to the existing
 * useConnectedIntegrations hook so both pages agree on what "Connected" means.
 * That hook only ever reports `true` (sticky, never flips back to false), so
 * we seed every known key at `false` first and let real detections override it —
 * otherwise cards that were never connected would render no badge at all.
 *
 * Merges in the registry's own active-connections view so a registered agent
 * shows Connected immediately, before it has any session-derived evidence
 * (see issue: integration cards ignored active agent connections until a
 * session existed).
 */
export function useAgentConnectionStatus(): Record<string, boolean> {
  const { cogniInstance, isInitializing } = useCogniInstance();
  const { tenant, tenantReady } = useTenant();
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [registryConnected, setRegistryConnected] = useState<Record<string, boolean>>({});

  useEffect(() => {
    if (!cogniInstance || isInitializing || !tenantReady) return;
    let cancelled = false;
    listSessions(cogniInstance, { range: "24h", limit: 50 })
      .then((page) => { if (!cancelled) setSessions(page.sessions); })
      .catch(() => { if (!cancelled) setSessions([]); });
    listActiveAgentConnections(cogniInstance).then((connections) => {
      if (!cancelled) setRegistryConnected(deriveConnectedFromAgentConnections(connections));
    });
    return () => { cancelled = true; };
  }, [cogniInstance, isInitializing, tenantReady]);

  const detected = useConnectedIntegrations(sessions, tenant?.tenant_id ?? null);
  return { ...UNCONNECTED, ...detected, ...registryConnected };
}
