"use client";

import { useState, useEffect } from "react";
import type { CogneeInstance } from "@/modules/instances/types";
import {
  getActiveAgentConnections,
  type ActiveAgentConnection,
} from "@/modules/agents/getActiveAgentConnections";
import type { SessionRow } from "@/modules/sessions/getSessions";
import {
  getConnectedIntegrations,
  setConnectedIntegrations as persistConnectedIntegrations,
} from "@/utils/browserStorage";

// Per-integration session_id prefix. Detection is coarse on purpose — any session
// whose id starts with the prefix counts as connected. Keep these in sync with
// the shipped integrations (claude-code → "cc_", codex → "codex_" as emitted by
// the plugins' _generate_session_id). Openclaw and API/MCP have no fixed prefix.
export const INTEGRATION_SESSION_PREFIX: Record<string, string> = {
  "claude-code": "cc_",
  codex: "codex_",
};

export function integrationsFromSessionIds(
  sessionIds: Array<string | null>,
): Record<string, boolean> {
  const connected: Record<string, boolean> = {};
  for (const [key, prefix] of Object.entries(INTEGRATION_SESSION_PREFIX)) {
    if (sessionIds.some((sessionId) => sessionId?.startsWith(prefix))) connected[key] = true;
  }
  return connected;
}

/**
 * Derives and persists per-integration "Connected" state from the session_id
 * prefixes. Sticky per tenant via localStorage so a card stays "Connected" after
 * its session ages out of the polled window.
 */
export function useConnectedIntegrations(
  sessions: SessionRow[],
  tenantId: string | null,
  instance?: CogneeInstance | null,
): Record<string, boolean> {
  const [connectedIntegrations, setConnectedIntegrations] = useState<Record<string, boolean>>({});
  const [activeConnections, setActiveConnections] = useState<{
    tenantId: string | null;
    connections: ActiveAgentConnection[];
  }>({ tenantId: null, connections: [] });

  useEffect(() => {
    if (!tenantId || !instance) {
      setActiveConnections({ tenantId, connections: [] });
      return;
    }

    const controller = new AbortController();
    let cancelled = false;
    getActiveAgentConnections(instance, controller.signal).then((connections) => {
      if (!cancelled) setActiveConnections({ tenantId, connections });
    });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [instance, tenantId]);

  useEffect(() => {
    if (!tenantId) return;
    const persisted = getConnectedIntegrations(tenantId);
    const connectionSessionIds =
      activeConnections.tenantId === tenantId
        ? activeConnections.connections.map((connection) => connection.session_id)
        : [];
    const observed = integrationsFromSessionIds([
      ...sessions.map((session) => session.session_id),
      ...connectionSessionIds,
    ]);
    const next = { ...persisted, ...observed };
    if (JSON.stringify(next) !== JSON.stringify(persisted)) {
      persistConnectedIntegrations(tenantId, next);
    }
    setConnectedIntegrations((prev) =>
      JSON.stringify(prev) !== JSON.stringify(next) ? next : prev,
    );
  }, [activeConnections, sessions, tenantId]);

  return connectedIntegrations;
}
