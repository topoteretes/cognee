"use client";

import { useState, useEffect } from "react";
import type { SessionRow } from "@/modules/sessions/getSessions";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import {
  getConnectedIntegrations,
  setConnectedIntegrations as persistConnectedIntegrations,
} from "@/utils/browserStorage";

const CONNECTION_REFRESH_INTERVAL_MS = 15_000;

interface AgentConnection {
  session_id?: string | null;
}

interface AgentConnectionsResponse {
  agents?: AgentConnection[];
}

// Per-integration session_id prefix. Detection is coarse on purpose — any session
// whose id starts with the prefix counts as connected. Keep these in sync with
// the shipped integrations (claude-code → "cc_", codex → "codex_" as emitted by
// the plugins' _generate_session_id). Openclaw and API/MCP have no fixed prefix.
export const INTEGRATION_SESSION_PREFIX: Record<string, string> = {
  "claude-code": "cc_",
  codex: "codex_",
};

/**
 * Derives and persists per-integration "Connected" state from session and active
 * connection ids. Sticky per tenant via localStorage so a card stays "Connected"
 * after its session or connection ages out of the polled window.
 */
export function useConnectedIntegrations(
  sessions: SessionRow[],
  tenantId: string | null,
): Record<string, boolean> {
  const { cogniInstance } = useCogniInstance();
  const [activeSessionIds, setActiveSessionIds] = useState<string[]>([]);
  const [connectedIntegrations, setConnectedIntegrations] = useState<Record<string, boolean>>({});

  useEffect(() => {
    if (!cogniInstance || !tenantId) {
      setActiveSessionIds([]);
      return;
    }

    let cancelled = false;
    const instance = cogniInstance;

    async function loadActiveConnections() {
      try {
        const response = await instance.fetch("/v1/agents/connections?active_only=true");
        if (!response.ok) return;

        const payload = (await response.json()) as AgentConnectionsResponse;
        if (!cancelled) {
          setActiveSessionIds(
            (payload.agents ?? [])
              .map((connection) => connection.session_id)
              .filter((sessionId): sessionId is string => Boolean(sessionId)),
          );
        }
      } catch {
        // Connection status is supplementary. Session-derived and persisted
        // state still provide a useful fallback when this endpoint is absent.
      }
    }

    void loadActiveConnections();
    const interval = window.setInterval(loadActiveConnections, CONNECTION_REFRESH_INTERVAL_MS);

    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [cogniInstance, tenantId]);

  useEffect(() => {
    if (!tenantId) return;
    const persisted = getConnectedIntegrations(tenantId);
    const next = { ...persisted };
    for (const [key, prefix] of Object.entries(INTEGRATION_SESSION_PREFIX)) {
      if (
        sessions.some((session) => session.session_id.startsWith(prefix)) ||
        activeSessionIds.some((sessionId) => sessionId.startsWith(prefix))
      ) {
        next[key] = true;
      }
    }
    if (JSON.stringify(next) !== JSON.stringify(persisted)) {
      persistConnectedIntegrations(tenantId, next);
    }
    setConnectedIntegrations((prev) =>
      JSON.stringify(prev) !== JSON.stringify(next) ? next : prev,
    );
  }, [activeSessionIds, sessions, tenantId]);

  return connectedIntegrations;
}
