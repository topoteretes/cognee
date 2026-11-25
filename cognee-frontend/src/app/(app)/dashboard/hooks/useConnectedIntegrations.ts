"use client";

import { useState, useEffect } from "react";
import { AGENT_CHANNEL_NAMES, channelForSessionId } from "@/modules/sessions/getSessions";
import type { SessionRow } from "@/modules/sessions/getSessions";
import {
  AGENT_INTEGRATION_BY_KEY,
  AGENT_INTEGRATION_DEFS,
  connectionIntegrationKey,
} from "@/modules/agents/getAgentConnections";
import type { AgentConnection } from "@/modules/agents/getAgentConnections";
import {
  getConnectedIntegrations,
  setConnectedIntegrations as persistConnectedIntegrations,
} from "@/utils/browserStorage";

/** The integration card keys this hook can report on. Cards keep their own
 *  slugs — AGENT_INTEGRATION_DEFS.key — because that is what is already
 *  persisted in users' browsers. */
export const INTEGRATION_KEYS: readonly string[] = AGENT_INTEGRATION_DEFS.map((def) => def.key);

// Stable reference for callers that don't have a connections list yet — a
// fresh `[]` on every render would re-run the effect below every render for
// no reason, since nothing in it depends on the array's identity otherwise.
const NO_CONNECTIONS: AgentConnection[] = [];

/**
 * Derives and persists which agents and integrations have ever reached this
 * tenant, keyed both by channel name (for the memory graph) and by integration
 * slug (for the Integrations cards).
 *
 * Sticky per tenant via localStorage, and that stickiness is load-bearing
 * rather than a nicety: the polled session page is scoped to the dashboard's
 * range toggle and capped at 50 rows, and the deployed pod caps it again on
 * its own side. A busy workspace pushes an agent's last session off that page
 * within the hour, and without a persisted answer the graph would then report
 * an installed agent as never connected.
 *
 * `connections` is optional and additive: a plugin registration alone (no
 * session or memory write yet) is real evidence a card should read
 * "Connected", ahead of what `sessions` can prove. Passing none keeps this
 * hook's session-only behavior for callers that haven't wired the endpoint up.
 */
export function useConnectedIntegrations(
  sessions: SessionRow[],
  tenantId: string | null,
  connections: AgentConnection[] = NO_CONNECTIONS,
): Record<string, boolean> {
  const [connectedIntegrations, setConnectedIntegrations] = useState<Record<string, boolean>>({});

  useEffect(() => {
    if (!tenantId) return;
    const persisted = getConnectedIntegrations(tenantId);
    const next = { ...persisted };
    const seen = new Set(sessions.map((s) => channelForSessionId(s.session_id)));
    for (const name of AGENT_CHANNEL_NAMES) {
      if (seen.has(name)) next[name] = true;
    }
    for (const def of AGENT_INTEGRATION_DEFS) {
      if (seen.has(def.channel)) next[def.key] = true;
    }
    for (const connection of connections) {
      const key = connectionIntegrationKey(connection);
      const def = key ? AGENT_INTEGRATION_BY_KEY.get(key) : undefined;
      if (!def) continue;
      next[def.key] = true;
      next[def.channel] = true;
    }
    if (JSON.stringify(next) !== JSON.stringify(persisted)) {
      persistConnectedIntegrations(tenantId, next);
    }
    setConnectedIntegrations((prev) =>
      JSON.stringify(prev) !== JSON.stringify(next) ? next : prev,
    );
  }, [sessions, connections, tenantId]);

  return connectedIntegrations;
}
