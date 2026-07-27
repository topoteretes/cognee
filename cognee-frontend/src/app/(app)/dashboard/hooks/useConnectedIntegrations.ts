"use client";

import { useState, useEffect } from "react";
import type { SessionRow } from "@/modules/sessions/getSessions";
import {
  getConnectedIntegrations,
  setConnectedIntegrations as persistConnectedIntegrations,
} from "@/utils/browserStorage";

// Per-integration session_id prefix. Detection is coarse on purpose — any session
// whose id starts with the prefix counts as connected. Keep these in sync with
// the shipped integrations (claude-code → "cc_", codex → "codex_" as emitted by
// the plugins' _generate_session_id). Openclaw and API/MCP have no fixed prefix.
const INTEGRATION_SESSION_PREFIX: Record<string, string> = {
  "claude-code": "cc_",
  codex: "codex_",
};

/**
 * Derives and persists per-integration "Connected" state from the session_id
 * prefixes. Sticky per tenant via localStorage so a card stays "Connected" after
 * its session ages out of the polled window.
 */
export function useConnectedIntegrations(
  sessions: SessionRow[],
  tenantId: string | null,
): Record<string, boolean> {
  const [connectedIntegrations, setConnectedIntegrations] = useState<Record<string, boolean>>({});

  useEffect(() => {
    if (!tenantId) return;
    const persisted = getConnectedIntegrations(tenantId);
    const next = { ...persisted };
    for (const [key, prefix] of Object.entries(INTEGRATION_SESSION_PREFIX)) {
      if (sessions.some((s) => s.session_id.startsWith(prefix))) next[key] = true;
    }
    if (JSON.stringify(next) !== JSON.stringify(persisted)) {
      persistConnectedIntegrations(tenantId, next);
    }
    setConnectedIntegrations((prev) =>
      JSON.stringify(prev) !== JSON.stringify(next) ? next : prev,
    );
  }, [sessions, tenantId]);

  return connectedIntegrations;
}
