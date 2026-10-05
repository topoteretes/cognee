"use client";

import { useEffect, useState } from "react";
import getConnectionStatus from "@/modules/integrations/getConnectionStatus";
import type { NodeStatus } from "@/app/(app)/dashboard/partials/redesign/MemoryFlowDiagram";

/**
 * Connection status per data-source provider, from the control-plane's
 * connection records — the same source the Integrations page reads. Providers
 * come from DATA_SOURCE_CARDS, so a connector added there appears in the
 * memory graph without touching this hook.
 *
 * A tenant that only receives channels routed from another workspace's install
 * still counts as connected: memory is flowing in either way. A failed read
 * stays "disconnected" — the graph has no third state, and the Integrations
 * page is where a degraded connection gets explained properly.
 */
export function useDataSourceStatuses(providers: string[], tenantId: string | null): Record<string, NodeStatus> {
  const [statuses, setStatuses] = useState<Record<string, NodeStatus>>({});
  // Providers is a fresh array each render; the joined key keeps the effect from
  // refiring on identity alone.
  const providerKey = providers.join(",");

  useEffect(() => {
    let cancelled = false;
    if (!tenantId) {
      setStatuses({});
      return;
    }
    const list = providerKey === "" ? [] : providerKey.split(",");
    Promise.all(
      list.map(async (provider): Promise<[string, NodeStatus]> => {
        try {
          const status = await getConnectionStatus(provider, tenantId);
          return [provider, status.connected || status.viaRouting ? "connected" : "disconnected"];
        } catch (e: unknown) {
          console.warn(`[dashboard] connection status failed for ${provider}:`, e);
          return [provider, "disconnected"];
        }
      }),
    ).then((entries) => {
      if (!cancelled) setStatuses(Object.fromEntries(entries));
    });
    return () => { cancelled = true; };
  }, [providerKey, tenantId]);

  return statuses;
}
