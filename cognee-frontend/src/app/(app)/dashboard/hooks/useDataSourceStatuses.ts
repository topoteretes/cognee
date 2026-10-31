"use client";

import { useEffect, useState } from "react";
import getConnectionStatus from "@/modules/integrations/getConnectionStatus";
import getConnectorCatalog from "@/modules/integrations/getConnectorCatalog";
import { DATA_SOURCE_CARDS } from "@/modules/integrations/dataSourceCards";
import { isOffered } from "@/modules/integrations/connectorAvailability";
import type { NodeStatus } from "@/app/(app)/dashboard/partials/redesign/MemoryFlowDiagram";

/**
 * The graph's own statuses plus one it never draws: "absent" is a framework
 * connector this deployment does not ship, which the graph leaves out rather
 * than showing as disconnected forever.
 */
export type DataSourceNodeStatus = NodeStatus | "absent";

async function readNodeStatus(provider: string, tenantId: string): Promise<NodeStatus> {
  try {
    const status = await getConnectionStatus(provider, tenantId);
    return status.connected || status.viaRouting ? "connected" : "disconnected";
  } catch (e: unknown) {
    console.warn(`[dashboard] connection status failed for ${provider}:`, e);
    return "disconnected";
  }
}

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
 *
 * The catalog is read first, and a framework provider it does not list is
 * never asked for: the backend has no routes for it, so the read could only
 * 404 into a node nobody here can connect (review-reported). A failed
 * catalog read asks for everything, as before — a failed read must not hide
 * a live connector.
 */
export function useDataSourceStatuses(providers: string[], tenantId: string | null): Record<string, DataSourceNodeStatus> {
  const [statuses, setStatuses] = useState<Record<string, DataSourceNodeStatus>>({});
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
    void getConnectorCatalog()
      .then((catalog) =>
        Promise.all(
          list.map(async (provider): Promise<[string, DataSourceNodeStatus]> => {
            const cfg = DATA_SOURCE_CARDS.find((card) => card.key === provider);
            if (cfg && isOffered(cfg, catalog) === false) return [provider, "absent"];
            return [provider, await readNodeStatus(provider, tenantId)];
          }),
        ),
      )
      .then((entries) => {
        if (!cancelled) setStatuses(Object.fromEntries(entries));
      });
    return () => { cancelled = true; };
  }, [providerKey, tenantId]);

  return statuses;
}
