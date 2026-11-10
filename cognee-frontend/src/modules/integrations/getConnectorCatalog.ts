"use server";

import type { ConnectorCatalogEntry } from "@/modules/integrations/connectorTypes";

/**
 * Open-source stub — the catalog reports which connectors this deployment holds
 * secrets for, a question only the hosted control plane can answer.
 *
 * `null` keeps the cards rendering their own static copy, which is what the
 * caller does when the catalog cannot be read.
 */
export default async function getConnectorCatalog(): Promise<ConnectorCatalogEntry[] | null> {
  return null;
}
