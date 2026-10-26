import type { ConnectorCatalogEntry } from "./connectorTypes";
import type { TeamConnectorCfg } from "./types";

/**
 * The catalog as a reader sees it over time: not yet read (undefined), read
 * and failed (null), or the deployment's own entries.
 */
export type CatalogRead = ConnectorCatalogEntry[] | null | undefined;

/**
 * Whether this deployment can answer for a data-source card at all.
 *
 * Slack is off the framework and never in the catalog, so it is always
 * offered. A framework provider is offered when the catalog lists it —
 * configured or not, since an install predating a lost secret must still
 * manage and disconnect. A catalog that could not be read offers everything:
 * a failed read must never hide a live connector. Only a catalog that read
 * fine and does not list the provider says no — the backend has no routes
 * for it, so every status read would 404 and the card would say "Couldn't
 * read the connection status" over a Retry that can never succeed
 * (review-reported, CLO-6620).
 *
 * Undefined while the catalog is still in flight: not yet known, which a
 * caller must not collapse to either answer.
 */
export function isOffered(cfg: TeamConnectorCfg, catalog: CatalogRead): boolean | undefined {
  if (cfg.api !== "framework") return true;
  if (catalog === undefined) return undefined;
  if (catalog === null) return true;
  return catalog.some((entry) => entry.provider === cfg.key);
}

/**
 * The entry a provider absent from a read catalog gets. To the reader, "not
 * shipped here" and "shipped without secrets" are the same fact — nothing on
 * this deployment can connect it — so the card says "not switched on here
 * yet" for both, and offers nothing.
 */
export function absentEntry(cfg: TeamConnectorCfg): ConnectorCatalogEntry {
  return {
    provider: cfg.key,
    displayName: cfg.name,
    configured: false,
    supportsResourceSelection: false,
    degraded: false,
  };
}
