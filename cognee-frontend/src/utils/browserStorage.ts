/**
 * Typed, safe accessors for every browser-storage key used across the app.
 *
 * All reads return null / false / empty-object on storage failure (private
 * browsing, quota exceeded, SSR) so callers never need try/catch. All writes
 * are best-effort and silently swallow quota errors.
 *
 * Keys are consolidated here so a typo in any one file can no longer silently
 * break the onboarding or billing flows.
 */

// ── Internal key registry ─────────────────────────────────────────────────

const KEYS = {
  // sessionStorage
  awaitingDataset:       "cognee-awaiting-dataset",
  graphNodes:            "cognee-graph-nodes",
  creditsBannerDismissed:"cognee-credits-banner-dismissed",
  // localStorage
  connectedIntegrations: (tenantId: string) => `cognee-connected-integrations-${tenantId}`,
} as const;

// ── Low-level safe I/O ────────────────────────────────────────────────────

function sessionGet(key: string): string | null {
  try { return sessionStorage.getItem(key); } catch { return null; }
}

function sessionSet(key: string, value: string): void {
  try { sessionStorage.setItem(key, value); } catch { /* quota or private browsing */ }
}

function sessionRemove(key: string): void {
  try { sessionStorage.removeItem(key); } catch { /* ignore */ }
}

function localGet(key: string): string | null {
  try { return localStorage.getItem(key); } catch { return null; }
}

function localSet(key: string, value: string): void {
  try { localStorage.setItem(key, value); } catch { /* quota or private browsing */ }
}

function localRemove(key: string): void {
  try { localStorage.removeItem(key); } catch { /* ignore */ }
}

// ── Public API ────────────────────────────────────────────────────────────

// cognee-awaiting-dataset (sessionStorage) --------------------------------

/** Returns the dataset id handed off from onboarding, or null if absent. */
export function getAwaitingDataset(): string | null {
  return sessionGet(KEYS.awaitingDataset);
}

/** Marks a dataset as being provisioned so the dashboard shows a skeleton. */
export function setAwaitingDataset(datasetId: string): void {
  sessionSet(KEYS.awaitingDataset, datasetId);
}

/** Clears the flag once the dataset has finished processing. */
export function clearAwaitingDataset(): void {
  sessionRemove(KEYS.awaitingDataset);
}

// cognee-graph-nodes (sessionStorage) -------------------------------------

/** Returns the last-known graph node count, or null if not yet fetched. */
export function getCachedGraphNodes(): number | null {
  const raw = sessionGet(KEYS.graphNodes);
  return raw !== null ? Number(raw) : null;
}

/** Persists the graph node count so the first paint shows a cached value. */
export function setCachedGraphNodes(count: number): void {
  sessionSet(KEYS.graphNodes, String(count));
}

// cognee-credits-banner-dismissed (sessionStorage) ------------------------

/** Returns true if the user has dismissed the low-credit banner this session. */
export function isCreditsBannerDismissed(): boolean {
  return sessionGet(KEYS.creditsBannerDismissed) === "1";
}

/** Records that the user dismissed the low-credit banner. */
export function dismissCreditsBanner(): void {
  sessionSet(KEYS.creditsBannerDismissed, "1");
}

// cognee-connected-integrations-{tenantId} (localStorage) -----------------

/** Returns the persisted set of integrations that have ever connected for a tenant. */
export function getConnectedIntegrations(tenantId: string): Record<string, boolean> {
  try {
    const raw = localGet(KEYS.connectedIntegrations(tenantId));
    return (JSON.parse(raw ?? "{}") as Record<string, boolean>);
  } catch {
    return {};
  }
}

/** Persists the set of connected integrations for a tenant. */
export function setConnectedIntegrations(tenantId: string, value: Record<string, boolean>): void {
  localSet(KEYS.connectedIntegrations(tenantId), JSON.stringify(value));
}

/** Clears integration state for a tenant (called on workspace switch). */
export function clearConnectedIntegrations(tenantId: string): void {
  localRemove(KEYS.connectedIntegrations(tenantId));
}
