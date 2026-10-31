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
  searchActiveConversation: (tenantId: string) => `cognee-search-active-conversation-${tenantId}`,
  // localStorage
  connectedIntegrations: (tenantId: string) => `cognee-connected-integrations-${tenantId}`,
  insufficientCreditsFailure: (tenantId: string) => `cognee-insufficient-credits-failure-${tenantId}`,
  searchType: (tenantId: string) => `cognee-search-type-${tenantId}`,
  searchCodeConversations: (tenantId: string) => `cognee-search-code-conversations-${tenantId}`,
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

/**
 * Returns the persisted set of integrations and agent channels that have ever
 * connected for a tenant.
 *
 * Only entries stored as boolean true are returned. This record is the
 * dashboard memory graph's registration source, so a value left by an older
 * shape has to read as "nothing recorded" rather than reach the graph as
 * something merely truthy.
 */
export function getConnectedIntegrations(tenantId: string): Record<string, boolean> {
  try {
    const raw = localGet(KEYS.connectedIntegrations(tenantId));
    const parsed: unknown = JSON.parse(raw ?? "{}");
    // `=== true` is the whole guard: an array, a string or a number yields no
    // entry whose value is boolean true, and `null` throws into the catch. An
    // explicit shape check ahead of it was unreachable.
    return Object.fromEntries(
      Object.entries(parsed as Record<string, unknown>).filter(
        (entry): entry is [string, boolean] => entry[1] === true,
      ),
    );
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

// cognee-insufficient-credits-failure-{tenantId} (localStorage) ----------

export interface InsufficientCreditsFailureRecord {
  operation: string | null;
  at: number;
}

/**
 * Returns the last unacknowledged credit failure for a tenant, or null if
 * none was recorded (or the stored value is corrupt). Persisted in
 * localStorage rather than sessionStorage so it survives a closed tab —
 * the failure can arrive after the request that caused it is long gone.
 */
export function getInsufficientCreditsFailure(tenantId: string): InsufficientCreditsFailureRecord | null {
  const raw = localGet(KEYS.insufficientCreditsFailure(tenantId));
  if (!raw) return null;
  try {
    return JSON.parse(raw) as InsufficientCreditsFailureRecord;
  } catch {
    return null;
  }
}

/** Records a credit failure so the next page load can surface it even if the live modal was missed. */
export function setInsufficientCreditsFailure(tenantId: string, record: InsufficientCreditsFailureRecord): void {
  localSet(KEYS.insufficientCreditsFailure(tenantId), JSON.stringify(record));
}

/** Clears the record once the user has acknowledged it (dismissed it, or gone to billing). */
export function clearInsufficientCreditsFailure(tenantId: string): void {
  localRemove(KEYS.insufficientCreditsFailure(tenantId));
}

// cognee-search-active-conversation-{tenantId} (sessionStorage) -------------

/**
 * The Search conversation the user last had open on this pod, or null.
 * sessionStorage on purpose: it is the tab's place in the chat, and the
 * conversation list it refers to is also cached per tab.
 */
export function getSearchActiveConversation(tenantId: string): string | null {
  return sessionGet(KEYS.searchActiveConversation(tenantId));
}

/** Remembers (or with null, forgets) the open Search conversation for a pod. */
export function setSearchActiveConversation(tenantId: string, conversationId: string | null): void {
  if (conversationId === null) sessionRemove(KEYS.searchActiveConversation(tenantId));
  else sessionSet(KEYS.searchActiveConversation(tenantId), conversationId);
}

// cognee-search-type-{tenantId} (localStorage) ------------------------------

/**
 * Which Search type the user last picked on this pod, or null if never set.
 * Cleared with the account (SESSION_SCOPED_STORAGE_PREFIXES): it picks the
 * retriever, so it must not pass to whoever signs in next on this browser.
 *
 * localStorage, not sessionStorage: the answer follows from what the workspace
 * holds (a code repo vs. documents), not from where the tab is in a chat, so
 * it should outlive the tab the way the choice it reflects does. Returned as a
 * raw string — this module stays free of feature types, and the caller
 * validates it against the ids it knows.
 */
export function getSearchType(tenantId: string): string | null {
  return localGet(KEYS.searchType(tenantId));
}

/** Remembers the Search type for a pod. */
export function setSearchType(tenantId: string, searchType: string): void {
  localSet(KEYS.searchType(tenantId), searchType);
}

// cognee-search-code-conversations-{tenantId} (localStorage) ----------------

/**
 * The Search tab's Code conversations for a pod, serialised, or null.
 *
 * Kept here because the pod keeps nothing: a Code turn is a code-graph lookup
 * on /v1/search, which writes no session, and the sessions API is read-only.
 * Text conversations need none of this — the pod records every recall turn.
 */
export function getSearchCodeConversations(tenantId: string): string | null {
  return localGet(KEYS.searchCodeConversations(tenantId));
}

/**
 * Stores them, reporting whether the write took — unlike the other setters,
 * because a quota failure here is worth retrying with less rather than losing
 * every conversation at once.
 */
export function setSearchCodeConversations(tenantId: string, value: string): boolean {
  try {
    localStorage.setItem(KEYS.searchCodeConversations(tenantId), value);
    return true;
  } catch {
    return false;
  }
}
