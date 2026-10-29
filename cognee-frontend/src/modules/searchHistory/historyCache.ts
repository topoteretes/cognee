import type { Conversation } from "./conversations";

/**
 * Search history for the current tab, so revisiting the page doesn't re-scan.
 *
 * Caching is safe here because the Search page is the only thing that writes
 * these conversations, and it writes its own back (see writeHistoryCache) —
 * a conversation you just had is in the cache before you navigate away. The
 * TTL covers only what this tab cannot see: the same account searching in
 * another tab or device, and the pod dropping a session's Q&A content after
 * its session-cache TTL. Both move slowly, so minutes is the right order.
 */
const HISTORY_TTL_MS = 5 * 60_000;

interface CacheEntry {
  conversations: Conversation[];
  storedAt: number;
}

const _history = new Map<string, CacheEntry>();

/** Cached conversations for this pod, or null when absent or past the TTL. */
export function readHistoryCache(instanceId: string): Conversation[] | null {
  const entry = _history.get(instanceId);
  if (!entry) return null;
  if (Date.now() - entry.storedAt > HISTORY_TTL_MS) {
    _history.delete(instanceId);
    return null;
  }
  return entry.conversations;
}

/**
 * Replaces what this pod has cached.
 *
 * Callers must only pass settled conversations: a message still marked loading
 * would be restored as a spinner that never resolves.
 *
 * Message count is deliberately not filtered here. An unused "+" click is
 * message-less and must not be cached, but so is a real conversation whose Q&A
 * the pod aged out after its cache TTL while keeping its label — and the two
 * are indistinguishable at this level. Only the caller knows which list a
 * conversation came from, so it does that filtering (see SearchPage).
 */
export function writeHistoryCache(instanceId: string, conversations: Conversation[]): void {
  // An empty history is never worth caching, and caching it is actively
  // harmful: "no conversations" is indistinguishable from "asked too early",
  // and a cache hit short-circuits the reload that would have corrected it.
  if (conversations.length === 0) return;
  if (conversations.some((c) => c.messages.some((m) => m.loading))) return;
  // storedAt is kept from the existing entry: the TTL measures how long ago the
  // pod was read, and the page rewrites this on every mount, so stamping it now
  // would push the refetch out indefinitely for anyone who visits often.
  const storedAt = _history.get(instanceId)?.storedAt ?? Date.now();
  _history.set(instanceId, { conversations, storedAt });
}

/** Test seam — the cache outlives a component, so suites must reset it. */
export function clearHistoryCache(): void {
  _history.clear();
}
