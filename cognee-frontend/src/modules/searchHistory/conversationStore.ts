import type { Conversation } from "./conversations";
import { getSearchActiveConversation, setSearchActiveConversation } from "@/utils/browserStorage";

/**
 * The Search page's live state, kept outside the component (CLO-846).
 *
 * The page is unmounted on every navigation, and `useState` went with it: the
 * conversation the user had open was closed, so the next question was sent
 * under a brand-new session id and the pod answered it with no memory of the
 * turns before — the context loss the ticket describes. An answer still being
 * generated was lost the same way. Keeping the state here, keyed by pod like
 * the history cache, lets a remount pick up exactly where the user left off,
 * including a stream that is still arriving. The open conversation's id is
 * additionally persisted per tab so a full reload restores it too.
 *
 * Only what this visit started or continued lives here; conversations restored
 * from the pod stay in the history cache.
 */
export interface SearchState {
  conversations: Conversation[];
  activeId: string | null;
  /** Conversation whose answer is being generated, or null when idle. */
  inFlightId: string | null;
}

export const EMPTY_SEARCH_STATE: SearchState = Object.freeze({
  conversations: [],
  activeId: null,
  inFlightId: null,
});

interface Store {
  state: SearchState;
  listeners: Set<() => void>;
}

const _stores = new Map<string, Store>();

function storeFor(instanceId: string): Store {
  const existing = _stores.get(instanceId);
  if (existing) return existing;
  const created: Store = {
    state: { ...EMPTY_SEARCH_STATE, activeId: getSearchActiveConversation(instanceId) },
    listeners: new Set(),
  };
  _stores.set(instanceId, created);
  return created;
}

/** Current state for a pod. Stable between updates, so it is safe as a snapshot. */
export function readSearchState(instanceId: string): SearchState {
  return storeFor(instanceId).state;
}

export function subscribeSearchState(instanceId: string, listener: () => void): () => void {
  const store = storeFor(instanceId);
  store.listeners.add(listener);
  return () => {
    store.listeners.delete(listener);
  };
}

/** Replace the state; an updater returning the same object is a no-op. */
export function updateSearchState(
  instanceId: string,
  update: (state: SearchState) => SearchState,
): void {
  const store = storeFor(instanceId);
  const next = update(store.state);
  if (next === store.state) return;
  if (next.activeId !== store.state.activeId) setSearchActiveConversation(instanceId, next.activeId);
  store.state = next;
  store.listeners.forEach((listener) => listener());
}

/** Test seam — the store outlives a component, so suites must reset it. */
export function clearSearchState(): void {
  for (const instanceId of _stores.keys()) setSearchActiveConversation(instanceId, null);
  _stores.clear();
}
