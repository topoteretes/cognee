import { useCallback, useSyncExternalStore } from "react";
import {
  EMPTY_SEARCH_STATE,
  readSearchState,
  subscribeSearchState,
  type SearchState,
} from "./conversationStore";

const noSubscription = (): (() => void) => () => undefined;
const emptySnapshot = (): SearchState => EMPTY_SEARCH_STATE;

/**
 * The Search page's state for a pod, re-rendering on every store update — an
 * answer streaming in after a remount included.
 *
 * The server snapshot is always empty: the persisted active id lives in the
 * browser's sessionStorage, so hydrating with it would not match the markup
 * the server produced. React re-renders with the client snapshot right after.
 */
export function useSearchState(instanceId: string | null): SearchState {
  const subscribe = useCallback(
    (listener: () => void) =>
      instanceId ? subscribeSearchState(instanceId, listener) : noSubscription(),
    [instanceId],
  );
  const getSnapshot = useCallback(
    () => (instanceId ? readSearchState(instanceId) : EMPTY_SEARCH_STATE),
    [instanceId],
  );
  return useSyncExternalStore(subscribe, getSnapshot, emptySnapshot);
}
