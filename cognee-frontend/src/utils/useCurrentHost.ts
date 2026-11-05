import { useSyncExternalStore } from "react";

const subscribe = (): (() => void) => () => {};

/**
 * The host the browser is on, or null while rendering on the server: a flag
 * decided on the host then reads as off first and settles after hydration,
 * instead of the two renders disagreeing.
 */
export function useCurrentHost(): string | null {
  return useSyncExternalStore(
    subscribe,
    () => window.location.host,
    () => null,
  );
}
