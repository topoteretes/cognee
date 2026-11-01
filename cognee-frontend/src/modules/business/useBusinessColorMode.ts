import { useCallback, useSyncExternalStore } from "react";
import type { ColorMode } from "./colorEncoding";

// How the graph is colored is a per-browser preference, not per tenant or
// dataset, so it carries no identity and needs no wipe on signout.
const STORAGE_KEY = "cognee_business_color_mode";
const DEFAULT_MODE: ColorMode = "type";

const listeners = new Set<() => void>();
// Holds the choice only while storage refuses the write, so the toggle still
// works for this session instead of silently doing nothing.
let unsavedMode: ColorMode | null = null;

function parseMode(value: string | null): ColorMode {
  return value === "source" || value === "type" ? value : DEFAULT_MODE;
}

function readMode(): ColorMode {
  if (unsavedMode) return unsavedMode;
  try {
    return parseMode(window.localStorage.getItem(STORAGE_KEY));
  } catch {
    // Storage can be unreachable (private mode, blocked cookies): the default
    // mode still renders correctly, it just is not remembered.
    return DEFAULT_MODE;
  }
}

function subscribe(onChange: () => void): () => void {
  listeners.add(onChange);
  // Another tab changing the preference arrives as a storage event; a null
  // key means that tab cleared storage.
  const onStorage = (event: StorageEvent) => {
    if (event.key === null || event.key === STORAGE_KEY) onChange();
  };
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(onChange);
    window.removeEventListener("storage", onStorage);
  };
}

// Read synchronously before paint (useSyncExternalStore) so a saved "source"
// never flashes the default first, without the hydration mismatch a lazy
// useState initializer would cause: the server snapshot is always the default.
export function useBusinessColorMode(): [ColorMode, (mode: ColorMode) => void] {
  const mode = useSyncExternalStore(subscribe, readMode, () => DEFAULT_MODE);
  const setMode = useCallback((next: ColorMode) => {
    try {
      window.localStorage.setItem(STORAGE_KEY, next);
      unsavedMode = null;
    } catch {
      unsavedMode = next;
    }
    listeners.forEach((notify) => notify());
  }, []);
  return [mode, setMode];
}
