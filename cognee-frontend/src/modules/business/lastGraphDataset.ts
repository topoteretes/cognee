// The dataset the graph last opened on, per pod instance. Available on the
// first render so the content fetch can start before `/v1/datasets/` answers:
// that list took 3626 ms on a local pod and, since the seed comes from it, it
// gated the whole graph behind itself even though the id it yields rarely
// changes between visits.
//
// Keyed by instanceId, and registered in sessionScopedStorage's PREFIXES so
// signout and the account-switch wipe clear it: instanceId is the TENANT id
// (pod.ts), so two users of the same tenant share this browser key, and
// without that registration the next one would inherit the previous one's
// dataset and eat the failed fetch this module exists to avoid.
const KEY_PREFIX = "cognee_last_graph_dataset:";

function keyFor(instanceId: string): string {
  return `${KEY_PREFIX}${instanceId}`;
}

export function readLastGraphDataset(instanceId: string | null): string | null {
  if (!instanceId || typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(keyFor(instanceId));
  } catch {
    // Storage can be unreachable (private mode, blocked cookies, quota). The
    // graph then just waits for the dataset list, as it did before this.
    return null;
  }
}

export function writeLastGraphDataset(instanceId: string | null, datasetId: string | null): void {
  if (!instanceId || typeof window === "undefined") return;
  try {
    if (datasetId) window.localStorage.setItem(keyFor(instanceId), datasetId);
    else window.localStorage.removeItem(keyFor(instanceId));
  } catch {
    // Same as the read above: losing this only costs the head start.
  }
}

// useSyncExternalStore plumbing. A plain effect reads this one render too
// late, and on in-app navigation that is the render that matters: the dataset
// list is already warm, so the seed latches onto its first entry and the
// write below then overwrites the very value this module stored. Reading it
// synchronously before paint is what makes the persisted id win that race,
// without the hydration mismatch a lazy useState initializer would cause
// (the trap FilterContext documents for the persisted workspace).
export function subscribeToLastGraphDataset(onStoreChange: () => void): () => void {
  if (typeof window === "undefined") return () => {};
  // Filtered by key: a `storage` event fires for every write another tab
  // makes, and waking the graph's seed for an unrelated one is pure noise.
  // A null key means the other tab called clear(), which does affect us.
  const onStorage = (event: StorageEvent) => {
    if (event.key === null || event.key.startsWith(KEY_PREFIX)) onStoreChange();
  };
  window.addEventListener("storage", onStorage);
  return () => window.removeEventListener("storage", onStorage);
}

// Always null on the server, matching hydration's first client pass; React
// swaps in the real value synchronously right after.
export function getServerLastGraphDatasetSnapshot(): string | null {
  return null;
}
