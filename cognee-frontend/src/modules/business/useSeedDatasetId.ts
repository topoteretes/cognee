"use client";

import { useEffect, useState, useSyncExternalStore } from "react";
import pickSeedDatasetId from "./pickSeedDatasetId";
import {
  getServerLastGraphDatasetSnapshot,
  readLastGraphDataset,
  subscribeToLastGraphDataset,
  writeLastGraphDataset,
} from "./lastGraphDataset";

interface SeedDatasetArgs {
  instanceId: string;
  activeDatasetId: string | null;
  setActiveDatasetId: (id: string) => void;
  selectedDatasetId: string | null;
  readableDatasetIds: string[];
  readableLoading: boolean;
  readableUnavailable: boolean;
  governanceDatasetIds: string[];
}

interface SeedDatasetResult {
  // The dataset we opened on last visit's memory and the readable list has
  // not vouched for yet, or null. Exposed so the view can tell an unconfirmed
  // guess apart from a real failure: a content fetch failing for OUR bet must
  // not be reported to the user as a connection problem.
  unconfirmedSeedId: string | null;
}

// Owns which dataset the graph opens on, and the storage behind it. Lives
// outside BusinessView because it is three effects around one decision, and
// because effects inside a 600-line view have nowhere to be tested.
export default function useSeedDatasetId({
  instanceId,
  activeDatasetId,
  setActiveDatasetId,
  selectedDatasetId,
  readableDatasetIds,
  readableLoading,
  readableUnavailable,
  governanceDatasetIds,
}: SeedDatasetArgs): SeedDatasetResult {
  const persistedDatasetId = useSyncExternalStore(
    subscribeToLastGraphDataset,
    () => readLastGraphDataset(instanceId),
    getServerLastGraphDatasetSnapshot,
  );

  // Which id we opened on as an unverified bet from last visit's memory.
  // Tracked explicitly rather than by comparing against the stored value:
  // the write effect below puts activeDatasetId INTO that store, and
  // useSyncExternalStore reads it straight back, so `persistedDatasetId`
  // follows the active dataset within a render or two. Comparing the two
  // would mean "correct any dataset that is not in the readable list, from
  // any source, forever" - including a brain the user picked by hand.
  const [unconfirmedSeedId, setUnconfirmedSeedId] = useState<string | null>(null);

  const seedDatasetId = pickSeedDatasetId({
    selectedDatasetId,
    persistedDatasetId,
    readableDatasetIds,
    readableLoading,
    governanceDatasetIds,
    readableUnavailable,
  });

  useEffect(() => {
    if (activeDatasetId || !seedDatasetId) return;
    // Unverified exactly when it came from storage while the readable list
    // was still answering; anything the list itself produced is already
    // vouched for.
    if (readableLoading && seedDatasetId === persistedDatasetId) {
      setUnconfirmedSeedId(seedDatasetId);
    }
    setActiveDatasetId(seedDatasetId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seedDatasetId, activeDatasetId]);

  // Settles that bet once the readable list lands: a dataset deleted or
  // unshared since last visit would otherwise be reopened on every visit,
  // each time costing a failed content fetch. Fires at most once per bet,
  // and never for a dataset the user chose themselves.
  useEffect(() => {
    if (!unconfirmedSeedId || readableLoading) return;
    if (activeDatasetId !== unconfirmedSeedId) {
      setUnconfirmedSeedId(null);
      return;
    }
    if (readableDatasetIds.includes(unconfirmedSeedId)) {
      setUnconfirmedSeedId(null);
      return;
    }
    const replacement = readableDatasetIds[0];
    // With nothing readable to move to, leaving the dead id in place beats
    // blanking the view: the readable list is empty or unavailable, and the
    // view's own empty/error states already speak to both.
    if (replacement) {
      setUnconfirmedSeedId(null);
      setActiveDatasetId(replacement);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [unconfirmedSeedId, readableDatasetIds, readableLoading, activeDatasetId]);

  useEffect(() => {
    if (!activeDatasetId) return;
    writeLastGraphDataset(instanceId, activeDatasetId);
  }, [instanceId, activeDatasetId]);

  return { unconfirmedSeedId };
}
