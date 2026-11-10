interface SeedSources {
  // The dataset the user picked in the app-wide filter, when there is one.
  selectedDatasetId: string | null;
  // What the graph opened on last time (lastGraphDataset). Used while the
  // readable list is still answering, so the content fetch is not gated on
  // it; once that list lands it decides.
  persistedDatasetId: string | null;
  // `/v1/datasets/` — get_all_user_permission_datasets(user, "read"), i.e.
  // exactly the set the graph endpoint will serve this caller.
  readableDatasetIds: string[];
  // Whether that list is still in flight. Passed in rather than inferred from
  // an empty array: a workspace with no datasets at all produces the same
  // empty array, and treating it as "still loading" kept reopening a deleted
  // dataset on a workspace that had nothing left to open.
  readableLoading: boolean;
  // Dataset nodes off the provenance graph. Scoped by tenant, NOT by read
  // grant, so an entry here can be one this caller cannot fetch.
  governanceDatasetIds: string[];
  // True only when the readable list failed to load, as opposed to being
  // empty because the workspace genuinely has no datasets (CLO-726).
  readableUnavailable: boolean;
}

// Which dataset the graph opens on. Prefers the caller's own pick, then last
// visit's while the readable list is still answering, then that list itself;
// governance is a last resort used only when the list could not be fetched at
// all, because its tenant scoping can name a dataset this caller has no read
// grant on.
export default function pickSeedDatasetId({
  selectedDatasetId,
  persistedDatasetId,
  readableDatasetIds,
  readableLoading,
  governanceDatasetIds,
  readableUnavailable,
}: SeedSources): string | null {
  if (selectedDatasetId) return selectedDatasetId;
  if (readableLoading) return persistedDatasetId;
  if (readableDatasetIds.length > 0) {
    if (persistedDatasetId !== null && readableDatasetIds.includes(persistedDatasetId)) {
      return persistedDatasetId;
    }
    return readableDatasetIds[0];
  }
  // The list answered. Empty means an empty workspace, and opening anything
  // there would only produce a failed fetch behind an error the user cannot
  // act on; the view's own empty state is the honest answer.
  if (readableUnavailable) return persistedDatasetId ?? governanceDatasetIds[0] ?? null;
  return null;
}
