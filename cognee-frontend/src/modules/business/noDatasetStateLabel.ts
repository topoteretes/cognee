// Which "there is no brain on screen" message the Business view owes the user,
// or null when some other overlay (the loading weave, a content-level empty
// state) is the one talking.
//
// Extracted from BusinessView's JSX because the combinations are what matter
// and they are not obvious: the gate that silences the "create a dataset"
// message while the readable list is still answering also silenced it when
// that list had FAILED, on a workspace whose provenance graph came back empty
// — first visit, no remembered dataset, nothing to select, and no copy on
// screen explaining any of it.
export interface NoDatasetState {
  // Any layer query still in flight.
  isLoading: boolean;
  // The dataset the canvas is showing, when there is one.
  activeDatasetId: string | null;
  // How `/v1/datasets/` (the read-scoped list) is doing.
  listState: "loading" | "unavailable" | "ready";
  // Datasets this caller may read, after narrowing.
  readableDatasetCount: number;
  // The provenance fetch itself failed.
  governanceFailed: boolean;
}

export const NO_DATASET_SELECTED =
  "no dataset selected — create a dataset and upload documents to see your business model";
export const WORKSPACE_UNREADABLE =
  "couldn't load this workspace's datasets — check your connection and reload";

export default function noDatasetStateLabel({
  isLoading,
  activeDatasetId,
  listState,
  readableDatasetCount,
  governanceFailed,
}: NoDatasetState): string | null {
  // A dataset is on screen, or something is still arriving: neither is this
  // function's business.
  if (isLoading || activeDatasetId) return null;
  // A failed fetch outranks an empty list, whichever of the two failed: being
  // told to create a dataset when the request for them just failed is the one
  // message that is certainly wrong.
  if (governanceFailed || listState === "unavailable") return WORKSPACE_UNREADABLE;
  // Still answering — the switcher's own label says so, and a moment later
  // this is answerable for real.
  if (listState === "loading") return null;
  return readableDatasetCount === 0 ? NO_DATASET_SELECTED : null;
}
