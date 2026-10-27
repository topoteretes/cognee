import type { DisplayStatus } from "@/app/(app)/datasets/brainsTypes";

export type StatusFilter = "all" | "ready" | "processing" | "attention";

export const STATUS_FILTERS: { value: StatusFilter; label: string }[] = [
  { value: "all", label: "All" },
  { value: "ready", label: "Ready" },
  { value: "processing", label: "Processing" },
  { value: "attention", label: "Attention" },
];

// One bucket per status, not status arrays per bucket: a Record keyed by the
// full DisplayStatus union means adding a status without placing it here is a
// compile error, instead of it silently matching no chip.
const STATUS_BUCKET: Record<Exclude<DisplayStatus, "loading">, Exclude<StatusFilter, "all">> = {
  completed: "ready",
  // An empty dataset has nothing left to process and nothing wrong with it,
  // it belongs with the other datasets that are ready to use.
  empty: "ready",
  pending: "processing",
  running: "processing",
  failed: "attention",
  failed_insufficient_credits: "attention",
};

interface FilterableDataset {
  id: string;
  name: string;
  status: DisplayStatus;
}

/** A dataset needs attention when it failed or was built under a config that
 *  has since changed (outdated); "ready" excludes the outdated ones. Outdated
 *  comes from the graph models config, not the processing status poll, so
 *  it's known even while the dataset itself reads "loading" (a reload in
 *  flight, or a just-submitted placeholder) — checked first, so an outdated
 *  dataset never shows under Ready just because its status hasn't resolved
 *  yet. A "loading" dataset that isn't outdated has nothing else yet to judge
 *  it against, so it matches every other chip: hiding it would be guessing
 *  wrong, not filtering correctly. */
function matchesStatus(dataset: FilterableDataset, filter: StatusFilter, outdatedIds: ReadonlySet<string>): boolean {
  if (filter === "all") return true;
  if (outdatedIds.has(dataset.id)) return filter === "attention";
  if (dataset.status === "loading") return true;
  return STATUS_BUCKET[dataset.status] === filter;
}

export function filterDatasetList<T extends FilterableDataset>(
  datasets: readonly T[],
  filter: StatusFilter,
  search: string,
  outdatedIds: ReadonlySet<string>,
): T[] {
  const query = search.trim().toLowerCase();
  return datasets.filter((dataset) =>
    (!query || dataset.name.toLowerCase().includes(query)) && matchesStatus(dataset, filter, outdatedIds));
}
