"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { ALL_FACET_ID, filterByFacet, withUnclaimedAsPersonal, type DatasetFacet } from "@/app/(app)/datasets/partials/datasetFacets";
import { filterDatasetList, type StatusFilter } from "@/app/(app)/datasets/partials/datasetListFilter";
import { isPendingDataset } from "@/app/(app)/datasets/brainsTypes";
import useDatasetFacets, { type FacetsStatus } from "@/app/(app)/datasets/partials/useDatasetFacets";
import type { DatasetAccess } from "@/modules/permissions/types";
import type { Dataset } from "@/app/(app)/datasets/brainsTypes";

export interface DatasetFinderFilters {
  facets: DatasetFacet[];
  facetId: string;
  facetsStatus: FacetsStatus;
  selectFacet: (facetId: string) => void;
  statusFilter: StatusFilter;
  setStatusFilter: (filter: StatusFilter) => void;
  search: string;
  setSearch: (value: string) => void;
  visible: Dataset[];
  /** True when a filter hides some datasets, so an empty list is a filter's doing. */
  narrowed: boolean;
  ownerLabel: (datasetId: string | null) => string | null;
  /** Who holds what on one dataset, as the pod last said. */
  accessFor: (datasetId: string) => DatasetAccess | undefined;
  /** Reads who holds what again, after a change to it. */
  refreshAccess: () => Promise<void>;
}

export function describeOwner(entry: DatasetAccess | undefined): string | null {
  if (!entry) return null;
  if (entry.isOwner) return "Owned by you";
  // The owner's address is only on the grants list, which a share holder sees.
  const owner = entry.sharedWith?.find((grant) => grant.principalType === "user" && grant.principalId === entry.ownerId);
  return owner?.name ? `Owned by ${owner.name}` : null;
}

export default function useDatasetFinderFilters(
  datasets: Dataset[],
  outdatedIds: Set<string>,
  sessionCreatedIds: ReadonlySet<string> = new Set(),
): DatasetFinderFilters {
  // Not a dataset still being created: it has no access to read yet, and a read
  // started now would only compete with the very request that makes it.
  const datasetKey = useMemo(
    () => datasets.filter((dataset) => !isPendingDataset(dataset.id)).map((dataset) => dataset.id).sort().join(","),
    [datasets],
  );
  const { facets: loadedFacets, accessById, status, refresh } = useDatasetFacets(datasetKey);
  // A placeholder still being created, or one this session has had confirmed,
  // is always mine until proven shared (see withUnclaimedAsPersonal) — fold it
  // into Personal so every reader of `facets` (this hook, GroupRail's counts)
  // agrees, one dataset doesn't look personal in one place and unclassified in
  // another. Anything else missing from the access list stays unclassified:
  // the list can be mid-refresh, or have dropped a row it couldn't parse, and
  // neither means the dataset is actually mine.
  const mineIds = useMemo(
    () => datasets.filter((dataset) => isPendingDataset(dataset.id) || sessionCreatedIds.has(dataset.id)).map((dataset) => dataset.id),
    [datasets, sessionCreatedIds],
  );
  const facets = useMemo(
    () => withUnclaimedAsPersonal(loadedFacets, mineIds),
    [loadedFacets, mineIds],
  );

  const [facetId, setFacetId] = useState(ALL_FACET_ID);
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [search, setSearch] = useState("");

  // A facet the rail no longer offers (its group was left, or the access
  // list simply hasn't loaded it yet) must not go on filtering by an id nobody
  // can see or reselect, so fall back to All, and commit the fallback so the
  // facet reappearing later (the group comes back, the list finishes loading)
  // doesn't silently resurrect a selection the user was never shown leaving.
  useEffect(() => {
    if (status === "loading" || facetId === ALL_FACET_ID) return;
    if (!facets.some((entry) => entry.id === facetId)) setFacetId(ALL_FACET_ID);
  }, [facets, facetId, status]);

  // Without the access list every facet but All is empty, so none of them may filter.
  const facet = status === "error" ? undefined : facets.find((entry) => entry.id === facetId);
  const visible = useMemo(
    () => filterDatasetList(filterByFacet(datasets, facet), statusFilter, search, outdatedIds),
    [datasets, facet, outdatedIds, search, statusFilter],
  );

  const ownerLabel = useCallback(
    (datasetId: string | null) => (datasetId ? describeOwner(accessById.get(datasetId)) : null),
    [accessById],
  );

  const accessFor = useCallback((datasetId: string) => accessById.get(datasetId), [accessById]);

  return {
    accessFor,
    refreshAccess: refresh,
    facets,
    facetId,
    facetsStatus: status,
    selectFacet: setFacetId,
    statusFilter,
    setStatusFilter,
    search,
    setSearch,
    visible,
    narrowed: visible.length !== datasets.length,
    ownerLabel,
  };
}
