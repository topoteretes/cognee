import type { DatasetAccess, WorkspaceGroup } from "@/modules/permissions/types";

export type FacetKind = "all" | "personal" | "shared" | "group";

/** One entry of the datasets page's left rail: a filter over one list. A
 *  dataset reachable two ways shows up under both, but it is one dataset. */
export interface DatasetFacet {
  id: string;
  kind: FacetKind;
  label: string;
  hint: string;
  datasetIds: ReadonlySet<string>;
}

export const ALL_FACET_ID = "all";
export const PERSONAL_FACET_ID = "personal";
const SHARED_FACET_ID = "shared";

function plural(count: number, word: string): string {
  return `${count} ${word}${count === 1 ? "" : "s"}`;
}

// Shared with me: not mine, and reaching me other than through a group —
// a personal share or a grant to the whole workspace.
function isSharedDirectly(entry: DatasetAccess): boolean {
  return !entry.isOwner && entry.sources.some((source) => source.principalType !== "role");
}

function groupFacets(
  access: readonly DatasetAccess[],
  groups: readonly WorkspaceGroup[],
  memberOf: ReadonlySet<string>,
): DatasetFacet[] {
  const memberCounts = new Map(groups.map((group) => [group.id, group.memberCount]));
  const byGroup = new Map<string, { name: string; datasetIds: Set<string> }>();
  for (const entry of access) {
    for (const source of entry.sources) {
      if (source.principalType !== "role") continue;
      const facet = byGroup.get(source.principalId) ?? { name: source.name ?? "Group", datasetIds: new Set<string>() };
      facet.datasetIds.add(entry.datasetId);
      byGroup.set(source.principalId, facet);
    }
  }
  // A group I am in appears even when it holds nothing yet, so a new group does
  // not look like it never reached me.
  for (const group of groups) {
    if (memberOf.has(group.id) && !byGroup.has(group.id)) {
      byGroup.set(group.id, { name: group.name, datasetIds: new Set<string>() });
    }
  }
  return [...byGroup.entries()]
    .map(([id, { name, datasetIds }]) => {
      const members = memberCounts.get(id);
      const people = members === undefined ? "Group" : plural(members, "member");
      return {
        id,
        kind: "group" as const,
        label: name,
        hint: datasetIds.size === 0 ? `${people} · no datasets yet` : people,
        datasetIds,
      };
    })
    .sort((a, b) => a.label.localeCompare(b.label));
}

/** *memberOf*: ids of the groups the signed-in user belongs to. */
export function buildFacets(
  access: readonly DatasetAccess[],
  groups: readonly WorkspaceGroup[],
  memberOf: ReadonlySet<string> = new Set(),
): DatasetFacet[] {
  const ids = (keep: (entry: DatasetAccess) => boolean): Set<string> =>
    new Set(access.filter(keep).map((entry) => entry.datasetId));
  return [
    { id: ALL_FACET_ID, kind: "all", label: "All datasets", hint: "Everything you can open", datasetIds: ids(() => true) },
    { id: PERSONAL_FACET_ID, kind: "personal", label: "Personal", hint: "Only you, unless shared", datasetIds: ids((entry) => entry.isOwner) },
    { id: SHARED_FACET_ID, kind: "shared", label: "Shared with me", hint: "Given to you directly", datasetIds: ids(isSharedDirectly) },
    ...groupFacets(access, groups, memberOf),
  ];
}

/** The datasets one facet shows. "All" shows every dataset on the page, even
 *  one the access list has not caught up with yet (just created, say). */
export function filterByFacet<T extends { id: string }>(datasets: readonly T[], facet: DatasetFacet | undefined): T[] {
  if (!facet || facet.kind === "all") return [...datasets];
  return datasets.filter((dataset) => facet.datasetIds.has(dataset.id));
}

/** A dataset this session itself just created (a submitted placeholder, or
 *  one the pod has already confirmed but the access list hasn't caught up
 *  with yet) is owned by nobody else: datasets are private until explicitly
 *  shared (see handleCreate), so it is Personal, not merely unclassified.
 *  Folding it into that facet here means Personal is correct the moment the
 *  dataset exists, without waiting on a round trip to the access endpoint.
 *
 *  *mineIds* must be scoped to datasets actually known to be the caller's own
 *  (handleCreate's placeholders and confirmations), never to "missing from
 *  the access list" — an access list that's mid-refresh, failed after a good
 *  load, or dropped a malformed row would otherwise read a dataset actually
 *  shared with someone else as Personal. */
export function withUnclaimedAsPersonal(
  facets: readonly DatasetFacet[],
  mineIds: readonly string[],
): DatasetFacet[] {
  if (mineIds.length === 0) return [...facets];
  return facets.map((facet) =>
    facet.id === PERSONAL_FACET_ID
      ? { ...facet, datasetIds: new Set([...facet.datasetIds, ...mineIds]) }
      : facet);
}
