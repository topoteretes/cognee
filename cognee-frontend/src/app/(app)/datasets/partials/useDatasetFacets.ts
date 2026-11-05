"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { notifications } from "@mantine/notifications";
import getDatasetAccess from "@/modules/permissions/getDatasetAccess";
import getGroupOverview from "@/modules/permissions/getGroupOverview";
import { useCogniInstance, useTenant } from "@/modules/tenant/TenantProvider";
import { useCurrentUser } from "@/modules/users/useCurrentUser";
import isCloudEnvironment from "@/utils/isCloudEnvironment";
import { buildFacets, type DatasetFacet } from "@/app/(app)/datasets/partials/datasetFacets";
import type { DatasetAccess, GroupOverview } from "@/modules/permissions/types";

export type FacetsStatus = "loading" | "error" | "ready";

export interface DatasetFacets {
  facets: DatasetFacet[];
  accessById: ReadonlyMap<string, DatasetAccess>;
  status: FacetsStatus;
  refresh: () => Promise<void>;
}

/**
 * The left rail of the datasets page, from who holds what on each dataset.
 * Reloads whenever *datasetKey* changes, so a dataset created or deleted on
 * the page lands in its facets without a manual refresh.
 */
export default function useDatasetFacets(datasetKey: string): DatasetFacets {
  const { cogniInstance } = useCogniInstance();
  const { tenant } = useTenant();
  const [access, setAccess] = useState<DatasetAccess[]>([]);
  const [groups, setGroups] = useState<GroupOverview[]>([]);
  const { data: me } = useCurrentUser();
  const myEmail = me?.email?.toLowerCase() ?? null;
  const [status, setStatus] = useState<FacetsStatus>(isCloudEnvironment() ? "loading" : "ready");
  const requestId = useRef(0);
  // True once a load has actually landed, so a later failure can be told
  // apart from the first one: a refresh that fails has something to fall
  // back to, a first load that fails does not.
  const hasLoaded = useRef(false);

  const refresh = useCallback(async (): Promise<void> => {
    // These overview endpoints belong to the Cloud pod, not the local SDK API.
    if (!isCloudEnvironment()) return;
    if (!tenant || !cogniInstance) return;
    const id = ++requestId.current;
    try {
      const [loadedAccess, loadedGroups] = await Promise.all([
        getDatasetAccess(cogniInstance),
        getGroupOverview(cogniInstance),
      ]);
      if (id !== requestId.current) return;
      setAccess(loadedAccess);
      setGroups(loadedGroups);
      setStatus("ready");
      hasLoaded.current = true;
    } catch (e) {
      if (id !== requestId.current) return;
      console.warn("[DATASETS] Could not load dataset access:", e instanceof Error ? e.message : String(e));
      if (!hasLoaded.current) {
        // Nothing to fall back to yet, so the rail collapses to "All" and
        // Share hides itself until a load actually succeeds.
        setStatus("error");
        return;
      }
      // A refresh failing after a good load keeps the facets and the access
      // map it already has: they are still probably correct, and throwing
      // them away would silently revoke Share and collapse every group the
      // user was looking at for no reason the backend gave us. Surface the
      // failure instead of pretending it didn't happen.
      notifications.show({
        title: "Couldn't refresh who has access",
        message: "Showing the last known groups and permissions, they may be out of date.",
        color: "yellow",
      });
    }
  }, [cogniInstance, tenant]);

  useEffect(() => {
    void refresh();
  }, [refresh, datasetKey]);

  const facets = useMemo(() => {
    const memberOf = new Set(
      groups.filter((group) => group.members.some((member) => member.email.toLowerCase() === myEmail)).map((group) => group.id),
    );
    const named = groups.map((group) => ({ id: group.id, name: group.name, memberCount: group.members.length }));
    return buildFacets(access, named, memberOf);
  }, [access, groups, myEmail]);
  const accessById = useMemo(() => new Map(access.map((entry) => [entry.datasetId, entry])), [access]);

  return { facets, accessById, status, refresh };
}
