"use client";

/**
 * Open-source stub — the workspace and its groups, without its people.
 *
 * The SaaS version also lists the workspace's members through the management
 * API (members/actions/listTenantMembers), which the public build excludes.
 */
import { useEffect, useState } from "react";
import getGroupOverview from "@/modules/permissions/getGroupOverview";
import { useCogniInstance, useTenant } from "@/modules/tenant/TenantProvider";

export interface ShareCandidate {
  id: string;
  kind: "user" | "role" | "tenant";
  label: string;
}

export const WORKSPACE_LABEL = "Everyone in the workspace";

export default function useShareCandidates(): ShareCandidate[] | null {
  const { cogniInstance } = useCogniInstance();
  const { tenant } = useTenant();
  const [candidates, setCandidates] = useState<ShareCandidate[] | null>(null);
  const tenantId = tenant?.tenant_id ?? null;

  useEffect(() => {
    if (!cogniInstance || !tenantId) return;
    let cancelled = false;
    getGroupOverview(cogniInstance)
      .then((groups) => {
        if (cancelled) return;
        setCandidates([
          { id: tenantId, kind: "tenant", label: WORKSPACE_LABEL },
          ...groups.map((group): ShareCandidate => ({ id: group.id, kind: "role", label: group.name })),
        ]);
      })
      .catch((e: unknown) => {
        console.error("[DATASETS] Could not load who a dataset can be shared with:", e instanceof Error ? e.message : String(e));
        if (!cancelled) setCandidates([]);
      });
    return () => { cancelled = true; };
  }, [cogniInstance, tenantId]);

  return candidates;
}
