"use client";

import { useQuery } from "@tanstack/react-query";
import getGovernanceGraph from "../getGovernanceGraph";
import type { BusinessLayer, BusinessLayerContext, BusinessLayerResult } from "./types";

export const GOVERNANCE_LAYER_ID = "governance";

// "Who may do what" — tenant/user/role membership, dataset ownership and ACL
// grants. Fixed regardless of which brain is focused, unlike the content
// layer below, which swaps with the selected dataset.
const governanceLayer: BusinessLayer = {
  id: GOVERNANCE_LAYER_ID,
  label: "Governance",
  defaultEnabled: true,
  useData(ctx: BusinessLayerContext): BusinessLayerResult {
    const { data, isLoading, error } = useQuery({
      // Scoped by instanceId — same class of bug as businessBrainsQueryKey:
      // an unscoped key served the PREVIOUS tenant's users/ACLs/emails for
      // up to staleTime after switching tenants (COG-6233).
      queryKey: ["business-governance", ctx.cogniInstance.instanceId],
      queryFn: () => getGovernanceGraph(ctx.cogniInstance),
      staleTime: 60_000,
    });
    return { data: data ?? null, isLoading, error };
  },
};

export default governanceLayer;
