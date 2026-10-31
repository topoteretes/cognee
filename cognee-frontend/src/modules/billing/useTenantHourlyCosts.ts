"use client";

import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import getTenantHourlyCosts, { type TenantHourlyCosts } from "./getTenantHourlyCosts";

type CostRange = "24h" | "7d" | "30d";

/**
 * Open-source stub — mirrors the cloud hook's signature so CostPanel needs no
 * conditional logic; getTenantHourlyCosts always resolves to null here.
 */
export function useTenantHourlyCosts(
  tenantId: string | null,
  range: CostRange,
): UseQueryResult<TenantHourlyCosts | null> {
  return useQuery({
    queryKey: ["tenant-hourly-costs", tenantId, range],
    queryFn: () => getTenantHourlyCosts(),
    enabled: !!tenantId,
    staleTime: 60_000,
    retry: false,
  });
}
