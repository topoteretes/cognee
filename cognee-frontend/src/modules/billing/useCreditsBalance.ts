"use client";

// Public copy of the SaaS module of the same path. The directory is excluded
// from the sync (it contains cloud-only billing/config actions), but this file
// itself is portable and its consumers are shared UI — keep it compatible with
// the SaaS original; the sync build gate fails if the signatures drift.

import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import type { CreditsOverview, CreditsOverviewResult } from "./creditsOverviewTypes";
import getCreditsOverview from "./getCreditsOverview";

// Only fetched on demand (enabled=false until the modal opens) — there is no
// app-wide balance polling today, and this hook isn't meant to introduce one.
export function useCreditsBalance(enabled: boolean): UseQueryResult<CreditsOverviewResult> {
  return useQuery({
    queryKey: ["credits-overview"],
    queryFn: getCreditsOverview,
    enabled,
    staleTime: 30_000,
    retry: false,
  });
}

export type TenantCreditsRow = CreditsOverview["tenants"][number];

// `overview.budget` is the account-wide aggregate across every tenant the
// user owns — the pod's credit guard is enforced per workspace, so callers
// must resolve the balance for the active tenant specifically (matches the
// pattern in BillingPage.tsx). Returns null if the overview hasn't loaded
// yet, the request failed, or the tenant has no budget row — every consumer
// of these two helpers treats "no number" the same way, so unwrapping the
// result here is what keeps them from each having to handle the union.
export function getTenantRow(result: CreditsOverviewResult | null | undefined, tenantId: string | null): TenantCreditsRow | null {
  if (result?.status !== "ok" || !tenantId) return null;
  return result.data.tenants.find((t) => t.tenantId === tenantId) ?? null;
}

export function getTenantRemainingUsd(result: CreditsOverviewResult | null | undefined, tenantId: string | null): number | null {
  return getTenantRow(result, tenantId)?.remainingUsd ?? null;
}
