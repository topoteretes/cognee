"use server";

export interface CreditsOverview {
  budget: { totalUsd: number; spentUsd: number | null; remainingUsd: number | null; spentPct: number | null };
  tenants: { tenantId: string; tenantName: string; spentUsd: number | null; maxBudgetUsd: number | null; remainingUsd: number | null }[];
  purchases: { id: string; tokensPurchased: number; amountPaidUsd: number; currency: string; createdAt: string }[];
}

/**
 * Open-source stub — credit overview requires the cloud billing backend.
 * Always returns null so the dashboard banner is simply not shown in OSS mode.
 */
export default async function getCreditsOverview(): Promise<CreditsOverview | null> {
  return null;
}
