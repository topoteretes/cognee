"use server";

export interface TenantHourlyCostPoint {
  hour: string;
  spendUsd: number;
}

export interface TenantHourlyCosts {
  tenantId: string;
  start: string;
  end: string;
  currency: string;
  points: TenantHourlyCostPoint[];
}

/**
 * Open-source stub — hourly cost tracking requires the cloud billing backend.
 * Callers render an empty chart on null, so this is indistinguishable from a
 * tenant with no spend.
 */
export default async function getTenantHourlyCosts(): Promise<TenantHourlyCosts | null> {
  return null;
}
