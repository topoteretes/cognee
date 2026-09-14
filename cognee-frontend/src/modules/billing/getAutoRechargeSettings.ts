"use server";

export interface AutoRechargeSettings {
  autoRechargeEnabled: boolean;
  rechargeThresholdUsd: number | null;
  rechargeAmountUsd: number | null;
  monthlyRechargeLimitUsd: number | null;
  monthlyRechargedUsd: number;
  notifyThresholdUsd: number | null;
  lastRechargeError: string | null;
}

/**
 * Open-source stub — auto-recharge requires the cloud billing backend.
 * Always returns null so the insufficient-credits modal simply omits
 * auto-recharge state in OSS mode.
 */
export default async function getAutoRechargeSettings(
  _tenantId: string,
): Promise<AutoRechargeSettings | null> {
  return null;
}
