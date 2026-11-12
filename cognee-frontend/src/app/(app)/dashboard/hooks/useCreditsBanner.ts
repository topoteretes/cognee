"use client";

import { useState, useCallback } from "react";
import { useTenant } from "@/modules/tenant/TenantProvider";
import { useCreditsBalance, getTenantRemainingUsd } from "@/modules/billing/useCreditsBalance";
import { isCreditsBannerDismissed, dismissCreditsBanner } from "@/utils/browserStorage";

export interface CreditsBannerState {
  creditsRemainingUsd: number | null;
  /** True when balance < $5. */
  showLowBalanceBanner: boolean;
  /** True when the low-balance banner is not active. */
  showVoucherBanner: boolean;
  dismiss: () => void;
}

const LOW_BALANCE_THRESHOLD_USD = 5;

/**
 * Fetches credit usage and tracks banner visibility.
 *
 * Only one banner may show at a time — priority:
 *   1. Low-balance banner (< $5)
 *   2. Voucher banner (promotional)
 */
export function useCreditsBanner(): CreditsBannerState {
  const { tenant } = useTenant();
  const { data: overview } = useCreditsBalance(true);
  const [dismissed, setDismissed] = useState<boolean>(isCreditsBannerDismissed);

  const creditsRemainingUsd = getTenantRemainingUsd(overview, tenant?.tenant_id ?? null);

  const isLowBalance = creditsRemainingUsd !== null && creditsRemainingUsd < LOW_BALANCE_THRESHOLD_USD;
  const showLowBalanceBanner = !dismissed && isLowBalance;
  const showVoucherBanner = !isLowBalance;

  const dismiss = useCallback(() => {
    dismissCreditsBanner();
    setDismissed(true);
  }, []);

  return {
    creditsRemainingUsd,
    showLowBalanceBanner,
    showVoucherBanner,
    dismiss,
  };
}
