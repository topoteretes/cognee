"use client";

import { useState, useCallback } from "react";
import { useTenant } from "@/modules/tenant/TenantProvider";
import { useCreditsBalance, getTenantRow } from "@/modules/billing/useCreditsBalance";
import { isCreditsBannerDismissed, dismissCreditsBanner } from "@/utils/browserStorage";

export interface CreditsBannerState {
  creditsSpentPct: number | null;
  creditsRemainingUsd: number | null;
  /** True when ≥ 90 % of credits are spent. Wins over all other banners. */
  showCreditPctBanner: boolean;
  /** True when balance < $1 and credit-pct banner is not showing. */
  showLowBalanceBanner: boolean;
  /** True when neither warning banner is active. */
  showVoucherBanner: boolean;
  dismiss: () => void;
}

/**
 * Fetches credit usage and tracks banner visibility.
 *
 * Only one banner may show at a time — priority:
 *   1. Credit-percentage banner (≥ 90 %)
 *   2. Low-balance banner (< $1)
 *   3. Voucher banner (promotional)
 */
export function useCreditsBanner(): CreditsBannerState {
  const { tenant } = useTenant();
  const { data: overview } = useCreditsBalance(true);
  const [dismissed, setDismissed] = useState<boolean>(isCreditsBannerDismissed);

  const row = getTenantRow(overview, tenant?.tenant_id ?? null);
  const creditsSpentPct =
    row?.spentUsd != null && row.maxBudgetUsd ? Math.round((row.spentUsd / row.maxBudgetUsd) * 100) : null;
  const creditsRemainingUsd = row?.remainingUsd ?? null;

  const showCreditPctBanner = !dismissed && creditsSpentPct !== null && creditsSpentPct >= 90;
  const showLowBalanceBanner = !showCreditPctBanner && creditsRemainingUsd !== null && creditsRemainingUsd < 1;
  const showVoucherBanner = !showCreditPctBanner && !showLowBalanceBanner;

  const dismiss = useCallback(() => {
    dismissCreditsBanner();
    setDismissed(true);
  }, []);

  return {
    creditsSpentPct,
    creditsRemainingUsd,
    showCreditPctBanner,
    showLowBalanceBanner,
    showVoucherBanner,
    dismiss,
  };
}
