"use client";

import { useState, useEffect, useCallback } from "react";
import { useTenant } from "@/modules/tenant/TenantProvider";
import getCreditsOverview from "@/modules/billing/getCreditsOverview";
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
  const { tenant, isOwner } = useTenant();
  const [creditsSpentPct, setCreditsSpentPct] = useState<number | null>(null);
  const [creditsRemainingUsd, setCreditsRemainingUsd] = useState<number | null>(null);
  const [dismissed, setDismissed] = useState<boolean>(isCreditsBannerDismissed);

  useEffect(() => {
    if (!tenant) return;
    getCreditsOverview()
      .then((ov) => {
        if (!ov) return;
        const row = ov.tenants.find((t) => t.tenantId === tenant.tenant_id);
        if (!row) return;
        if (row.spentUsd != null && row.maxBudgetUsd) {
          setCreditsSpentPct(Math.round((row.spentUsd / row.maxBudgetUsd) * 100));
        }
        if (row.remainingUsd != null) {
          setCreditsRemainingUsd(row.remainingUsd);
        }
      })
      .catch((err) => console.error("Failed to fetch credits overview:", err));
  }, [isOwner, tenant]);

  const showCreditPctBanner = !dismissed && creditsSpentPct !== null && creditsSpentPct >= 90;
  const showLowBalanceBanner = !showCreditPctBanner && creditsRemainingUsd !== null && creditsRemainingUsd < 1;
  // Promotional-only banner, hidden — voucher redemption isn't offered here.
  const showVoucherBanner = false;

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
