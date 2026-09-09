"use client";

import { useCallback, useEffect, useState, type ReactNode } from "react";
import { setInsufficientCreditsListener, type InsufficientCreditsEvent } from "@/services/credits/insufficientCreditsBridge";
import { useCreditsBalance, getTenantRemainingUsd } from "@/modules/billing/useCreditsBalance";
import { useInsufficientCreditsNotice } from "@/modules/billing/useInsufficientCreditsNotice";
import getAutoRechargeSettings from "@/modules/billing/getAutoRechargeSettings";
import { useTenant } from "@/modules/tenant/TenantProvider";
import { trackEvent } from "@/modules/analytics";
import InsufficientCreditsModal, { type AutoRechargeModalInfo } from "@/ui/elements/InsufficientCreditsModal";
import InsufficientCreditsNotice from "@/ui/elements/InsufficientCreditsNotice";

interface InsufficientCreditsState {
  isOpen: boolean;
  operation: string | null;
  // The tenant that actually failed (from the request URL), NOT necessarily
  // the active tenant — the user may switch workspaces before a delayed 402
  // lands, and this modal must show that tenant's balance/auto-recharge
  // state, not whichever workspace happens to be active when it renders.
  tenantId: string | null;
}

// Single mount point (in the (app) layout) that reacts to every 402 the pod
// client's error interceptor reports, from any page. Pages that make pod
// calls (upload, search, recall, ...) no longer need to know this modal (or
// the persistent notice below) exists — see the interceptor registration in
// services/http/pod.ts.
export default function InsufficientCreditsProvider({ children }: { children: ReactNode }): React.ReactElement {
  const [state, setState] = useState<InsufficientCreditsState>({
    isOpen: false,
    operation: null,
    tenantId: null,
  });
  const [autoRecharge, setAutoRecharge] = useState<AutoRechargeModalInfo | null>(null);
  // Distinct from `autoRecharge === null` (which the modal's copy reads as a
  // confident "auto-recharge is off/unconfigured"): this is true while we
  // genuinely don't know yet, so a slow or failed fetch can never be shown
  // to the user as a false "it's off" or stale "it's on" from a prior tenant.
  const [autoRechargeLoading, setAutoRechargeLoading] = useState(false);
  const { data: credits, isFetching: isLoadingBalance } = useCreditsBalance(state.isOpen);
  const { tenant } = useTenant();
  const notice = useInsufficientCreditsNotice(tenant?.tenant_id ?? null);

  // Fetched only while the modal is open, keyed on the FAILING tenant
  // (state.tenantId), not whichever workspace is currently active — the
  // modal's copy changes when that tenant has auto-recharge on (top-up is
  // coming / the last charge failed), and switching workspaces while the
  // modal is open must not silently swap in a different tenant's state.
  useEffect(() => {
    if (!state.isOpen || !state.tenantId) return;
    let cancelled = false;
    setAutoRechargeLoading(true);
    getAutoRechargeSettings(state.tenantId)
      .then((settings) => {
        if (cancelled) return;
        setAutoRecharge(
          settings
            ? { enabled: settings.autoRechargeEnabled, lastError: settings.lastRechargeError }
            : null,
        );
      })
      .catch((e: unknown) => console.error("Failed to load auto-recharge settings:", e))
      .finally(() => { if (!cancelled) setAutoRechargeLoading(false); });
    return () => { cancelled = true; };
  }, [state.isOpen, state.tenantId]);

  const { dismiss: dismissNotice, record: recordNotice } = notice;

  const handleClose = useCallback(() => {
    setState({ isOpen: false, operation: null, tenantId: null });
    // The user just saw this failure live — don't resurrect it as a
    // persistent notice on their next page load.
    dismissNotice();
  }, [dismissNotice]);

  useEffect(() => {
    function handleInsufficientCredits(event: InsufficientCreditsEvent): void {
      setState({ isOpen: true, operation: event.operation, tenantId: event.tenantId });
      // Clear any prior tenant's result immediately — otherwise a second 402
      // for a DIFFERENT tenant could briefly show while the fetch below is
      // still in flight, the leftover autoRecharge value from the previous
      // tenant.
      setAutoRecharge(null);
      // Persisted in case the tab is closed or reloaded before this modal
      // is dismissed — the notice below is what surfaces it on next load.
      recordNotice({ operation: event.operation, at: event.at, tenantId: event.tenantId });
      trackEvent({ pageName: "Insufficient Credits Modal", eventName: "insufficient_credits_modal_opened", additionalProperties: { operation: event.operation ?? "unknown" } });
    }
    setInsufficientCreditsListener(handleInsufficientCredits);
    return () => setInsufficientCreditsListener(null);
  }, [recordNotice]);

  return (
    <>
      {children}
      <InsufficientCreditsNotice
        isVisible={notice.isVisible}
        operation={notice.operation}
        onDismiss={notice.dismiss}
      />
      <InsufficientCreditsModal
        isOpen={state.isOpen}
        operation={state.operation}
        remainingUsd={getTenantRemainingUsd(credits, state.tenantId)}
        isLoadingBalance={isLoadingBalance}
        autoRecharge={autoRecharge}
        autoRechargeLoading={autoRechargeLoading}
        onClose={handleClose}
      />
    </>
  );
}
