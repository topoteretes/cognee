"use client";

import { useCallback, useEffect, useState, type ReactNode } from "react";
import { setInsufficientCreditsListener, type InsufficientCreditsEvent } from "@/services/credits/insufficientCreditsBridge";
import { useCreditsBalance, getTenantRemainingUsd } from "@/modules/billing/useCreditsBalance";
import { useInsufficientCreditsNotice } from "@/modules/billing/useInsufficientCreditsNotice";
import { useTenant } from "@/modules/tenant/TenantProvider";
import { trackEvent } from "@/modules/analytics";
import InsufficientCreditsModal from "@/ui/elements/InsufficientCreditsModal";
import InsufficientCreditsNotice from "@/ui/elements/InsufficientCreditsNotice";

interface InsufficientCreditsState {
  isOpen: boolean;
  operation: string | null;
}

// Single mount point (in the (app) layout) that reacts to every 402 the pod
// client's error interceptor reports, from any page. Pages that make pod
// calls (upload, search, recall, ...) no longer need to know this modal (or
// the persistent notice below) exists — see the interceptor registration in
// services/http/pod.ts.
export default function InsufficientCreditsProvider({ children }: { children: ReactNode }): React.ReactElement {
  const [state, setState] = useState<InsufficientCreditsState>({ isOpen: false, operation: null });
  const { data: credits, isFetching: isLoadingBalance } = useCreditsBalance(state.isOpen);
  const { tenant } = useTenant();
  const notice = useInsufficientCreditsNotice(tenant?.tenant_id ?? null);

  const { dismiss: dismissNotice, record: recordNotice } = notice;

  const handleClose = useCallback(() => {
    setState({ isOpen: false, operation: null });
    // The user just saw this failure live — don't resurrect it as a
    // persistent notice on their next page load.
    dismissNotice();
  }, [dismissNotice]);

  useEffect(() => {
    function handleInsufficientCredits(event: InsufficientCreditsEvent): void {
      setState({ isOpen: true, operation: event.operation });
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
        remainingUsd={getTenantRemainingUsd(credits, tenant?.tenant_id ?? null)}
        isLoadingBalance={isLoadingBalance}
        onClose={handleClose}
      />
    </>
  );
}
