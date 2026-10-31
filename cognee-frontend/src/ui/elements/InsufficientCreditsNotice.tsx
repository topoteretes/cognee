"use client";

import { trackEvent } from "@/modules/analytics";

interface InsufficientCreditsNoticeProps {
  isVisible: boolean;
  operation: string | null;
  onDismiss: () => void;
}

const OPERATION_LABELS: Record<string, string> = {
  remember: "upload",
  cognify: "processing",
  improve: "processing",
  search: "search",
  recall: "search",
};

const WARN_ICON = (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
    <line x1="12" y1="9" x2="12" y2="13" />
    <line x1="12" y1="17" x2="12.01" y2="17" />
  </svg>
);

const CLOSE_ICON = (
  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
    <line x1="18" y1="6" x2="6" y2="18" />
    <line x1="6" y1="6" x2="18" y2="18" />
  </svg>
);

// App-wide fallback for when the reactive InsufficientCreditsModal was missed
// (tab closed before the 402 arrived, client-side timeout beat the pod's
// response, etc.) — see useInsufficientCreditsNotice for the persistence
// logic. Fixed below TopBar (which is height:53/zIndex:300 — see
// ui/layout/TopBar.tsx) rather than inside CustomAppShell's layout flow, so
// this ticket doesn't have to touch that component's existing conditional
// rendering.
export default function InsufficientCreditsNotice({
  isVisible,
  operation,
  onDismiss,
}: InsufficientCreditsNoticeProps): React.ReactElement | null {
  if (!isVisible) return null;

  const actionLabel = operation ? OPERATION_LABELS[operation] ?? operation : "last action";

  function handleDismiss(): void {
    trackEvent({ pageName: "Insufficient Credits Notice", eventName: "insufficient_credits_notice_dismissed", additionalProperties: { operation: operation ?? "unknown" } });
    onDismiss();
  }

  function handleBillingClick(): void {
    trackEvent({ pageName: "Insufficient Credits Notice", eventName: "insufficient_credits_notice_billing_clicked", additionalProperties: { operation: operation ?? "unknown" } });
  }

  return (
    <div
      style={{
        position: "fixed", top: 53, left: 0, right: 0, zIndex: 310,
        display: "flex", alignItems: "center", justifyContent: "center", gap: 12,
        padding: "10px 16px",
        background: "rgba(239,68,68,0.96)",
        backdropFilter: "blur(8px)",
        boxShadow: "0 2px 12px rgba(0,0,0,0.25)",
      }}
    >
      <span style={{ color: "#fff", display: "flex", alignItems: "center" }}>{WARN_ICON}</span>
      <span style={{ fontSize: 13, color: "#fff" }}>
        Your last {actionLabel} failed — your workspace balance is too low.
      </span>
      <a
        href="/billing"
        onClick={handleBillingClick}
        style={{ fontSize: 13, fontWeight: 700, color: "#fff", textDecoration: "underline", textUnderlineOffset: 3, whiteSpace: "nowrap" }}
      >
        Top up credits →
      </a>
      <button
        onClick={handleDismiss}
        aria-label="Dismiss"
        className="cursor-pointer"
        style={{ background: "none", border: "none", padding: 2, color: "rgba(255,255,255,0.85)", lineHeight: 1 }}
      >
        {CLOSE_ICON}
      </button>
    </div>
  );
}
