"use client";

interface CreditBannersProps {
  creditsSpentPct: number | null;
  showCreditPctBanner: boolean;
  showLowBalanceBanner: boolean;
  showVoucherBanner: boolean;
  onDismiss: () => void;
  isOwner: boolean;
}

const WARN_ICON = (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
    <line x1="12" y1="9" x2="12" y2="13" />
    <line x1="12" y1="17" x2="12.01" y2="17" />
  </svg>
);

const CLOSE_ICON = (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
    <line x1="18" y1="6" x2="6" y2="18" />
    <line x1="6" y1="6" x2="18" y2="18" />
  </svg>
);

/**
 * Renders at most one credit-related banner at a time. Priority is enforced
 * by the parent hook (useCreditsBanner) — only one show-flag will be true.
 */
export function CreditBanners({
  creditsSpentPct,
  showCreditPctBanner,
  showLowBalanceBanner,
  showVoucherBanner,
  onDismiss,
  isOwner,
}: CreditBannersProps): React.ReactElement | null {
  if (showCreditPctBanner && creditsSpentPct !== null) {
    const isOut = creditsSpentPct >= 100;
    const accent = isOut ? "#EF4444" : "#EAB308";
    const textColor = isOut ? "#FCA5A5" : "#FDE047";
    return (
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        gap: 12, flexWrap: "wrap",
        background: isOut ? "rgba(239,68,68,0.10)" : "rgba(234,179,8,0.10)",
        border: `1px solid ${isOut ? "rgba(239,68,68,0.30)" : "rgba(234,179,8,0.30)"}`,
        borderRadius: 10, padding: "12px 16px",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ color: accent }}>{WARN_ICON}</span>
          <span style={{ fontSize: 13, color: textColor }}>
            {isOut
              ? "Your workspace has used all available credits — agent requests may fail."
              : `Your workspace has used ${creditsSpentPct}% of available credits.`}
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
          {isOwner ? (
            <a href="/billing" style={{ fontSize: 13, fontWeight: 500, color: textColor, textDecoration: "underline", textUnderlineOffset: 3 }}>
              Top up credits →
            </a>
          ) : (
            <span style={{ fontSize: 13, color: "rgba(237,236,234,0.5)" }}>Ask the workspace owner to top up.</span>
          )}
          <button
            onClick={onDismiss}
            aria-label="Dismiss"
            style={{ background: "none", border: "none", cursor: "pointer", padding: 2, color: "rgba(237,236,234,0.4)", lineHeight: 1 }}
          >
            {CLOSE_ICON}
          </button>
        </div>
      </div>
    );
  }

  if (showLowBalanceBanner) {
    return (
      <div style={{
        display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap",
        background: "rgba(239,68,68,0.10)",
        border: "1px solid rgba(239,68,68,0.30)",
        borderRadius: 10, padding: "12px 16px",
      }}>
        <span style={{ color: "#EF4444" }}>{WARN_ICON}</span>
        <span style={{ fontSize: 13, color: "#FCA5A5" }}>
          Your Token Balance is below 1 USD. You can recharge credits{" "}
          <a href="/billing" style={{ color: "#FCA5A5", textDecoration: "underline", textUnderlineOffset: 3 }}>
            here
          </a>
          .
        </span>
      </div>
    );
  }

  if (showVoucherBanner) {
    return (
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        gap: 12, flexWrap: "wrap",
        background: "var(--color-cognee-lavender-tint-10)",
        border: "1px solid rgba(188,155,255,0.30)",
        borderRadius: 10, padding: "12px 16px",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--color-cognee-lavender)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M20 12a2 2 0 0 1 2-2V7a2 2 0 0 0-2-2H4a2 2 0 0 0-2 2v3a2 2 0 0 1 0 4v3a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-3a2 2 0 0 1-2-2z" />
            <line x1="13" y1="5" x2="13" y2="19" />
          </svg>
          <span style={{ fontSize: 13, color: "#D9C7FF" }}>You have a voucher? Redeem it here.</span>
        </div>
        <a
          href="/billing"
          style={{
            flexShrink: 0,
            background: "var(--color-cognee-lavender)",
            color: "var(--color-cognee-lavender-text)",
            borderRadius: 8,
            padding: "8px 16px",
            fontSize: 13,
            fontWeight: 500,
            textDecoration: "none",
            whiteSpace: "nowrap",
          }}
        >
          Redeem voucher →
        </a>
      </div>
    );
  }

  return null;
}
