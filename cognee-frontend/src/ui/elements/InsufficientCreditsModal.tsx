"use client";

import { useRouter } from "next/navigation";
import { Loader } from "@mantine/core";
import { trackEvent } from "@/modules/analytics";

export interface AutoRechargeModalInfo {
  enabled: boolean;
  lastError: string | null;
}

interface InsufficientCreditsModalProps {
  isOpen: boolean;
  operation: string | null;
  remainingUsd: number | null;
  isLoadingBalance: boolean;
  autoRecharge: AutoRechargeModalInfo | null;
  // While true, autoRecharge's current value (including a leftover null from
  // a previous tenant) must not be treated as a confident answer — show a
  // neutral message instead of asserting the wrong thing.
  autoRechargeLoading: boolean;
  onClose: () => void;
}

const OPERATION_LABELS: Record<string, string> = {
  remember: "upload",
  cognify: "processing",
  improve: "processing",
  search: "search",
  recall: "search",
};

// Fed by InsufficientCreditsProvider, which is the single mount point that
// listens for 402s across every pod call (upload, cognify, search, recall,
// improve) — no page wires this modal itself.
export default function InsufficientCreditsModal({
  isOpen,
  operation,
  remainingUsd,
  isLoadingBalance,
  autoRecharge,
  autoRechargeLoading,
  onClose,
}: InsufficientCreditsModalProps): React.ReactElement | null {
  const router = useRouter();

  if (!isOpen) return null;

  const actionLabel = operation ? OPERATION_LABELS[operation] ?? operation : "this action";

  function goToBilling(): void {
    trackEvent({ pageName: "Insufficient Credits Modal", eventName: "insufficient_credits_billing_clicked", additionalProperties: { operation: operation ?? "unknown" } });
    onClose();
    router.push("/billing");
  }

  function handleDismiss(): void {
    trackEvent({ pageName: "Insufficient Credits Modal", eventName: "insufficient_credits_modal_dismissed", additionalProperties: { operation: operation ?? "unknown" } });
    onClose();
  }

  return (
    <div
      style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.5)", backdropFilter: "blur(4px)", WebkitBackdropFilter: "blur(4px)", zIndex: 1000, display: "flex", alignItems: "center", justifyContent: "center" }}
      onClick={handleDismiss}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{ background: "rgba(15,15,15,0.92)", backdropFilter: "blur(16px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, padding: 24, width: 420, display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 16px 48px rgba(0,0,0,0.12)" }}
      >
        <h2 style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", margin: 0 }}>
          Not enough credits to run {actionLabel}
        </h2>
        {/* While autoRechargeLoading, we genuinely don't know the tenant's
            auto-recharge state yet, so this branch must not assert either
            "it's on" or "it's off" — both are confident claims that could
            flip a moment later once the fetch resolves. */}
        {autoRechargeLoading ? (
          <p style={{ fontSize: 13, color: "rgba(237,236,234,0.55)", margin: 0 }}>
            Checking your workspace&apos;s credit balance…
          </p>
        ) : autoRecharge?.enabled && autoRecharge.lastError ? (
          <p style={{ fontSize: 13, color: "#FCA5A5", margin: 0 }}>
            Auto recharge is on but the last automatic charge failed: {autoRecharge.lastError} Add
            credits manually or update your card on the billing page.
          </p>
        ) : autoRecharge?.enabled ? (
          <p style={{ fontSize: 13, color: "rgba(237,236,234,0.55)", margin: 0 }}>
            Auto recharge is on — a top-up may take a couple of minutes. Retry shortly, or add
            credits on the billing page to continue right away.
          </p>
        ) : (
          <p style={{ fontSize: 13, color: "rgba(237,236,234,0.55)", margin: 0 }}>
            Your workspace doesn&apos;t have enough credits left. Add credits on the billing page to continue.
          </p>
        )}

        <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 12px", borderRadius: 8, border: "1px solid rgba(101,16,244,0.45)", background: "rgba(101,16,244,0.08)" }}>
          <span style={{ fontSize: 12, color: "rgba(237,236,234,0.55)" }}>Current balance</span>
          <span style={{ flex: 1, textAlign: "right", fontSize: 15, fontWeight: 700, color: "#EDECEA", fontVariantNumeric: "tabular-nums" }}>
            {isLoadingBalance ? <Loader size={14} color="#EDECEA" /> : remainingUsd !== null ? `$${remainingUsd.toFixed(2)}` : "—"}
          </span>
        </div>

        <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
          <button
            onClick={handleDismiss}
            className="cursor-pointer"
            style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "rgba(237,236,234,0.7)", fontFamily: "inherit" }}
          >
            Dismiss
          </button>
          <button
            onClick={goToBilling}
            className="cursor-pointer"
            style={{ background: "#6510F4", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "#fff", fontFamily: "inherit" }}
          >
            Go to billing
          </button>
        </div>
      </div>
    </div>
  );
}
