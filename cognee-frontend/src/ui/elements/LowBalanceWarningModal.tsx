"use client";

import { useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { trackEvent } from "@/modules/analytics";
import type { PendingLowBalanceWarning } from "@/modules/billing/useLowBalanceUploadWarning";

interface LowBalanceWarningModalProps {
  warning: PendingLowBalanceWarning | null;
  onCancel: () => void;
}

// Deliberately phrased as a ceiling ("up to"), never a prediction: the
// underlying estimate already bakes in a safety margin, and for PDF/DOCX the
// range's own high end assumes a text density most real files don't reach
// (see estimateUploadCostUsd.ts) — so the true cost is almost always lower
// than this number. Framing it as "up to" makes that the expected, unremarkable
// outcome instead of a broken promise (mirrors why the pod's own 402 message
// omits a cost figure entirely — see check_credits.py's _enforce).
function costLabel(warning: PendingLowBalanceWarning): string {
  const ceiling = warning.kind === "point" ? warning.estimatedUsd : warning.highUsd;
  return `Up to ~$${ceiling.toFixed(2)}`;
}

function bodyCopy(warning: PendingLowBalanceWarning): string {
  return warning.kind === "point"
    ? "Based on the file size, this could use more credits than your current balance — the actual cost is usually lower than this ceiling. Top up to continue with this upload."
    : "We can't tell exactly how much text a PDF/DOCX contains before uploading, so this is a worst-case ceiling, not a prediction — the actual cost is usually well below it. Top up to continue with this upload.";
}

function buildTrackingProps(warning: PendingLowBalanceWarning): { [key: string]: string } {
  return {
    estimate_kind: warning.kind,
    remaining_usd: warning.remainingUsd.toFixed(2),
    ...(warning.kind === "point"
      ? { estimated_usd: warning.estimatedUsd.toFixed(2) }
      : { estimated_low_usd: warning.lowUsd.toFixed(2), estimated_high_usd: warning.highUsd.toFixed(2) }),
  };
}

// Pre-flight, non-blocking version of InsufficientCreditsModal: shown BEFORE
// the upload request fires, using a client-side cost estimate (see
// estimateUploadCostUsd), instead of reacting to the pod's own 402 after the
// fact. Since commit 2c99316 it is a hard gate: the only way forward is to top
// up, so both exits abandon the upload.
export default function LowBalanceWarningModal({
  warning,
  onCancel,
}: LowBalanceWarningModalProps): React.ReactElement | null {
  const router = useRouter();
  // Denominator for the two exit events: without it the gate's volume is
  // invisible, since neither exit fires when the user simply abandons the tab.
  // useLowBalanceUploadWarning builds a fresh warning object per gate and
  // nulls it on dismiss, so object identity IS the appearance — and comparing
  // against it also defeats StrictMode's dev-only double effect invocation.
  const trackedWarning = useRef<PendingLowBalanceWarning | null>(null);
  useEffect(() => {
    if (!warning || trackedWarning.current === warning) return;
    trackedWarning.current = warning;
    trackEvent({ pageName: "Low Balance Warning Modal", eventName: "low_balance_upload_blocked", additionalProperties: buildTrackingProps(warning) });
  }, [warning]);

  if (!warning) return null;
  // Aliased so the nested handlers below narrow correctly — TS doesn't carry
  // the null-check narrowing of a prop into inner function closures.
  const activeWarning = warning;
  const trackingProps = buildTrackingProps(activeWarning);

  function handleCancel(): void {
    trackEvent({ pageName: "Low Balance Warning Modal", eventName: "low_balance_upload_cancelled", additionalProperties: trackingProps });
    onCancel();
  }

  function goToBilling(): void {
    trackEvent({ pageName: "Low Balance Warning Modal", eventName: "low_balance_upload_billing_clicked", additionalProperties: trackingProps });
    onCancel();
    router.push("/billing");
  }

  return (
    <div
      style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.5)", backdropFilter: "blur(4px)", WebkitBackdropFilter: "blur(4px)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }}
      onClick={handleCancel}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{ background: "rgba(15,15,15,0.92)", backdropFilter: "blur(16px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, padding: 24, width: 420, display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 16px 48px rgba(0,0,0,0.12)" }}
      >
        <h2 style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", margin: 0 }}>
          This upload might use more credits than you have
        </h2>
        <p style={{ fontSize: 13, color: "rgba(237,236,234,0.55)", margin: 0 }}>
          {bodyCopy(activeWarning)}
        </p>

        <div style={{ display: "flex", gap: 8 }}>
          <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 2, padding: "10px 12px", borderRadius: 8, border: "1px solid rgba(255,255,255,0.1)", background: "rgba(255,255,255,0.04)" }}>
            <span style={{ fontSize: 11, color: "rgba(237,236,234,0.5)" }}>Could cost</span>
            <span style={{ fontSize: 15, fontWeight: 700, color: "#EDECEA", fontVariantNumeric: "tabular-nums" }}>{costLabel(activeWarning)}</span>
          </div>
          <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 2, padding: "10px 12px", borderRadius: 8, border: "1px solid rgba(101,16,244,0.45)", background: "rgba(101,16,244,0.08)" }}>
            <span style={{ fontSize: 11, color: "rgba(237,236,234,0.5)" }}>Current balance</span>
            <span style={{ fontSize: 15, fontWeight: 700, color: "#EDECEA", fontVariantNumeric: "tabular-nums" }}>${activeWarning.remainingUsd.toFixed(2)}</span>
          </div>
        </div>

        <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
          <button
            onClick={handleCancel}
            className="cursor-pointer"
            style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "rgba(237,236,234,0.7)", fontFamily: "inherit" }}
          >
            Cancel
          </button>
          <button
            onClick={goToBilling}
            className="cursor-pointer"
            style={{ background: "#6510F4", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "#fff", fontFamily: "inherit" }}
          >
            Top up first
          </button>
        </div>
      </div>
    </div>
  );
}
