"use client";

import { tokens } from "@/ui/theme/tokens";

// $5/mo workspace base fee — shown in the create-workspace modal.
export const WORKSPACE_PRICE_LABEL = "$5/month";

interface NameWorkspaceModalProps {
  name: string;
  setName: (v: string) => void;
  submitting: boolean;
  // Covered by the free allowance — no price notice, no trip to Stripe.
  isFree: boolean;
  error: string | null;
  onSubmit: () => void;
  onClose: () => void;
}

export default function NameWorkspaceModal({
  name, setName, submitting, isFree, error, onSubmit, onClose,
}: NameWorkspaceModalProps): React.JSX.Element {
  const valid = name.trim().length >= 2 && name.trim().length <= 50;
  return (
    <div
      style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.45)", backdropFilter: "blur(4px)", WebkitBackdropFilter: "blur(4px)", zIndex: 1000, display: "flex", alignItems: "center", justifyContent: "center" }}
      onClick={onClose}
    >
      <form
        onClick={(e) => e.stopPropagation()}
        onSubmit={(e) => { e.preventDefault(); if (valid && !submitting) onSubmit(); }}
        style={{
          background: "rgba(15,15,15,0.92)",
          backdropFilter: "blur(16px)",
          border: "1px solid rgba(255,255,255,0.1)",
          borderRadius: 12,
          padding: 24,
          width: 400,
          boxShadow: "0 20px 60px rgba(0,0,0,0.6)",
        }}
      >
        <div style={{ marginBottom: 16 }}>
          <h2 style={{ fontSize: 16, fontWeight: 700, color: "#EDECEA", margin: 0, fontFamily: '"TWKLausanne", sans-serif' }}>
            Create a new workspace
          </h2>
          <p style={{ fontSize: 13, color: "rgba(237,236,234,0.55)", margin: "4px 0 0" }}>
            Name your workspace. You can switch between workspaces from the top bar.
          </p>
        </div>
        <div style={{ display: "flex", alignItems: "flex-start", gap: 8, marginBottom: 14, padding: "10px 12px", borderRadius: 8, background: "rgba(188,155,255,0.1)", border: "1px solid rgba(188,155,255,0.25)" }}>
          <span style={{ fontSize: 12.5, color: "rgba(237,236,234,0.8)", lineHeight: 1.45 }}>
            {isFree ? (
              <>Your first workspace is <strong style={{ color: "#EDECEA" }}>free</strong>. Workspaces you were invited to don&apos;t count, so this one costs nothing.</>
            ) : (
              <>A new workspace costs <strong style={{ color: "#EDECEA" }}>{WORKSPACE_PRICE_LABEL}</strong>. You&apos;ll be taken to Stripe to confirm payment — the workspace is created once payment succeeds.</>
            )}
          </span>
        </div>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          autoFocus
          maxLength={50}
          placeholder="Workspace name"
          style={{
            width: "100%",
            boxSizing: "border-box",
            background: "rgba(255,255,255,0.04)",
            border: "1px solid rgba(255,255,255,0.12)",
            borderRadius: 8,
            padding: "10px 12px",
            fontSize: 14,
            color: "#EDECEA",
            outline: "none",
            fontFamily: "inherit",
          }}
        />
        {error && (
          <div style={{ marginTop: 10, fontSize: 13, color: "#F87171" }}>{error}</div>
        )}
        <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 18 }}>
          {/* Never disabled while submitting: gating it meant a request that
              never came back left the user with no way out but a reload
              (CLO-747). The submit is NOT fire-and-forget — it ends in a
              full-page navigation to Stripe or a full-screen takeover by
              WorkspaceSetupScreen — so dismissing has to actually cancel those:
              useWorkspaceCreation tracks a dismissal generation and drops the
              result on every branch that has not already been charged. */}
          <button
            type="button"
            onClick={onClose}
            style={{ padding: "8px 14px", borderRadius: 8, border: "1px solid rgba(255,255,255,0.12)", background: "transparent", color: "#EDECEA", fontSize: 13, cursor: "pointer" }}
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={!valid || submitting}
            style={{ padding: "8px 14px", borderRadius: 8, border: "none", background: tokens.purple, color: "#fff", fontSize: 13, fontWeight: 500, cursor: !valid || submitting ? "default" : "pointer", opacity: !valid || submitting ? 0.6 : 1 }}
          >
            {submitting
              ? (isFree ? "Creating..." : "Redirecting...")
              : (isFree ? "Create workspace" : `Continue to payment · ${WORKSPACE_PRICE_LABEL}`)}
          </button>
        </div>
      </form>
    </div>
  );
}
