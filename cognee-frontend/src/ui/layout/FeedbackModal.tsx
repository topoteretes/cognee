"use client";

import { useCallback, useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { usePathname } from "next/navigation";
import { http } from "@/services/http/client";
import { useTenant } from "@/modules/tenant/TenantContext";

const SUCCESS_AUTO_CLOSE_MS = 2200;

interface Props {
  onClose: () => void;
}

export default function FeedbackModal({ onClose }: Props) {
  const { tenant } = useTenant();
  const pathname = usePathname();

  const [message, setMessage] = useState("");
  const [sending, setSending] = useState(false);
  const [sent, setSent] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [portalTarget, setPortalTarget] = useState<Element | null>(null);

  useEffect(() => {
    setPortalTarget(document.body);
  }, []);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  useEffect(() => {
    if (!sent) return;
    const timer = setTimeout(onClose, SUCCESS_AUTO_CLOSE_MS);
    return () => clearTimeout(timer);
  }, [sent, onClose]);

  const send = useCallback(async () => {
    if (!message.trim() || sending) return;
    setSending(true);
    setError(null);
    try {
      await http.post("/api/feedback", {
        message: message.trim(),
        tenantId: tenant?.tenant_id ?? null,
        tenantName: tenant?.tenant_name ?? null,
        page: pathname,
      });
      setSent(true);
    } catch (e) {
      console.error("[feedback] send failed:", e instanceof Error ? e.message : String(e));
      setError("Could not send your feedback. Please try again.");
    } finally {
      setSending(false);
    }
  }, [message, sending, tenant, pathname]);

  if (!portalTarget) return null;
  return createPortal(
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.35)", backdropFilter: "blur(4px)", WebkitBackdropFilter: "blur(4px)",
        zIndex: 1000,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
      onClick={onClose}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="feedback-modal-title"
        style={{
          background: "rgba(15,15,15,0.92)",
          backdropFilter: "blur(16px)",
          border: "1px solid rgba(255,255,255,0.1)",
          borderRadius: 12,
          padding: 20,
          width: 420,
          maxWidth: "calc(100vw - 48px)",
          boxShadow: "0 20px 60px rgba(0,0,0,0.6)",
          display: "flex",
          flexDirection: "column",
          gap: 14,
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {sent ? (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 10, padding: "22px 0 14px", textAlign: "center" }}>
            <div style={{ width: 44, height: 44, borderRadius: "50%", border: "1.5px solid rgba(34,197,94,0.5)", background: "rgba(34,197,94,0.12)", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#22C55E" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M20 6L9 17l-5-5" /></svg>
            </div>
            <span style={{ fontSize: 14, fontWeight: 700, color: "#EDECEA" }}>Thank you!</span>
            <p style={{ fontSize: 13, color: "rgba(237,236,234,0.55)", margin: 0, maxWidth: 300 }}>
              Your feedback is on its way to the team. We read every message.
            </p>
          </div>
        ) : (
          <>
            <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 12 }}>
              <div>
                <h2 id="feedback-modal-title" style={{ fontSize: 15, fontWeight: 700, color: "#EDECEA", margin: "0 0 3px" }}>
                  Give feedback
                </h2>
                <p style={{ fontSize: 12.5, color: "rgba(237,236,234,0.55)", margin: 0 }}>
                  What&rsquo;s broken, missing, or great? It goes straight to the team.
                </p>
              </div>
              <button
                onClick={onClose}
                aria-label="Close"
                className="cursor-pointer"
                style={{ background: "none", border: "none", color: "rgba(237,236,234,0.35)", fontSize: 16, lineHeight: 1, padding: 4, borderRadius: 5 }}
              >
                &#10005;
              </button>
            </div>

            <textarea
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="e.g. The session page takes forever to load when I have many sessions…"
              aria-label="Your feedback"
              autoFocus
              style={{
                width: "100%",
                boxSizing: "border-box",
                minHeight: 120,
                resize: "vertical",
                background: "rgba(0,0,0,0.45)",
                border: "1px solid rgba(255,255,255,0.12)",
                borderRadius: 8,
                padding: "10px 12px",
                fontFamily: "inherit",
                fontSize: 13,
                lineHeight: 1.55,
                color: "#EDECEA",
                outline: "none",
              }}
            />

            {error && <p style={{ fontSize: 12, color: "#EF4444", margin: 0 }}>{error}</p>}

            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <span style={{ flex: 1, fontSize: 11, color: "rgba(237,236,234,0.35)" }}>
                Sent with your account email so we can reply
              </span>
              <button
                onClick={onClose}
                className="cursor-pointer"
                style={{ background: "transparent", border: "1px solid rgba(255,255,255,0.12)", color: "rgba(237,236,234,0.55)", borderRadius: 7, padding: "8px 14px", fontSize: 13, fontWeight: 500, fontFamily: "inherit" }}
              >
                Cancel
              </button>
              <button
                onClick={send}
                disabled={!message.trim() || sending}
                className="cursor-pointer"
                style={{
                  background: "#BC9BFF",
                  color: "#1e1e1c",
                  border: "none",
                  borderRadius: 7,
                  padding: "8px 14px",
                  fontSize: 13,
                  fontWeight: 500,
                  fontFamily: "inherit",
                  minWidth: 118,
                  opacity: !message.trim() || sending ? 0.45 : 1,
                  cursor: !message.trim() || sending ? "not-allowed" : "pointer",
                }}
              >
                {sending ? "Sending…" : "Send feedback"}
              </button>
            </div>
          </>
        )}
      </div>
    </div>,
    portalTarget,
  );
}
