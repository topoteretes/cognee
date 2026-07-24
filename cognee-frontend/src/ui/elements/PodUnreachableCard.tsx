"use client";

import { trackEvent } from "@/modules/analytics";

/**
 * Terminal error state for when pod-readiness polling has genuinely given up
 * (podUnreachable), as opposed to still connecting. Scoped to the content
 * area — not a full-page takeover — so the app shell (nav, top bar) stays
 * visible, matching the rest of the dashboard's loading states.
 */
export default function PodUnreachableCard({ pageName = "Dashboard" }: { pageName?: string }) {
  return (
    <div style={{ minHeight: "100%", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center", padding: "clamp(16px, 3vw, 32px)" }}>
      <div style={{
        border: "1px solid rgba(245,158,11,0.3)",
        borderRadius: 12,
        background: "rgba(245,158,11,0.06)",
        padding: "32px 40px",
        maxWidth: 440,
        width: "100%",
        textAlign: "center",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 12,
      }}>
        <svg width="32" height="32" viewBox="0 0 16 16" fill="none">
          <path d="M8 1L1 14h14L8 1z" fill="rgba(245,158,11,0.25)" stroke="#F59E0B" strokeWidth="1" />
          <text x="8" y="12" textAnchor="middle" fontSize="9" fontWeight="700" fill="#FBBF24">!</text>
        </svg>
        <h2 style={{ margin: 0, fontSize: 18, fontWeight: 700, color: "#EDECEA" }}>
          We&apos;re having trouble reaching your workspace
        </h2>
        <p style={{ margin: 0, fontSize: 13, color: "rgba(237,236,234,0.65)", lineHeight: "20px" }}>
          This can happen during setup or a temporary hiccup. Try again, or sign out and back in.
        </p>
        <div style={{ display: "flex", gap: 10, marginTop: 8 }}>
          <button
            onClick={() => window.location.reload()}
            className="cursor-pointer"
            style={{ background: "none", border: "1px solid rgba(255,255,255,0.2)", borderRadius: 8, padding: "8px 18px", fontSize: 13, fontWeight: 500, color: "rgba(237,236,234,0.8)" }}
          >
            Try again
          </button>
          <a
            href="/api/signout"
            onClick={() => trackEvent({ pageName, eventName: "sign_out" })}
            style={{ background: "#6510F4", border: "none", borderRadius: 8, padding: "8px 18px", fontSize: 13, fontWeight: 500, color: "#fff", textDecoration: "none", display: "inline-flex", alignItems: "center" }}
          >
            Sign out
          </a>
        </div>
      </div>
    </div>
  );
}
