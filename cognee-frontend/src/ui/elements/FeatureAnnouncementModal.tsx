"use client";

import { useRouter } from "next/navigation";
import type { FeatureAnnouncementContent } from "@/modules/featureAnnouncements/featureAnnouncementContent";

interface FeatureAnnouncementModalProps {
  content: FeatureAnnouncementContent;
  onDismiss: () => void;
}

const CLOSE_ICON = (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
    <line x1="18" y1="6" x2="6" y2="18" />
    <line x1="6" y1="6" x2="18" y2="18" />
  </svg>
);

// Split-panel "what's new" modal — illustration on one side, copy + CTA on
// the other, in the style of a release announcement rather than a form.
// Mounted once by FeatureAnnouncementsProvider; content is looked up by
// feature_key from FEATURE_ANNOUNCEMENT_CONTENT.
export default function FeatureAnnouncementModal({
  content,
  onDismiss,
}: FeatureAnnouncementModalProps): React.ReactElement {
  const router = useRouter();
  const Illustration = content.illustration;

  function handleCta(): void {
    onDismiss();
    router.push(content.ctaHref);
  }

  return (
    <div
      style={{
        position: "fixed", inset: 0, background: "rgba(0,0,0,0.5)",
        backdropFilter: "blur(4px)", WebkitBackdropFilter: "blur(4px)",
        zIndex: 1000, display: "flex", alignItems: "center", justifyContent: "center",
        padding: 16,
      }}
      onClick={onDismiss}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: "rgba(15,15,15,0.96)", backdropFilter: "blur(16px)",
          border: "1px solid rgba(255,255,255,0.1)", borderRadius: 16,
          width: 640, maxWidth: "100%", display: "flex",
          boxShadow: "0 24px 64px rgba(0,0,0,0.5)", overflow: "hidden",
        }}
      >
        <div
          style={{
            flex: "0 0 42%", background: "rgba(101,16,244,0.12)",
            display: "flex", alignItems: "center", justifyContent: "center", padding: 32,
          }}
        >
          <div style={{ width: "100%", maxWidth: 220 }}>
            <Illustration />
          </div>
        </div>

        <div style={{ flex: 1, padding: 32, display: "flex", flexDirection: "column", gap: 16, position: "relative" }}>
          <button
            onClick={onDismiss}
            aria-label="Dismiss"
            style={{
              position: "absolute", top: 16, right: 16, background: "none", border: "none",
              cursor: "pointer", padding: 4, color: "rgba(237,236,234,0.4)", lineHeight: 1,
            }}
          >
            {CLOSE_ICON}
          </button>

          <h2 style={{ fontSize: 20, fontWeight: 700, color: "#EDECEA", margin: 0, paddingRight: 24 }}>
            {content.title}
          </h2>
          <p style={{ fontSize: 14, color: "rgba(237,236,234,0.65)", margin: 0, lineHeight: 1.6 }}>
            {content.description}
          </p>

          <div style={{ marginTop: "auto", display: "flex", gap: 10 }}>
            <button
              onClick={handleCta}
              style={{
                background: "#6510F4", border: "none", borderRadius: 8, padding: "10px 20px",
                fontSize: 13, fontWeight: 500, color: "#fff", cursor: "pointer",
              }}
            >
              {content.ctaLabel}
            </button>
            <button
              onClick={onDismiss}
              style={{
                background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: 8, padding: "10px 20px", fontSize: 13, fontWeight: 500,
                color: "#EDECEA", cursor: "pointer",
              }}
            >
              Maybe later
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
