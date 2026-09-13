/**
 * Open-source stub — Memory Coverage replays every question ever asked
 * against the tenant's real brains and is a Cognee Cloud feature. Renders a
 * text-only notice instead of syncing the real page.
 */
import { AsciiFrame } from "@/app/(app)/dashboard/partials/redesign/AsciiFrame";
import { FONT, T } from "@/app/(app)/dashboard/partials/redesign/mono";

export default function Page() {
  return (
    <div style={{ minHeight: "100%", padding: "24px 32px 32px" }}>
      <AsciiFrame label="Memory Coverage" minHeight={320}>
        <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 10, textAlign: "center", padding: 20 }}>
          <span style={{ ...FONT, fontSize: 16, fontWeight: 500, color: T.text }}>
            Memory Coverage is a Cognee Cloud feature
          </span>
          <span style={{ ...FONT, fontSize: 13, color: T.muted, maxWidth: 380 }}>
            Build your own dashboard from the API, or use the hosted one in Cognee Cloud.
          </span>
          <a
            href="https://www.cognee.ai"
            target="_blank"
            rel="noopener noreferrer"
            style={{ ...FONT, marginTop: 4, background: T.lavender, color: "#000000", borderRadius: 8, padding: "8px 20px", fontSize: 13, fontWeight: 600, textDecoration: "none" }}
          >
            Open Cognee Cloud
          </a>
        </div>
      </AsciiFrame>
    </div>
  );
}
