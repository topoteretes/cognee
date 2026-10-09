"use client";

import React from "react";
import { FONT, SANS, T } from "./redesign/mono";
import type { AciAgentKey, AciCardConfig } from "./agentConnectionSteps";

// Brand mark tucked in the card's bottom-right corner (contained, not bleeding).
const LOGO_H = 64;

interface AciCardProps {
  card: AciCardConfig;
  activeKey: AciAgentKey | null;
  isUploading: boolean;
  hasDocuments: boolean;
  integrationConnected: Record<string, boolean>;
  onCardClick: (key: AciAgentKey) => void;
}

export function AciCard({
  card,
  activeKey,
  isUploading,
  hasDocuments,
  integrationConnected,
  onCardClick,
}: AciCardProps): React.ReactElement {
  const connected = card.key === "upload" ? hasDocuments : !!integrationConnected[card.key];
  const isActive = activeKey === card.key;
  const isUpload = card.key === "upload";

  const logoNode = buildLogoNode(card.key, card.name);
  const ctaLabel = isUpload ? (connected ? "Add more data" : "Upload data") : "Connect";

  return (
    <button
      className="aci-card"
      onClick={() => onCardClick(card.key)}
      aria-haspopup={!isUpload ? "dialog" : undefined}
      disabled={isUploading && isUpload}
      style={{
        position: "relative",
        flex: "0 0 auto",
        width: card.width,
        background: isActive ? T.purpleSoft : T.chrome,
        border: `1px solid ${isActive ? T.lavender : T.frame}`,
        borderRadius: 0,
        padding: 14,
        height: 104,
        overflow: "hidden",
        cursor: isUploading && isUpload ? "wait" : "pointer",
        textAlign: "left",
        display: "flex",
        flexDirection: "column",
        transition: "border-color 150ms, background 150ms",
      }}
    >
      {/* 1. Title + status on one line */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
        <span style={{ fontFamily: SANS, fontSize: 14, fontWeight: 600, color: T.text, letterSpacing: "-0.01em", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", minWidth: 0 }}>
          {card.name}
        </span>
        {connected && (
          <span style={{ display: "inline-flex", alignItems: "center", gap: 5, flexShrink: 0 }}>
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: T.green }} />
            <span style={{ ...FONT, fontSize: 11, fontWeight: 500, color: T.green }}>Connected</span>
          </span>
        )}
      </div>

      {/* 2. Description */}
      <span style={{ ...FONT, fontSize: 11.5, color: T.muted, lineHeight: 1.35, marginTop: 3, paddingRight: 78, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
        {card.description}
      </span>

      {/* 3a. Button, bottom-left */}
      <div style={{ marginTop: "auto" }}>
        {isUploading && isUpload ? (
          <div style={{ display: "flex", alignItems: "center", gap: 6, height: 34 }}>
            <div style={{ width: 10, height: 10, borderRadius: "50%", border: `1.5px solid ${T.frameStrong}`, borderTopColor: T.lavender, animation: "aci-spin 0.8s linear infinite" }} />
            <span style={{ ...FONT, fontSize: 12, fontWeight: 500, color: T.muted }}>Uploading…</span>
          </div>
        ) : (
          <span className="aci-cta-chip" style={{ ...FONT, display: "inline-flex", alignItems: "center", justifyContent: "center", height: 34, minWidth: 96, padding: "0 14px", background: "transparent", border: `1px solid ${T.frameStrong}`, borderRadius: 0, fontSize: 12.5, fontWeight: 500, color: T.text, whiteSpace: "nowrap", transition: "background 150ms, border-color 150ms, color 150ms" }}>
            {ctaLabel}
          </span>
        )}
      </div>

      {/* 3b. Illustration, bottom-right */}
      <div className="aci-card-logo" style={{ position: "absolute", bottom: 12, right: 12, display: "flex", alignItems: "flex-end", pointerEvents: "none" }}>
        {logoNode}
      </div>
    </button>
  );
}

function buildLogoNode(key: AciAgentKey, name: string): React.ReactElement {
  if (key === "upload") {
    return (
      <svg height={LOGO_H} viewBox="0 0 80 100" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect x="16" y="6" width="54" height="70" rx="6" fill="#D4D4D8" stroke="#71717A" strokeWidth="3.5" />
        <rect x="8" y="14" width="54" height="70" rx="6" fill="#E4E4E7" stroke="#71717A" strokeWidth="3.5" />
        <rect x="2" y="22" width="54" height="70" rx="6" fill="#F4F4F5" stroke="#52525B" strokeWidth="3.5" />
        <path d="M38 22v16h18" stroke="#52525B" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round" />
        <line x1="12" y1="52" x2="46" y2="52" stroke="#52525B" strokeWidth="3" strokeLinecap="round" />
        <line x1="12" y1="63" x2="46" y2="63" stroke="#52525B" strokeWidth="3" strokeLinecap="round" />
        <line x1="12" y1="74" x2="30" y2="74" stroke="#52525B" strokeWidth="3" strokeLinecap="round" />
      </svg>
    );
  }
  if (key === "api-mcp") {
    return (
      <svg height={LOGO_H} viewBox="0 0 90 110" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect x="5" y="20" width="80" height="50" rx="10" fill="#1a1a2e" stroke="rgba(255,255,255,0.15)" strokeWidth="2" />
        <path d="M25 35L16 45L25 55" stroke="var(--color-cognee-lavender-tint-60)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M65 35L74 45L65 55" stroke="var(--color-cognee-lavender-tint-60)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
        <line x1="50" y1="30" x2="40" y2="60" stroke="rgba(255,255,255,0.4)" strokeWidth="2.5" strokeLinecap="round" />
      </svg>
    );
  }
  const src = key === "claude-code" ? "/visuals/logos/claude.svg" : key === "codex" ? "/visuals/logos/codex.svg" : "/visuals/logos/openclaw.svg";
  return <img src={src} alt={name} style={{ height: LOGO_H, width: "auto", maxWidth: 72, objectFit: "contain" }} />;
}
