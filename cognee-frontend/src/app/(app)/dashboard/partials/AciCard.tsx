"use client";

import React from "react";
import type { AciAgentKey, AciCardConfig } from "./agentConnectionSteps";

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
  const logoRight = isUpload ? -12 : card.key === "api-mcp" ? -28 : -36;
  const ctaLabel = isUpload
    ? (connected ? "Add more data" : "Upload data")
    : card.key === "api-mcp" ? "Connect" : "Connect agent";

  return (
    <button
      className="aci-card"
      onClick={() => onCardClick(card.key)}
      aria-haspopup={!isUpload ? "dialog" : undefined}
      disabled={isUploading && isUpload}
      style={{
        position: "relative",
        background: isActive ? "var(--color-cognee-lavender-tint-20)" : "rgba(255,255,255,0.06)",
        backdropFilter: "blur(12px)",
        border: `1px solid ${isActive ? "var(--color-cognee-lavender-tint-35)" : "rgba(255,255,255,0.1)"}`,
        borderRadius: 12,
        padding: "20px 16px 0 16px",
        height: 160,
        overflow: "hidden",
        cursor: isUploading && isUpload ? "wait" : "pointer",
        textAlign: "left",
        display: "flex",
        flexDirection: "column",
        transition: "border-color 150ms, background 150ms",
      }}
    >
      {connected && (
        <div style={{ position: "absolute", top: 12, right: 12, display: "flex", alignItems: "center", gap: 4, zIndex: 1 }}>
          <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#22C55E", flexShrink: 0 }} />
          <span style={{ fontSize: 11, fontWeight: 500, color: "#16A34A", whiteSpace: "nowrap" }}>Connected</span>
        </div>
      )}
      <span style={{ fontSize: 16, fontWeight: 300, color: "#EDECEA", lineHeight: 1.25, letterSpacing: "-0.01em", fontFamily: '"TWKLausanne", sans-serif', paddingRight: connected ? 90 : 16 }}>
        {card.name}
      </span>
      <div style={{ position: "absolute", bottom: 14, left: 16, zIndex: 1 }}>
        {isUploading && isUpload ? (
          <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <div style={{ width: 9, height: 9, borderRadius: "50%", border: "1.5px solid #D1D5DB", borderTopColor: "var(--color-cognee-purple)", animation: "aci-spin 0.8s linear infinite", flexShrink: 0 }} />
            <span style={{ fontSize: 11, fontWeight: 500, color: "rgba(237,236,234,0.65)" }}>Uploading…</span>
          </div>
        ) : (
          <span className="aci-cta-chip" style={{ display: "inline-flex", alignItems: "center", gap: 4, background: "rgba(20,20,22,0.92)", backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)", border: "1px solid rgba(237,236,234,0.65)", borderRadius: 6, padding: "5px 10px", fontSize: 12, fontWeight: 500, color: "rgba(237,236,234,0.65)", whiteSpace: "nowrap", transition: "background 150ms" }}>
            {ctaLabel}
          </span>
        )}
      </div>
      <div className="aci-card-logo" style={{ position: "absolute", bottom: -18, right: logoRight, pointerEvents: "none" }}>
        {logoNode}
      </div>
    </button>
  );
}

function buildLogoNode(key: AciAgentKey, name: string): React.ReactElement {
  if (key === "upload") {
    return (
      <svg height="110" viewBox="0 0 80 100" fill="none" xmlns="http://www.w3.org/2000/svg">
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
      <svg height="110" viewBox="0 0 90 110" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect x="5" y="20" width="80" height="50" rx="10" fill="#1a1a2e" stroke="rgba(255,255,255,0.15)" strokeWidth="2" />
        <path d="M25 35L16 45L25 55" stroke="var(--color-cognee-lavender-tint-60)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M65 35L74 45L65 55" stroke="var(--color-cognee-lavender-tint-60)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
        <line x1="50" y1="30" x2="40" y2="60" stroke="rgba(255,255,255,0.4)" strokeWidth="2.5" strokeLinecap="round" />
      </svg>
    );
  }
  const src = key === "claude-code" ? "/visuals/logos/claude.svg" : key === "codex" ? "/visuals/logos/codex.svg" : "/visuals/logos/openclaw.svg";
  return <img src={src} alt={name} style={{ height: 110, width: "auto" }} />;
}
