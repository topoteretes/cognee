"use client";

import { useState } from "react";
import { trackEvent } from "@/modules/analytics";
import type { AciAgentKey } from "./agentConnectionSteps";

interface SkillCopyBlockProps {
  path: string;
  content: string;
  card?: AciAgentKey;
}

export function SkillCopyBlock({ path, content, card }: SkillCopyBlockProps): React.ReactElement {
  const [phase, setPhase] = useState<"idle" | "copying" | "done">("idle");

  function handleCopy(e: React.MouseEvent) {
    e.stopPropagation();
    trackEvent({ pageName: "Dashboard", eventName: "agent_config_copied", additionalProperties: { card: card ?? "unknown", block: "skill_install" } });
    navigator.clipboard.writeText(content);
    setPhase("copying");
    setTimeout(() => setPhase("done"), 900);
    setTimeout(() => setPhase("idle"), 3800);
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }} onClick={(e) => e.stopPropagation()}>
      {/* Destination path — purely informational. Mono grey to match InlineCodeBlock. */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, background: "var(--color-cognee-dark)", borderRadius: 8, padding: "10px 14px" }}>
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.45)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
          <polyline points="14 2 14 8 20 8" />
        </svg>
        <code style={{
          fontSize: 12,
          fontFamily: 'ui-monospace, Menlo, Monaco, "Cascadia Mono", "Segoe UI Mono", "Roboto Mono", monospace',
          color: "rgba(237,236,234,0.85)",
          flex: 1,
        }}>
          {path}
        </code>
      </div>

      {/* Primary action — solid lavender reads as the primary CTA. */}
      <button
        onClick={handleCopy}
        disabled={phase !== "idle"}
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: 7,
          background: phase === "done" ? "rgba(34,197,94,0.15)" : phase === "copying" ? "var(--color-cognee-lavender-hover)" : "var(--color-cognee-lavender)",
          border: `1px solid ${phase === "done" ? "rgba(34,197,94,0.4)" : "transparent"}`,
          borderRadius: 8,
          padding: "9px 16px",
          fontSize: 13,
          fontWeight: 500,
          cursor: phase === "idle" ? "pointer" : "default",
          color: phase === "done" ? "#22C55E" : "var(--color-cognee-lavender-text)",
          fontFamily: "inherit",
          transition: "background 200ms, border-color 200ms, color 200ms",
          width: "100%",
        }}
      >
        {phase === "idle" && (
          <>
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none">
              <rect x="5" y="5" width="8" height="8" rx="1.5" stroke="var(--color-cognee-lavender-text)" strokeWidth="1.5" />
              <path d="M11 3H4.5A1.5 1.5 0 003 4.5V11" stroke="var(--color-cognee-lavender-text)" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
            Copy install command
          </>
        )}
        {phase === "copying" && (
          <>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="var(--color-cognee-lavender-text)" strokeWidth="2.5" strokeLinecap="round" style={{ animation: "aci-spin 0.7s linear infinite" }}>
              <path d="M21 12a9 9 0 11-6.219-8.56" />
            </svg>
            Copying to clipboard…
          </>
        )}
        {phase === "done" && (
          <>
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none">
              <path d="M3 8.5L6.5 12L13 5" stroke="#22C55E" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            Copied — paste &amp; run in your local terminal
          </>
        )}
      </button>

      {/* Preview of what gets written — runs entirely on the user's machine. */}
      {phase === "done" && (
        <div style={{
          background: "var(--color-cognee-dark)",
          borderRadius: 8,
          padding: "10px 14px",
          fontFamily: 'ui-monospace, Menlo, Monaco, "Cascadia Mono", "Segoe UI Mono", "Roboto Mono", monospace',
          fontSize: 11,
          lineHeight: "18px",
        }}>
          <div style={{ color: "#585B70" }}>$ <span style={{ color: "#CDD6F4" }}>paste &amp; run the command in your terminal</span></div>
          <div style={{ color: "#A6E3A1", marginTop: 3 }}>↳ writes {path} on your local machine</div>
        </div>
      )}
    </div>
  );
}
