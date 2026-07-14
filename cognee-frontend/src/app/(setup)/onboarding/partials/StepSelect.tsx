"use client";

import { useState } from "react";
import Image from "next/image";
import { trackEvent } from "@/modules/analytics";
import { SkipLink } from "./Shared";

export type OnboardingPath = "claude-code" | "codex" | "company";

function CompanyBrainIcon() {
  // Stacked-document icon — mirrors the Company Brain card on the dashboard.
  return (
    <svg height="72" viewBox="0 0 80 100" fill="none" xmlns="http://www.w3.org/2000/svg">
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

export function StepSelect({ onSelect }: { onSelect: (path: OnboardingPath) => void }) {
  const cards: { key: OnboardingPath; name: string; description: string; logo: React.ReactNode }[] = [
    { key: "claude-code", name: "Claude Code", description: "Give Claude Code persistent memory across all your projects", logo: <Image src="/visuals/logos/claude.svg" alt="Claude Code" width={72} height={72} style={{ height: 72, width: "auto" }} /> },
    { key: "codex", name: "Codex", description: "Connect OpenAI Codex to your knowledge graph via a skill", logo: <Image src="/visuals/logos/codex.svg" alt="Codex" width={72} height={72} style={{ height: 72, width: "auto" }} /> },
    { key: "company", name: "Company Brain", description: "Upload PDFs, docs, and data to build your knowledge graph", logo: <CompanyBrainIcon /> },
  ];
  const [hovered, setHovered] = useState<OnboardingPath | null>(null);

  return (
    <div style={{
      minHeight: "100vh",
      backgroundColor: "#000000",
      backgroundImage: "linear-gradient(rgba(244,244,244,0.10) 1px, transparent 1px), linear-gradient(90deg, rgba(244,244,244,0.10) 1px, transparent 1px)",
      backgroundSize: "33px 33px",
      display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
      padding: "56px 24px", boxSizing: "border-box",
    }}>
      <div className="flex flex-col items-center gap-2" style={{ paddingBottom: 36 }}>
        <h1 style={{ fontSize: 30, fontWeight: 300, color: "#EDECEA", margin: 0, fontFamily: '"TWKLausanne", sans-serif', letterSpacing: "-0.02em" }}>How do you want to start?</h1>
        <p style={{ fontSize: 15, color: "rgba(237,236,234,0.65)", margin: 0, textAlign: "center", maxWidth: 460, lineHeight: "22px" }}>
          Connect a coding agent to your memory, or upload your own data to build a company brain.
        </p>
      </div>

      <div style={{ display: "flex", gap: 16, flexWrap: "wrap", justifyContent: "center", maxWidth: 880, width: "100%" }}>
        {cards.map((card) => {
          const active = hovered === card.key;
          return (
            <button
              key={card.key}
              onClick={() => { trackEvent({ pageName: "Onboarding", eventName: "onboarding_path_selected", additionalProperties: { path: card.key } }); onSelect(card.key); }}
              onMouseEnter={() => setHovered(card.key)}
              onMouseLeave={() => setHovered(null)}
              className="cursor-pointer"
              style={{
                flex: "1 1 240px", maxWidth: 280, minWidth: 220,
                background: active ? "rgba(188,155,255,0.12)" : "#2a2a2e",
                border: `1px solid ${active ? "rgba(188,155,255,0.45)" : "rgba(255,255,255,0.1)"}`,
                borderRadius: 16,
                padding: "28px 24px",
                display: "flex", flexDirection: "column", alignItems: "center", gap: 14,
                textAlign: "center",
                boxShadow: "0 20px 60px rgba(0,0,0,0.5)",
                transition: "background 150ms, border-color 150ms",
              }}
            >
              <div style={{ height: 72, display: "flex", alignItems: "center", justifyContent: "center" }}>{card.logo}</div>
              <div style={{ fontSize: 18, fontWeight: 400, color: "#EDECEA", fontFamily: '"TWKLausanne", sans-serif', letterSpacing: "-0.01em" }}>{card.name}</div>
              <div style={{ fontSize: 13, color: "rgba(237,236,234,0.6)", lineHeight: "19px" }}>{card.description}</div>
              <span style={{ marginTop: 4, display: "inline-flex", alignItems: "center", gap: 5, background: "#BC9BFF", color: "#1e1e1c", borderRadius: 8, padding: "8px 18px", fontSize: 13, fontWeight: 500 }}>
                {card.key === "company" ? "Upload data" : "Connect agent"} →
              </span>
            </button>
          );
        })}
      </div>

      <SkipLink />
    </div>
  );
}
