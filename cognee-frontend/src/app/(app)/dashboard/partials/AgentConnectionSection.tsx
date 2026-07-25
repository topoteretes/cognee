"use client";

import React, { useState, useRef, useEffect } from "react";
import { useRouter } from "next/navigation";
import { trackEvent } from "@/modules/analytics";
import { SEARCH_SESSION_PREFIX } from "@/modules/sessions/getSessions";
import type { SessionRow } from "@/modules/sessions/getSessions";
import Modal from "@/ui/elements/Modal/Modal";
import { CARDS_CFG, getSteps } from "./agentConnectionSteps";
import type { AciAgentKey } from "./agentConnectionSteps";
import { AciCard } from "./AciCard";
import { AciStepRow } from "./AciStepRow";

// Index of the "Upload something" step in the claude-code flow — used to
// auto-advance when a new session is detected while the modal is open.
const CLAUDE_CONNECT_STEP = 2;

interface AgentConnectionSectionProps {
  onUploadClick: () => void;
  isUploading: boolean;
  serviceUrl: string | null;
  apiKey: string | null;
  isInitializing: boolean;
  hasDocuments: boolean;
  sessions: SessionRow[];
  integrationConnected?: Record<string, boolean>;
}

export function AgentConnectionSection({
  onUploadClick,
  isUploading,
  serviceUrl,
  apiKey,
  isInitializing,
  hasDocuments,
  sessions,
  integrationConnected = {},
}: AgentConnectionSectionProps): React.ReactElement {
  const router = useRouter();
  const [activeKey, setActiveKey] = useState<AciAgentKey | null>(null);
  const [stepIndexMap, setStepIndexMap] = useState<Partial<Record<AciAgentKey, number>>>({});
  const [connectVerified, setConnectVerified] = useState(false);
  const sessionBaselineRef = useRef<Set<string> | null>(null);

  const baseUrl = serviceUrl ?? "https://your-tenant.aws.cognee.ai";
  const resolvedKey = apiKey ?? "your-api-key";
  const credsCode = `export COGNEE_BASE_URL="${baseUrl}"\nexport COGNEE_API_KEY="${resolvedKey}"`;
  const stepOpts = { baseUrl, resolvedKey, credsCode, isInitializing, connectVerified };

  // Detect a new Claude Code session while the modal is open. Derives connection
  // state from the `sessions` prop (circuit-breaker-protected, 15s poll) —
  // no rogue setInterval bypassing the breaker.
  useEffect(() => {
    if (activeKey !== "claude-code") {
      sessionBaselineRef.current = null;
      setConnectVerified(false);
      return;
    }
    if (connectVerified) return;

    const realIds = sessions
      .map((s) => s.session_id)
      .filter((id) => !id.startsWith(SEARCH_SESSION_PREFIX));

    if (sessionBaselineRef.current === null) {
      // First render with modal open — snapshot the pre-existing session set.
      sessionBaselineRef.current = new Set(realIds);
      return;
    }

    const baseline = sessionBaselineRef.current;
    if (realIds.some((id) => !baseline.has(id))) {
      setConnectVerified(true);
      setStepIndexMap((prev) => {
        const cur = prev["claude-code"] ?? 0;
        return cur <= CLAUDE_CONNECT_STEP ? { ...prev, "claude-code": CLAUDE_CONNECT_STEP + 1 } : prev;
      });
    }
  }, [sessions, activeKey, connectVerified]);

  function handleCardClick(key: AciAgentKey) {
    trackEvent({ pageName: "Dashboard", eventName: "agent_card_clicked", additionalProperties: { card: key } });
    if (key === "upload") { onUploadClick(); return; }
    setActiveKey(key);
    if (stepIndexMap[key] === undefined) {
      setStepIndexMap((s) => ({ ...s, [key]: 0 }));
    }
  }

  function goToStep(key: AciAgentKey, idx: number) {
    trackEvent({ pageName: "Dashboard", eventName: "agent_step_viewed", additionalProperties: { card: key, step: String(idx) } });
    setStepIndexMap((prev) => ({ ...prev, [key]: idx }));
  }

  const popupOpen = activeKey !== null && activeKey !== "upload";
  const activeCfg = CARDS_CFG.find((c) => c.key === activeKey);
  const activeSteps = activeKey ? getSteps(activeKey, stepOpts) : [];
  const currentStep = activeKey ? (stepIndexMap[activeKey] ?? 0) : 0;

  const logoSrc =
    activeKey === "claude-code"
      ? "/visuals/logos/claude.svg"
      : activeKey === "codex"
      ? "/visuals/logos/codex.svg"
      : "/visuals/logos/openclaw.svg";

  const popupContent =
    popupOpen && activeCfg && activeKey ? (
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="aci-popup-title"
        onClick={(e) => e.stopPropagation()}
        className="aci-popup"
        style={{
          background: "rgba(15,15,15,0.92)",
          backdropFilter: "blur(16px)",
          borderRadius: 14,
          width: 520,
          maxWidth: "calc(100vw - 32px)",
          boxShadow: "0 20px 60px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.1)",
          overflow: "hidden",
          animation: "aci-popup 200ms cubic-bezier(0.22,1,0.36,1) forwards",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "16px 20px", borderBottom: "1px solid rgba(255,255,255,0.1)" }}>
          {activeKey === "api-mcp" ? (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" style={{ flexShrink: 0 }}>
              <rect x="3" y="6" width="18" height="12" rx="2" stroke="rgba(237,236,234,0.7)" strokeWidth="1.5" />
              <path d="M7 9L4 12L7 15" stroke="var(--color-cognee-lavender-tint-60)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M17 9L20 12L17 15" stroke="var(--color-cognee-lavender-tint-60)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              <line x1="13" y1="8" x2="11" y2="16" stroke="rgba(237,236,234,0.5)" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          ) : (
            <img src={logoSrc} alt={activeCfg.name} style={{ width: 24, height: 24, objectFit: "contain", flexShrink: 0 }} />
          )}
          <div style={{ flex: 1, minWidth: 0 }}>
            <div id="aci-popup-title" style={{ fontSize: 15, fontWeight: 700, color: "#EDECEA", lineHeight: "20px" }}>Connect {activeCfg.name}</div>
            <div style={{ fontSize: 12, color: "rgba(237,236,234,0.45)", marginTop: 1 }}>Step {currentStep + 1} of {activeSteps.length}</div>
          </div>
          <button
            onClick={() => setActiveKey(null)}
            aria-label="Close"
            style={{ background: "none", border: "none", color: "rgba(237,236,234,0.65)", cursor: "pointer", padding: 4, borderRadius: 6, lineHeight: 1, flexShrink: 0 }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>
        {activeSteps.map((step, i) => (
          <AciStepRow
            key={i}
            step={step}
            index={i}
            total={activeSteps.length}
            isActive={currentStep === i}
            isDone={i < currentStep}
            card={activeKey}
            onClick={() => goToStep(activeKey, i)}
            onNavigate={(path) => router.push(path)}
          />
        ))}
      </div>
    ) : null;

  return (
    <div>
      <style>{`
        @keyframes aci-check  { 0% { transform: scale(0.4); opacity: 0; } 100% { transform: scale(1); opacity: 1; } }
        @keyframes aci-spin   { to { transform: rotate(360deg); } }
        @keyframes aci-popup  { 0% { opacity: 0; transform: scale(0.97) translateY(6px); } 100% { opacity: 1; transform: scale(1) translateY(0); } }
        .aci-card-logo { transition: transform 300ms ease; }
        .aci-card:hover .aci-card-logo { transform: scale(1.15); }
        .aci-card:hover .aci-cta-chip { background: rgba(101,16,244,0.85) !important; }
        .aci-step-row:hover { background: rgba(255,255,255,0.04); }
        .aci-step-row[data-active="true"]:hover { background: transparent; }
        .aci-card-grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 16px; }
        @media (max-width: 1100px) { .aci-card-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
        @media (max-width: 800px)  { .aci-card-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
        @media (max-width: 480px)  { .aci-card-grid { grid-template-columns: 1fr; } }
        @media (prefers-reduced-motion: reduce) {
          .aci-card-logo, .aci-step-body, .aci-popup { transition: none !important; animation: none !important; }
        }
      `}</style>

      <div className="aci-card-grid">
        {CARDS_CFG.map((card) => (
          <AciCard
            key={card.key}
            card={card}
            activeKey={activeKey}
            isUploading={isUploading}
            hasDocuments={hasDocuments}
            integrationConnected={integrationConnected}
            onCardClick={handleCardClick}
          />
        ))}
      </div>

      <Modal isOpen={popupOpen} onClose={() => setActiveKey(null)}>
        {popupContent}
      </Modal>
    </div>
  );
}
