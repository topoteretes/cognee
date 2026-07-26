"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import Image from "next/image";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import { useUser } from "@/modules/users/UserContext";
import { completeOnboardingAndNavigate } from "../completeOnboardingAndNavigate";
import { useAgentConnectionDetection } from "../hooks/useAgentConnectionDetection";
import { useOnboardingTrackEvent } from "../useOnboardingTrackEvent";
import { buildAgentOnboardingCards } from "./agentOnboardingCards";
import { SkipLink } from "./Shared";

// Steps that show the live "waiting for connection" indicator. Both Claude
// Code and Codex share the same flow: step 3 = Upload (index 2), step 4 =
// Recall (index 3).
const UPLOAD_STEP = 2;
const CONNECT_STEP = 3;

export function AgentOnboarding({ agent, serviceUrl, apiKey, cogniInstance, onRestart }: {
  agent: "claude-code" | "codex";
  serviceUrl: string | null;
  apiKey: string;
  cogniInstance: ReturnType<typeof useCogniInstance>["cogniInstance"];
  onRestart: () => void;
}) {
  const router = useRouter();
  const { markOnboardingComplete } = useUser();
  const track = useOnboardingTrackEvent();
  const name = agent === "claude-code" ? "Claude Code" : "Codex";
  const credsReady = Boolean(serviceUrl && apiKey);
  const baseUrl = serviceUrl || "https://your-tenant.aws.cognee.ai";
  const resolvedKey = apiKey || "your-api-key";
  // API key + base url are all the plugin/skill needs.
  const credsCode = `export COGNEE_BASE_URL="${baseUrl}"\nexport COGNEE_API_KEY="${resolvedKey}"`;

  // Accordion: exactly one step expanded at a time. Clicking a later step
  // collapses the current one (which turns green/"Done") and expands the next.
  const [currentStep, setCurrentStep] = useState(0);

  const detecting = currentStep === UPLOAD_STEP || currentStep === CONNECT_STEP;
  const connectVerified = useAgentConnectionDetection(cogniInstance, detecting);

  function goToDashboard(): void {
    track({ pageName: "Onboarding", eventName: "onboarding_completed", additionalProperties: { destination: "dashboard", path: agent } });
    completeOnboardingAndNavigate(markOnboardingComplete, () => router.push("/dashboard"));
  }

  const cards = buildAgentOnboardingCards({ agent, name, baseUrl, credsCode, credsReady, connectVerified, goToDashboard });

  useEffect(() => {
    track({ pageName: "Onboarding", eventName: "agent_step_viewed", additionalProperties: { agent, step: String(currentStep), title: cards[currentStep]?.title ?? "" } });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentStep, agent]);

  // Fires exactly once per mount, the moment the live-detection hook flips —
  // this is the only real proof the agent actually connected during onboarding.
  const connectedTracked = useRef(false);
  useEffect(() => {
    if (!connectVerified || connectedTracked.current) return;
    connectedTracked.current = true;
    track({ pageName: "Onboarding", eventName: "agent_connected", additionalProperties: { agent, step: String(currentStep) } });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connectVerified, agent]);

  return (
    <div style={{
      minHeight: "100vh",
      backgroundColor: "#000000",
      backgroundImage: "linear-gradient(rgba(244,244,244,0.10) 1px, transparent 1px), linear-gradient(90deg, rgba(244,244,244,0.10) 1px, transparent 1px)",
      backgroundSize: "33px 33px",
      display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
      padding: "56px 24px", boxSizing: "border-box",
    }}>
      <div style={{
        background: "#2a2a2e",
        border: "1px solid rgba(255,255,255,0.1)",
        borderRadius: 16,
        padding: "40px 48px",
        boxShadow: "0 20px 60px rgba(0,0,0,0.5)",
        maxWidth: 620, width: "100%", boxSizing: "border-box",
        display: "flex", flexDirection: "column", alignItems: "center", gap: 8,
      }}>
        <div className="flex items-center gap-2" style={{ marginBottom: 4 }}>
          <Image src={agent === "claude-code" ? "/visuals/logos/claude.svg" : "/visuals/logos/codex.svg"} alt={name} width={28} height={28} style={{ width: 28, height: 28, objectFit: "contain" }} />
          <h1 style={{ fontSize: 26, fontWeight: 300, color: "#EDECEA", margin: 0, fontFamily: '"TWKLausanne", sans-serif', letterSpacing: "-0.02em" }}>Connect {name}</h1>
        </div>
        <p style={{ fontSize: 14, color: "rgba(237,236,234,0.6)", margin: "0 0 12px", textAlign: "center" }}>
          A few quick steps to give {name} persistent memory.
        </p>

        <style>{`
          @keyframes ob-check { 0% { transform: scale(0.4); opacity: 0; } 100% { transform: scale(1); opacity: 1; } }
          @keyframes ob-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
          .ob-pulse { animation: ob-pulse 1.4s ease-in-out infinite; }
        `}</style>
        {/* Reserve a constant height for the steps so the buttons below never
            shift when switching steps (expanded bodies differ in height). The
            leftover space below the box acts as a cushion before the buttons. */}
        <div style={{ width: "100%", minHeight: 490, display: "flex", flexDirection: "column" }}>
        <div style={{ width: "100%", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 12, overflow: "hidden", background: "rgba(255,255,255,0.04)" }}>
          {cards.map((card, i) => {
            const isActive = currentStep === i;
            const isDone = i < currentStep;
            const isLast = i === cards.length - 1;
            return (
              <div
                key={i}
                onClick={() => setCurrentStep(i)}
                style={{ borderBottom: !isLast ? "1px solid rgba(255,255,255,0.07)" : "none", cursor: isActive ? "default" : "pointer" }}
              >
                {/* Row header — always visible */}
                <div style={{ display: "flex", alignItems: "center", gap: 12, padding: isActive ? "14px 18px 0" : "14px 18px" }}>
                  <div style={{
                    width: 26, height: 26, borderRadius: "50%", flexShrink: 0,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    background: isDone ? "rgba(34,197,94,0.18)" : isActive ? "#BC9BFF" : "rgba(255,255,255,0.1)",
                    transition: "background 200ms ease",
                  }}>
                    {isDone ? (
                      <svg width="11" height="11" viewBox="0 0 16 16" fill="none" style={{ animation: "ob-check 220ms cubic-bezier(0.22,1,0.36,1) forwards" }}>
                        <path d="M3 8.5L6.5 12L13 5" stroke="#22C55E" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    ) : (
                      <span style={{ fontSize: 12, fontWeight: 700, color: isActive ? "#1e1e1c" : "rgba(237,236,234,0.6)", lineHeight: 1 }}>{i + 1}</span>
                    )}
                  </div>
                  <span style={{ flex: 1, fontSize: 14, fontWeight: isActive ? 500 : 400, color: isDone ? "rgba(237,236,234,0.45)" : isActive ? "#EDECEA" : "rgba(237,236,234,0.35)" }}>
                    {card.title}
                  </span>
                  {isDone && (
                    <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase", background: "rgba(34,197,94,0.18)", color: "#22C55E", borderRadius: 100, padding: "2px 8px", flexShrink: 0, animation: "ob-check 200ms ease forwards" }}>
                      Done
                    </span>
                  )}
                </div>
                {/* Expanded body — animates open/closed; only the active step shows it */}
                <div style={{ display: "grid", gridTemplateRows: isActive ? "1fr" : "0fr", opacity: isActive ? 1 : 0, transition: "grid-template-rows 260ms ease, opacity 200ms ease" }}>
                  <div style={{ overflow: "hidden" }}>
                    <div onClick={(e) => e.stopPropagation()} style={{ padding: "8px 18px 18px 50px", display: "flex", flexDirection: "column", gap: 10 }}>
                      <p style={{ fontSize: 13, color: "rgba(237,236,234,0.6)", lineHeight: "19px", margin: 0 }}>{card.description}</p>
                      {card.node}
                      {!isLast && (
                        <p style={{ margin: "2px 0 0", fontSize: 12, color: "rgba(237,236,234,0.5)" }}>Click step {i + 2} when ready ↓</p>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4, marginTop: 8, width: "100%" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <button
              onClick={onRestart}
              className="cursor-pointer"
              style={{ display: "inline-flex", alignItems: "center", gap: 7, background: "transparent", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 8, padding: "9px 18px", fontSize: 13, fontWeight: 500, color: "rgba(237,236,234,0.65)", fontFamily: "inherit", cursor: "pointer" }}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M1 4v6h6" /><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" /></svg>
              Start over
            </button>
            {currentStep < cards.length - 1 && (
              <button
                onClick={() => setCurrentStep((s) => Math.min(s + 1, cards.length - 1))}
                className="cursor-pointer"
                style={{ display: "inline-flex", alignItems: "center", gap: 7, background: "#BC9BFF", border: "none", borderRadius: 8, padding: "9px 18px", fontSize: 13, fontWeight: 500, color: "#1e1e1c", fontFamily: "inherit", cursor: "pointer" }}
              >
                Next
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="5" y1="12" x2="19" y2="12" /><polyline points="12 5 19 12 12 19" /></svg>
              </button>
            )}
          </div>
          <SkipLink label="Skip onboarding" compact />
        </div>
      </div>
    </div>
  );
}
