"use client";

import { useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import Image from "next/image";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import { useUser } from "@/modules/users/UserContext";
import { OsToggle } from "@/ui/elements/OsToggle";
import { useOsPreference } from "@/ui/layout/OsPreferenceContext";
import { TerminalBlock } from "@/ui/elements/AgentSetupBlocks";
import { agentSetupSteps, setupIntro } from "@/modules/integrations/agentSetupSteps";
import { completeOnboardingAndNavigate } from "../completeOnboardingAndNavigate";
import { useAgentConnectionDetection } from "../hooks/useAgentConnectionDetection";
import { useOnboardingTrackEvent } from "../useOnboardingTrackEvent";

function Action({ index, label, description, children }: { index: number; label: string; description?: string; children?: React.ReactNode }) {
  return (
    <div style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
      <div style={{
        width: 22, height: 22, borderRadius: "50%", flexShrink: 0, marginTop: 1,
        display: "flex", alignItems: "center", justifyContent: "center",
        background: "rgba(188,155,255,0.18)", border: "1px solid rgba(188,155,255,0.35)",
      }}>
        <span style={{ fontSize: 11, fontWeight: 700, color: "#BC9BFF", lineHeight: 1 }}>{index}</span>
      </div>
      <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 8 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          <span style={{ fontSize: 15.5, fontWeight: 500, color: "#EDECEA", lineHeight: "22px" }}>{label}</span>
          {description && (
            <span style={{ fontSize: 14, color: "rgba(237,236,234,0.55)", lineHeight: "20px" }}>{description}</span>
          )}
        </div>
        {children}
      </div>
    </div>
  );
}

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
  const { os } = useOsPreference();
  const name = agent === "claude-code" ? "Claude Code" : "Codex";
  const credsReady = Boolean(serviceUrl && apiKey);
  const baseUrl = serviceUrl || "https://your-tenant.aws.cognee.ai";
  const resolvedKey = apiKey || "your-api-key";
  const steps = agentSetupSteps(agent, { os, baseUrl, apiKey: resolvedKey, loading: !credsReady });

  // Still polled, purely to keep the `agent_connected` analytics signal. It no
  // longer drives any UI: there is no connection indicator, and gating Continue
  // on it dead-ended the page, because the plugin's startup only registers an
  // agent connection and never writes the session row this hook looks for.
  const connectVerified = useAgentConnectionDetection(cogniInstance, true);

  function finish(): void {
    track({ pageName: "Onboarding", eventName: "onboarding_completed", additionalProperties: { destination: "dashboard", path: agent } });
    completeOnboardingAndNavigate(markOnboardingComplete, () => router.push("/dashboard"));
  }

  function setUpLater(): void {
    track({ pageName: "Onboarding", eventName: "onboarding_skipped" });
    completeOnboardingAndNavigate(markOnboardingComplete, () => router.push("/dashboard"));
  }

  useEffect(() => {
    track({ pageName: "Onboarding", eventName: "agent_step_viewed", additionalProperties: { agent, step: "0", title: name } });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agent]);

  // Analytics only. It deliberately no longer auto-advances: with the status
  // strip gone there would be nothing on screen to explain why the page
  // suddenly navigated away mid-read.
  const connectedTracked = useRef(false);
  useEffect(() => {
    if (!connectVerified || connectedTracked.current) return;
    connectedTracked.current = true;
    track({ pageName: "Onboarding", eventName: "agent_connected", additionalProperties: { agent, step: "0" } });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connectVerified, agent]);

  return (
    <div style={{
      minHeight: "100vh",
      backgroundColor: "#000000",
      backgroundImage: "linear-gradient(rgba(244,244,244,0.10) 1px, transparent 1px), linear-gradient(90deg, rgba(244,244,244,0.10) 1px, transparent 1px)",
      backgroundSize: "24px 24px",
      display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
      padding: "40px 24px", boxSizing: "border-box",
    }}>
      <div style={{
        background: "#2a2a2e",
        border: "1px solid rgba(255,255,255,0.1)",
        boxShadow: "0 20px 60px rgba(0,0,0,0.5)",
        maxWidth: 620, width: "100%", boxSizing: "border-box",
        display: "flex", flexDirection: "column",
      }}>
        {/* Header — the one sentence that matters: where these commands go. */}
        <div style={{ padding: "26px 30px 18px", borderBottom: "1px solid rgba(255,255,255,0.07)" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <Image src={agent === "claude-code" ? "/visuals/logos/claude.svg" : "/visuals/logos/codex.svg"} alt={name} width={26} height={26} style={{ width: 26, height: 26, objectFit: "contain" }} />
              <h1 style={{ fontSize: 24, fontWeight: 300, color: "#EDECEA", margin: 0, fontFamily: '"TWKLausanne", sans-serif', letterSpacing: "-0.02em" }}>
                Connect {name}
              </h1>
            </div>
            <OsToggle />
          </div>
          <p style={{ fontSize: 15, color: "rgba(237,236,234,0.65)", margin: "10px 0 0", lineHeight: "22px" }}>
            {setupIntro(agent, os)}
          </p>
        </div>

        {/* The three actions */}
        <div style={{ padding: "22px 30px 24px", display: "flex", flexDirection: "column", gap: 20 }}>
          {/* Rendered straight from the shared step definitions, so the
              dashboard panel and integrations wizard cannot say anything
              different from what onboarding says. */}
          {steps.map((step, i) => (
            <Action key={step.title} index={i + 1} label={step.title} description={step.description || undefined}>
              {step.lines && (
                <TerminalBlock
                  lines={step.lines}
                  loading={step.loading}
                  placeholder="Preparing your credentials…"
                  copyLabel={step.copyLabel}
                  onCopied={() => track({
                    pageName: "Onboarding",
                    eventName: "onboarding_creds_copied",
                    additionalProperties: { copy_target: i === 1 ? "setup_block" : "start_command", agent },
                  })}
                />
              )}
              {step.content}
            </Action>
          ))}
        </div>

        {/* Footer */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, padding: "14px 30px", borderTop: "1px solid rgba(255,255,255,0.07)" }}>
          <button
            onClick={onRestart}
            className="cursor-pointer"
            style={{ display: "inline-flex", alignItems: "center", gap: 7, background: "transparent", border: "1px solid rgba(255,255,255,0.12)", padding: "9px 16px", fontSize: 13, fontWeight: 500, color: "rgba(237,236,234,0.65)", fontFamily: "inherit" }}
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="19" y1="12" x2="5" y2="12" /><polyline points="12 19 5 12 12 5" /></svg>
            Back
          </button>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <button
              onClick={setUpLater}
              className="cursor-pointer"
              style={{ background: "none", border: "none", color: "rgba(237,236,234,0.5)", fontSize: 13, fontFamily: "inherit", padding: "9px 6px" }}
            >
              Set up later
            </button>
            {/* Unconditionally enabled. It used to be gated on connection
                detection, which never fires for this flow — with the status
                strip gone there would also be nothing explaining the block. */}
            <button
              onClick={finish}
              className="cursor-pointer"
              style={{
                display: "inline-flex", alignItems: "center", gap: 7,
                background: "#BC9BFF", border: "none", padding: "9px 20px",
                fontSize: 13, fontWeight: 500, color: "#1e1e1c", fontFamily: "inherit",
              }}
            >
              Continue
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="5" y1="12" x2="19" y2="12" /><polyline points="12 5 19 12 12 19" /></svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
