"use client";

import { useState, useEffect, useRef } from "react";
import { useCogniInstance, useTenant } from "@/modules/tenant/TenantProvider";
import TetrisBackground from "@/ui/elements/Auth/TetrisBackground";
import { GraphBuildingAnimation } from "./GraphBuildingAnimation";
import { useOnboardingTrackEvent } from "../useOnboardingTrackEvent";

// Bridges the gap between "Get started" and a live cogniInstance so path
// selection and Step 2's own pod-readiness bar never race a null connection.
// Step 2 already visualizes the remaining wait for a fully warm pod — this
// only waits for the API key + tenant record to exist.

// A pending workspace checklist item is shown as a spinner, a done one as a
// checkmark — cogniInstance/apiKey resolve together almost immediately, but
// tenantReady is the real wait: it only flips once waitForPodReady confirms
// the pod actually answers AND the default dataset has been created (see
// waitForPodReady.ts), not just once a cogniInstance object exists — that
// earlier, too-early gate was releasing users into path selection well
// before the backend had actually finished, which is what caused the 401s
// and failed requests seen on the dashboard right after onboarding.
function PreparingChecklistItem({ label, done }: { label: string; done: boolean }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      {done ? (
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" style={{ flexShrink: 0 }}>
          <circle cx="8" cy="8" r="8" fill="#22C55E" />
          <path d="M4.5 8.5L6.8 10.8L11.5 5.5" stroke="#0A0A0A" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      ) : (
        <div style={{ width: 16, height: 16, borderRadius: "50%", border: "2px solid rgba(237,236,234,0.15)", borderTopColor: "#BC9BFF", animation: "ob-spin 0.8s linear infinite", flexShrink: 0 }} />
      )}
      <span style={{ fontSize: 13, color: done ? "rgba(237,236,234,0.5)" : "#EDECEA" }}>{label}</span>
    </div>
  );
}

export function StepPreparing({ onReady }: { onReady: () => void }) {
  const { cogniInstance, apiKey } = useCogniInstance();
  const { tenantReady } = useTenant();
  const [elapsedMs, setElapsedMs] = useState(0);
  const track = useOnboardingTrackEvent();
  const completedTracked = useRef(false);

  useEffect(() => {
    track({ pageName: "Onboarding", eventName: "provisioning_started" });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!tenantReady) return;
    if (!completedTracked.current) {
      completedTracked.current = true;
      track({ pageName: "Onboarding", eventName: "provisioning_completed", additionalProperties: { duration_ms: String(elapsedMs) } });
    }
    onReady();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tenantReady, onReady]);

  useEffect(() => {
    const start = Date.now();
    const id = setInterval(() => setElapsedMs(Date.now() - start), 1000);
    return () => clearInterval(id);
  }, []);

  const takingLonger = elapsedMs > 30_000;

  return (
    <div style={{
      minHeight: "100vh",
      position: "relative",
      backgroundColor: "#000000",
      backgroundImage: "linear-gradient(rgba(244,244,244,0.10) 1px, transparent 1px), linear-gradient(90deg, rgba(244,244,244,0.10) 1px, transparent 1px)",
      backgroundSize: "33px 33px",
      display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
      padding: "56px 24px", boxSizing: "border-box",
      overflow: "hidden",
    }}>
      <TetrisBackground />

      <div style={{
        position: "relative", zIndex: 3,
        display: "flex", flexDirection: "column", alignItems: "center", gap: 20,
        background: "#2a2a2e",
        border: "1px solid rgba(255,255,255,0.1)",
        borderRadius: 16,
        padding: "48px 64px",
        boxShadow: "0 20px 60px rgba(0,0,0,0.5)",
        maxWidth: 480, width: "100%",
      }}>
        <style>{`@keyframes ob-spin { to { transform: rotate(360deg); } }`}</style>
        <GraphBuildingAnimation />

        <h1 style={{
          fontSize: 26, fontWeight: 300, color: "#EDECEA", margin: 0,
          textAlign: "center", letterSpacing: "-0.02em", lineHeight: 1.2,
          fontFamily: '"TWKLausanne", sans-serif',
        }}>
          Building your workspace…
        </h1>
        <p style={{ margin: 0, fontSize: 14, color: "rgba(237,236,234,0.65)", textAlign: "center", lineHeight: "22px", maxWidth: 340 }}>
          {takingLonger
            ? "Still going — this can take a couple of minutes on a brand-new account."
            : "Setting up your memory layer. This usually takes under a minute."}
        </p>

        <div style={{ display: "flex", flexDirection: "column", gap: 12, alignSelf: "stretch" }}>
          {/* Ordered to match the actual completion sequence in useTenantInit.ts:
              the API key resolves before cogniInstance is set, which resolves
              before tenantReady — showing them out of order made "API key"
              flip to done before "workspace" ever left its spinner. */}
          <PreparingChecklistItem label="Creating your API key" done={!!apiKey} />
          <PreparingChecklistItem label="Preparing your workspace" done={!!cogniInstance} />
          <PreparingChecklistItem label="Preparing your data" done={tenantReady} />
        </div>

        {takingLonger && (
          <button
            // This is BEFORE path selection — the user hasn't chosen how they
            // want to onboard yet, so bailing out here must continue into
            // onboarding (path selection), never mark it complete or leave
            // for the dashboard. That's what the "Skip onboarding" links
            // deeper in the flow are for — this button just stops waiting.
            onClick={() => {
              track({ pageName: "Onboarding", eventName: "provisioning_continue_without_waiting", additionalProperties: { duration_ms: String(elapsedMs) } });
              onReady();
            }}
            className="cursor-pointer"
            style={{ background: "none", border: "1px solid rgba(255,255,255,0.2)", borderRadius: 8, padding: "8px 18px", fontSize: 13, fontWeight: 500, color: "rgba(237,236,234,0.8)" }}
          >
            Continue without waiting
          </button>
        )}
      </div>
    </div>
  );
}
