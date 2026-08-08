"use client";

import { useState, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import { useUser } from "@/modules/users/UserContext";
import ServeOnboarding from "./ServeOnboarding";
import { completeOnboardingAndNavigate } from "./completeOnboardingAndNavigate";
import getDatasets from "@/modules/datasets/getDatasets";
import { TrackPageView } from "@/modules/analytics";
import { useOnboardingTrackEvent } from "./useOnboardingTrackEvent";
import { type OnboardingDemoEntry, DEMO_QUERIES } from "@/ui/elements/AgentActivityTerminal";
import recallKnowledge from "@/modules/datasets/recallKnowledge";
import { StepPreparing } from "./partials/StepPreparing";
import { Step1 } from "./partials/Step1";
import { Step2 } from "./partials/Step2";
import { Step3, extractFirstAnswer } from "./partials/Step3";
import { StepSelect, type OnboardingPath } from "./partials/StepSelect";
import { AgentOnboarding } from "./partials/AgentOnboarding";

// Completes onboarding and navigates to the dashboard, where the skeleton
// takes over until the workspace is fully ready. Used by the "Skip to
// dashboard" escape so a slow pod/dataset never traps the user.
function skipToDashboard(
  router: ReturnType<typeof useRouter>,
  markOnboardingComplete: () => Promise<void>,
  track: ReturnType<typeof useOnboardingTrackEvent>,
): void {
  track({ pageName: "Onboarding", eventName: "onboarding_skipped_provisioning" });
  void completeOnboardingAndNavigate(markOnboardingComplete, () => router.push("/dashboard"));
}

export default function OnboardingPage() {
  const { cogniInstance, isInitializing, serviceUrl, apiKey } = useCogniInstance();
  const { markOnboardingComplete } = useUser();
  const router = useRouter();
  const track = useOnboardingTrackEvent();
  const searchParams = useSearchParams();
  const isServeMode = searchParams.get("source") === "serve";
  const rawStep = parseInt(searchParams.get("step") ?? "0", 10);
  const initialStep = Math.min(Math.max(isNaN(rawStep) ? 0 : rawStep, 0), 3);
  const [step, setStep] = useState(initialStep);
  // Top-level branch: preparing → path selection → either the Claude/Codex
  // card flow ("agent") or the existing Company Brain upload flow ("company").
  // A ?step= deep link jumps straight into the Company Brain sub-steps. The
  // former "welcome" step here duplicated the real /welcome page's content —
  // that content now lives there instead, so onboarding starts on preparing.
  const [view, setView] = useState<"preparing" | "select" | "company" | "agent">(initialStep >= 1 ? "company" : "preparing");
  const [agent, setAgent] = useState<"claude-code" | "codex">("claude-code");
  const [files, setFiles] = useState<File[]>([]);
  const [datasetId, setDatasetId] = useState<string | null>(null);
  // Demo recalls are kicked off the moment cognify finishes (datasetId is set).
  // The Step 3 terminal reveals them sequentially with a typewriter — by the
  // time the user reads the first question, the answers are usually already in.
  const [demoEntries, setDemoEntries] = useState<OnboardingDemoEntry[] | null>(null);

  // When entering mid-flow (step > 1), resolve the existing dataset
  useEffect(() => {
    if (initialStep <= 1 || !cogniInstance) return;
    getDatasets(cogniInstance)
      .then((data: Array<{ id: string }>) => {
        if (Array.isArray(data) && data.length > 0) setDatasetId(data[0].id);
      })
      .catch((err) => console.error("Failed to resolve existing dataset:", err));
  }, [cogniInstance, initialStep]);

  // Preload demo recalls as soon as the dataset exists (cognify just finished
  // — Step 2 advances to Step 3 via setDatasetId+setStep, this fires on the
  // same render). All three queries run in parallel.
  useEffect(() => {
    if (!datasetId || !cogniInstance || demoEntries) return;
    const initial: OnboardingDemoEntry[] = DEMO_QUERIES.map((q) => ({ query: q, result: null, status: "pending" }));
    setDemoEntries(initial);
    DEMO_QUERIES.forEach((q, i) => {
      recallKnowledge(cogniInstance, { query: q, scope: "graph", datasetIds: [datasetId] })
        .then((data) => {
          const text = extractFirstAnswer(data);
          setDemoEntries((prev) => prev?.map((e, j) => j === i ? { ...e, status: text ? "done" : "error", result: text } : e) ?? prev);
        })
        .catch(() => {
          setDemoEntries((prev) => prev?.map((e, j) => j === i ? { ...e, status: "error" } : e) ?? prev);
        });
    });
  }, [datasetId, cogniInstance, demoEntries]);

  const darkPage: React.CSSProperties = {
    backgroundColor: "#000000",
    backgroundImage: "linear-gradient(rgba(244,244,244,0.10) 1px, transparent 1px), linear-gradient(90deg, rgba(244,244,244,0.10) 1px, transparent 1px)",
    backgroundSize: "33px 33px",
  };

  if (isInitializing) {
    return (
      <><TrackPageView page="Onboarding" /><div className="flex items-center justify-center h-screen" style={darkPage}>
        <span style={{ fontSize: 14, color: "rgba(237,236,234,0.65)" }}>Connecting...</span>
      </div></>
    );
  }

  if (isServeMode) {
    // Local-mode onboarding requires a working cognee-cli backend connection.
    if (!cogniInstance) {
      return (
        <><TrackPageView page="Onboarding" /><div className="flex items-center justify-center h-screen" style={darkPage}>
          <span style={{ fontSize: 14, color: "rgba(237,236,234,0.65)" }}>Connecting...</span>
        </div></>
      );
    }
    return <ServeOnboarding />;
  }

  // Within the Company Brain flow, steps 1 and 2 render fine while the tenant
  // pod is still being provisioned in the background (Step 2 owns its own
  // loading state). Step 3 renders recall results, so it needs both the pod
  // (cogniInstance) AND the dataset before it can show anything.
  const podPending = view === "company" && step >= 3 && (!cogniInstance || !datasetId);

  return (
    <div style={{ minHeight: "100vh", overflowY: "auto", ...darkPage }}>
      <TrackPageView page="Onboarding" additionalProperties={{ view, step: String(step) }} />
      {view === "preparing" && (
        // Reaching path selection only proves the workspace is functional —
        // it says nothing about whether the user actually went through
        // onboarding. Completion is marked on explicit finish/skip only
        // (see markOnboardingComplete call sites below), never here.
        <StepPreparing onReady={() => setView("select")} />
      )}
      {view === "select" && (
        <StepSelect onSelect={(path: OnboardingPath) => {
          if (path === "company") { setView("company"); setStep(1); }
          else { setAgent(path); setView("agent"); }
        }} />
      )}
      {view === "agent" && (
        <AgentOnboarding
          agent={agent}
          serviceUrl={serviceUrl}
          apiKey={apiKey}
          cogniInstance={cogniInstance}
          onRestart={() => { setView("select"); setStep(0); }}
        />
      )}
      {view === "company" && (
        podPending ? (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: "100vh", gap: 16 }}>
            <style>{`@keyframes ob-spin { to { transform: rotate(360deg); } }`}</style>
            <div style={{ width: 36, height: 36, borderRadius: "50%", border: "3px solid rgba(237,236,234,0.12)", borderTopColor: "#BC9BFF", animation: "ob-spin 0.8s linear infinite" }} />
            <p style={{ margin: 0, fontSize: 14, color: "rgba(237,236,234,0.75)" }}>Still preparing your memory…</p>
            <p style={{ margin: 0, fontSize: 13, color: "rgba(237,236,234,0.5)" }}>This usually takes about a minute on first sign-up.</p>
            <button
              onClick={() => skipToDashboard(router, markOnboardingComplete, track)}
              className="cursor-pointer"
              style={{ marginTop: 8, background: "none", border: "1px solid rgba(255,255,255,0.2)", borderRadius: 8, padding: "8px 18px", fontSize: 13, fontWeight: 500, color: "rgba(237,236,234,0.8)" }}
            >
              Skip to dashboard
            </button>
          </div>
        ) : (
          <>
            {step === 1 && <Step1 files={files} setFiles={setFiles} onNext={() => setStep(2)} />}
            {step === 2 && <Step2 files={files} datasetId={datasetId} onNext={(id) => { setDatasetId(id); setStep(3); }} cogniInstance={cogniInstance} />}
            {step === 3 && cogniInstance && datasetId && (
              <Step3 datasetId={datasetId} cogniInstance={cogniInstance} demoEntries={demoEntries} />
            )}
          </>
        )
      )}
    </div>
  );
}
