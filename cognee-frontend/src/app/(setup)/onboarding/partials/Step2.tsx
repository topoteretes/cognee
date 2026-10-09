"use client";

import { useState, useRef, useEffect } from "react";
import { useCogniInstance, useTenant } from "@/modules/tenant/TenantProvider";
import createDataset from "@/modules/datasets/createDataset";
import rememberData from "@/modules/ingestion/rememberData";
import pollDatasetStatus from "@/modules/datasets/pollDatasetStatus";
import { setAwaitingDataset, clearAwaitingDataset } from "@/utils/browserStorage";
import { StepBadge, StepDots, SkipLink } from "./Shared";
import { useOnboardingTrackEvent } from "../useOnboardingTrackEvent";

interface ProcessingStep {
  label: string;
  progress: number;
  status: "pending" | "active" | "done" | "error";
}

export function Step2({ files, datasetId, onNext, cogniInstance }: {
  files: File[];
  datasetId: string | null;
  onNext: (dsId: string) => void;
  // May be null while the freshly-created tenant is still provisioning — Step 2
  // shows its loading bars until both the instance and the pod are ready.
  cogniInstance: ReturnType<typeof useCogniInstance>["cogniInstance"];
}) {
  const { tenantReady } = useTenant();
  const track = useOnboardingTrackEvent();
  const [steps, setSteps] = useState<ProcessingStep[]>([
    { label: "Setting up workspace", progress: 0, status: "active" },
    { label: "Uploading files", progress: 0, status: "pending" },
    { label: "Building knowledge graph", progress: 0, status: "pending" },
  ]);
  const [error, setError] = useState<string | null>(null);
  const [dsId, setDsId] = useState<string | null>(datasetId);
  const cancelledRef = useRef(false);
  const tickerIds = useRef<ReturnType<typeof setInterval>[]>([]);

  const allDone = steps.every((s) => s.status === "done");

  // While the tenant is still provisioning, fill the "Setting up workspace" bar
  // from 0 → 95% over ~180s (the provisioning poll's max window) so it's always
  // visibly progressing instead of parked. It snaps to 100% the moment the
  // tenant is ready (runPipeline below), so a fast provision just completes it
  // early.
  useEffect(() => {
    if (tenantReady) return;
    const stop = smoothTo(0, 95, 0.02); // slow creep over the ~180s provisioning window
    return stop;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tenantReady]);

  // In StrictMode, React mounts → unmounts → remounts every component.
  // The cleanup cancels any in-flight run so the remount starts fresh.
  // Wait for the tenant to be FULLY provisioned before kicking off the
  // pipeline: both the cogni instance (the tenant has been resolved/created)
  // AND tenantReady (its pod answers). For fresh tenants both arrive while the
  // user reads through Step 0/1; until then the loading bars creep.
  useEffect(() => {
    if (!tenantReady || !cogniInstance) return;
    cancelledRef.current = false;
    tickerIds.current.forEach(clearInterval);
    tickerIds.current = [];
    runPipeline();

    return () => {
      cancelledRef.current = true;
      tickerIds.current.forEach(clearInterval);
      tickerIds.current = [];
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tenantReady, cogniInstance]);

  // Auto-advance to Step 3 as soon as processing completes
  useEffect(() => {
    if (!allDone || !dsId) return;
    const timer = setTimeout(() => {
      track({ pageName: "Onboarding", eventName: "onboarding_step_completed", additionalProperties: { step: "2" } });
      onNext(dsId);
    }, 800);
    return () => clearTimeout(timer);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [allDone, dsId]);

  function updateStep(index: number, update: Partial<ProcessingStep>) {
    setSteps((prev) => prev.map((s, i) => i === index ? { ...s, ...update } : s));
  }

  // Eases step[index].progress asymptotically toward `cap` at a fluid 60ms cadence.
  // `approachPerSec` is the fraction of the remaining gap closed per second, so motion
  // is always visible during an indeterminate wait and never lands hard on a round
  // number. Never sets an initial value — always continues from wherever the bar is —
  // and holds just shy of `cap` (so an unfinished stage shows slow motion, not a park),
  // until it's re-armed (e.g. accelerated) or the stage is completed to 100 elsewhere.
  // Returns a stop function; also pushed to tickerIds for bulk cleanup.
  function smoothTo(index: number, cap: number, approachPerSec: number): () => void {
    const TICK_MS = 60;
    const k = 1 - Math.pow(1 - approachPerSec, TICK_MS / 1000); // per-tick gap fraction
    const ceiling = cap - 0.4; // approach but never reach the cap
    const id = setInterval(() => {
      if (cancelledRef.current) { clearInterval(id); return; }
      setSteps((prev) => {
        const cur = prev[index].progress;
        if (cur >= ceiling) { clearInterval(id); return prev; } // gap closed; stop cleanly
        const next = Math.min(cur + (cap - cur) * k, ceiling);
        return prev.map((s, i) => i === index ? { ...s, progress: next } : s);
      });
    }, TICK_MS);
    tickerIds.current.push(id);
    return () => clearInterval(id);
  }

  async function runPipeline() {
    const cancelled = () => cancelledRef.current;
    if (!cogniInstance) return; // only runs once the tenant is ready (see effect guard)

    try {
      // The tenant is provisioned by the time we get here — complete the
      // "Setting up workspace" bar (it was creeping toward 95% while we waited).
      updateStep(0, { progress: 100, status: "done" });

      let currentDsId = dsId;
      if (!currentDsId) {
        const ds = await createDataset({ name: "default_dataset" }, cogniInstance);
        if (cancelled()) return;
        currentDsId = ds.id;
        setDsId(currentDsId);
      }

      // ── Uploading files (bar 1): one remember call uploads + kicks off processing ──
      // No explicit progress:0 — on a StrictMode/re-invoke the bar keeps its value
      // instead of visibly snapping backward; on a genuine first run it's already 0.
      updateStep(1, { status: "active" });
      const stopUpload = smoothTo(1, 90, 0.9);
      // currentDsId is always a string here: either it was non-null at entry or
      // we just assigned it from createDataset above. TypeScript can't narrow it
      // through the conditional assignment, so we guard explicitly.
      if (!currentDsId) return;
      await rememberData({ id: currentDsId, name: "default_dataset" }, files, cogniInstance, { runInBackground: true });
      if (cancelled()) return;
      // Hand off to the dashboard: if the user skips before processing finishes,
      // the dashboard keeps its skeleton until this dataset reaches a terminal
      // status. Cleared below when the poll completes (full-onboarding path).
      setAwaitingDataset(currentDsId);
      stopUpload();
      updateStep(1, { progress: 100, status: "done" });

      // ── Building knowledge graph (bar 2): poll the background pipeline ──
      if (cancelled()) return;
      updateStep(2, { status: "active" });
      // Baseline: an always-moving creep toward a high ceiling, so the bar advances
      // even if the poller never observes an intermediate status. (A fresh cognify
      // run goes straight to STARTED/COMPLETED — INITIATED is reserved for
      // reset/recovery — and during the add phase there's no cognify status row at
      // all, so we must NOT depend on a specific status to un-park the bar.)
      let stopGraph = smoothTo(2, 90, 0.04);

      // The backend collapses INITIATED/STARTED/COMPLETED and may skip the in-progress
      // states entirely. Treat the FIRST in-progress signal (whichever it is) as a
      // single "accelerate" event rather than a required, ordered sequence of caps.
      let seenInProgress = false;

      await Promise.all([
        pollDatasetStatus(currentDsId!, cogniInstance, {
          intervalMs: 1500,
          initialDelayMs: 750,
          onStatus: (status) => {
            if (cancelled()) return;
            if (
              (status === "DATASET_PROCESSING_INITIATED" || status === "DATASET_PROCESSING_STARTED") &&
              !seenInProgress
            ) {
              seenInProgress = true;
              stopGraph();
              stopGraph = smoothTo(2, 90, 0.25); // accelerate toward the same ceiling
            }
          },
        }),
      ]);

      if (cancelled()) return;

      // Pipeline complete — finish the graph bar smoothly.
      stopGraph();
      updateStep(2, { progress: 100, status: "done" });
      // Dataset is processed — clear the dashboard hand-off flag so the normal
      // completion path doesn't leave the dashboard waiting.
      clearAwaitingDataset();

    } catch (err) {
      // Clear the hand-off flag even when the tab was abandoned mid-flow —
      // a failed pipeline must never leave the dashboard skeleton waiting on
      // a dataset that will not finish.
      clearAwaitingDataset();
      if (cancelled()) return;
      const message = err instanceof Error ? err.message : "Processing failed";
      setError(message);
      setSteps((prev) => prev.map((s) => s.status === "active" ? { ...s, status: "error" } : s));
      track({ pageName: "Onboarding", eventName: "onboarding_processing_failed", additionalProperties: { step: "2", error: message } });
    }
  }

  return (
    <div style={{
      minHeight: "100vh",
      backgroundColor: "#000000",
      position: "relative",
      display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
      padding: "56px 24px", boxSizing: "border-box",
      overflow: "hidden",
    }}>
      <div style={{
        position: "absolute", inset: -20, zIndex: 0,
        backgroundImage: "linear-gradient(rgba(244,244,244,0.10) 1px, transparent 1px), linear-gradient(90deg, rgba(244,244,244,0.10) 1px, transparent 1px)",
        backgroundSize: "33px 33px",
        pointerEvents: "none",
      }} />
      <div className="flex flex-col items-center justify-center gap-6" style={{
        position: "relative", zIndex: 1,
        background: "#2a2a2e",
        border: "1px solid rgba(255,255,255,0.1)",
        borderRadius: 16,
        padding: "48px 64px",
        boxShadow: "0 20px 60px rgba(0,0,0,0.5)",
        maxWidth: 560, width: "100%", boxSizing: "border-box",
      }}>
      <StepBadge step={2} total={3} />
      <h1 style={{ fontSize: 28, fontWeight: 300, color: "#EDECEA", margin: 0, fontFamily: '"TWKLausanne", sans-serif', letterSpacing: "-0.02em" }}>Building your memory</h1>
      <p style={{ fontSize: 15, color: "rgba(237,236,234,0.65)", margin: 0, textAlign: "center", maxWidth: 480, lineHeight: "22px" }}>
        Cognee is extracting entities, building relationships, and generating embeddings.
      </p>

      <div style={{ width: 480, maxWidth: "100%", background: "#2a2a2e", border: "1px solid rgba(188,155,255,0.20)", borderRadius: 12, padding: 24 }}>
        <div className="flex flex-col gap-5">
          {steps.map((step, i) => (
            <div key={i} className="flex items-center gap-3">
              {step.status === "done" ? (
                <div className="flex items-center justify-center flex-shrink-0 rounded-full" style={{ width: 32, height: 32, background: "#22C55E" }}>
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M3.5 8.5L6.5 11.5L12.5 4.5" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>
                </div>
              ) : step.status === "error" ? (
                <div className="flex items-center justify-center flex-shrink-0 rounded-full" style={{ width: 32, height: 32, background: "#EF4444" }}>
                  <span style={{ color: "#fff", fontSize: 13, fontWeight: 700 }}>!</span>
                </div>
              ) : step.status === "active" ? (
                <div className="flex items-center justify-center flex-shrink-0 rounded-full" style={{ width: 32, height: 32, background: "#BC9BFF" }}>
                  <span style={{ color: "#1e1e1c", fontSize: 13, fontWeight: 700 }}>{i + 1}</span>
                </div>
              ) : (
                <div className="flex items-center justify-center flex-shrink-0 rounded-full" style={{ width: 32, height: 32, border: "1.5px solid rgba(255,255,255,0.2)" }}>
                  <span style={{ color: "rgba(237,236,234,0.4)", fontSize: 13, fontWeight: 700 }}>{i + 1}</span>
                </div>
              )}
              <div className="flex-1 flex flex-col gap-1">
                <div className="flex justify-between">
                  <span style={{ fontSize: 14, fontWeight: 500, color: step.status === "pending" ? "rgba(237,236,234,0.4)" : step.status === "error" ? "#EF4444" : "#EDECEA" }}>{step.label}</span>
                  {step.status === "done" && <span style={{ fontSize: 12, fontWeight: 500, color: "#22C55E" }}>Done</span>}
                  {step.status === "active" && <span style={{ fontSize: 12, fontWeight: 500, color: "#BC9BFF" }}>{Math.round(step.progress)}%</span>}
                  {step.status === "error" && <span style={{ fontSize: 12, fontWeight: 500, color: "#EF4444" }}>Failed</span>}
                </div>
                <div style={{ height: 4, borderRadius: 2, background: "rgba(188,155,255,0.10)" }}>
                  <div style={{ height: 4, borderRadius: 2, background: step.status === "done" ? "#22C55E" : step.status === "active" ? "#8CFF86" : step.status === "error" ? "#EF4444" : "transparent", width: `${step.progress}%`, transition: "width 0.15s linear" }} />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {error && (
        <div style={{ background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: 8, padding: "10px 16px", fontSize: 13, color: "#FCA5A5", maxWidth: 480 }}>
          {error}
        </div>
      )}

      {allDone && (
        <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 14, color: "#22C55E" }}>
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M3.5 8.5L6.5 11.5L12.5 4.5" stroke="#22C55E" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>
          Memory built — continuing…
        </div>
      )}

      <StepDots current={2} total={3} />

      <SkipLink />
      </div>
    </div>
  );
}
