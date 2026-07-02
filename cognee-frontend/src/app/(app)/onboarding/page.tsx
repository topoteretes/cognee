"use client";

import Image from "next/image";
import { useState, useRef, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useCogniInstance, useTenant } from "@/modules/tenant/TenantProvider";
import markOnboardingComplete from "@/modules/users/markOnboardingComplete";
import TetrisBackground from "@/ui/elements/Auth/TetrisBackground";
import ServeOnboarding from "./ServeOnboarding";
import rememberData from "@/modules/ingestion/rememberData";
import createDataset from "@/modules/datasets/createDataset";
import getDatasets from "@/modules/datasets/getDatasets";
import getDatasetData from "@/modules/datasets/getDatasetData";
import pollDatasetStatus from "@/modules/datasets/pollDatasetStatus";
import { trackEvent, TrackPageView } from "@/modules/analytics";
import { AgentActivityTerminal, type OnboardingDemoEntry, DEMO_QUERIES } from "@/ui/elements/AgentActivityTerminal";
import recallKnowledge from "@/modules/datasets/recallKnowledge";
import { listSessions, SEARCH_SESSION_PREFIX } from "@/modules/sessions/getSessions";
import { CLAUDE_MARKETPLACE_ADD, CLAUDE_PLUGIN_INSTALL, CODEX_HOOKS_ENABLE, CODEX_MARKETPLACE_ADD, CODEX_PLUGIN_INSTALL, UPLOAD_MEMORY_PROMPT, UPLOAD_SAMPLE_PROMPT, RECALL_SAMPLE_PROMPT } from "@/data/prompts";

// ── Shared ──

function StepBadge({ step, total = 4 }: { step: number; total?: number }) {
  return (
    <div style={{ background: "rgba(188,155,255,0.20)", borderRadius: 100, border: "1px solid rgba(188,155,255,0.35)", padding: "5px 12px" }}>
      <span style={{ color: "#EDECEA", fontSize: 13, fontWeight: 500 }}>Step {step} of {total}</span>
    </div>
  );
}

function StepDots({ current, total = 4 }: { current: number; total?: number }) {
  return (
    <div className="flex items-center gap-2">
      {Array.from({ length: total }).map((_, i) => (
        <div key={i} style={{ width: 24, height: 4, borderRadius: 2, background: i + 1 === current ? "#BC9BFF" : "rgba(255,255,255,0.2)" }} />
      ))}
    </div>
  );
}

function SkipLink({ label = "Skip onboarding and go to dashboard", compact = false }: { label?: string; compact?: boolean } = {}) {
  const router = useRouter();
  return (
    <button onClick={() => { trackEvent({ pageName: "Onboarding", eventName: "onboarding_skipped" }); sessionStorage.setItem("cognee-onboarding-skipped", "1"); localStorage.setItem("cognee-onboarding-complete", "1"); markOnboardingComplete().catch(() => {}); router.push("/dashboard"); }} className="cursor-pointer" style={{ background: "none", border: "none", color: "rgba(237,236,234,0.65)", fontSize: 13, paddingTop: compact ? 12 : 32, paddingBottom: compact ? 0 : 24 }}>
      {label}
    </button>
  );
}

// ── Step 0: Welcome screen ──

function Step0({ onNext }: { onNext: () => void }) {
  const [primaryHover, setPrimaryHover] = useState(false);
  const [secondaryHover, setSecondaryHover] = useState(false);
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
      {/* Falling tetrominoes — same canvas the auth hero uses. Sits behind
          the focal container; pointer events stay on the card. */}
      <TetrisBackground />

      {/* Focal container */}
      <div style={{
        position: "relative", zIndex: 3,
        display: "flex", flexDirection: "column", alignItems: "center",
        background: "#2a2a2e",
        border: "1px solid rgba(255,255,255,0.1)",
        borderRadius: 16,
        padding: "48px 64px",
        boxShadow: "0 20px 60px rgba(0,0,0,0.5)",
        maxWidth: 540, width: "100%",
      }}>

        {/* Heading */}
        <h1 style={{
          fontSize: 34, fontWeight: 300, color: "#EDECEA", margin: "0 0 12px",
          textAlign: "center", letterSpacing: "-0.02em", lineHeight: 1.15,
          fontFamily: '"TWKLausanne", sans-serif',
        }}>
          Welcome to Cognee Cloud
        </h1>
        <p style={{ fontSize: 15, color: "rgba(237,236,234,0.65)", margin: "0 0 36px", textAlign: "center", lineHeight: "24px", maxWidth: 400 }}>
          Let's take a minute to set up your account.
        </p>

        {/* CTAs — lavender purple */}
        <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap", justifyContent: "center" }}>
          <button
            onClick={() => { trackEvent({ pageName: "Onboarding", eventName: "onboarding_welcome_cta" }); onNext(); }}
            onMouseEnter={() => setPrimaryHover(true)}
            onMouseLeave={() => setPrimaryHover(false)}
            style={{
              background: primaryHover ? "#A87CFF" : "#BC9BFF",
              border: "none", borderRadius: 8,
              padding: "11px 32px", fontSize: 14, fontWeight: 500, color: "#1e1e1c",
              cursor: "pointer", letterSpacing: "-0.01em",
              transition: "background 150ms ease",
            }}
          >
            Get started →
          </button>
          <a
            href="https://docs.cognee.ai"
            target="_blank"
            rel="noopener noreferrer"
            onMouseEnter={() => setSecondaryHover(true)}
            onMouseLeave={() => setSecondaryHover(false)}
            style={{
              display: "inline-flex", alignItems: "center", gap: 7,
              background: secondaryHover ? "rgba(188,155,255,0.10)" : "transparent",
              border: `1px solid ${secondaryHover ? "#A87CFF" : "#BC9BFF"}`,
              borderRadius: 8,
              padding: "11px 20px", fontSize: 14, fontWeight: 500,
              color: secondaryHover ? "#A87CFF" : "#BC9BFF",
              textDecoration: "none",
              cursor: "pointer",
              transition: "background 150ms ease, border-color 150ms ease, color 150ms ease",
            }}
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill="#BC9BFF"><polygon points="5 3 19 12 5 21 5 3"/></svg>
            Video tour (2 min)
          </a>
        </div>

      </div>
    </div>
  );
}

// ── Step 1: Connect your data ──

function Step1({ onNext, files, setFiles }: {
  onNext: () => void;
  files: File[];
  setFiles: React.Dispatch<React.SetStateAction<File[]>>;
}) {
  const [isDragging, setIsDragging] = useState(false);
  const [showPaste, setShowPaste] = useState(false);
  const [pasteText, setPasteText] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Files are only collected here; the actual upload + processing happens
  // in Step2 as a single remember call.
  const handleFiles = (newFiles: FileList | File[]) => {
    const fileArray = Array.from(newFiles);
    setFiles((prev) => [...prev, ...fileArray]);
    trackEvent({ pageName: "Onboarding", eventName: "onboarding_files_added", additionalProperties: { file_count: String(fileArray.length), step: "1" } });
  };

  const removeFile = (index: number) => setFiles((prev) => prev.filter((_, i) => i !== index));

  const handlePasteSubmit = () => {
    if (!pasteText.trim()) return;
    trackEvent({ pageName: "Onboarding", eventName: "onboarding_text_pasted", additionalProperties: { text_length: String(pasteText.length), step: "1" } });
    const blob = new Blob([pasteText], { type: "text/plain" });
    const file = new File([blob], "pasted-text.txt", { type: "text/plain" });
    setFiles((prev) => [...prev, file]);
    setPasteText("");
    setShowPaste(false);
  };

  return (
    <div style={{
      minHeight: "100vh",
      backgroundColor: "#000000",
      backgroundImage: "linear-gradient(rgba(244,244,244,0.10) 1px, transparent 1px), linear-gradient(90deg, rgba(244,244,244,0.10) 1px, transparent 1px)",
      backgroundSize: "33px 33px",
      display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
      boxSizing: "border-box",
    }}>
      <div style={{ alignItems: "center", display: "flex", flexDirection: "column", paddingBottom: 40, paddingLeft: "clamp(16px, 5vw, 80px)", paddingRight: "clamp(16px, 5vw, 80px)", paddingTop: 48, width: "100%", boxSizing: "border-box" }}>

        {/* Header */}
        <div style={{ alignItems: "center", display: "flex", flexDirection: "column", gap: 8, paddingBottom: 40 }}>
          <div style={{ background: "rgba(188,155,255,0.20)", borderRadius: 100, border: "1px solid rgba(188,155,255,0.35)", padding: "5px 12px" }}>
            <div style={{ color: "#EDECEA", fontSize: 13, lineHeight: "16px" }}>Step 1 of 3</div>
          </div>
          <div style={{ color: "#EDECEA", fontSize: 28, fontWeight: 300, lineHeight: "34px", paddingTop: 8, letterSpacing: "-0.02em", fontFamily: '"TWKLausanne", sans-serif' }}>Connect your data</div>
          <div style={{ color: "rgba(237,236,234,0.65)", fontSize: 15, lineHeight: "22px", textAlign: "center", maxWidth: 440 }}>Choose how to get your data into Cognee. You can always add or change sources later.</div>
        </div>

        {/* Card: Add new data — solid #2a2a2e to match the rest of onboarding */}
        <div style={{ maxWidth: 880, width: "100%", background: "#2a2a2e", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 16, boxShadow: "0 20px 60px rgba(0,0,0,0.5)", display: "flex", flexDirection: "column", gap: 20, padding: "48px 64px" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <div style={{ color: "#EDECEA", fontSize: 17, lineHeight: "22px" }}>Add new data</div>
            <div style={{ color: "rgba(237,236,234,0.65)", fontSize: 13, lineHeight: "16px" }}>Upload files or paste content directly</div>
          </div>

          {/* Hidden file input */}
          <input ref={fileInputRef} type="file" multiple accept=".pdf,.csv,.txt,.md,.json,.docx" className="hidden" onChange={(e) => { if (e.target.files) handleFiles(e.target.files); e.target.value = ""; }} />

          {/* Drop zone */}
          <div
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(e) => { e.preventDefault(); setIsDragging(false); if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files); }}
            style={{ alignItems: "center", background: isDragging ? "rgba(188,155,255,0.20)" : "rgba(255,255,255,0.04)", border: `2px dashed ${isDragging ? "#BC9BFF" : "rgba(255,255,255,0.18)"}`, borderRadius: 12, cursor: "pointer", display: "flex", flexDirection: "column", flexShrink: 0, gap: 8, height: 200, justifyContent: "center", paddingBlock: 40, paddingInline: 20, transition: "background 200ms, border-color 200ms" }}
          >
            <div style={{ alignItems: "center", background: "rgba(255,255,255,0.1)", borderRadius: 10, display: "flex", flexShrink: 0, height: 40, justifyContent: "center", width: 40 }}>
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#BC9BFF" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" /></svg>
            </div>
            <div style={{ color: "#BC9BFF", fontSize: 14, lineHeight: "18px" }}>Drop files here or browse</div>
            <div style={{ color: "rgba(237,236,234,0.4)", fontSize: 12, lineHeight: "16px" }}>PDF, CSV, TXT, Markdown, JSON</div>
          </div>

          {/* File list */}
          {files.length > 0 && (
            <div className="flex flex-col gap-2">
              {files.map((f, i) => (
                <div key={`f-${i}`} className="flex items-center justify-between" style={{ background: "rgba(255,255,255,0.07)", borderRadius: 8, padding: "8px 12px" }}>
                  <div className="flex items-center gap-2">
                    <span style={{ fontSize: 13, color: "#EDECEA" }}>{f.name}</span>
                    <span style={{ fontSize: 11, color: "rgba(237,236,234,0.4)" }}>({(f.size / 1024).toFixed(0)} KB)</span>
                  </div>
                  <button onClick={() => removeFile(i)} className="cursor-pointer bg-transparent border-none p-1" style={{ color: "rgba(237,236,234,0.4)", fontSize: 14 }}>&#10005;</button>
                </div>
              ))}
            </div>
          )}

          {files.length > 0 && (
            <div style={{ fontSize: 13, color: "#22C55E" }}>{files.length} file{files.length !== 1 ? "s" : ""} ready to process</div>
          )}

          {/* Paste text button / area */}
          {!showPaste ? (
            <div onClick={() => setShowPaste(true)} style={{ alignItems: "center", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 10, cursor: "pointer", display: "flex", gap: 8, paddingBlock: 12, paddingInline: 16 }}>
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#BC9BFF" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="8" y="2" width="8" height="4" rx="1" ry="1" /><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2" /><line x1="12" y1="11" x2="12" y2="17" /><line x1="9" y1="14" x2="15" y2="14" /></svg>
              <div style={{ color: "rgba(237,236,234,0.7)", flexShrink: 0, fontSize: 13, lineHeight: "16px" }}>Paste text</div>
            </div>
          ) : (
            <div className="flex flex-col gap-2">
              <textarea
                autoFocus
                value={pasteText}
                onChange={(e) => setPasteText(e.target.value)}
                placeholder="Paste your text content here..."
                style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.15)", borderRadius: 10, color: "#EDECEA", fontSize: 13, minHeight: 80, padding: 12, resize: "vertical", outline: "none" }}
              />
              <div className="flex gap-2">
                <button onClick={handlePasteSubmit} className="cursor-pointer" style={{ background: "#BC9BFF", border: "none", borderRadius: 8, color: "#1e1e1c", fontSize: 13, padding: "6px 16px" }}>Add text</button>
                <button onClick={() => { setShowPaste(false); setPasteText(""); }} className="cursor-pointer" style={{ background: "none", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 8, color: "rgba(237,236,234,0.6)", fontSize: 13, padding: "6px 16px" }}>Cancel</button>
              </div>
            </div>
          )}
        </div>

        {/* Continue button when files selected */}
        {files.length > 0 && (
          <div style={{ paddingTop: 24 }}>
            <button onClick={() => { trackEvent({ pageName: "Onboarding", eventName: "onboarding_step_completed", additionalProperties: { step: "1", file_count: String(files.length) } }); onNext(); }} className="cursor-pointer" style={{ background: "#BC9BFF", border: "none", borderRadius: 8, padding: "11px 32px", fontSize: 14, fontWeight: 500, color: "#1e1e1c", letterSpacing: "-0.01em" }}>
              Continue with {files.length} file{files.length !== 1 ? "s" : ""} →
            </button>
          </div>
        )}

        <div style={{ marginTop: 24 }}>
          <StepDots current={1} total={3} />
        </div>
        <SkipLink />
      </div>
    </div>
  );
}

// ── Step 2: Build memory (upload + cognify) ──

interface ProcessingStep {
  label: string;
  progress: number;
  status: "pending" | "active" | "done" | "error";
}

function Step2({ files, datasetId, onNext, cogniInstance }: {
  files: File[];
  datasetId: string | null;
  onNext: (dsId: string) => void;
  // May be null while the freshly-created tenant is still provisioning — Step 2
  // shows its loading bars until both the instance and the pod are ready.
  cogniInstance: ReturnType<typeof useCogniInstance>["cogniInstance"];
}) {
  const { tenantReady } = useTenant();
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
      trackEvent({ pageName: "Onboarding", eventName: "onboarding_step_completed", additionalProperties: { step: "2" } });
      onNext(dsId);
    }, 800);
    return () => clearTimeout(timer);
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
      await rememberData({ id: currentDsId!, name: "default_dataset" }, files, cogniInstance, { runInBackground: true });
      if (cancelled()) return;
      // Hand off to the dashboard: if the user skips before processing finishes,
      // the dashboard keeps its skeleton until this dataset reaches a terminal
      // status. Cleared below when the poll completes (full-onboarding path).
      try { sessionStorage.setItem("cognee-awaiting-dataset", currentDsId!); } catch { /* ignore */ }
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
      try { sessionStorage.removeItem("cognee-awaiting-dataset"); } catch { /* ignore */ }

    } catch (err) {
      if (cancelled()) return;
      setError(err instanceof Error ? err.message : "Processing failed");
      setSteps((prev) => prev.map((s) => s.status === "active" ? { ...s, status: "error" } : s));
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

// ── Step 3: Agent Activity Terminal ──

// Mirror of the "no answer" detection used in the terminal so the parent
// can store a non-boilerplate top result. Kept narrow on purpose.
function extractFirstAnswer(data: unknown): string | null {
  if (!Array.isArray(data)) return null;
  const noAnswerPatterns: RegExp[] = [
    /no (relevant |specific |such )?(information|data|context|knowledge|results?|answer)/i,
    /\b(does not|doesn'?t|do not|don'?t)\b[^.]{0,60}\b(contain|include|provide|mention|have|appear)/i,
    /unable to (find|answer|provide|determine|locate)/i,
    /there is no (information|mention|data|reference)/i,
  ];
  for (const row of data as Array<{ text?: unknown; answer?: unknown; search_result?: unknown }>) {
    let text: string | null = null;
    if (typeof row.text === "string") text = row.text;
    else if (typeof row.answer === "string") text = row.answer;
    else if (typeof row.search_result === "string") text = row.search_result;
    else if (Array.isArray(row.search_result) && row.search_result.every((x) => typeof x === "string")) text = (row.search_result as string[]).join("\n\n");
    if (!text || !text.trim()) continue;
    const t = text.trim();
    if (t.length < 300 && noAnswerPatterns.some((p) => p.test(t))) continue;
    return t.slice(0, 220);
  }
  return null;
}

function Step3({ datasetId, cogniInstance, demoEntries }: {
  datasetId: string;
  cogniInstance: NonNullable<ReturnType<typeof useCogniInstance>["cogniInstance"]>;
  demoEntries: OnboardingDemoEntry[] | null;
}) {
  const router = useRouter();
  const dataset = { id: datasetId, name: "default_dataset" };

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
      <div style={{
        position: "relative", zIndex: 1,
        padding: "48px 64px",
        display: "flex", flexDirection: "column", alignItems: "center", gap: 24,
        background: "#2a2a2e",
        border: "1px solid rgba(255,255,255,0.1)",
        borderRadius: 16,
        boxShadow: "0 20px 60px rgba(0,0,0,0.5)",
        maxWidth: 860, width: "100%", boxSizing: "border-box",
      }}>
      <StepBadge step={3} total={3} />
      <h1 style={{ fontSize: 28, fontWeight: 300, color: "#EDECEA", margin: 0, fontFamily: '"TWKLausanne", sans-serif', letterSpacing: "-0.02em" }}>Ask cognee anything</h1>
      <p style={{ fontSize: 15, color: "rgba(237,236,234,0.65)", margin: 0, textAlign: "center", lineHeight: "22px" }}>
        Your memory is ready. Ask anything about your data below.
      </p>

      <div style={{ width: "100%", maxWidth: 780 }}>
        <AgentActivityTerminal
          sessions={[]}
          runs={[]}
          agents={[]}
          datasets={[dataset]}
          selectedDataset={dataset}
          cogniInstance={cogniInstance}
          dataLoading={false}
          range="24h"
          onNavigate={router.push}
          variant="onboarding"
          onboardingDemo={demoEntries}
        />
      </div>

      <StepDots current={3} total={3} />

      <button
        onClick={() => {
          trackEvent({ pageName: "Onboarding", eventName: "onboarding_completed", additionalProperties: { destination: "dashboard" } });
          localStorage.setItem("cognee-onboarding-complete", "1");
          router.push("/dashboard");
        }}
        className="cursor-pointer"
        style={{ background: "#BC9BFF", border: "none", borderRadius: 8, padding: "11px 32px", fontSize: 14, fontWeight: 500, color: "#1e1e1c", letterSpacing: "-0.01em" }}
      >
        Connect my agent now →
      </button>
      </div>
    </div>
  );
}

// ── Path selection: Claude / Codex / Company Brain ──

type OnboardingPath = "claude-code" | "codex" | "company";

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

function StepSelect({ onSelect }: { onSelect: (path: OnboardingPath) => void }) {
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

// ── Claude / Codex card onboarding ──

// Single-line code block: shows ONE line, truncates the rest with an ellipsis (…)
// so long commands never wrap or overflow on small screens. `code` is what's
// shown; `toCopy` (when set) is the full multi-line command that's copied.
function OnboardingInlineCode({ code, toCopy, loading, placeholder = "Preparing…" }: {
  code: string; toCopy?: string; loading?: boolean; placeholder?: string;
}) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    if (loading) return;
    navigator.clipboard.writeText(toCopy ?? code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
    trackEvent({ pageName: "Onboarding", eventName: "onboarding_creds_copied" });
  };
  return (
    <div
      onClick={copy}
      className="cursor-pointer"
      style={{ background: "#18181B", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: "11px 14px", display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, cursor: loading ? "wait" : "pointer", width: "100%" }}
    >
      <pre style={{ margin: 0, fontSize: 12.5, fontFamily: 'ui-monospace, Menlo, Monaco, "Cascadia Mono", "Segoe UI Mono", "Roboto Mono", monospace', color: loading ? "#585B70" : "rgba(237,236,234,0.85)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", flex: 1, minWidth: 0 }}>
        <code>{loading ? placeholder : code}</code>
      </pre>
      <button
        onClick={(e) => { e.stopPropagation(); copy(); }}
        className="cursor-pointer"
        style={{ background: "#27272A", border: "1px solid #3F3F46", borderRadius: 4, padding: "4px 8px", fontSize: 11, color: loading ? "rgba(237,236,234,0.35)" : "rgba(237,236,234,0.65)", flexShrink: 0 }}
      >
        {copied ? "Copied!" : "Copy"}
      </button>
    </div>
  );
}

// Live connection indicator for the "connect & recall" step: a pulsing dim dot
// while we wait for the agent's first session, flipping to a solid green dot +
// "Connected" once a new session is detected in Cognee Cloud.
function ConnectStatus({ verified }: { verified: boolean }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12.5 }}>
      <span
        className={verified ? undefined : "ob-pulse"}
        style={{
          width: 8, height: 8, borderRadius: "50%", flexShrink: 0,
          background: verified ? "#22C55E" : "rgba(237,236,234,0.4)",
          boxShadow: verified ? "0 0 0 3px rgba(34,197,94,0.18)" : "none",
        }}
      />
      <span style={{ color: verified ? "#22C55E" : "rgba(237,236,234,0.5)" }}>
        {verified ? "Connected — activity detected in Cognee Cloud" : "Waiting for a connection…"}
      </span>
    </div>
  );
}

function AgentOnboarding({ agent, serviceUrl, apiKey, cogniInstance, onRestart }: {
  agent: "claude-code" | "codex";
  serviceUrl: string | null;
  apiKey: string;
  cogniInstance: ReturnType<typeof useCogniInstance>["cogniInstance"];
  onRestart: () => void;
}) {
  const router = useRouter();
  const name = agent === "claude-code" ? "Claude Code" : "Codex";
  const credsReady = Boolean(serviceUrl && apiKey);
  const baseUrl = serviceUrl || "https://your-tenant.aws.cognee.ai";
  const resolvedKey = apiKey || "your-api-key";
  // API key + base url are all the plugin/skill needs.
  const credsCode = `export COGNEE_BASE_URL="${baseUrl}"\nexport COGNEE_API_KEY="${resolvedKey}"`;

  // Accordion: exactly one step expanded at a time. Clicking a later step
  // collapses the current one (which turns green/"Done") and expands the next.
  const [currentStep, setCurrentStep] = useState(0);

  // Steps that show the live "waiting for connection" indicator. Both Claude Code
  // and Codex share the same flow: step 3 = Upload (index 2), step 4 = Recall
  // (index 3).
  const UPLOAD_STEP = 2;
  const CONNECT_STEP = 3;
  const [connectVerified, setConnectVerified] = useState(false);

  // Detect agent activity while either the Upload or Recall step is active.
  // There is NO generic "did we get an API call" endpoint, so we watch two
  // concrete signals and flip on whichever appears first:
  //   • a NEW session  → recall / session-scoped calls (carry a session_id)
  //   • NEW data docs  → graph-direct uploads (no session_id, so no session)
  // Both are baselined when the step opens so only activity AFTER that counts.
  useEffect(() => {
    const onDetectStep = currentStep === UPLOAD_STEP || currentStep === CONNECT_STEP;
    if (!onDetectStep || !cogniInstance || connectVerified) return;
    let cancelled = false;
    let primed = false;
    const baselineSessions = new Set<string>();
    let baselineDocs = -1;

    const realSessionIds = (rows: { session_id: string }[]) =>
      rows.map((s) => s.session_id).filter((id) => !id.startsWith(SEARCH_SESSION_PREFIX));

    // Doc count of the default dataset (the onboarding target). -1 = unknown.
    async function docCount(): Promise<number> {
      try {
        const datasets = await getDatasets(cogniInstance!);
        if (!Array.isArray(datasets) || datasets.length === 0) return 0;
        const target = datasets.find((d: { name?: string }) => d.name === "default_dataset") ?? datasets[0];
        if (!target?.id) return 0;
        const data = await getDatasetData(target.id, cogniInstance!);
        return Array.isArray(data) ? data.length : 0;
      } catch {
        return -1;
      }
    }

    async function check() {
      const [page, docs] = await Promise.all([
        listSessions(cogniInstance!, { range: "24h", limit: 50 }).catch(() => null),
        docCount(),
      ]);
      if (cancelled) return;
      const ids = page ? realSessionIds(page.sessions) : [];
      if (!primed) {
        ids.forEach((id) => baselineSessions.add(id));
        baselineDocs = docs;
        primed = true;
        return;
      }
      const newSession = ids.some((id) => !baselineSessions.has(id));
      const newDocs = docs >= 0 && baselineDocs >= 0 && docs > baselineDocs;
      if (newSession || newDocs) setConnectVerified(true);
    }

    check();
    const id = setInterval(check, 7000);
    return () => { cancelled = true; clearInterval(id); };
  }, [currentStep, cogniInstance, connectVerified]);

  function goToDashboard() {
    trackEvent({ pageName: "Onboarding", eventName: "onboarding_completed", additionalProperties: { destination: "dashboard", path: agent } });
    localStorage.setItem("cognee-onboarding-complete", "1");
    router.push("/dashboard");
  }

  const credsCard = {
    title: "Copy your API credentials",
    description: "Open a terminal and run these to point your agent at your Cognee memory.",
    node: <OnboardingInlineCode code={`export COGNEE_BASE_URL="${baseUrl}"`} toCopy={credsCode} loading={!credsReady} placeholder="Preparing your credentials…" />,
  };
  const allSetCard = {
    title: "You're all set",
    description: "The loop you just saw — upload, exit, reopen, and your agent still remembers — is the whole point. Here's why it works:",
    node: (
      <div style={{ display: "flex", flexDirection: "column", gap: 12, width: "100%" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 8, padding: 14, borderRadius: 10, background: "rgba(237,236,234,0.04)", border: "1px solid rgba(237,236,234,0.10)" }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: "#EDECEA" }}>What just happened</div>
          <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13, lineHeight: "19px", color: "rgba(237,236,234,0.6)" }}>
            <li>The Cognee plugin hooks into your agent&apos;s lifecycle — no curl or manual API calls — and captures your session as you work.</li>
            <li>When a session ends (e.g. you <strong style={{ color: "#EDECEA" }}>exit</strong>), it consolidates that session into your Cognee Cloud knowledge graph.</li>
            <li>On every new session, it automatically recalls your memory back from the cloud.</li>
          </ul>
          <div style={{ fontSize: 13, lineHeight: "19px", color: "rgba(237,236,234,0.6)" }}>
            That&apos;s why, after running <code>/exit</code> and reopening, your agent still knew what you uploaded — sessions are disposable; your memory isn&apos;t.
          </div>
        </div>
        <button onClick={goToDashboard} className="cursor-pointer" style={{ background: "#BC9BFF", border: "none", borderRadius: 8, padding: "11px 32px", fontSize: 14, fontWeight: 500, color: "#1e1e1c", letterSpacing: "-0.01em" }}>
          Go to Dashboard →
        </button>
      </div>
    ),
  };

  const cards: { title: string; description: string; node?: React.ReactNode }[] =
    agent === "claude-code"
      ? [
          credsCard,
          {
            title: "Install the Cognee plugin",
            description: "Run these in your terminal one at a time — register the Cognee marketplace, then install the memory plugin.",
            node: (
              <div style={{ display: "flex", flexDirection: "column", gap: 8, width: "100%" }}>
                <OnboardingInlineCode code={CLAUDE_MARKETPLACE_ADD} />
                <OnboardingInlineCode code={CLAUDE_PLUGIN_INSTALL} />
              </div>
            ),
          },
          {
            title: "Upload something to Cognee",
            description: "Pick one and paste it into Claude — it stores the content in your Cognee memory so you can recall it in the next step.",
            node: (
              <div style={{ display: "flex", flexDirection: "column", gap: 14, width: "100%" }}>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: "#EDECEA", marginBottom: 6 }}>Option A · Your existing memory</div>
                  <OnboardingInlineCode code={UPLOAD_MEMORY_PROMPT} />
                </div>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: "#EDECEA", marginBottom: 6 }}>Option B · Try it with a sample</div>
                  <OnboardingInlineCode code={UPLOAD_SAMPLE_PROMPT} />
                </div>
                <ConnectStatus verified={connectVerified} />
              </div>
            ),
          },
          {
            title: connectVerified ? "Connected — activity detected" : "Recall it from Cognee",
            description: connectVerified
              ? "We detected your new session in Cognee Cloud — you're connected. You're all set."
              : "Now ask Claude a question about what you just uploaded — it should answer from Cognee Cloud. (For the sample, use the question below.) This step completes on its own once your session shows up.",
            node: (
              <div style={{ display: "flex", flexDirection: "column", gap: 10, width: "100%" }}>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: "#EDECEA", marginBottom: 6 }}>First, run this to start a fresh session</div>
                  <OnboardingInlineCode code="/exit" />
                </div>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: "#EDECEA", marginBottom: 6 }}>Then ask</div>
                  <OnboardingInlineCode code={RECALL_SAMPLE_PROMPT} />
                </div>
                <ConnectStatus verified={connectVerified} />
              </div>
            ),
          },
          allSetCard,
        ]
      : [
          credsCard,
          {
            title: "Install the Cognee plugin",
            description: "Run these in your terminal one at a time — enable Codex hooks, register the Cognee marketplace, then install the memory plugin.",
            node: (
              <div style={{ display: "flex", flexDirection: "column", gap: 8, width: "100%" }}>
                <OnboardingInlineCode code={CODEX_HOOKS_ENABLE} />
                <OnboardingInlineCode code={CODEX_MARKETPLACE_ADD} />
                <OnboardingInlineCode code={CODEX_PLUGIN_INSTALL} />
              </div>
            ),
          },
          {
            title: "Upload something to Cognee",
            description: `Pick one and paste it into ${name} — it stores the content in your Cognee memory so you can recall it in the next step.`,
            node: (
              <div style={{ display: "flex", flexDirection: "column", gap: 14, width: "100%" }}>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: "#EDECEA", marginBottom: 6 }}>Option A · Your existing memory</div>
                  <OnboardingInlineCode code={UPLOAD_MEMORY_PROMPT} />
                </div>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: "#EDECEA", marginBottom: 6 }}>Option B · Try it with a sample</div>
                  <OnboardingInlineCode code={UPLOAD_SAMPLE_PROMPT} />
                </div>
                <ConnectStatus verified={connectVerified} />
              </div>
            ),
          },
          {
            title: connectVerified ? "Connected — activity detected" : "Recall it from Cognee",
            description: connectVerified
              ? "We detected your new session in Cognee Cloud — you're connected. You're all set."
              : `Now ask ${name} a question about what you just uploaded — it should answer from Cognee Cloud. (For the sample, use the question below.) This step completes on its own once your session shows up.`,
            node: (
              <div style={{ display: "flex", flexDirection: "column", gap: 10, width: "100%" }}>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: "#EDECEA", marginBottom: 6 }}>First, run this to start a fresh session</div>
                  <OnboardingInlineCode code="/exit" />
                </div>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: "#EDECEA", marginBottom: 6 }}>Then ask</div>
                  <OnboardingInlineCode code={RECALL_SAMPLE_PROMPT} />
                </div>
                <ConnectStatus verified={connectVerified} />
              </div>
            ),
          },
          allSetCard,
        ];

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

// ── Main ──

// Completes onboarding (persists flags) and navigates to the dashboard, where
// the skeleton takes over until the workspace is fully ready. Used by the
// "Skip to dashboard" escape so a slow pod/dataset never traps the user.
function skipToDashboard(router: ReturnType<typeof useRouter>) {
  try {
    localStorage.setItem("cognee-onboarding-complete", "1");
    sessionStorage.setItem("cognee-onboarding-skipped", "1");
  } catch { /* ignore */ }
  markOnboardingComplete().catch(() => {});
  trackEvent({ pageName: "Onboarding", eventName: "onboarding_skipped_provisioning" });
  router.push("/dashboard");
}

export default function OnboardingPage() {
  const { cogniInstance, isInitializing, serviceUrl, apiKey } = useCogniInstance();
  const router = useRouter();
  const searchParams = useSearchParams();
  const isServeMode = searchParams.get("source") === "serve";
  const initialStep = Math.min(Math.max(parseInt(searchParams.get("step") ?? "0", 10), 0), 3);
  const [step, setStep] = useState(initialStep);
  // Top-level branch: welcome → path selection → either the Claude/Codex card
  // flow ("agent") or the existing Company Brain upload flow ("company").
  // A ?step= deep link jumps straight into the Company Brain sub-steps.
  const [view, setView] = useState<"welcome" | "select" | "company" | "agent">(initialStep >= 1 ? "company" : "welcome");
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
      .catch(() => {});
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
      {view === "welcome" && <Step0 onNext={() => setView("select")} />}
      {view === "select" && (
        <StepSelect onSelect={(path) => {
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
              onClick={() => skipToDashboard(router)}
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
