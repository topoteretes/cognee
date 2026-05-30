"use client";

import React, { useEffect, useState, useRef } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import { useFilter } from "@/ui/layout/FilterContext";
import searchDataset from "@/modules/datasets/searchDataset";
import addData from "@/modules/ingestion/addData";
import cognifyDataset from "@/modules/datasets/cognifyDataset";
import createDataset from "@/modules/datasets/createDataset";
import pollDatasetStatus from "@/modules/datasets/pollDatasetStatus";
import { loadGraphModelsConfig, findModelForDataset, findPromptForDataset, findOntologyForDataset } from "@/modules/configuration/userConfiguration";
import { toCleanSchema } from "@/modules/graphModels/types";
import { toGraphModelSchema } from "@/modules/graphModels/toGraphModelSchema";
import { listSessions, getSessionStats } from "@/modules/sessions/getSessions";
import getDatasetGraph from "@/modules/datasets/getDatasetGraph";
import { notifications } from "@mantine/notifications";
import { Tooltip } from "@mantine/core";
import { trackEvent, TrackPageView } from "@/modules/analytics";
import QuickstartCards from "@/ui/elements/QuickstartCards";


interface PipelineRun { id: string; pipeline_name: string; status: string; dataset_id: string | null; dataset_name: string | null; owner_email: string | null; created_at: string | null; pipeline_run_id: string | null }

function timeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

function pipelineLabel(name: string): string {
  if (name.includes("cognify")) return "cognee.cognify";
  if (name.includes("add")) return "cognee.add";
  if (name.includes("search")) return "cognee.search";
  return name;
}

function ownerDisplayName(email: string | null): string {
  if (!email) return "System";
  if (email.endsWith("@cognee.agent")) {
    const local = email.split("@")[0];
    const parts = local.split("-");
    const shortId = parts.pop() || "";
    const type = parts.join(" ");
    return `${type} ${shortId}`;
  }
  if (email === "default_user@example.com") return "You";
  return email.split("@")[0];
}

function statusDot(status: string): string {
  if (status.includes("COMPLETED")) return "#22C55E";
  if (status.includes("ERRORED")) return "#EF4444";
  if (status.includes("STARTED") || status.includes("INITIATED")) return "#F59E0B";
  return "#A1A1AA";
}

type Range = "24h" | "7d" | "30d";

export default function OverviewPage() {
  const { cogniInstance, isInitializing } = useCogniInstance();
  const { agents, datasets, selectedAgent, setSelectedDataset, refreshDatasets, loading: filterLoading } = useFilter();
  const [runs, setRuns] = useState<PipelineRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [showDatasetPicker, setShowDatasetPicker] = useState(false);
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [range, setRange] = useState<Range>("24h");
  const [sessions, setSessions] = useState<import("@/modules/sessions/getSessions").SessionRow[]>([]);
  const [sessionStats, setSessionStats] = useState<import("@/modules/sessions/getSessions").SessionStats | null>(null);
  const [activityFilter, setActivityFilter] = useState<"all" | "sessions" | "datasets" | "failed">("all");
  const [graphNodes, setGraphNodes] = useState<number | null>(null);
  const [graphEdges, setGraphEdges] = useState<number | null>(null);
  const [showSdkModal, setShowSdkModal] = useState(false);
  const [showUploadDoneModal, setShowUploadDoneModal] = useState<{ datasetName: string; datasetId: string } | null>(null);
  const [bannerDismissed, setBannerDismissed] = useState(() =>
    typeof window !== "undefined" && !!sessionStorage.getItem("cognee-welcome-banner-dismissed")
  );
  const [animKey, setAnimKey] = useState(0);
  const animTriggered = useRef(false);
  const uploadInputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();

  // Trigger onboarding stream animation once after loading completes (fires on every page open)
  useEffect(() => {
    if (loading || isInitializing || filterLoading) return;
    if (animTriggered.current) return;
    animTriggered.current = true;
    setAnimKey(k => k + 1);
  }, [loading, isInitializing, filterLoading]);

  async function uploadToDataset(ds: { id: string; name: string }, files: File[]) {
    if (!cogniInstance) return;
    setIsUploading(true);
    try {
      await addData({ id: ds.id }, files, cogniInstance);
      trackEvent({ pageName: "Dashboard", eventName: "dashboard_files_uploaded", additionalProperties: { dataset_id: ds.id, dataset_name: ds.name, file_count: String(files.length) } });
      notifications.show({ title: `Files uploaded to "${ds.name}"`, message: `${files.length} file(s) added. Cognify running.`, color: "blue", autoClose: 5000 });
      // Load graph model, custom prompt, and ontology assignments for this dataset
      const cfg = await loadGraphModelsConfig(cogniInstance);
      const cognifyOpts: { graphModel?: object; customPrompt?: string; ontologyKey?: string[] } = {};
      const assignedModel = findModelForDataset(cfg.models, ds.id);
      if (assignedModel) {
        const cleanSchema = toCleanSchema(assignedModel.schema);
        cognifyOpts.graphModel = toGraphModelSchema(cleanSchema);
      }
      const promptName = findPromptForDataset(cfg.promptAssignments ?? {}, ds.id);
      if (promptName && cfg.customPrompts?.[promptName]) {
        cognifyOpts.customPrompt = cfg.customPrompts[promptName];
      }
      const ontologyKey = findOntologyForDataset(cfg.ontologyAssignments ?? {}, ds.id);
      if (ontologyKey) {
        cognifyOpts.ontologyKey = [ontologyKey];
      }
      await cognifyDataset({ id: ds.id, name: ds.name, data: [], status: "" }, cogniInstance, cognifyOpts);
      await pollDatasetStatus(ds.id, cogniInstance, { intervalMs: 5000 });
      refreshDatasets();
      setShowUploadDoneModal({ datasetName: ds.name, datasetId: ds.id });
    } catch (err) {
      console.error("Dashboard upload failed:", err);
      notifications.show({ title: "Upload failed", message: err instanceof Error ? err.message : String(err), color: "red" });
    } finally {
      setIsUploading(false);
    }
  }

  async function handleDashboardUpload(e: React.ChangeEvent<HTMLInputElement>) {
    if (!cogniInstance || !e.target.files?.length) return;
    const files = Array.from(e.target.files);
    e.target.value = "";

    // If only one dataset exists, upload to it
    if (datasets.length === 1) {
      await uploadToDataset(datasets[0], files);
      return;
    }

    // If no datasets exist, create default and upload
    if (datasets.length === 0) {
      const ds = await createDataset({ name: "default_dataset" }, cogniInstance);
      refreshDatasets();
      await uploadToDataset(ds, files);
      return;
    }

    // Multiple datasets, none selected — show picker
    setPendingFiles(files);
    setShowDatasetPicker(true);
  }

  async function handlePickDataset(ds: { id: string; name: string }) {
    setShowDatasetPicker(false);
    setSelectedDataset(ds);
    trackEvent({ pageName: "Dashboard", eventName: "dashboard_dataset_picked", additionalProperties: { dataset_id: ds.id, dataset_name: ds.name } });
    await uploadToDataset(ds, pendingFiles);
    setPendingFiles([]);
  }

  useEffect(() => {
    if (!cogniInstance || isInitializing || filterLoading) return;

    let cancelled = false;

    // Poll telemetry + sessions every 15s. The Activity & Memory
    // table binds to `sessions`, so each tick refreshes it.
    function fetchTelemetry() {
      return Promise.all([
        cogniInstance!
          .fetch("/v1/activity/pipeline-runs")
          .then((r) => (r.ok ? r.json() : []))
          .catch(() => []),
        listSessions(cogniInstance!, { range, limit: 20 }),
        getSessionStats(cogniInstance!, range),
      ]).then(([runData, sessionsPage, stats]) => {
        if (cancelled) return;
        setRuns(Array.isArray(runData) ? runData : []);
        setSessions(sessionsPage?.sessions ?? []);
        setSessionStats(stats);
      });
    }

    fetchTelemetry()
      .then(() => {
        if (cancelled) return;
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    const interval = setInterval(fetchTelemetry, 15000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [cogniInstance, isInitializing, filterLoading, range]);

  // Fetch graph node/edge counts whenever datasets or instance change
  useEffect(() => {
    if (!cogniInstance || !datasets.length) return;
    let cancelled = false;
    Promise.all(
      datasets.map((ds) => getDatasetGraph(ds, cogniInstance).catch(() => null))
    ).then((graphs) => {
      if (cancelled) return;
      let totalNodes = 0;
      let totalEdges = 0;
      for (const g of graphs) {
        if (g && Array.isArray(g.nodes)) totalNodes += g.nodes.length;
        if (g && Array.isArray(g.edges)) totalEdges += g.edges.length;
      }
      setGraphNodes(totalNodes);
      setGraphEdges(totalEdges);
    }).catch(() => {});
    return () => { cancelled = true; };
  }, [cogniInstance, datasets]);

  // Onboarding redirect: send fresh users to onboarding until they have pipeline activity
  useEffect(() => {
    if (loading || filterLoading) return;
    if (
      runs.length === 0 &&
      !localStorage.getItem("cognee-onboarding-complete") &&
      !sessionStorage.getItem("cognee-onboarding-skipped")
    ) {
      router.replace("/onboarding");
    }
  }, [loading, filterLoading, runs, router]);

  if (loading || isInitializing || filterLoading) {
    return <><TrackPageView page="Dashboard" /><div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", background: "#FFFFFF" }}><video src="/videos/mascot-waiting.mp4" autoPlay loop muted playsInline style={{ width: 200, height: "auto" }} /></div></>;
  }

  // Deduplicate runs
  const latestRuns: PipelineRun[] = [];
  const seen = new Set<string>();
  for (const r of runs) {
    const key = r.pipeline_run_id || r.id;
    if (!seen.has(key)) { seen.add(key); latestRuns.push(r); }
  }

  const filteredRuns = latestRuns;
  const filteredDatasets = datasets;

  const apiCalls = filteredRuns.length;
  const errorCount = filteredRuns.filter((r) => r.status.includes("ERRORED")).length;

  const apiCallsTrend = apiCalls > 0 ? "+12%" : "steady";
  const connectedAgents = agents.filter((a) => a.is_agent && !a.is_default);
  const liveAgentIds = new Set(
    sessions.filter((s) => s.effective_status === "running").map((s) => s.user_id)
  );
  const liveAgents = connectedAgents.filter((a) => liveAgentIds.has(a.id) || a.status === "LIVE");
  const hasSessionData = sessionStats != null && (sessionStats.completed > 0 || sessionStats.failed > 0);
  const successRate = hasSessionData ? sessionStats!.success_rate * 100 : (apiCalls > 0 ? ((1 - errorCount / apiCalls) * 100) : 100);

  // Merge sessions + datasets into one activity feed, sort by recency.
  type ActivityRow =
    | { kind: "session"; session: import("@/modules/sessions/getSessions").SessionRow; agent: typeof agents[number] | null; dataset: typeof datasets[number] | null; timeStr: number }
    | { kind: "dataset"; dataset: typeof datasets[number] & { _ds_data?: unknown[]; updated_at?: string | null }; timeStr: number };

  const activity: ActivityRow[] = [];
  for (const s of sessions) {
    const agent = agents.find((a) => a.id === s.user_id) || null;
    const ds = s.dataset_id ? datasets.find((d) => d.id === s.dataset_id) || null : null;
    const ts = s.last_activity_at ? new Date(s.last_activity_at).getTime() : 0;
    activity.push({ kind: "session", session: s, agent, dataset: ds, timeStr: ts });
  }
  for (const d of filteredDatasets) {
    // Datasets don't currently carry updated_at via the list endpoint; fall back to recent runs.
    const latestRun = latestRuns.find((r) => r.dataset_id === d.id && r.created_at);
    const ts = latestRun?.created_at ? new Date(latestRun.created_at).getTime() : 0;
    activity.push({ kind: "dataset", dataset: d, timeStr: ts });
  }
  activity.sort((a, b) => b.timeStr - a.timeStr);

  const failedSessions = sessions.filter((s) => s.effective_status === "failed" || s.error_count > 0).length;
  const sessionCount = sessions.length;

  const filteredActivity = activity.filter((row) => {
    if (activityFilter === "all") return true;
    if (activityFilter === "sessions") return row.kind === "session";
    if (activityFilter === "datasets") return row.kind === "dataset";
    if (activityFilter === "failed") {
      return row.kind === "session" && (row.session.effective_status === "failed" || row.session.error_count > 0);
    }
    return true;
  });

  const greeting = greetingForTime();

  // Onboarding completion state — shared between banner and progress bar
  const onboardingSteps = [
    { done: true },
    { done: runs.length > 0 || (graphNodes ?? 0) > 0 },
    { done: (graphNodes ?? 0) > 0 || runs.some(r => r.status?.includes("COMPLETED")) },
    { done: agents.some(a => a.is_agent) },
  ];
  const onboardingCompletedCount = onboardingSteps.filter(s => s.done).length;
  // Onboarding is complete when the 3 core steps are done (account + data + graph),
  // when all 4 steps including agent connection are done, or when the wizard was
  // finished (localStorage flag). Agent connection (step 4) is optional — not
  // requiring it prevents the banner from reappearing on every new login.
  const onboardingComplete = onboardingSteps.slice(0, 3).every(s => s.done)
    || onboardingCompletedCount === onboardingSteps.length
    || !!localStorage.getItem("cognee-onboarding-complete");
  const onboardingNextIncomplete = onboardingSteps.findIndex(s => !s.done);
  const onboardingNextStep = onboardingNextIncomplete === -1 ? 4 : onboardingNextIncomplete + 1;
  // The dashboard progress steps are offset by 1 from the wizard steps (step 1 = "account created" = always done).
  const wizardNextStep = Math.max(1, onboardingNextStep - 1);

  return (
    <div style={{ backgroundColor: "#FAFAFA", minHeight: "100%", padding: 32, display: "flex", flexDirection: "column", gap: 40, fontFamily: "system-ui, sans-serif" }}>
      {/* Hidden file input for dashboard upload */}
      <input ref={uploadInputRef} type="file" multiple accept=".pdf,.csv,.txt,.md,.json,.docx" className="hidden" onChange={handleDashboardUpload} />

      {/* Dataset picker modal */}
      {showDatasetPicker && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.3)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={() => { setShowDatasetPicker(false); setPendingFiles([]); }}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "#fff", borderRadius: 12, padding: 24, width: 420, display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 16px 48px rgba(0,0,0,0.12)" }}>
            <h2 style={{ fontSize: 18, fontWeight: 600, color: "#18181B", margin: 0 }}>Upload to which brain?</h2>
            <p style={{ fontSize: 13, color: "#71717A", margin: 0 }}>
              {pendingFiles.length} file{pendingFiles.length !== 1 ? "s" : ""} selected. Choose a brain to upload to.
            </p>
            <div style={{ display: "flex", flexDirection: "column", gap: 4, maxHeight: 300, overflow: "auto" }}>
              {datasets.map((ds) => (
                <button key={ds.id} onClick={() => handlePickDataset(ds)} className="cursor-pointer hover:bg-cognee-hover" style={{ display: "flex", alignItems: "center", gap: 10, padding: "12px 14px", borderRadius: 8, border: "1px solid #F4F4F5", background: "none", textAlign: "left", fontFamily: "inherit", width: "100%" }}>
                  <div style={{ width: 8, height: 8, borderRadius: 2, background: "#6510F4", flexShrink: 0 }} />
                  <span style={{ fontSize: 14, fontWeight: 500, color: "#18181B" }}>{ds.name}</span>
                </button>
              ))}
            </div>
            <div style={{ display: "flex", justifyContent: "flex-end" }}>
              <button onClick={() => { setShowDatasetPicker(false); setPendingFiles([]); }} className="cursor-pointer" style={{ background: "#fff", border: "1px solid #E4E4E7", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "#3F3F46", fontFamily: "inherit" }}>Cancel</button>
            </div>
          </div>
        </div>
      )}

      {/* Welcome banner — shown only while onboarding is incomplete */}
      {!bannerDismissed && !onboardingComplete && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.35)", zIndex: 400, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={() => { setBannerDismissed(true); sessionStorage.setItem("cognee-welcome-banner-dismissed", "1"); }}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "#FFFFFF", borderRadius: 20, width: "min(860px, 92vw)", overflow: "hidden", display: "flex", boxShadow: "0 24px 64px rgba(0,0,0,0.16)", position: "relative" }}>
            {/* Close */}
            <button onClick={() => { setBannerDismissed(true); sessionStorage.setItem("cognee-welcome-banner-dismissed", "1"); }} style={{ position: "absolute", top: 20, right: 20, background: "none", border: "none", cursor: "pointer", color: "#A1A1AA", fontSize: 18, lineHeight: 1, zIndex: 1 }}>&#10005;</button>

            {/* Left: text + button */}
            <div style={{ flex: "0 0 50%", padding: "52px 48px 52px 52px", display: "flex", flexDirection: "column", justifyContent: "space-between", gap: 32 }}>
              <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
                <h2 style={{ margin: 0, fontSize: 52, fontWeight: 300, lineHeight: 1.05, letterSpacing: "-0.02em", color: "#18181B", fontFamily: '"TWK Lausanne", system-ui, sans-serif' }}>
                  {onboardingNextStep <= 2 ? "Let's start building\nthe graph!" : "Let's finish\nsetting up!"}
                </h2>
                <p style={{ margin: 0, fontSize: 16, fontWeight: 400, color: "#3F3F46", lineHeight: 1.6, fontFamily: '"Inter", system-ui, sans-serif' }}>
                  {onboardingNextStep <= 2
                    ? <>Upload any document to see<br />what Cognee can do.</>
                    : onboardingNextStep === 3
                    ? <>Your data is uploaded. Now let<br />Cognee build your knowledge graph.</>
                    : <>Your graph is ready. Connect an<br />agent to put it to work.</>}
                </p>
              </div>
              <button
                onClick={() => { setBannerDismissed(true); sessionStorage.setItem("cognee-welcome-banner-dismissed", "1"); router.push(`/onboarding?step=${wizardNextStep}`); }}
                style={{ background: "#6510F4", color: "#FFFFFF", border: "none", borderRadius: 100, padding: "16px 0", fontSize: 16, fontWeight: 400, fontFamily: '"Inter", system-ui, sans-serif', cursor: "pointer", width: 220, letterSpacing: "0.01em" }}
              >
                {wizardNextStep <= 1 ? "Start onboarding" : `Continue — step ${wizardNextStep} of 4`}
              </button>
            </div>

            {/* Right: mascot video */}
            <div style={{ flex: "0 0 50%", background: "#FFFFFF", display: "flex", alignItems: "center", justifyContent: "center", minHeight: 340 }}>
              <video src="/videos/mascot-purple-glow.mp4" autoPlay loop muted playsInline style={{ width: "80%", height: "auto" }} />
            </div>
          </div>
        </div>
      )}

      {/* Onboarding progress bar — only when not complete */}
      {!onboardingComplete && (() => {
        const steps = onboardingSteps;
        const completed = onboardingCompletedCount;
        const nextStep = wizardNextStep;
        // Stream timing: circle=0.55s, line=0.35s, flowing left→right
        const CDUR = 0.55, LDUR = 0.35, START = 0.15;
        const circleDelay = (i: number) => START + i * (CDUR + LDUR);
        const lineDelay = (i: number) => START + (i - 1) * (CDUR + LDUR) + CDUR;
        return (
          <div onClick={() => router.push(`/onboarding?step=${nextStep}`)} style={{ position: "sticky", top: 0, zIndex: 50, margin: "-32px -32px 0 -32px", padding: "14px 32px", background: "#FFFFFF", borderBottom: "1px solid #E4E4E7", display: "flex", alignItems: "center", justifyContent: "space-between", cursor: "pointer" }}>
            <span style={{ fontSize: 20, fontWeight: 300, color: "#1A1A1A", fontFamily: '"TWK Lausanne", system-ui, sans-serif', whiteSpace: "nowrap" }}>
              Onboarding {completed}/{steps.length} complete
            </span>
            <div style={{ display: "flex", alignItems: "center" }}>
              {steps.map((s, i) => (
                <span key={i} style={{ display: "flex", alignItems: "center" }}>
                  {/* Connector line: gray base + purple fill wipes left→right */}
                  {i > 0 && (
                    <span style={{ position: "relative", width: 28, height: 2, background: "#D4D4D8", flexShrink: 0, overflow: "hidden" }}>
                      {s.done && animKey > 0 && (
                        <span key={`line-${i}-${animKey}`} style={{ position: "absolute", inset: 0, background: "#6510F4", animation: `streamFillLine ${LDUR}s cubic-bezier(0.4,0,0.2,1) ${lineDelay(i)}s backwards` }} />
                      )}
                    </span>
                  )}
                  {/* Circle: gray border, transparent bg; purple fill wipes left→right when done */}
                  <span style={{ position: "relative", width: 48, height: 48, borderRadius: "50%", display: "inline-flex", alignItems: "center", justifyContent: "center", flexShrink: 0, border: `2px solid ${s.done ? "#6510F4" : "#ADADAD"}`, overflow: "hidden", background: "transparent", transition: s.done ? `border-color 0.2s ease ${circleDelay(i) + CDUR * 0.3}s` : "none" }}>
                    {s.done && animKey > 0 && (
                      <span key={`fill-${i}-${animKey}`} style={{ position: "absolute", inset: 0, background: "#6510F4", animation: `streamFill ${CDUR}s cubic-bezier(0.4,0,0.2,1) ${circleDelay(i)}s backwards` }} />
                    )}
                    <span style={{ position: "relative", zIndex: 1, fontSize: 15, fontWeight: 600, color: s.done ? "#FFFFFF" : "#8A8A8A", fontFamily: '"TWK Lausanne", system-ui, sans-serif', transition: s.done && animKey > 0 ? `color 0.2s ease ${circleDelay(i) + CDUR * 0.6}s` : "none" }}>{i + 1}</span>
                  </span>
                </span>
              ))}
            </div>
            <button onClick={(e) => { e.stopPropagation(); router.push(`/onboarding?step=${nextStep}`); }} style={{ background: "#6510F4", color: "#FFFFFF", border: "none", borderRadius: 6, padding: "8px 16px", fontSize: 13, fontWeight: 500, fontFamily: '"TWK Lausanne", system-ui, sans-serif', cursor: "pointer", whiteSpace: "nowrap" }}>
              {completed <= 1 ? "Start the onboarding" : "Finish the onboarding"}
            </button>
          </div>
        );
      })()}

      <style>{`
        @keyframes streamFill {
          from { clip-path: inset(0 100% 0 0 round 999px); }
          to   { clip-path: inset(0 0% 0 0 round 999px); }
        }
        @keyframes streamFillLine {
          from { clip-path: inset(0 100% 0 0); }
          to   { clip-path: inset(0 0% 0 0); }
        }
      `}</style>

      {/* Greeting */}
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <div style={{ color: "#A1A1AA", fontSize: 11, letterSpacing: "0.14em", textTransform: "uppercase", lineHeight: "14px" }}>
          Overview · {formatToday()}
        </div>
        <div style={{ color: "#18181B", fontSize: 32, letterSpacing: "-0.02em", lineHeight: "36px" }}>
          {greeting}{selectedAgent ? `, ${ownerDisplayName(selectedAgent.email)}` : ""}
        </div>
        {selectedAgent && (
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 4 }}>
            <span style={{ fontSize: 12, color: "#A1A1AA" }}>Showing data for:</span>
            <span style={{ background: "#F0EDFF", borderRadius: 4, padding: "2px 8px", fontSize: 12, fontWeight: 500, color: "#6510F4" }}>{selectedAgent.agent_type}</span>
          </div>
        )}
      </div>

      {/* Integrate cognee into your system */}
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <span style={{ fontSize: 20, fontWeight: 300, color: "#18181B", fontFamily: '"TWK Lausanne", system-ui, sans-serif' }}>Integrate cognee into your system</span>
        <QuickstartCards />
      </div>

      {/* Stat cards */}
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div style={{ display: "flex", justifyContent: "flex-end" }}>
        <RangePicker value={range} onChange={setRange} />
      </div>
      <div style={{ display: "flex", gap: 16 }}>
        {/* Merged metrics box — takes 3 of the 5 flex units */}
        <div style={{ flex: 3, border: "1px solid #E4E4E7", borderRadius: 12, overflow: "hidden", display: "flex" }}>
          {/* Column 1 */}
          <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
            <div style={{ padding: "20px 24px", display: "flex", flexDirection: "column", gap: 4 }}>
              <span style={{ color: "#71717A", fontSize: 11, letterSpacing: "0.14em", lineHeight: "14px", textTransform: "uppercase" }}>Connected agents</span>
              <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
                <span style={{ color: "#18181B", fontSize: 32, letterSpacing: "-0.02em", lineHeight: "36px", fontVariantNumeric: "tabular-nums" }}>{connectedAgents.length}</span>
                <span style={{ color: "#71717A", fontSize: 13, lineHeight: "16px" }}>{liveAgents.length > 0 ? `${liveAgents.length} live now` : "none live"}</span>
              </div>
            </div>
            <div style={{ height: 1, background: "#E4E4E7" }} />
            <div style={{ padding: "20px 24px", display: "flex", flexDirection: "column", gap: 4 }}>
              <span style={{ color: "#71717A", fontSize: 11, letterSpacing: "0.14em", lineHeight: "14px", textTransform: "uppercase" }}>API calls, {range}</span>
              <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
                <span style={{ color: "#18181B", fontSize: 32, letterSpacing: "-0.02em", lineHeight: "36px", fontVariantNumeric: "tabular-nums" }}>{apiCalls.toLocaleString()}</span>
                <span style={{ color: "#71717A", fontSize: 13, lineHeight: "16px" }}>across {sessionCount.toLocaleString()} session{sessionCount !== 1 ? "s" : ""}</span>
              </div>
            </div>
          </div>
          {/* Vertical divider */}
          <div style={{ width: 1, background: "#E4E4E7", flexShrink: 0 }} />
          {/* Column 2 */}
          <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
            <div style={{ padding: "20px 24px", display: "flex", flexDirection: "column", gap: 4 }}>
              <span style={{ color: "#71717A", fontSize: 11, letterSpacing: "0.14em", lineHeight: "14px", textTransform: "uppercase" }}>Brains</span>
              <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
                <span style={{ color: "#18181B", fontSize: 32, letterSpacing: "-0.02em", lineHeight: "36px", fontVariantNumeric: "tabular-nums" }}>{filteredDatasets.length.toLocaleString()}</span>
                <span style={{ color: "#71717A", fontSize: 13, lineHeight: "16px" }}>{sessionCount.toLocaleString()} session{sessionCount !== 1 ? "s" : ""}</span>
              </div>
            </div>
            <div style={{ height: 1, background: "#E4E4E7" }} />
            <div style={{ padding: "20px 24px", display: "flex", flexDirection: "column", gap: 4 }}>
              <span style={{ color: "#71717A", fontSize: 11, letterSpacing: "0.14em", lineHeight: "14px", textTransform: "uppercase" }}>Success rate</span>
              <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
                <span style={{ color: "#18181B", fontSize: 32, letterSpacing: "-0.02em", lineHeight: "36px", fontVariantNumeric: "tabular-nums" }}>{successRate.toFixed(1)}%</span>
                <span style={{ color: successRate >= 95 ? "#16A34A" : successRate >= 80 ? "#D97706" : "#DC2626", fontSize: 13, lineHeight: "16px" }}>
                  {successRate >= 95 ? "healthy" : successRate >= 80 ? "degraded" : "critical"}
                </span>
              </div>
            </div>
          </div>
          {/* Vertical divider */}
          <div style={{ width: 1, background: "#E4E4E7", flexShrink: 0 }} />
          {/* Column 3 */}
          <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
            <div style={{ padding: "20px 24px", display: "flex", flexDirection: "column", gap: 4 }}>
              <span style={{ color: "#71717A", fontSize: 11, letterSpacing: "0.14em", lineHeight: "14px", textTransform: "uppercase" }}>Nodes</span>
              <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
                <span style={{ color: "#18181B", fontSize: 32, letterSpacing: "-0.02em", lineHeight: "36px", fontVariantNumeric: "tabular-nums" }}>{graphNodes != null ? graphNodes.toLocaleString() : "—"}</span>
              </div>
            </div>
            <div style={{ height: 1, background: "#E4E4E7" }} />
            <div style={{ padding: "20px 24px", display: "flex", flexDirection: "column", gap: 4 }}>
              <span style={{ color: "#71717A", fontSize: 11, letterSpacing: "0.14em", lineHeight: "14px", textTransform: "uppercase" }}>Edges</span>
              <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
                <span style={{ color: "#18181B", fontSize: 32, letterSpacing: "-0.02em", lineHeight: "36px", fontVariantNumeric: "tabular-nums" }}>{graphEdges != null ? graphEdges.toLocaleString() : "—"}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Activity & Memory — takes 2 of the 5 flex units */}
        <div style={{ flex: 2, border: "1px solid #E4E4E7", borderRadius: 12, padding: "20px 24px", display: "flex", flexDirection: "column", gap: 16 }}>
          <span style={{ color: "#71717A", fontSize: 11, letterSpacing: "0.14em", lineHeight: "14px", textTransform: "uppercase" }}>Activity & Memory</span>
          <div style={{ display: "flex", flexDirection: "column", flex: 1 }}>
            {filteredActivity.slice(0, 3).map((row, i) => (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, padding: "9px 0", borderBottom: i < Math.min(filteredActivity.length, 3) - 1 ? "1px solid #F4F4F5" : "none", minWidth: 0 }}>
                {row.kind === "session" ? (
                  <>
                    <span style={{ width: 6, height: 6, borderRadius: "50%", flexShrink: 0, background: row.session.effective_status === "failed" ? "#EF4444" : row.session.effective_status === "running" ? "#3B82F6" : "#22C55E" }} />
                    <span style={{ fontSize: 11, fontWeight: 500, color: row.session.effective_status === "failed" ? "#EF4444" : row.session.effective_status === "running" ? "#3B82F6" : "#16A34A", flexShrink: 0, textTransform: "capitalize" }}>{row.session.effective_status || "session"}</span>
                    <span style={{ fontSize: 12, color: "#18181B", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1 }}>{labelFromSession(row.session)}</span>
                    <span style={{ fontSize: 11, color: "#A1A1AA", flexShrink: 0, whiteSpace: "nowrap" }}>{row.session.last_activity_at ? timeAgo(row.session.last_activity_at) : ""}</span>
                  </>
                ) : (
                  <>
                    <span style={{ width: 6, height: 6, borderRadius: "50%", flexShrink: 0, background: "#6510F4" }} />
                    <span style={{ fontSize: 11, fontWeight: 500, color: "#16A34A", flexShrink: 0 }}>ready</span>
                    <span style={{ fontSize: 12, color: "#18181B", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1 }}>{row.dataset.name}</span>
                    <span style={{ fontSize: 11, color: "#A1A1AA", flexShrink: 0, whiteSpace: "nowrap" }}>{row.timeStr ? timeAgo(new Date(row.timeStr).toISOString()) : ""}</span>
                  </>
                )}
              </div>
            ))}
            {filteredActivity.length === 0 && (
              <span style={{ fontSize: 13, color: "#A1A1AA" }}>No activity yet.</span>
            )}
          </div>
        </div>
      </div>
      </div>

      {/* SDK modal */}
      {showSdkModal && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.3)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={() => setShowSdkModal(false)}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "#fff", borderRadius: 12, width: 520, boxShadow: "0 16px 48px rgba(0,0,0,0.12)", overflow: "hidden", fontFamily: '"Inter", system-ui, sans-serif' }}>
            {/* Header */}
            <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "16px 20px", borderBottom: "1px solid #E4E4E7" }}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none"><path d="M7 8l5 4 5-4M7 16l5-4 5 4" stroke="#6510F4" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
              <span style={{ fontSize: 14, fontWeight: 600, color: "#18181B" }}>Connect with Python SDK</span>
              <button onClick={() => setShowSdkModal(false)} className="cursor-pointer" style={{ marginLeft: "auto", background: "none", border: "none", color: "#A1A1AA", fontSize: 16, padding: 2 }}>&#10005;</button>
            </div>
            {/* Steps */}
            <div style={{ padding: "16px 20px", display: "flex", flexDirection: "column", gap: 16 }}>
              {/* Step 1 */}
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <div className="flex items-center justify-center flex-shrink-0 rounded-full" style={{ width: 24, height: 24, background: "#F0EDFF" }}>
                  <span style={{ color: "#6510F4", fontSize: 12, fontWeight: 600 }}>1</span>
                </div>
                <span style={{ fontSize: 13, fontWeight: 500, color: "#18181B" }}>Install Cognee</span>
              </div>
              <div className="flex items-center justify-between" style={{ background: "#18181B", borderRadius: 8, padding: "10px 16px" }}>
                <span style={{ fontSize: 13, color: "#A1A1AA", fontFamily: '"Fira Code", monospace' }}>pip install cognee</span>
                <CopyButton text="pip install cognee" />
              </div>

              <div style={{ height: 1, background: "#E4E4E7" }} />

              {/* Step 2 */}
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <div className="flex items-center justify-center flex-shrink-0 rounded-full" style={{ width: 24, height: 24, background: "#F0EDFF" }}>
                  <span style={{ color: "#6510F4", fontSize: 12, fontWeight: 600 }}>2</span>
                </div>
                <span style={{ fontSize: 13, fontWeight: 500, color: "#18181B" }}>Get your API Base URL and API key</span>
              </div>
              <Link href="/api-keys" style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12, color: "#6510F4", textDecoration: "none", fontWeight: 500 }}>
                Go to API Keys
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#6510F4" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14" /><path d="M12 5l7 7-7 7" /></svg>
              </Link>

              <div style={{ height: 1, background: "#E4E4E7" }} />

              {/* Step 3 */}
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <div className="flex items-center justify-center flex-shrink-0 rounded-full" style={{ width: 24, height: 24, background: "#F0EDFF" }}>
                  <span style={{ color: "#6510F4", fontSize: 12, fontWeight: 600 }}>3</span>
                </div>
                <span style={{ fontSize: 13, fontWeight: 500, color: "#18181B" }}>Connect your agent</span>
              </div>
              <div style={{ background: "#18181B", borderRadius: 8, padding: "12px 16px", position: "relative" }}>
                <pre style={{ margin: 0, fontSize: 12, lineHeight: "20px", fontFamily: '"Fira Code", monospace', color: "#A1A1AA" }}>
{`import asyncio
import cognee

async def main():
    await cognee.serve(
        url="<your-api-base-url>",
        api_key="<your-api-key>"
    )

    await cognee.remember("Your data here")

    results = await cognee.recall("What do we know?")
    print(results)

asyncio.run(main())`}
                </pre>
                <span style={{ position: "absolute", top: 12, right: 16 }}>
                  <CopyButton text={`import asyncio\nimport cognee\n\nasync def main():\n    await cognee.serve(\n        url="<your-api-base-url>",\n        api_key="<your-api-key>"\n    )\n\n    await cognee.remember("Your data here")\n\n    results = await cognee.recall("What do we know?")\n    print(results)\n\nasyncio.run(main())`} />
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Upload done modal */}
      {showUploadDoneModal && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.3)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={() => setShowUploadDoneModal(null)}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "#fff", borderRadius: 12, padding: 28, width: 440, display: "flex", flexDirection: "column", gap: 20, boxShadow: "0 16px 48px rgba(0,0,0,0.12)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div style={{ width: 36, height: 36, borderRadius: 8, background: "#DCFCE7", display: "flex", alignItems: "center", justifyContent: "center" }}>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#16A34A" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 6L9 17l-5-5" /></svg>
              </div>
              <div>
                <h2 style={{ fontSize: 17, fontWeight: 600, color: "#18181B", margin: 0 }}>Knowledge graph built</h2>
                <p style={{ fontSize: 13, color: "#71717A", margin: 0 }}>&ldquo;{showUploadDoneModal.datasetName}&rdquo; is now searchable.</p>
              </div>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <button
                onClick={() => { setShowUploadDoneModal(null); router.push("/search"); }}
                className="cursor-pointer hover:bg-cognee-hover"
                style={{ display: "flex", alignItems: "center", gap: 10, padding: "12px 14px", borderRadius: 8, border: "1px solid #E4E4E7", background: "#fff", textAlign: "left", fontFamily: "inherit", width: "100%" }}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#6510F4" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" /></svg>
                <div>
                  <div style={{ fontSize: 14, fontWeight: 500, color: "#18181B" }}>Search your data</div>
                  <div style={{ fontSize: 12, color: "#71717A" }}>Ask questions about your knowledge graph</div>
                </div>
              </button>
              <button
                onClick={() => { setShowUploadDoneModal(null); router.push(`/datasets/${showUploadDoneModal.datasetId}`); }}
                className="cursor-pointer hover:bg-cognee-hover"
                style={{ display: "flex", alignItems: "center", gap: 10, padding: "12px 14px", borderRadius: 8, border: "1px solid #E4E4E7", background: "#fff", textAlign: "left", fontFamily: "inherit", width: "100%" }}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#6510F4" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="6" cy="6" r="3" /><circle cx="18" cy="6" r="3" /><circle cx="12" cy="18" r="3" /><line x1="8.5" y1="7.5" x2="10.5" y2="16" /><line x1="15.5" y1="7.5" x2="13.5" y2="16" /></svg>
                <div>
                  <div style={{ fontSize: 14, fontWeight: 500, color: "#18181B" }}>Inspect the knowledge graph</div>
                  <div style={{ fontSize: 12, color: "#71717A" }}>View entities and relationships</div>
                </div>
              </button>
              <button
                onClick={() => { setShowUploadDoneModal(null); router.push("/knowledge-graph"); }}
                className="cursor-pointer hover:bg-cognee-hover"
                style={{ display: "flex", alignItems: "center", gap: 10, padding: "12px 14px", borderRadius: 8, border: "1px solid #E4E4E7", background: "#fff", textAlign: "left", fontFamily: "inherit", width: "100%" }}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#6510F4" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" /><path d="M3 9h18M9 3v18" /></svg>
                <div>
                  <div style={{ fontSize: 14, fontWeight: 500, color: "#18181B" }}>Explore the knowledge graph</div>
                  <div style={{ fontSize: 12, color: "#71717A" }}>Open the full graph visualization</div>
                </div>
              </button>
            </div>
            <button
              onClick={() => setShowUploadDoneModal(null)}
              className="cursor-pointer"
              style={{ background: "none", border: "1px solid #E4E4E7", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "#71717A", fontFamily: "inherit", alignSelf: "flex-end" }}
            >
              Stay here
            </button>
          </div>
        </div>
      )}

      {/* Search */}
      <DashboardSearch datasets={filteredDatasets} cogniInstance={cogniInstance} sessions={sessions} hasDocuments={filteredRuns.length > 0 || (graphNodes ?? 0) > 0} />
    </div>
  );
}

// ── Helpers ──────────────────────────────────────────────────────────────

function formatToday(): string {
  const now = new Date();
  const weekday = now.toLocaleDateString("en-US", { weekday: "long" });
  const day = now.getDate();
  const month = now.toLocaleDateString("en-US", { month: "long" });
  return `${weekday}, ${day} ${month}`;
}

function greetingForTime(): string {
  const h = new Date().getHours();
  if (h < 12) return "Good morning";
  if (h < 18) return "Good afternoon";
  return "Good evening";
}

function durationString(startedISO: string | null, endedISO: string | null): string {
  if (!startedISO) return "—";
  const start = new Date(startedISO).getTime();
  const end = endedISO ? new Date(endedISO).getTime() : Date.now();
  const s = Math.max(0, (end - start) / 1000);
  if (s < 1) return "0s";
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${Math.round(s % 60)}s`;
}

// ── Sub-components ───────────────────────────────────────────────────────

function RangePicker({ value, onChange }: { value: Range; onChange: (v: Range) => void }) {
  const opts: Range[] = ["24h", "7d", "30d"];
  return (
    <div style={{ display: "flex", alignItems: "center", background: "#fff", border: "1px solid #E4E4E7", borderRadius: 8, overflow: "hidden" }}>
      {opts.map((o) => (
        <button
          key={o}
          onClick={() => onChange(o)}
          style={{
            paddingBlock: 8,
            paddingInline: 14,
            fontFamily: "inherit",
            fontSize: 13,
            lineHeight: "16px",
            color: value === o ? "#FFFFFF" : "#3F3F46",
            background: value === o ? "#18181B" : "transparent",
            border: "none",
            cursor: "pointer",
          }}
        >
          {o === "24h" ? "Last 24 hours" : o === "7d" ? "Last 7 days" : "Last 30 days"}
        </button>
      ))}
    </div>
  );
}

function StatCard({ label, value, suffix, badge, icon, iconBg, chip }: {
  label: string;
  value: string;
  suffix?: string;
  badge?: React.ReactNode;
  icon?: React.ReactNode;
  iconBg?: string;
  chip?: { label: string; color: string; bg: string };
}) {
  return (
    <div style={{ flex: 1, background: "#fff", border: "1px solid #E4E4E7", borderRadius: 12, paddingBlock: 20, paddingInline: 20, display: "flex", flexDirection: "column", gap: 12 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span style={{ color: "#71717A", fontSize: 11, letterSpacing: "0.14em", lineHeight: "14px", textTransform: "uppercase" }}>{label}</span>
        {chip ? (
          <div style={{ background: chip.bg, borderRadius: 4, paddingBlock: 2, paddingInline: 8 }}>
            <span style={{ color: chip.color, fontSize: 11, lineHeight: "14px" }}>{chip.label}</span>
          </div>
        ) : icon ? (
          <div style={{ background: iconBg || "#F4F4F5", borderRadius: 6, width: 24, height: 24, display: "flex", alignItems: "center", justifyContent: "center" }}>{icon}</div>
        ) : null}
      </div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
        <span style={{ color: "#18181B", fontSize: 32, letterSpacing: "-0.02em", lineHeight: "36px", fontVariantNumeric: "tabular-nums" }}>{value}</span>
        {suffix && <span style={{ color: "#71717A", fontSize: 13, lineHeight: "16px" }}>{suffix}</span>}
      </div>
      {badge}
    </div>
  );
}

function AgentStack({ agents, live }: { agents: { id: string; agent_type: string }[]; live?: Set<string> }) {
  const palette = ["#18181B", "#6510F4", "#D97706", "#16A34A"];
  return (
    <div style={{ display: "flex", alignItems: "center" }}>
      {agents.slice(0, 3).map((a, i) => {
        const initials = a.agent_type
          .split(/[\s-]+/)
          .filter(Boolean)
          .map((p) => p[0]?.toUpperCase())
          .join("")
          .slice(0, 2) || "?";
        const isLive = live?.has(a.id);
        return (
          <div key={a.id} style={{ position: "relative", marginLeft: i === 0 ? 0 : -6 }}>
            <div style={{ width: 24, height: 24, borderRadius: "50%", border: "2px solid #FFFFFF", background: palette[i % palette.length], display: "flex", alignItems: "center", justifyContent: "center", color: "#FFFFFF", fontSize: 9, lineHeight: "12px" }}>
              {initials}
            </div>
            {isLive && <div style={{ position: "absolute", top: -1, right: -1, width: 8, height: 8, borderRadius: "50%", background: "#22C55E", border: "1.5px solid #FFFFFF" }} />}
          </div>
        );
      })}
      {agents.length === 0 && <span style={{ fontSize: 12, color: "#A1A1AA" }}>none yet</span>}
    </div>
  );
}

function DatasetTags({ datasets }: { datasets: { id: string; name: string }[] }) {
  if (datasets.length === 0) return <span style={{ fontSize: 11, color: "#A1A1AA" }}>no brains yet</span>;
  const visible = datasets.slice(0, 2);
  const more = datasets.length - visible.length;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      {visible.map((d) => (
        <span key={d.id} style={{ background: "#F4F4F5", borderRadius: 4, paddingBlock: 2, paddingInline: 8, color: "#3F3F46", fontSize: 11, lineHeight: "14px" }}>{d.name}</span>
      ))}
      {more > 0 && <span style={{ color: "#A1A1AA", fontSize: 11 }}>+{more} more</span>}
    </div>
  );
}

function Sparkline() {
  return (
    <svg width="100%" height="28" viewBox="0 0 240 28" fill="none" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg" style={{ flexShrink: 0 }}>
      <path d="M0 22 L15 20 L30 17 L45 18 L60 13 L75 15 L90 10 L105 12 L120 8 L135 11 L150 6 L165 9 L180 4 L195 7 L210 3 L225 5 L240 2" stroke="#71717A" strokeWidth="1.5" fill="none" strokeLinecap="round" />
      <path d="M0 22 L15 20 L30 17 L45 18 L60 13 L75 15 L90 10 L105 12 L120 8 L135 11 L150 6 L165 9 L180 4 L195 7 L210 3 L225 5 L240 2 L240 28 L0 28 Z" fill="#71717A" style={{ opacity: 0.08 }} />
    </svg>
  );
}

function ErrorPipelineTag({ runs }: { runs: PipelineRun[] }) {
  const errored = runs.find((r) => r.status.includes("ERRORED"));
  if (!errored) return <span style={{ color: "#71717A", fontSize: 12 }}>no errors</span>;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#DC2626" }} />
      <span style={{ color: "#71717A", fontSize: 12, lineHeight: "16px" }}>{errored.pipeline_name}</span>
    </div>
  );
}

function OnboardingItem({ title, subtitle, icon, done, accent, loading, href, onClick, highlightSubtitle }: {
  title: string; subtitle: string; icon: React.ReactNode;
  done?: boolean; accent?: boolean; loading?: boolean;
  href?: string; onClick?: () => void; highlightSubtitle?: boolean;
}) {
  const content = (
    <div style={{ display: "flex", alignItems: "center", flex: 1, gap: 12, paddingBlock: 14, paddingInline: 16, background: accent ? "#F0EDFF" : "transparent", borderRadius: 8, borderRight: accent ? "none" : "1px solid #F4F4F5" }}>
      <div style={{ position: "relative", width: 32, height: 32, flexShrink: 0, borderRadius: 8, background: accent ? "#6510F4" : "#F4F4F5", display: "flex", alignItems: "center", justifyContent: "center" }}>
        {loading ? (
          <div style={{ width: 16, height: 16, border: "2px solid #6510F4", borderTopColor: "transparent", borderRadius: "50%", animation: "spin 0.8s linear infinite" }} />
        ) : icon}
        {done && !loading && (
          <div style={{ position: "absolute", top: -2, right: -2, width: 14, height: 14, borderRadius: "50%", background: "#16A34A", border: "2px solid #FFFFFF", display: "flex", alignItems: "center", justifyContent: "center" }}>
            <svg width="7" height="7" viewBox="0 0 16 16"><path d="M3.5 8.5 L6.5 11.5 L12.5 4.5" stroke="#FFFFFF" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" fill="none" /></svg>
          </div>
        )}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 2, minWidth: 0 }}>
        <span style={{ color: "#18181B", fontSize: 13, lineHeight: "16px" }}>{title}</span>
        <span style={{ color: accent || highlightSubtitle ? "#6510F4" : "#A1A1AA", fontSize: 12, lineHeight: "14px" }}>{subtitle}</span>
      </div>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
  if (href) return <Link href={href} className="cursor-pointer" style={{ flex: 1, textDecoration: "none" }}>{content}</Link>;
  if (onClick) return <button onClick={onClick} className="cursor-pointer" style={{ flex: 1, border: "none", background: "transparent", padding: 0, textAlign: "left" }}>{content}</button>;
  return <div style={{ flex: 1 }}>{content}</div>;
}

function FilterPill({ active, onClick, children, dot, icon }: { active: boolean; onClick: () => void; children: React.ReactNode; dot?: string; icon?: React.ReactNode }) {
  return (
    <button onClick={onClick} className="cursor-pointer" style={{ display: "flex", alignItems: "center", gap: 6, background: active ? "#18181B" : "#FFFFFF", color: active ? "#FFFFFF" : "#3F3F46", border: active ? "none" : "1px solid #E4E4E7", borderRadius: 100, paddingBlock: 6, paddingInline: 12, fontSize: 12, lineHeight: "16px", fontFamily: "inherit" }}>
      {dot && <span style={{ width: 6, height: 6, borderRadius: "50%", background: dot, flexShrink: 0 }} />}
      {icon}
      {children}
    </button>
  );
}

type ActivityRow =
  | { kind: "session"; session: import("@/modules/sessions/getSessions").SessionRow; agent: unknown; dataset: unknown; timeStr: number }
  | { kind: "dataset"; dataset: { id: string; name: string }; timeStr: number };

function ActivityTable({ rows, agents }: { rows: ActivityRow[]; agents: { id: string; agent_type: string; is_agent: boolean; is_default: boolean }[] }) {
  if (rows.length === 0) {
    return (
      <div style={{ background: "#FFFFFF", border: "1px solid #E4E4E7", borderRadius: 12, padding: 32, textAlign: "center" }}>
        <span style={{ color: "#A1A1AA", fontSize: 13 }}>No activity in this time range.</span>
      </div>
    );
  }
  return (
    <div style={{ background: "#FFFFFF", border: "1px solid #E4E4E7", borderRadius: 12, overflow: "hidden" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 16, paddingBlock: 12, paddingInline: 20, background: "#FFFFFF", borderBottom: "1px solid #E4E4E7" }}>
        <span style={{ width: 80, flexShrink: 0, color: "#71717A", fontSize: 11, letterSpacing: "0.08em", lineHeight: "14px", textTransform: "uppercase" }}>Type</span>
        <span style={{ flex: 1, color: "#71717A", fontSize: 11, letterSpacing: "0.08em", lineHeight: "14px", textTransform: "uppercase" }}>Name</span>
        <span style={{ width: 140, flexShrink: 0, color: "#71717A", fontSize: 11, letterSpacing: "0.08em", lineHeight: "14px", textTransform: "uppercase" }}>Source</span>
        <span style={{ width: 100, flexShrink: 0, textAlign: "right", color: "#71717A", fontSize: 11, letterSpacing: "0.08em", lineHeight: "14px", textTransform: "uppercase" }}>Size / Dur</span>
        <span style={{ width: 80, flexShrink: 0, textAlign: "right", color: "#71717A", fontSize: 11, letterSpacing: "0.08em", lineHeight: "14px", textTransform: "uppercase" }}>Cost</span>
        <span style={{ width: 80, flexShrink: 0, textAlign: "right", color: "#71717A", fontSize: 11, letterSpacing: "0.08em", lineHeight: "14px", textTransform: "uppercase" }}>Updated</span>
      </div>
      {rows.map((row, idx) => {
        if (row.kind === "session") {
          const s = row.session;
          const failed = s.effective_status === "failed" || s.error_count > 0;
          const dotColor = failed ? "#DC2626" : s.effective_status === "abandoned" ? "#D97706" : "#16A34A";
          const label = labelFromSession(s);
          const agentEntry = agents.find((a) => a.is_agent && a.id === s.user_id);
          const agentName = agentEntry ? agentEntry.agent_type : s.user_id.slice(0, 8);
          const agentInitial = agentName.split(/[\s-]+/).filter(Boolean).map((p) => p[0]?.toUpperCase()).join("").slice(0, 2) || "?";
          return (
            <Link
              key={`session-${s.session_id}-${s.user_id}`}
              href={`/activity?session=${encodeURIComponent(s.session_id)}`}
              className="cursor-pointer hover:bg-cognee-hover"
              style={{ display: "flex", alignItems: "center", gap: 16, paddingBlock: 14, paddingInline: 20, borderBottom: "1px solid #F4F4F5", background: "transparent", textDecoration: "none", color: "inherit", transition: "background 150ms" }}
              title="Open in Activity"
            >
              <div style={{ width: 80, flexShrink: 0, display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{ width: 6, height: 6, borderRadius: "50%", background: dotColor, flexShrink: 0 }} />
                <span style={{ color: failed ? "#DC2626" : "#52525B", fontSize: 11, letterSpacing: "0.04em", textTransform: "uppercase", lineHeight: "14px" }}>Session</span>
              </div>
              <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 2 }}>
                <span style={{ color: "#18181B", fontSize: 14, lineHeight: "18px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{label}</span>
                <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
                  <span style={{ color: "#A1A1AA", fontSize: 11, lineHeight: "14px" }}>
                    {(s.last_model ?? "—")} · {s.tokens_in + s.tokens_out} tokens · {s.error_count} errors
                  </span>
                </div>
              </div>
              <div style={{ width: 140, flexShrink: 0, display: "flex", alignItems: "center", gap: 6 }}>
                <div style={{ width: 18, height: 18, background: "#18181B", borderRadius: 4, display: "flex", alignItems: "center", justifyContent: "center", color: "#FFFFFF", fontSize: 9, lineHeight: "12px" }}>{agentInitial}</div>
                <span style={{ color: "#3F3F46", fontSize: 13, lineHeight: "16px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{agentName}</span>
              </div>
              <span style={{ width: 100, flexShrink: 0, textAlign: "right", color: "#18181B", fontSize: 13, lineHeight: "16px", fontVariantNumeric: "tabular-nums" }}>{durationString(s.started_at, s.ended_at)}</span>
              <span style={{ width: 80, flexShrink: 0, textAlign: "right", color: s.cost_usd > 0 ? "#18181B" : "#A1A1AA", fontSize: 13, lineHeight: "16px", fontVariantNumeric: "tabular-nums" }}>{s.cost_usd > 0 ? `$${s.cost_usd.toFixed(2)}` : "—"}</span>
              <span style={{ width: 80, flexShrink: 0, textAlign: "right", color: "#A1A1AA", fontSize: 12, lineHeight: "16px" }}>{s.last_activity_at ? timeAgo(s.last_activity_at) : "—"}</span>
            </Link>
          );
        }
        const d = row.dataset;
        return (
          <div key={`dataset-${d.id}`} style={{ display: "flex", alignItems: "center", gap: 16, paddingBlock: 14, paddingInline: 20, borderBottom: "1px solid #F4F4F5", background: "transparent" }}>
            <div style={{ width: 80, flexShrink: 0, display: "flex", alignItems: "center", gap: 6 }}>
              <DatasetIconXs color="#6510F4" />
              <span style={{ color: "#6510F4", fontSize: 11, letterSpacing: "0.04em", textTransform: "uppercase", lineHeight: "14px" }}>Brain</span>
            </div>
            <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 2 }}>
              <span style={{ color: "#18181B", fontSize: 14, lineHeight: "18px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{d.name}</span>
              <span style={{ color: "#16A34A", fontSize: 11, lineHeight: "14px" }}>ready</span>
            </div>
            <span style={{ width: 140, flexShrink: 0, color: "#71717A", fontSize: 12, lineHeight: "16px" }}>—</span>
            <span style={{ width: 100, flexShrink: 0, textAlign: "right", color: "#A1A1AA", fontSize: 13, lineHeight: "16px" }}>—</span>
            <span style={{ width: 80, flexShrink: 0, textAlign: "right", color: "#A1A1AA", fontSize: 13, lineHeight: "16px" }}>—</span>
            <span style={{ width: 80, flexShrink: 0, textAlign: "right", color: "#A1A1AA", fontSize: 12, lineHeight: "16px" }}>{row.timeStr ? timeAgo(new Date(row.timeStr).toISOString()) : "—"}</span>
          </div>
        );
      })}
    </div>
  );
}

function labelFromSession(s: import("@/modules/sessions/getSessions").SessionRow): string {
  // Without hitting the detail endpoint we don't have a semantic label —
  // surface the session id trimmed to something readable. Backend should
  // expose label on the list endpoint as a follow-up.
  const id = s.session_id;
  if (id.length <= 48) return id;
  return id.slice(0, 48) + "…";
}

// Tiny icons (kept minimal to match the Paper reference).
function AgentIconSm() { return <svg width="12" height="12" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="8" r="4" stroke="#6510F4" strokeWidth="1.75" /><path d="M5.5 21a6.5 6.5 0 0113 0" stroke="#6510F4" strokeWidth="1.75" strokeLinecap="round" /></svg>; }
function DatasetIconSm() { return <svg width="12" height="12" viewBox="0 0 24 24" fill="none"><ellipse cx="12" cy="5" rx="9" ry="3" stroke="#6510F4" strokeWidth="1.75" /><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" stroke="#6510F4" strokeWidth="1.75" /></svg>; }
function DatasetIconXs({ color = "#71717A" }: { color?: string }) { return <svg width="10" height="10" viewBox="0 0 14 14" fill="none"><ellipse cx="7" cy="3" rx="5" ry="1.5" stroke={color} strokeWidth="1.2" /><path d="M12 6.5c0 .83-2.24 1.5-5 1.5s-5-.67-5-1.5M2 3v7.5C2 11.33 4.24 12 7 12s5-.67 5-1.5V3" stroke={color} strokeWidth="1.2" /></svg>; }
function SdkIcon() { return <svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M7 8l5 4 5-4M7 16l5-4 5 4" stroke="#52525B" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" /></svg>; }
function KeyIconSm() { return <svg width="16" height="16" viewBox="0 0 24 24" fill="none"><rect x="4" y="8" width="16" height="8" rx="2" stroke="#52525B" strokeWidth="1.75" /><circle cx="9" cy="12" r="1.5" fill="#52525B" /></svg>; }
function UploadIconSm() { return <svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M12 16V8M12 8L8 12M12 8L16 12" stroke="#FFFFFF" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" /><path d="M4 17v2a2 2 0 002 2h12a2 2 0 002-2v-2" stroke="#FFFFFF" strokeWidth="1.75" strokeLinecap="round" /></svg>; }

// QuickstartCards is imported from @/ui/elements/QuickstartCards

// ── Scope pills (shared) ─────────────────────────────────────────────────

const SCOPE_DESCRIPTIONS: Record<string, string> = {
  documents: "Search entities and relationships in your knowledge graph.",
  agent: "Search recent agent sessions and execution traces.",
};

function ScopePills({
  value,
  onChange,
}: {
  value: "documents" | "agent";
  onChange: (v: "documents" | "agent") => void;
}) {
  const items: { key: "documents" | "agent"; label: string }[] = [
    { key: "documents", label: "Brain" },
    { key: "agent", label: "Agent Memory" },
  ];
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      {items.map((it) => {
        const active = value === it.key;
        return (
          <Tooltip key={it.key} label={SCOPE_DESCRIPTIONS[it.key]} withArrow multiline w={220} position="bottom">
            <button
              type="button"
              onClick={() => onChange(it.key)}
              className="cursor-pointer"
              style={{
                background: active ? "#18181B" : "#FFFFFF",
                color: active ? "#FFFFFF" : "#3F3F46",
                border: active ? "none" : "1px solid #E4E4E7",
                borderRadius: 100,
                paddingBlock: 5,
                paddingInline: 11,
                fontSize: 12,
                lineHeight: "16px",
                fontFamily: "inherit",
                cursor: "pointer",
              }}
            >
              {it.label}
            </button>
          </Tooltip>
        );
      })}
    </div>
  );
}

// ── Inline dashboard search ──

type SearchScope = "documents" | "agent";

function DashboardSearch({
  datasets,
  cogniInstance,
  sessions,
  hasDocuments,
}: {
  datasets: { id: string; name: string }[];
  cogniInstance: ReturnType<typeof useCogniInstance>["cogniInstance"];
  sessions: import("@/modules/sessions/getSessions").SessionRow[];
  hasDocuments?: boolean;
}) {
  const [query, setQuery] = useState("");
  const [scope, setScope] = useState<SearchScope>("documents");
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>(datasets[0]?.id ?? "");
  const [results, setResults] = useState<unknown[]>([]);
  const [searching, setSearching] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);

  // Pick the most recently active session the user can see. Required
  // for session / trace / all scope — recall returns nothing otherwise.
  const mostRecentSessionId = sessions[0]?.session_id ?? "";

  async function handleSearch(q: string) {
    if (!q.trim() || !cogniInstance) return;
    setSearching(true);
    setResults([]);
    setHasSearched(true);
    try {
      const { default: recallKnowledge } = await import("@/modules/datasets/recallKnowledge");
      // Map UI scope → recall scope. Non-graph scopes require a
      // session_id; we fall back to graph when none is available.
      const sendScope = scope === "agent" ? ["session", "trace"] : "graph";
      const sendSessionId = scope === "agent" ? mostRecentSessionId : undefined;
      const data = await recallKnowledge(cogniInstance, {
        query: q,
        scope: sendScope as never,
        sessionId: sendSessionId,
        datasetIds: selectedDatasetId ? [selectedDatasetId] : datasets.map((d) => d.id),
      });
      setResults(Array.isArray(data) ? data : []);
    } catch {
      setResults([]);
    } finally {
      setSearching(false);
    }
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12, minHeight: 200, opacity: hasDocuments === false ? 0.45 : 1, pointerEvents: hasDocuments === false ? "none" : "auto", transition: "opacity 200ms" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontSize: 20, fontWeight: 300, color: "#1A1A1A", fontFamily: '"TWK Lausanne", system-ui, sans-serif' }}>Search your knowledge</span>
      </div>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <ScopePills value={scope} onChange={setScope} />
        <div style={{ display: "flex", alignItems: "center", gap: 16, flexWrap: "nowrap", marginRight: 24 }}>
          <span style={{ fontSize: 11, color: "#A1A1AA", letterSpacing: "0.1em", textTransform: "uppercase", whiteSpace: "nowrap", flexShrink: 0 }}>Search in</span>
          <select
            value={selectedDatasetId}
            onChange={(e) => setSelectedDatasetId(e.target.value)}
            style={{ appearance: "none", background: "none", border: "none", outline: "none", fontSize: 14, fontWeight: 400, color: "#333333", fontFamily: "inherit", cursor: "pointer", padding: 0, paddingRight: 14, backgroundImage: "url(\"data:image/svg+xml,%3Csvg width='8' height='5' viewBox='0 0 8 5' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M1 1l3 3 3-3' stroke='%23333333' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E\")", backgroundRepeat: "no-repeat", backgroundPosition: "right center", maxWidth: 180, overflow: "hidden", textOverflow: "ellipsis", lineHeight: "20px" }}
          >
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>{d.name}</option>
            ))}
          </select>
        </div>
      </div>
      <div style={{ background: "#fff", border: `1px solid ${query ? "#6510F4" : "#EEEEEE"}`, borderRadius: 10, padding: "12px 16px", display: "flex", alignItems: "center", gap: 12, transition: "border-color 0.2s" }}>
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none" style={{ flexShrink: 0 }}>
          <circle cx="8" cy="8" r="5.5" stroke="#A1A1AA" strokeWidth="1.5" />
          <path d="M12.5 12.5L16 16" stroke="#A1A1AA" strokeWidth="1.5" strokeLinecap="round" />
        </svg>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") handleSearch(query); }}
          placeholder="Ask a question about your data..."
          style={{ flex: 1, border: "none", outline: "none", fontSize: 14, color: "#18181B", fontFamily: "inherit", background: "transparent" }}
        />
        <button onClick={() => handleSearch(query)} disabled={!query} className="cursor-pointer" style={{ background: query ? "#6510F4" : "#E4E4E7", border: "none", borderRadius: 6, padding: "8px 10px", display: "flex", alignItems: "center", justifyContent: "center", transition: "background 150ms", cursor: query ? "pointer" : "default", flexShrink: 0 }}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={query ? "#fff" : "#A1A1AA"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="5" y1="12" x2="19" y2="12" /><polyline points="12 5 19 12 12 19" /></svg>
        </button>
      </div>

      {searching && (
        <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "12px 0" }}>
          <div style={{ width: 14, height: 14, borderRadius: "50%", border: "2px solid #E4E4E7", borderTopColor: "#6510F4", animation: "spin 1s linear infinite" }} />
          <span style={{ fontSize: 13, color: "#71717A" }}>Searching...</span>
        </div>
      )}

      {results.length > 0 && !searching && (
        <div style={{ background: "#fff", border: "1px solid #E5E7EB", borderRadius: 10, overflow: "hidden" }}>
          {results.map((raw, i) => {
            // searchDataset may return different shapes depending on
            // search_type / route: { dataset_name, search_result:[...] },
            // plain strings, or recall rows tagged with _source.
            const r = raw as { dataset_name?: string; search_result?: unknown; text?: string; raw?: { value?: string }; answer?: string; session_feedback?: string };
            const label = r.dataset_name || "Result";
            let lines: string[] = [];
            if (r.text) {
              lines = [r.text];
            } else if (r.raw?.value) {
              lines = [typeof r.raw.value === "string" ? r.raw.value : JSON.stringify(r.raw.value)];
            } else if (Array.isArray(r.search_result)) {
              lines = (r.search_result as unknown[]).map((x) => (typeof x === "string" ? x : JSON.stringify(x)));
            } else if (typeof r.search_result === "string") {
              lines = [r.search_result];
            } else if (r.answer) {
              lines = [r.answer];
            } else if (r.session_feedback) {
              lines = [r.session_feedback];
            } else if (typeof raw === "string") {
              lines = [raw as unknown as string];
            } else {
              try { lines = [JSON.stringify(raw)]; } catch { lines = [String(raw)]; }
            }
            return (
              <div key={i} style={{ padding: "14px 16px", borderBottom: i < results.length - 1 ? "1px solid #F4F4F5" : "none" }}>
                <span style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.04em", color: "#6510F4", textTransform: "uppercase" }}>{label}</span>
                {lines.map((text, j) => (
                  <div key={j} style={{ fontSize: 13, color: "#18181B", lineHeight: "20px", whiteSpace: "pre-wrap", wordBreak: "break-word", marginTop: 4 }}>
                    {text}
                  </div>
                ))}
              </div>
            );
          })}
        </div>
      )}

      {hasSearched && !searching && results.length === 0 && (
        <span style={{ fontSize: 13, color: "#A1A1AA", padding: "8px 0" }}>No results found.</span>
      )}

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}

// ── Python SDK card with expandable connect-to-cloud instructions ──

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      onClick={(e) => { e.stopPropagation(); navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 1500); }}
      className="cursor-pointer hover:bg-white/10 rounded p-1 -m-1 active:scale-90 transition-all"
      style={{ background: "none", border: "none", flexShrink: 0 }}
      title="Copy"
    >
      {copied ? (
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M3.5 8.5L6.5 11.5L12.5 4.5" stroke="#22C55E" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
      ) : (
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none"><rect x="5" y="5" width="8" height="8" rx="1.5" stroke="#71717A" strokeWidth="1.5" /><path d="M11 3H4.5A1.5 1.5 0 003 4.5V11" stroke="#71717A" strokeWidth="1.5" strokeLinecap="round" /></svg>
      )}
    </button>
  );
}

function SdkCard() {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      onClick={() => setExpanded(!expanded)}
      className="cursor-pointer hover:bg-cognee-hover transition-colors"
      style={{
        flex: expanded ? 2 : 1,
        background: "#fff",
        border: expanded ? "1px solid #DDD6FE" : "1px solid #EEEEEE",
        borderRadius: 10,
        padding: expanded ? 0 : "20px 16px",
        display: "flex",
        flexDirection: "column",
        alignItems: expanded ? "stretch" : "center",
        gap: expanded ? 0 : 8,
        textDecoration: "none",
        transition: "flex 200ms ease, border-color 200ms",
        overflow: "hidden",
      }}
    >
      {!expanded && (
        <>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <path d="M7 8l5 4 5-4M7 16l5-4 5 4" stroke="#6C47FF" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <span style={{ fontSize: 13, fontWeight: 500, color: "#333333" }}>Python SDK</span>
          <span style={{ fontSize: 11, color: "#999999" }}>pip install cognee</span>
        </>
      )}

      {expanded && (
        <div onClick={(e) => e.stopPropagation()} style={{ display: "flex", flexDirection: "column", gap: 0 }}>
          {/* Header */}
          <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "16px 20px", borderBottom: "1px solid #E4E4E7" }}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M7 8l5 4 5-4M7 16l5-4 5 4" stroke="#6510F4" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <span style={{ fontSize: 14, fontWeight: 600, color: "#18181B" }}>Connect with Python SDK</span>
            <button onClick={() => setExpanded(false)} className="cursor-pointer" style={{ marginLeft: "auto", background: "none", border: "none", color: "#A1A1AA", fontSize: 16, padding: 2 }}>&#10005;</button>
          </div>

          {/* Steps */}
          <div style={{ padding: "16px 20px", display: "flex", flexDirection: "column", gap: 16 }}>
            {/* Step 1 */}
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div className="flex items-center justify-center flex-shrink-0 rounded-full" style={{ width: 24, height: 24, background: "#F0EDFF" }}>
                <span style={{ color: "#6510F4", fontSize: 12, fontWeight: 600 }}>1</span>
              </div>
              <span style={{ fontSize: 13, fontWeight: 500, color: "#18181B" }}>Install the SDK</span>
            </div>
            <div className="flex items-center justify-between" style={{ background: "#18181B", borderRadius: 8, padding: "10px 16px" }}>
              <span style={{ fontSize: 13, color: "#A1A1AA", fontFamily: '"Fira Code", monospace' }}>pip install cognee</span>
              <CopyButton text="pip install cognee" />
            </div>

            <div style={{ height: 1, background: "#E4E4E7" }} />

            {/* Step 2 */}
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div className="flex items-center justify-center flex-shrink-0 rounded-full" style={{ width: 24, height: 24, background: "#F0EDFF" }}>
                <span style={{ color: "#6510F4", fontSize: 12, fontWeight: 600 }}>2</span>
              </div>
              <span style={{ fontSize: 13, fontWeight: 500, color: "#18181B" }}>Set your API key</span>
            </div>
            <div className="flex items-center justify-between" style={{ background: "#18181B", borderRadius: 8, padding: "10px 16px" }}>
              <span style={{ fontSize: 13, color: "#A1A1AA", fontFamily: '"Fira Code", monospace' }}>export COGNEE_API_KEY=your-key</span>
              <CopyButton text="export COGNEE_API_KEY=your-key" />
            </div>

            <div style={{ height: 1, background: "#E4E4E7" }} />

            {/* Step 3 */}
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div className="flex items-center justify-center flex-shrink-0 rounded-full" style={{ width: 24, height: 24, background: "#F0EDFF" }}>
                <span style={{ color: "#6510F4", fontSize: 12, fontWeight: 600 }}>3</span>
              </div>
              <span style={{ fontSize: 13, fontWeight: 500, color: "#18181B" }}>Connect to your instance</span>
            </div>
            <div style={{ background: "#18181B", borderRadius: 8, padding: "12px 16px", position: "relative" }}>
              <pre style={{ margin: 0, fontSize: 12, lineHeight: "20px", fontFamily: '"Fira Code", monospace', color: "#A1A1AA" }}>
{`import asyncio
import cognee

async def main():
    cognee.config.set_llm_api_key("your-key")

    await cognee.add("Your text data")
    await cognee.cognify()

    results = await cognee.search("query")
    print(results)

asyncio.run(main())`}
              </pre>
              <span style={{ position: "absolute", top: 12, right: 16 }}>
                <CopyButton text={`import asyncio\nimport cognee\n\nasync def main():\n    cognee.config.set_llm_api_key("your-key")\n\n    await cognee.add("Your text data")\n    await cognee.cognify()\n\n    results = await cognee.search("query")\n    print(results)\n\nasyncio.run(main())`} />
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Dev: onboarding toggle */}
      <div style={{ display: "flex", gap: 8, paddingTop: 8 }}>
        <Link href="/onboarding" style={{ fontSize: 12, color: "#A1A1AA", textDecoration: "none" }}>
          Open onboarding
        </Link>
        <span style={{ color: "#E4E4E7" }}>|</span>
        <Link href="/onboarding?source=serve" style={{ fontSize: 12, color: "#A1A1AA", textDecoration: "none" }}>
          Serve onboarding
        </Link>
      </div>
    </div>
  );
}

// ── Icons ──

function UploadIcon() { return <svg width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M12 16V8M12 8L8 12M12 8L16 12" stroke="#6C47FF" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /><path d="M4 17v2a2 2 0 002 2h12a2 2 0 002-2v-2" stroke="#6C47FF" strokeWidth="1.5" strokeLinecap="round" /></svg>; }
function KeyNavIcon() { return <svg width="24" height="24" viewBox="0 0 24 24" fill="none"><rect x="4" y="8" width="16" height="8" rx="2" stroke="#6C47FF" strokeWidth="1.5" /><circle cx="9" cy="12" r="1.5" fill="#6C47FF" /><path d="M14 10h3M14 14h3" stroke="#6C47FF" strokeWidth="1.5" strokeLinecap="round" /></svg>; }
function AgentNavIcon() { return <svg width="24" height="24" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="8" r="4" stroke="#6C47FF" strokeWidth="1.5" /><path d="M5.5 21a6.5 6.5 0 0113 0" stroke="#6C47FF" strokeWidth="1.5" strokeLinecap="round" /><path d="M15 3l2-2M9 3L7 1" stroke="#6C47FF" strokeWidth="1.5" strokeLinecap="round" /></svg>; }
