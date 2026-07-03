"use client";

import React, { useEffect, useState, useRef } from "react";
import { useRouter } from "next/navigation";
import { useCogniInstance, useTenant } from "@/modules/tenant/TenantProvider";
import getCreditsOverview from "@/modules/billing/getCreditsOverview";
import { useFilter } from "@/ui/layout/FilterContext";
import rememberData from "@/modules/ingestion/rememberData";
import createDataset from "@/modules/datasets/createDataset";
import pollDatasetStatus from "@/modules/datasets/pollDatasetStatus";
import { loadGraphModelsConfig, findModelForDataset, findPromptForDataset, findOntologyForDataset } from "@/modules/configuration/userConfiguration";
import { toCleanSchema } from "@/modules/graphModels/types";
import { toGraphModelSchema } from "@/modules/graphModels/toGraphModelSchema";
import { listSessions, SEARCH_SESSION_PREFIX } from "@/modules/sessions/getSessions";
import getDatasetGraph from "@/modules/datasets/getDatasetGraph";
import { notifications } from "@mantine/notifications";
import { trackEvent, TrackPageView } from "@/modules/analytics";
import { markOnboardingCompleteLocally } from "@/utils/onboardingFlag";
import { CLAUDE_MARKETPLACE_ADD, CLAUDE_PLUGIN_INSTALL, CODEX_HOOKS_ENABLE, CODEX_MARKETPLACE_ADD, CODEX_PLUGIN_INSTALL, OPENCLAW_SKILL_INSTALL, GENERIC_SKILL_INSTALL, UPLOAD_MEMORY_PROMPT, UPLOAD_SAMPLE_PROMPT, RECALL_SAMPLE_PROMPT } from "@/data/prompts";
import { AgentActivityTerminal, PipelineRun, Range, ownerDisplayName } from "@/ui/elements/AgentActivityTerminal";
import SkeletonBar from "@/ui/elements/SkeletonBar";
import DashboardSkeleton from "./DashboardSkeleton";
import isCloudEnvironment from "@/utils/isCloudEnvironment";

const AWAITING_DATASET_KEY = "cognee-awaiting-dataset";

// Per-integration "Connected" detection. The Claude Code and Codex plugins mint
// their session_id as `{prefix}_{dir}_{token}` with a fixed prefix per agent
// (see topoteretes/cognee-integrations: claude-code → "cc", codex → "codex"), so
// we can tell which agent connected from the session list alone. Detection is
// coarse on purpose — any session whose id starts with the prefix counts. Openclaw
// forwards the host session id (no Cognee prefix) and API/MCP is user-defined, so
// neither is auto-detected. Keep these in sync with the shipped integrations
// (the plugins' _generate_session_id + the skill prompts in prompts.ts).
const INTEGRATION_SESSION_PREFIX: Record<string, string> = {
  "claude-code": "cc_",
  codex: "codex_",
};

export default function OverviewPage() {
  const { cogniInstance, isInitializing, serviceUrl, apiKey } = useCogniInstance();
  const { tenantReady, tenant, isOwner } = useTenant();
  const { agents, datasets, selectedDataset, selectedAgent, setSelectedDataset, refreshDatasets, loading: filterLoading } = useFilter();
  const [runs, setRuns] = useState<PipelineRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const isHandlingUploadRef = useRef(false);
  const [showDatasetPicker, setShowDatasetPicker] = useState(false);
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [range] = useState<Range>("24h");
  const [sessions, setSessions] = useState<import("@/modules/sessions/getSessions").SessionRow[]>([]);
  // Which integrations have ever connected (by session_id prefix). Sticky per
  // tenant via localStorage so the badge survives the 24h session window.
  const [connectedIntegrations, setConnectedIntegrations] = useState<Record<string, boolean>>({});
  const [graphNodes, setGraphNodes] = useState<number | null>(() => {
    try { const v = sessionStorage.getItem("cognee-graph-nodes"); return v !== null ? Number(v) : null; } catch { return null; }
  });
  const [graphEdges, setGraphEdges] = useState<number | null>(null);
  const [showUploadDoneModal, setShowUploadDoneModal] = useState<{ datasetName: string; datasetId: string } | null>(null);
  const uploadInputRef = useRef<HTMLInputElement>(null);
  const sliderRef = useRef<HTMLDivElement>(null);
  const sliderDragging = useRef(false);
  const sliderStartX = useRef(0);
  const sliderScrollLeft = useRef(0);
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(false);
  const [creditsBannerDismissed, setCreditsBannerDismissed] = useState<boolean>(() => {
    try { return sessionStorage.getItem("cognee-credits-banner-dismissed") === "1"; } catch { return false; }
  });
  const [creditsSpentPct, setCreditsSpentPct] = useState<number | null>(null);
  const [creditsRemainingUsd, setCreditsRemainingUsd] = useState<number | null>(null);
  // True while a freshly-provisioned default dataset (handed off from onboarding
  // via sessionStorage) is still processing. Init from the flag so the skeleton
  // shows on first paint without waiting for the effect below.
  const [awaitingDataset, setAwaitingDataset] = useState<boolean>(() => {
    try { return !!sessionStorage.getItem(AWAITING_DATASET_KEY); } catch { return false; }
  });
  // Gates the dashboard render until the onboarding-redirect check below has
  // actually resolved. Without this, workspaceReady flips true (fast) before
  // that async check finishes, so the full dashboard flashes on screen for a
  // moment right before router.replace("/onboarding") kicks in.
  const [onboardingDecided, setOnboardingDecided] = useState(false);
  const router = useRouter();
  // Workspace is functionally ready only when the pod is up AND any in-flight
  // default-dataset processing has finished.
  const workspaceReady = !!cogniInstance && tenantReady && !awaitingDataset;
  const prevWorkspaceReady = useRef(workspaceReady);

  async function uploadToDataset(ds: { id: string; name: string }, files: File[]) {
    if (!cogniInstance) return;
    setIsUploading(true);
    try {
      // Load graph model, custom prompt, and ontology assignments for this dataset
      const cfg = await loadGraphModelsConfig(cogniInstance);
      const rememberOpts: { graphModel?: object; customPrompt?: string; ontologyKey?: string[] } = {};
      const assignedModel = findModelForDataset(cfg.models, ds.id);
      if (assignedModel) {
        const cleanSchema = toCleanSchema(assignedModel.schema);
        rememberOpts.graphModel = toGraphModelSchema(cleanSchema);
      }
      const promptName = findPromptForDataset(cfg.promptAssignments ?? {}, ds.id);
      if (promptName && cfg.customPrompts?.[promptName]) {
        rememberOpts.customPrompt = cfg.customPrompts[promptName];
      }
      const ontologyKey = findOntologyForDataset(cfg.ontologyAssignments ?? {}, ds.id);
      if (ontologyKey) {
        rememberOpts.ontologyKey = [ontologyKey];
      }
      await rememberData({ id: ds.id }, files, cogniInstance, rememberOpts);
      trackEvent({ pageName: "Dashboard", eventName: "dashboard_files_uploaded", additionalProperties: { dataset_id: ds.id, dataset_name: ds.name, file_count: String(files.length) } });
      notifications.show({ title: `Files uploaded to "${ds.name}"`, message: `${files.length} file(s) added. Cognify running.`, color: "blue", autoClose: 5000 });
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
    // Ref (not state) so the re-entrancy check is synchronous — guards against
    // a stray double-fired change event racing two dataset-creation calls for
    // the same deterministic dataset id, which would hit a UNIQUE constraint
    // on the backend (create_new_dataset's existence check + insert isn't atomic).
    if (isHandlingUploadRef.current || !cogniInstance || !e.target.files?.length) return;
    isHandlingUploadRef.current = true;
    const files = Array.from(e.target.files);
    e.target.value = "";

    try {
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
    } catch (err) {
      console.error("Dashboard upload failed:", err);
      notifications.show({ title: "Upload failed", message: err instanceof Error ? err.message : String(err), color: "red" });
    } finally {
      isHandlingUploadRef.current = false;
    }
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
        listSessions(cogniInstance!, { range, limit: 50 }),
      ]).then(([runData, sessionsPage]) => {
        if (cancelled) return;
        setRuns(Array.isArray(runData) ? runData : []);
        setSessions(sessionsPage?.sessions ?? []);
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

  // Derive per-integration "Connected" state from the session_id prefixes, and
  // persist it per tenant so a card stays "Connected" after its session ages out
  // of the polled window. Hydrates from localStorage and merges in live matches.
  const tenantId = tenant?.tenant_id ?? null;
  useEffect(() => {
    if (!tenantId) return;
    const storeKey = `cognee-connected-integrations-${tenantId}`;
    let persisted: Record<string, boolean> = {};
    try { persisted = JSON.parse(localStorage.getItem(storeKey) || "{}"); } catch { /* ignore */ }
    const next = { ...persisted };
    for (const [key, prefix] of Object.entries(INTEGRATION_SESSION_PREFIX)) {
      if (sessions.some((s) => s.session_id.startsWith(prefix))) next[key] = true;
    }
    if (JSON.stringify(next) !== JSON.stringify(persisted)) {
      try { localStorage.setItem(storeKey, JSON.stringify(next)); } catch { /* ignore */ }
    }
    setConnectedIntegrations((prev) => (JSON.stringify(prev) !== JSON.stringify(next) ? next : prev));
  }, [sessions, tenantId]);

  // Fetch graph node/edge counts whenever datasets or selected brain changes
  useEffect(() => {
    if (!cogniInstance || !datasets.length) return;
    const datasetsToFetch = selectedDataset ? datasets.filter(d => d.id === selectedDataset.id) : datasets;
    if (!datasetsToFetch.length) { setGraphNodes(0); setGraphEdges(0); return; }
    let cancelled = false;
    Promise.all(
      datasetsToFetch.map((ds) => getDatasetGraph(ds, cogniInstance).catch(() => null))
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
      try { sessionStorage.setItem("cognee-graph-nodes", String(totalNodes)); } catch {}
    }).catch(() => {});
    return () => { cancelled = true; };
  }, [cogniInstance, datasets, selectedDataset]);

  // Track whether the "What you can build" slider can scroll left/right, so
  // we can show/hide the edge fades + arrow buttons accordingly.
  useEffect(() => {
    const el = sliderRef.current;
    if (!el) return;
    const update = () => {
      setCanScrollLeft(el.scrollLeft > 4);
      setCanScrollRight(el.scrollLeft + el.clientWidth < el.scrollWidth - 4);
    };
    update();
    el.addEventListener("scroll", update, { passive: true });
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => { el.removeEventListener("scroll", update); ro.disconnect(); };
  }, []);

  function scrollSlider(delta: number) {
    sliderRef.current?.scrollBy({ left: delta, behavior: "smooth" });
  }

  // Onboarding redirect: send fresh users to onboarding until they have pipeline activity.
  // First-login case (no cogniInstance yet because the tenant is still being
  // provisioned in the background) redirects immediately — the data-fetch
  // effect above doesn't fire without an instance, so `loading` would
  // otherwise stay true forever.
  // Notify user when workspace becomes functionally ready (false→true transition).
  useEffect(() => {
    if (!prevWorkspaceReady.current && workspaceReady) {
      notifications.show({
        title: "Your workspace is ready",
        message: "All features are now available.",
        color: "teal",
        autoClose: 5000,
      });
    }
    prevWorkspaceReady.current = workspaceReady;
  }, [workspaceReady]);

  // Wait for a freshly-provisioned default dataset (handed off from onboarding)
  // to finish processing, then clear the flag. Best-effort: any error or a
  // missing dataset resolves to "ready" so the UI is never blocked indefinitely.
  useEffect(() => {
    let datasetId: string | null = null;
    try { datasetId = sessionStorage.getItem(AWAITING_DATASET_KEY); } catch { /* ignore */ }
    if (!datasetId) return;

    let cancelled = false;
    const clear = () => {
      if (cancelled) return;
      try { sessionStorage.removeItem(AWAITING_DATASET_KEY); } catch { /* ignore */ }
      setAwaitingDataset(false);
    };

    // Safety net: never block the dashboard longer than 30s regardless of pod state.
    // The polling below also has its own error handling, but if cogniInstance is null
    // (pod still starting) the poll never runs — this timeout prevents that deadlock.
    const safetyTimeout = setTimeout(clear, 30000);

    if (!cogniInstance) {
      return () => { cancelled = true; clearTimeout(safetyTimeout); };
    }

    pollDatasetStatus(datasetId, cogniInstance, { intervalMs: 5000 })
      .then(clear)
      .catch(clear);

    return () => { cancelled = true; clearTimeout(safetyTimeout); };
  }, [cogniInstance]);

  // Fetch credit usage for the low-balance warning banner. Not gated on
  // dismissal: the below-$1 red banner must show even after the percentage
  // banner is dismissed.
  useEffect(() => {
    if (!tenant) return;
    getCreditsOverview().then((ov) => {
      if (!ov) return;
      const t = ov.tenants.find((t) => t.tenantId === tenant.tenant_id);
      if (!t) return;
      if (t.spentUsd != null && t.maxBudgetUsd) {
        setCreditsSpentPct(Math.round((t.spentUsd / t.maxBudgetUsd) * 100));
      }
      if (t.remainingUsd != null) {
        setCreditsRemainingUsd(t.remainingUsd);
      }
    }).catch(() => {});
  }, [isOwner, tenant]);

  // Onboarding redirect: check Auth0 app state, then verify user actually has
  // datasets or runs before trusting onboarding_complete. The flag can be set
  // incorrectly by the backfill path, so we double-check with real data.
  useEffect(() => {
    // tenant === null means new user is in the welcome/provisioning flow — TenantProvider
    // will redirect to /welcome. Don't race it with an /onboarding redirect.
    if (isInitializing || !tenant) return;
    // Don't redirect until both data fetches have settled.
    if (loading || filterLoading) return;

    let cancelled = false;
    (async () => {
      const localSkipped = sessionStorage.getItem("cognee-onboarding-skipped");
      if (localSkipped) {
        setOnboardingDecided(true);
        return;
      }

      // Fast path: if user has runs OR datasets they've genuinely onboarded.
      if (runs.length > 0 || datasets.length > 0) {
        markOnboardingCompleteLocally();
        setOnboardingDecided(true);
        return;
      }

      // No local activity — verify via Auth0 before redirecting.
      // onboarding_complete is only trusted here when there's NO activity
      // (datasets + runs both empty), which means we should redirect unless
      // the flag was legitimately set AND the pod is still warming up.
      // Gate on tenantReady so we don't redirect during pod cold-start.
      if (!tenantReady) return;

      try {
        const res = await fetch("/api/user-app-state");
        if (cancelled) return;
        const appState = res.ok ? await res.json() : null;
        if (appState?.onboarding_complete) {
          // Flag is set but user has zero activity — could be a new workspace
          // that hasn't finished provisioning, or a genuinely empty account.
          // Re-check localStorage to see if this session already saw activity.
          const localComplete = localStorage.getItem("cognee-onboarding-complete");
          if (localComplete) {
            setOnboardingDecided(true);
            return; // already validated this session
          }
          // Otherwise redirect — the onboarding flow will handle re-entry gracefully.
        }
      } catch { /* fallback to redirect below */ }

      if (cancelled) return;
      // Deliberately leave onboardingDecided false here — we're navigating
      // away, so the skeleton should stay up through the route change
      // instead of flashing the dashboard for a frame first.
      router.replace("/onboarding");
    })();
    return () => { cancelled = true; };
  }, [cogniInstance, tenantReady, isInitializing, loading, filterLoading, runs, datasets, router, tenant]);

  // Show skeleton until pod is up (cogniInstance + tenantReady), any
  // freshly-provisioned default dataset has finished processing, AND the
  // onboarding-redirect check above has resolved (see onboardingDecided).
  // Workspace still provisioning: we already know this user is staying on
  // the dashboard, so the dashboard-shaped skeleton communicates progress.
  if (!workspaceReady) {
    return <DashboardSkeleton />;
  }
  // Onboarding decision still pending: this user might get redirected to
  // /onboarding in a moment, so deliberately show a neutral, non-dashboard
  // loading state here instead of DashboardSkeleton — otherwise the
  // dashboard-shaped skeleton itself reads as "the dashboard flashed" even
  // though no real data was ever rendered.
  if (!onboardingDecided) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: "100%" }}>
        <div
          style={{
            width: 28,
            height: 28,
            borderRadius: "50%",
            border: "2px solid rgba(255,255,255,0.15)",
            borderTopColor: "#BC9BFF",
            animation: "cognee-spin 0.8s linear infinite",
          }}
        />
        <style>{"@keyframes cognee-spin { to { transform: rotate(360deg); } }"}</style>
      </div>
    );
  }

  const dataLoading = loading || isInitializing || filterLoading;

  // Deduplicate runs
  const latestRuns: PipelineRun[] = [];
  const seen = new Set<string>();
  for (const r of runs) {
    const key = r.pipeline_run_id || r.id;
    if (!seen.has(key)) { seen.add(key); latestRuns.push(r); }
  }

  const filteredRuns = latestRuns;
  const filteredDatasets = selectedDataset ? datasets.filter(d => d.id === selectedDataset.id) : datasets;

  const apiCalls = filteredRuns.length;
  const connectedAgents = agents.filter((a) => a.is_agent && !a.is_default);
  const liveAgentIds = new Set(
    sessions.filter((s) => s.effective_status === "running").map((s) => s.user_id)
  );
  const liveAgents = connectedAgents.filter((a) => liveAgentIds.has(a.id) || a.status === "LIVE");
  const sessionCount = sessions.length;

  const greeting = greetingForTime();

  // Only one banner may show at a time. Priority: the percentage low-credit /
  // out-of-credits banner wins, then the below-$1 balance banner, then the
  // promotional voucher banner. The below-$1 banner still resurfaces once the
  // percentage banner is dismissed, so an out-of-credits workspace is never
  // left without a warning.
  const showCreditPctBanner =
    !creditsBannerDismissed && creditsSpentPct !== null && creditsSpentPct >= 90;
  const showLowBalanceBanner =
    !showCreditPctBanner && creditsRemainingUsd !== null && creditsRemainingUsd < 1;
  // Redeeming a voucher for cloud credits is meaningless in a self-hosted
  // install (there's no billing backend to redeem against), so this banner
  // — which would otherwise be the default fallback whenever the other two
  // credit banners are inactive (i.e. always, in OSS mode) — is cloud-only.
  const showVoucherBanner = isCloudEnvironment() && !showCreditPctBanner && !showLowBalanceBanner;

  return (
    <div style={{ minHeight: "100%", flexShrink: 0 }}>
      {/* Hidden file input for dashboard upload */}
      <input ref={uploadInputRef} type="file" multiple accept=".pdf,.csv,.txt,.md,.json,.docx" className="hidden" onChange={handleDashboardUpload} />

      {/* Dataset picker modal */}
      {showDatasetPicker && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.35)", backdropFilter: "blur(4px)", WebkitBackdropFilter: "blur(4px)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={() => { setShowDatasetPicker(false); setPendingFiles([]); }}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "rgba(15,15,15,0.92)", backdropFilter: "blur(16px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, padding: 24, width: 420, maxWidth: "calc(100vw - 32px)", display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 20px 60px rgba(0,0,0,0.6)" }}>
            <h2 style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", margin: 0 }}>Upload to which brain?</h2>
            <p style={{ fontSize: 13, color: "rgba(237,236,234,0.65)", margin: 0 }}>
              {pendingFiles.length} file{pendingFiles.length !== 1 ? "s" : ""} selected. Choose a brain to upload to.
            </p>
            <div style={{ display: "flex", flexDirection: "column", gap: 4, maxHeight: 300, overflow: "auto" }}>
              {datasets.map((ds) => (
                <button key={ds.id} onClick={() => handlePickDataset(ds)} className="cursor-pointer hover:bg-white/10" style={{ display: "flex", alignItems: "center", gap: 10, padding: "12px 14px", borderRadius: 8, border: "1px solid rgba(255,255,255,0.1)", background: "none", textAlign: "left", fontFamily: "inherit", width: "100%" }}>
                  <div style={{ width: 8, height: 8, borderRadius: 2, background: "#6510F4", flexShrink: 0 }} />
                  <span style={{ fontSize: 14, fontWeight: 500, color: "#EDECEA" }}>{ds.name}</span>
                </button>
              ))}
            </div>
            <div style={{ display: "flex", justifyContent: "flex-end" }}>
              <button onClick={() => { setShowDatasetPicker(false); setPendingFiles([]); }} className="cursor-pointer hover:bg-white/10" style={{ background: "transparent", border: "1px solid rgba(255,255,255,0.2)", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "rgba(237,236,234,0.65)", fontFamily: "inherit" }}>Cancel</button>
            </div>
          </div>
        </div>
      )}



      <div style={{ padding: "clamp(16px, 3vw, 32px)", display: "flex", flexDirection: "column", gap: 40 }}>

      {/* Compact greeting */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        <h1 style={{ margin: 0, fontSize: 26, fontWeight: 700, color: "#EDECEA", letterSpacing: "-0.02em", lineHeight: "32px" }}>
          {greeting}{selectedAgent ? `, ${ownerDisplayName(selectedAgent.email)}` : ""}
        </h1>
        {selectedAgent && (
          <span style={{ background: "#F0EDFF", borderRadius: 4, padding: "2px 8px", fontSize: 12, fontWeight: 500, color: "#6510F4" }}>{selectedAgent.agent_type}</span>
        )}
      </div>

      {/* ── Low-credit warning banner ────────────────────────────────────── */}
      {showCreditPctBanner && (
        <div style={{
          display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, flexWrap: "wrap",
          background: creditsSpentPct >= 100 ? "rgba(239,68,68,0.10)" : "rgba(234,179,8,0.10)",
          border: `1px solid ${creditsSpentPct >= 100 ? "rgba(239,68,68,0.30)" : "rgba(234,179,8,0.30)"}`,
          borderRadius: 10, padding: "12px 16px",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
              stroke={creditsSpentPct >= 100 ? "#EF4444" : "#EAB308"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
              <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
            </svg>
            <span style={{ fontSize: 13, color: creditsSpentPct >= 100 ? "#FCA5A5" : "#FDE047" }}>
              {creditsSpentPct >= 100
                ? "Your workspace has used all available credits — agent requests may fail."
                : `Your workspace has used ${creditsSpentPct}% of available credits.`}
            </span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
            {isOwner ? (
              <a href="/billing" style={{ fontSize: 13, fontWeight: 500, color: creditsSpentPct >= 100 ? "#FCA5A5" : "#FDE047", textDecoration: "underline", textUnderlineOffset: 3 }}>
                Top up credits →
              </a>
            ) : (
              <span style={{ fontSize: 13, color: "rgba(237,236,234,0.5)" }}>Ask the workspace owner to top up.</span>
            )}
            <button
              onClick={() => {
                try { sessionStorage.setItem("cognee-credits-banner-dismissed", "1"); } catch {}
                setCreditsBannerDismissed(true);
              }}
              aria-label="Dismiss"
              style={{ background: "none", border: "none", cursor: "pointer", padding: 2, color: "rgba(237,236,234,0.4)", lineHeight: 1 }}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
              </svg>
            </button>
          </div>
        </div>
      )}

      {/* ── Voucher banner / low-balance red banner ──────────────────────── */}
      {showLowBalanceBanner ? (
        <div style={{
          display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, flexWrap: "wrap",
          background: "rgba(239,68,68,0.10)",
          border: "1px solid rgba(239,68,68,0.30)",
          borderRadius: 10, padding: "12px 16px",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
              stroke="#EF4444" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
              <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
            </svg>
            <span style={{ fontSize: 13, color: "#FCA5A5" }}>
              Your Token Balance is below 1 USD. You can recharge credits{" "}
              <a href="/billing" style={{ color: "#FCA5A5", textDecoration: "underline", textUnderlineOffset: 3 }}>
                here
              </a>
              .
            </span>
          </div>
        </div>
      ) : showVoucherBanner ? (
        <div style={{
          display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, flexWrap: "wrap",
          background: "rgba(188,155,255,0.10)",
          border: "1px solid rgba(188,155,255,0.30)",
          borderRadius: 10, padding: "12px 16px",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
              stroke="#BC9BFF" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M20 12a2 2 0 0 1 2-2V7a2 2 0 0 0-2-2H4a2 2 0 0 0-2 2v3a2 2 0 0 1 0 4v3a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-3a2 2 0 0 1-2-2z"/>
              <line x1="13" y1="5" x2="13" y2="19"/>
            </svg>
            <span style={{ fontSize: 13, color: "#D9C7FF" }}>
              You have a voucher? Redeem it here.
            </span>
          </div>
          <a href="/billing" style={{
            flexShrink: 0, background: "#BC9BFF", color: "#1e1e1c",
            borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500,
            textDecoration: "none", whiteSpace: "nowrap",
          }}>
            Redeem voucher →
          </a>
        </div>
      ) : null}

      {/* ── KPI strip ────────────────────────────────────────────────────── */}
      <CompactStatsStrip
        liveAgents={liveAgents.length}
        apiCalls={apiCalls}
        sessionCount={sessionCount}
        graphNodes={graphNodes}
        graphEdges={graphEdges}
        brains={filteredDatasets.length}
        range={range}
        dataLoading={dataLoading}
      />

      {/* Agent + brain connection cards */}
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <div>
          <div style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", lineHeight: "24px" }}>Get started</div>
          <div style={{ fontSize: 13, color: "rgba(237,236,234,0.65)", marginTop: 3 }}>Connect your AI agents to give them persistent memory</div>
        </div>
        <AgentConnectionSection
          onUploadClick={() => uploadInputRef.current?.click()}
          isUploading={isUploading}
          serviceUrl={serviceUrl}
          apiKey={apiKey}
          isInitializing={isInitializing}
          hasDocuments={datasets.length > 0}
          cogniInstance={cogniInstance}
          integrationConnected={connectedIntegrations}
        />
      </div>

      {/* ── HERO: Agent memory terminal ─────────────────────────────────── */}
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>

        {/* Section header */}
        <div>
          <h2 style={{ margin: 0, fontSize: 18, fontWeight: 700, color: "#EDECEA", letterSpacing: "-0.01em", lineHeight: "24px" }}>Memory Activity</h2>
          <p style={{ margin: "3px 0 0", fontSize: 13, color: "rgba(237,236,234,0.55)" }}>A live log of every search against your memory — by your agents and by you. Click any row to see what was searched and why it answered.</p>
        </div>

        <AgentActivityTerminal
          sessions={sessions}
          runs={filteredRuns}
          agents={agents}
          datasets={filteredDatasets}
          selectedDataset={selectedDataset}
          cogniInstance={cogniInstance}
          dataLoading={dataLoading}
          range={range}
          onNavigate={(path) => router.push(path)}
        />
      </div>

      {/* ── Use-case cards — infinite slider ─────────────────────────── */}
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 18, fontWeight: 700, color: "#EDECEA", letterSpacing: "-0.01em", lineHeight: "24px" }}>What you can build</h2>
          <p style={{ margin: "3px 0 0", fontSize: 13, color: "rgba(237,236,234,0.65)" }}>Persistent memory and knowledge graphs for any domain</p>
        </div>

        <style>{`
          .usecase-slider { overflow-x: auto; scrollbar-width: none; -ms-overflow-style: none; cursor: grab; user-select: none; }
          .usecase-slider::-webkit-scrollbar { display: none; }
          .usecase-slider.is-dragging { cursor: grabbing; }
          .usecase-card { transition: border-color 200ms, box-shadow 200ms, background 200ms; flex: 0 0 280px; height: 160px; border-radius: 14px; overflow: hidden; border: 1px solid rgba(255,255,255,0.1); text-decoration: none; display: flex; align-items: center; justify-content: center; padding: 24px; background: rgba(0,0,0,0.45); backdrop-filter: blur(12px); text-align: center; }
          .usecase-card:hover { border-color: rgba(188,155,255,0.35); box-shadow: 0 8px 32px rgba(188,155,255,0.20); background: rgba(0,0,0,0.6); }
        `}</style>

        {/* Drag-to-scroll slider with edge fades + arrow affordances.
            Fades only appear when there's actually more content in that
            direction — otherwise the leftmost card's left border would be
            occluded by a constant fade. */}
        <div style={{ position: "relative" }}>
          {canScrollLeft && (
            <div style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: 64, background: "linear-gradient(to right, #000000, rgba(0,0,0,0))", zIndex: 2, pointerEvents: "none" }} />
          )}
          {canScrollRight && (
            <div style={{ position: "absolute", right: 0, top: 0, bottom: 0, width: 64, background: "linear-gradient(to left, #000000, rgba(0,0,0,0))", zIndex: 2, pointerEvents: "none" }} />
          )}
          {canScrollLeft && (
            <button
              onClick={() => scrollSlider(-320)}
              aria-label="Scroll use cases left"
              style={{ position: "absolute", left: 8, top: "50%", transform: "translateY(-50%)", zIndex: 3, width: 36, height: 36, borderRadius: "50%", border: "1px solid rgba(255,255,255,0.15)", background: "rgba(20,20,22,0.85)", backdropFilter: "blur(8px)", color: "#EDECEA", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="15 18 9 12 15 6" /></svg>
            </button>
          )}
          {canScrollRight && (
            <button
              onClick={() => scrollSlider(320)}
              aria-label="Scroll use cases right"
              style={{ position: "absolute", right: 8, top: "50%", transform: "translateY(-50%)", zIndex: 3, width: 36, height: 36, borderRadius: "50%", border: "1px solid rgba(255,255,255,0.15)", background: "rgba(20,20,22,0.85)", backdropFilter: "blur(8px)", color: "#EDECEA", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="9 18 15 12 9 6" /></svg>
            </button>
          )}

          <div
            ref={sliderRef}
            className="usecase-slider"
            onMouseDown={(e) => {
              sliderDragging.current = true;
              sliderStartX.current = e.pageX - (sliderRef.current?.offsetLeft ?? 0);
              sliderScrollLeft.current = sliderRef.current?.scrollLeft ?? 0;
              sliderRef.current?.classList.add("is-dragging");
            }}
            onMouseMove={(e) => {
              if (!sliderDragging.current || !sliderRef.current) return;
              e.preventDefault();
              const x = e.pageX - sliderRef.current.offsetLeft;
              sliderRef.current.scrollLeft = sliderScrollLeft.current - (x - sliderStartX.current) * 1.2;
            }}
            onMouseUp={() => { sliderDragging.current = false; sliderRef.current?.classList.remove("is-dragging"); }}
            onMouseLeave={() => { sliderDragging.current = false; sliderRef.current?.classList.remove("is-dragging"); }}
            style={{ display: "flex", gap: 16, padding: "4px 0 8px" }}
          >
            {([
              "A second brain",
              "Sales & deal intelligence",
              "Investment & research",
              "Docs & manuals",
              "Memory for coding agents",
            ]).map((title) => (
              <a
                key={title}
                href="https://docs.cognee.ai"
                target="_blank"
                rel="noopener noreferrer"
                className="usecase-card"
                draggable={false}
                onClick={(e) => { if (sliderRef.current && sliderRef.current.scrollLeft !== sliderScrollLeft.current) e.preventDefault(); }}
              >
                <span style={{ fontSize: 22, fontWeight: 500, color: "#EDECEA", fontFamily: '"TWKLausanne", sans-serif', letterSpacing: "0.08em", lineHeight: 1.25, textTransform: "uppercase" }}>
                  {title}
                </span>
              </a>
            ))}
          </div>
        </div>
      </div>

      {/* Upload done modal */}
      {showUploadDoneModal && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.35)", backdropFilter: "blur(4px)", WebkitBackdropFilter: "blur(4px)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={() => setShowUploadDoneModal(null)}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "rgba(15,15,15,0.92)", backdropFilter: "blur(16px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, padding: 28, width: 440, maxWidth: "calc(100vw - 32px)", display: "flex", flexDirection: "column", gap: 20, boxShadow: "0 20px 60px rgba(0,0,0,0.6)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div style={{ width: 36, height: 36, borderRadius: 8, background: "rgba(34,197,94,0.15)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#22C55E" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 6L9 17l-5-5" /></svg>
              </div>
              <div>
                <h2 style={{ fontSize: 17, fontWeight: 700, color: "#EDECEA", margin: 0 }}>Knowledge graph built</h2>
                <p style={{ fontSize: 13, color: "rgba(237,236,234,0.65)", margin: 0 }}>&ldquo;{showUploadDoneModal.datasetName}&rdquo; is now searchable.</p>
              </div>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <button
                onClick={() => { setShowUploadDoneModal(null); router.push("/search"); }}
                className="cursor-pointer hover:bg-white/10"
                style={{ display: "flex", alignItems: "center", gap: 10, padding: "12px 14px", borderRadius: 8, border: "1px solid rgba(255,255,255,0.1)", background: "rgba(255,255,255,0.06)", textAlign: "left", fontFamily: "inherit", width: "100%" }}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="rgba(188,155,255,0.60)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" /></svg>
                <div>
                  <div style={{ fontSize: 14, fontWeight: 500, color: "#EDECEA" }}>Search your data</div>
                  <div style={{ fontSize: 12, color: "rgba(237,236,234,0.65)" }}>Ask questions about your knowledge graph</div>
                </div>
              </button>
              <button
                onClick={() => { setShowUploadDoneModal(null); router.push(`/datasets/${showUploadDoneModal.datasetId}`); }}
                className="cursor-pointer hover:bg-white/10"
                style={{ display: "flex", alignItems: "center", gap: 10, padding: "12px 14px", borderRadius: 8, border: "1px solid rgba(255,255,255,0.1)", background: "rgba(255,255,255,0.06)", textAlign: "left", fontFamily: "inherit", width: "100%" }}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="rgba(188,155,255,0.60)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="6" cy="6" r="3" /><circle cx="18" cy="6" r="3" /><circle cx="12" cy="18" r="3" /><line x1="8.5" y1="7.5" x2="10.5" y2="16" /><line x1="15.5" y1="7.5" x2="13.5" y2="16" /></svg>
                <div>
                  <div style={{ fontSize: 14, fontWeight: 500, color: "#EDECEA" }}>Inspect the knowledge graph</div>
                  <div style={{ fontSize: 12, color: "rgba(237,236,234,0.65)" }}>View entities and relationships</div>
                </div>
              </button>
              <button
                onClick={() => { setShowUploadDoneModal(null); router.push("/knowledge-graph"); }}
                className="cursor-pointer hover:bg-white/10"
                style={{ display: "flex", alignItems: "center", gap: 10, padding: "12px 14px", borderRadius: 8, border: "1px solid rgba(255,255,255,0.1)", background: "rgba(255,255,255,0.06)", textAlign: "left", fontFamily: "inherit", width: "100%" }}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="rgba(188,155,255,0.60)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" /><path d="M3 9h18M9 3v18" /></svg>
                <div>
                  <div style={{ fontSize: 14, fontWeight: 500, color: "#EDECEA" }}>Explore the knowledge graph</div>
                  <div style={{ fontSize: 12, color: "rgba(237,236,234,0.65)" }}>Open the full graph visualization</div>
                </div>
              </button>
            </div>
            <button
              onClick={() => setShowUploadDoneModal(null)}
              className="cursor-pointer hover:bg-white/10"
              style={{ background: "none", border: "1px solid rgba(255,255,255,0.2)", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "rgba(237,236,234,0.65)", fontFamily: "inherit", alignSelf: "flex-end" }}
            >
              Stay here
            </button>
          </div>
        </div>
      )}

      </div>{/* end content zIndex:3 wrapper */}
    </div>
  );
}

// ── Helpers ──────────────────────────────────────────────────────────────

function greetingForTime(): string {
  const h = new Date().getHours();
  if (h < 12) return "Good morning";
  if (h < 18) return "Good afternoon";
  return "Good evening";
}

// ── Sub-components ───────────────────────────────────────────────────────

function CompactStatsStrip({
  liveAgents, apiCalls, sessionCount, graphNodes, graphEdges, brains, dataLoading,
}: {
  liveAgents: number; apiCalls: number; sessionCount: number;
  graphNodes: number | null; graphEdges: number | null;
  brains: number; range: Range; dataLoading: boolean;
}) {
  // Graph counts arrive on a separate async path (null until fetched).
  // Gate each metric on ALL inputs needed for its final value being ready,
  // so we go skeleton → final number ONCE, no intermediate 0 or "—".
  const graphLoading = dataLoading || graphNodes === null || graphEdges === null;
  const metrics: { label: string; value: number; loading: boolean; skeletonWidth: number }[] = [
    {
      // "Active Agents" — agents with a session currently running. Label stays "Agents".
      label: "Agents",
      value: liveAgents,
      loading: dataLoading,
      skeletonWidth: 24,
    },
    {
      label: "Sessions",
      value: sessionCount,
      loading: dataLoading,
      skeletonWidth: 36,
    },
    {
      label: "API calls",
      value: apiCalls,
      loading: dataLoading,
      skeletonWidth: 36,
    },
    {
      label: "Graph nodes",
      value: graphNodes ?? 0,
      loading: graphLoading,
      skeletonWidth: 48,
    },
    {
      label: "Graph edges",
      value: graphEdges ?? 0,
      loading: graphLoading,
      skeletonWidth: 48,
    },
    {
      label: "Brains",
      value: brains,
      loading: dataLoading,
      skeletonWidth: 20,
    },
  ];

  return (
    <div style={{ background: "rgba(255,255,255,0.06)", backdropFilter: "blur(12px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, overflow: "hidden" }}>
      <div style={{ display: "flex", overflowX: "auto" }}>
        {metrics.map((m, i) => {
          const isZero = !m.loading && m.value === 0;
          return (
            <div key={m.label} style={{ display: "flex", alignItems: "stretch", flex: "1 1 0", minWidth: 88 }}>
              <div style={{ flex: 1, padding: "14px 18px", display: "flex", flexDirection: "column", gap: 4 }}>
                <span style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", letterSpacing: "0.05em", textTransform: "uppercase", whiteSpace: "nowrap" }}>{m.label}</span>
                <span style={{ fontSize: 22, fontWeight: 700, color: isZero ? "rgba(255,255,255,0.2)" : "#EDECEA", fontVariantNumeric: "tabular-nums", lineHeight: "28px", display: "flex", alignItems: "center", minHeight: 28 }}>
                  {m.loading ? <SkeletonBar width={m.skeletonWidth} height={18} /> : m.value.toLocaleString()}
                </span>
              </div>
              {i < metrics.length - 1 && (
                <div style={{ width: 1, background: "rgba(255,255,255,0.08)", alignSelf: "stretch", flexShrink: 0 }} />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── AgentConnectionSection (popup modal flow) ────────────────────────────

// ── Skill copy block ──────────────────────────────────────────────────────

function SkillCopyBlock({ path, content, card }: { path: string; content: string; card?: AciAgentKey }) {
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
      {/* Destination path — purely informational, nothing runs here. Mono grey
          to match the other code snippets in the modal (InlineCodeBlock uses
          the same 0.85 alpha). */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, background: "#18181B", borderRadius: 8, padding: "10px 14px" }}>
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.45)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/>
        </svg>
        <code style={{ fontSize: 12, fontFamily: 'ui-monospace, Menlo, Monaco, "Cascadia Mono", "Segoe UI Mono", "Roboto Mono", monospace', color: "rgba(237,236,234,0.85)", flex: 1 }}>{path}</code>
      </div>

      {/* Copy button — solid lavender so it reads as the primary action. */}
      <button
        onClick={handleCopy}
        disabled={phase !== "idle"}
        style={{
          display: "flex", alignItems: "center", justifyContent: "center", gap: 7,
          background: phase === "done" ? "rgba(34,197,94,0.15)" : phase === "copying" ? "#A87CFF" : "#BC9BFF",
          border: `1px solid ${phase === "done" ? "rgba(34,197,94,0.4)" : "transparent"}`,
          borderRadius: 8, padding: "9px 16px", fontSize: 13, fontWeight: 500,
          cursor: phase === "idle" ? "pointer" : "default",
          color: phase === "done" ? "#22C55E" : "#1e1e1c", fontFamily: "inherit",
          transition: "background 200ms, border-color 200ms, color 200ms",
          width: "100%",
        }}
      >
        {phase === "idle" && (
          <>
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none"><rect x="5" y="5" width="8" height="8" rx="1.5" stroke="#1e1e1c" strokeWidth="1.5"/><path d="M11 3H4.5A1.5 1.5 0 003 4.5V11" stroke="#1e1e1c" strokeWidth="1.5" strokeLinecap="round"/></svg>
            Copy install command
          </>
        )}
        {phase === "copying" && (
          <>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#1e1e1c" strokeWidth="2.5" strokeLinecap="round" style={{ animation: "aci-spin 0.7s linear infinite" }}><path d="M21 12a9 9 0 11-6.219-8.56"/></svg>
            Copying to clipboard…
          </>
        )}
        {phase === "done" && (
          <>
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none"><path d="M3 8.5L6.5 12L13 5" stroke="#22C55E" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
            Copied — paste &amp; run in your local terminal
          </>
        )}
      </button>

      {/* Shows what the user will paste — runs entirely on their machine, not ours */}
      {phase === "done" && (
        <div style={{ background: "#18181B", borderRadius: 8, padding: "10px 14px", fontFamily: 'ui-monospace, Menlo, Monaco, "Cascadia Mono", "Segoe UI Mono", "Roboto Mono", monospace', fontSize: 11, lineHeight: "18px" }}>
          <div style={{ color: "#585B70" }}>$ <span style={{ color: "#CDD6F4" }}>paste &amp; run the command in your terminal</span></div>
          <div style={{ color: "#A6E3A1", marginTop: 3 }}>↳ writes {path} on your local machine</div>
        </div>
      )}
    </div>
  );
}

function InlineCodeBlock({ code, toCopy, loading, card, block }: { code: string; toCopy?: string; loading?: boolean; card?: AciAgentKey; block?: string }) {
  const [copied, setCopied] = useState(false);
  function doCopy() {
    if (loading) return;
    trackEvent({ pageName: "Dashboard", eventName: "agent_config_copied", additionalProperties: { card: card ?? "unknown", block: block ?? "code" } });
    navigator.clipboard.writeText(toCopy ?? code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  }
  return (
    <div
      onClick={(e) => { e.stopPropagation(); doCopy(); }}
      style={{ background: "#18181B", borderRadius: 8, padding: "11px 14px", display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, cursor: loading ? "wait" : "pointer" }}
    >
      <pre style={{ margin: 0, fontSize: 12.5, fontFamily: 'ui-monospace, Menlo, Monaco, "Cascadia Mono", "Segoe UI Mono", "Roboto Mono", monospace', color: loading ? "#585B70" : "rgba(237,236,234,0.65)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", flex: 1 }}>
        {loading ? "Loading…" : code}
      </pre>
      <button
        onClick={(e) => { e.stopPropagation(); doCopy(); }}
        aria-label={copied ? "Copied" : "Copy"}
        style={{ background: "none", border: "none", cursor: loading ? "wait" : "pointer", flexShrink: 0, padding: 2, borderRadius: 4 }}
      >
        {copied ? (
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M3.5 8.5L6.5 11.5L12.5 4.5" stroke="#22C55E" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
        ) : (
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none"><rect x="5" y="5" width="8" height="8" rx="1.5" stroke="#6B7280" strokeWidth="1.5" /><path d="M11 3H4.5A1.5 1.5 0 003 4.5V11" stroke="#6B7280" strokeWidth="1.5" strokeLinecap="round" /></svg>
        )}
      </button>
    </div>
  );
}

interface AciStepDef {
  title: string;
  description: string;
  code?: string;
  codeToCopy?: string;
  loading?: boolean;
  /** When set, renders multiple separately-copyable code blocks (e.g. commands run one at a time) */
  codeBlocks?: { code: string; toCopy?: string; label?: string }[];
  /** When set, renders a SkillCopyBlock instead of (or alongside) the code block */
  skillPath?: string;
  skillContent?: string;
}

type AciAgentKey = "upload" | "claude-code" | "codex" | "openclaw" | "api-mcp";

function AgentConnectionSection({
  onUploadClick, isUploading, serviceUrl, apiKey, isInitializing, hasDocuments, cogniInstance, integrationConnected = {},
}: {
  onUploadClick: () => void; isUploading: boolean;
  serviceUrl: string | null; apiKey: string | null; isInitializing: boolean;
  hasDocuments: boolean;
  cogniInstance: ReturnType<typeof useCogniInstance>["cogniInstance"];
  integrationConnected?: Record<string, boolean>;
}) {
  const router = useRouter();
  const [activeKey, setActiveKey] = useState<AciAgentKey | null>(null);
  const [stepIndexMap, setStepIndexMap] = useState<Partial<Record<AciAgentKey, number>>>({});
  // Live connection check for the Claude Code flow: while the modal is open we
  // poll for new sessions. A session that wasn't present when the modal opened
  // means the user's agent successfully connected to Cognee Cloud — we then mark
  // the "connect" step done and jump to the final step. Index of the connect
  // step within the claude-code steps (creds, plugin, → connect, review).
  const CLAUDE_CONNECT_STEP = 2;
  const [connectVerified, setConnectVerified] = useState(false);

  const baseUrl = serviceUrl || "https://your-tenant.aws.cognee.ai";
  const resolvedKey = apiKey || "your-api-key";
  // Cognee Cloud standardizes on COGNEE_BASE_URL everywhere user-facing — the
  // plugins/skills (claude-code, codex, …) and the REST flows all read it.
  const CREDS_CODE = `export COGNEE_BASE_URL="${baseUrl}"\nexport COGNEE_API_KEY="${resolvedKey}"`;

  // Poll for a freshly-created session while the Claude Code modal is open.
  // Baseline the existing session ids on open; the first new (non-search-ui)
  // session that appears flips the connect step to "connected".
  useEffect(() => {
    if (activeKey !== "claude-code" || !cogniInstance || connectVerified) return;
    let cancelled = false;
    const baseline = new Set<string>();
    let primed = false;

    const realSessionIds = (rows: { session_id: string }[]) =>
      rows.map((s) => s.session_id).filter((id) => !id.startsWith(SEARCH_SESSION_PREFIX));

    async function check() {
      const page = await listSessions(cogniInstance!, { range: "24h", limit: 50 });
      if (cancelled) return;
      const ids = realSessionIds(page.sessions);
      if (!primed) {
        // First tick establishes the baseline of pre-existing sessions.
        ids.forEach((id) => baseline.add(id));
        primed = true;
        return;
      }
      if (ids.some((id) => !baseline.has(id))) {
        setConnectVerified(true);
        // Mark the connect step done and reveal the final "review" step.
        setStepIndexMap((prev) => {
          const cur = prev["claude-code"] ?? 0;
          return cur <= CLAUDE_CONNECT_STEP ? { ...prev, "claude-code": CLAUDE_CONNECT_STEP + 1 } : prev;
        });
      }
    }

    check();
    const id = setInterval(check, 7000);
    return () => { cancelled = true; clearInterval(id); };
  }, [activeKey, cogniInstance, connectVerified]);

  // Reset the verified flag whenever the Claude Code modal is (re)opened.
  useEffect(() => {
    if (activeKey === "claude-code") setConnectVerified(false);
  }, [activeKey]);

  const CARDS_CFG: { key: AciAgentKey; name: string; description: string }[] = [
    { key: "claude-code", name: "Claude Code", description: "Give Claude Code persistent memory across all your projects" },
    { key: "codex",       name: "Codex",       description: "Connect OpenAI Codex to your knowledge graph via the Cognee plugin" },
    { key: "openclaw",    name: "Openclaw",     description: "Connect Openclaw to your knowledge graph via AGENTS.md" },
    { key: "api-mcp",     name: "API / MCP",    description: "Connect any agent or app via the REST API or MCP" },
    { key: "upload",      name: "Company Brain", description: "Upload PDFs, docs, and data to build your knowledge graph" },
  ];

  function getSteps(key: AciAgentKey): AciStepDef[] {
    const credStep: AciStepDef = {
      title: "Set your API credentials",
      description: "Open a terminal and run these commands to configure your Cognee endpoint and key.",
      code: `export COGNEE_BASE_URL="${baseUrl}"`,
      codeToCopy: CREDS_CODE,
      loading: isInitializing,
    };
    if (key === "claude-code") return [
      credStep,
      {
        title: "Install the Cognee plugin",
        description: "Run these in your terminal one at a time — register the Cognee marketplace, then install the memory plugin.",
        codeBlocks: [
          { code: CLAUDE_MARKETPLACE_ADD },
          { code: CLAUDE_PLUGIN_INSTALL },
        ],
      },
      {
        title: "Upload something to Cognee",
        description: "Pick one and paste it into Claude — it stores the content in your Cognee memory so you can recall it in the next step.",
        codeBlocks: [
          { label: "Option A · Your existing memory", code: UPLOAD_MEMORY_PROMPT },
          { label: "Option B · Try it with a sample", code: UPLOAD_SAMPLE_PROMPT },
        ],
      },
      {
        title: connectVerified ? "Connected — session detected ✓" : "Recall it from Cognee",
        description: connectVerified
          ? "We detected your new session in Cognee Cloud — you're connected."
          : "First run /exit to close the session — that syncs it into Cognee Cloud — then reopen Claude Code and ask the question below. Answering from a fresh session proves it's recalling from your cloud memory.",
        codeBlocks: [
          { code: "/exit" },
          { code: RECALL_SAMPLE_PROMPT },
        ],
      },
      {
        title: "You're all set",
        description: "The Cognee plugin hooks into Claude Code's lifecycle — no curl or manual API calls — and captures your session as you work. When a session ends (e.g. /exit), it consolidates that session into your Cognee Cloud knowledge graph, and every new session automatically recalls it back. Sessions are disposable; your memory isn't.",
      },
    ];
    if (key === "codex") return [
      credStep,
      {
        title: "Install the Cognee plugin",
        description: "Run these in your terminal one at a time — enable Codex hooks, register the Cognee marketplace, then install the memory plugin.",
        codeBlocks: [
          { code: CODEX_HOOKS_ENABLE },
          { code: CODEX_MARKETPLACE_ADD },
          { code: CODEX_PLUGIN_INSTALL },
        ],
      },
      {
        title: "Upload something to Cognee",
        description: "Pick one and paste it into Codex — it stores the content in your Cognee memory so you can recall it in the next step.",
        codeBlocks: [
          { label: "Option A · Your existing memory", code: UPLOAD_MEMORY_PROMPT },
          { label: "Option B · Try it with a sample", code: UPLOAD_SAMPLE_PROMPT },
        ],
      },
      {
        title: "Recall it from Cognee",
        description: "First run /exit to close the session — that syncs it into Cognee Cloud — then reopen Codex and ask the question below. Answering from a fresh session proves it's recalling from your cloud memory.",
        codeBlocks: [
          { code: "/exit" },
          { code: RECALL_SAMPLE_PROMPT },
        ],
      },
      {
        title: "You're all set",
        description: "The Cognee plugin hooks into Codex's lifecycle — no curl or manual API calls — and captures your session as you work. When a session ends (e.g. /exit), it consolidates that session into your Cognee Cloud knowledge graph, and every new session automatically recalls it back. Sessions are disposable; your memory isn't.",
      },
    ];
    if (key === "openclaw") return [
      credStep,
      {
        title: "Install the Cognee skill",
        description: "Click below to copy the install command to your clipboard, then paste and run it in your local terminal. Nothing is sent to our servers — the skill file is written on your own machine.",
        skillPath: "~/.openclaw/skills/cognee/SKILL.md",
        skillContent: OPENCLAW_SKILL_INSTALL,
      },
      {
        title: "Test the connection",
        description: `Open Openclaw in your project and ask: "What do you know from cognee?" — if it responds with knowledge from your brain, you're connected.`,
      },
    ];
    if (key === "api-mcp") return [
      credStep,
      {
        title: "Query the REST API",
        description: "Send a recall query to your Cognee endpoint from any HTTP client or language.",
        code: `curl -X POST ${baseUrl}/api/v1/recall`,
        codeToCopy: `curl -X POST ${baseUrl}/api/v1/recall \\\n  -H "X-Api-Key: ${resolvedKey}" \\\n  -H "Content-Type: application/json" \\\n  -d '{"query": "What are the main entities?"}'`,
        loading: isInitializing,
      },
      {
        title: "Or install the Cognee skill",
        description: "Prefer skills? Run this command from your project root to create the skill file, then point your agent at it (skills directory, instructions file, or system prompt). The skill teaches your agent to call the Cognee API using the credentials from step 1.",
        code: "skills/cognee/SKILL.md",
        codeToCopy: GENERIC_SKILL_INSTALL,
      },
      {
        title: "Test the connection",
        description: `Ask your agent: "What do you know from cognee?" — Cognee's memory should respond with knowledge from your brain.`,
      },
    ];
    return [];
  }

  function handleCardClick(key: AciAgentKey) {
    trackEvent({ pageName: "Dashboard", eventName: "agent_card_clicked", additionalProperties: { card: key } });
    if (key === "upload") { onUploadClick(); return; }
    setActiveKey(key);
    // Preserve position if user re-opens the same agent
    if (stepIndexMap[key] === undefined) {
      setStepIndexMap(s => ({ ...s, [key]: 0 }));
    }
  }

  function goToStep(key: AciAgentKey, idx: number) {
    trackEvent({ pageName: "Dashboard", eventName: "agent_step_viewed", additionalProperties: { card: key, step: String(idx) } });
    setStepIndexMap(prev => ({ ...prev, [key]: idx }));
  }

  const popupOpen = activeKey !== null && activeKey !== "upload";
  const activeCfg = CARDS_CFG.find(c => c.key === activeKey);

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
        @media (max-width: 800px) { .aci-card-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
        @media (max-width: 480px) { .aci-card-grid { grid-template-columns: 1fr; } }
        @media (prefers-reduced-motion: reduce) {
          .aci-card-logo, .aci-step-body, .aci-popup { transition: none !important; animation: none !important; }
        }
      `}</style>

      {/* Agent + brain card grid — visual style from connect-agent page */}
      <div className="aci-card-grid">
        {CARDS_CFG.map((card) => {
          const cardSteps = card.key !== "upload" ? getSteps(card.key as AciAgentKey) : [];
          const connected = card.key === "upload" ? hasDocuments : !!integrationConnected[card.key];
          const isActive = activeKey === card.key;
          const isUpload = card.key === "upload";

          const logoNode = isUpload ? (
            // Company Brain stacked-document icon (matches main branch CompanyDataIcon)
            <svg height="110" viewBox="0 0 80 100" fill="none" xmlns="http://www.w3.org/2000/svg">
              <rect x="16" y="6" width="54" height="70" rx="6" fill="#D4D4D8" stroke="#71717A" strokeWidth="3.5"/>
              <rect x="8" y="14" width="54" height="70" rx="6" fill="#E4E4E7" stroke="#71717A" strokeWidth="3.5"/>
              <rect x="2" y="22" width="54" height="70" rx="6" fill="#F4F4F5" stroke="#52525B" strokeWidth="3.5"/>
              <path d="M38 22v16h18" stroke="#52525B" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round"/>
              <line x1="12" y1="52" x2="46" y2="52" stroke="#52525B" strokeWidth="3" strokeLinecap="round"/>
              <line x1="12" y1="63" x2="46" y2="63" stroke="#52525B" strokeWidth="3" strokeLinecap="round"/>
              <line x1="12" y1="74" x2="30" y2="74" stroke="#52525B" strokeWidth="3" strokeLinecap="round"/>
            </svg>
          ) : card.key === "api-mcp" ? (
            <svg height="110" viewBox="0 0 90 110" fill="none" xmlns="http://www.w3.org/2000/svg">
              <rect x="5" y="20" width="80" height="50" rx="10" fill="#1a1a2e" stroke="rgba(255,255,255,0.15)" strokeWidth="2"/>
              <path d="M25 35L16 45L25 55" stroke="rgba(188,155,255,0.60)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M65 35L74 45L65 55" stroke="rgba(188,155,255,0.60)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
              <line x1="50" y1="30" x2="40" y2="60" stroke="rgba(255,255,255,0.4)" strokeWidth="2.5" strokeLinecap="round"/>
            </svg>
          ) : (
            <img
              src={card.key === "claude-code" ? "/visuals/logos/claude.svg" : card.key === "codex" ? "/visuals/logos/codex.svg" : "/visuals/logos/openclaw.svg"}
              alt={card.name}
              style={{ height: 110, width: "auto" }}
            />
          );

          const logoRight = isUpload ? -12 : card.key === "api-mcp" ? -28 : -36;

          const ctaLabel = isUpload
            ? (connected ? "Add more data" : "Upload data")
            : card.key === "api-mcp" ? "Connect" : "Connect agent";

          return (
            <button
              key={card.key}
              className="aci-card"
              onClick={() => handleCardClick(card.key)}
              aria-haspopup={!isUpload ? "dialog" : undefined}
              disabled={isUploading && isUpload}
              style={{
                position: "relative",
                background: isActive ? "rgba(188,155,255,0.20)" : "rgba(255,255,255,0.06)",
                backdropFilter: "blur(12px)",
                border: `1px solid ${isActive ? "rgba(188,155,255,0.35)" : "rgba(255,255,255,0.1)"}`,
                borderRadius: 12,
                padding: "20px 16px 0 16px",
                height: 160,
                overflow: "hidden",
                cursor: (isUploading && isUpload) ? "wait" : "pointer",
                textAlign: "left",
                display: "flex",
                flexDirection: "column",
                transition: "border-color 150ms, background 150ms",
              }}
            >
              {/* Top-right: connected status only */}
              {connected && (
                <div style={{ position: "absolute", top: 12, right: 12, display: "flex", alignItems: "center", gap: 4, zIndex: 1 }}>
                  <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#22C55E", flexShrink: 0 }} />
                  <span style={{ fontSize: 11, fontWeight: 500, color: "#16A34A", whiteSpace: "nowrap" }}>Connected</span>
                </div>
              )}

              {/* Card name — thin weight, TWK Lausanne matches subpage exactly */}
              <span style={{
                fontSize: 16, fontWeight: 300, color: "#EDECEA",
                lineHeight: 1.25, letterSpacing: "-0.01em",
                fontFamily: '"TWKLausanne", sans-serif',
                paddingRight: connected ? 90 : 16,
              }}>
                {card.name}
              </span>

              {/* Bottom-left action button chip */}
              <div style={{ position: "absolute", bottom: 14, left: 16, zIndex: 1 }}>
                {isUploading && isUpload ? (
                  <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                    <div style={{ width: 9, height: 9, borderRadius: "50%", border: "1.5px solid #D1D5DB", borderTopColor: "#6510F4", animation: "aci-spin 0.8s linear infinite", flexShrink: 0 }} />
                    <span style={{ fontSize: 11, fontWeight: 500, color: "rgba(237,236,234,0.65)" }}>Uploading…</span>
                  </div>
                ) : (
                  <span
                    className="aci-cta-chip"
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: 4,
                      background: "rgba(20,20,22,0.92)",
                      backdropFilter: "blur(12px)",
                      WebkitBackdropFilter: "blur(12px)",
                      border: "1px solid rgba(237,236,234,0.65)",
                      borderRadius: 6,
                      padding: "5px 10px",
                      fontSize: 12,
                      fontWeight: 500,
                      color: "rgba(237,236,234,0.65)",
                      whiteSpace: "nowrap",
                      transition: "background 150ms",
                    }}
                  >
                    {ctaLabel}
                  </span>
                )}
              </div>

              {/* Large overflowing logo — mirrors connect-agent page layout */}
              <div className="aci-card-logo" style={{ position: "absolute", bottom: -18, right: logoRight, pointerEvents: "none" }}>
                {logoNode}
              </div>
            </button>
          );
        })}
      </div>

      {/* Popup modal */}
      {popupOpen && activeCfg && activeKey && (() => {
        const steps = getSteps(activeKey);
        const currentStep = stepIndexMap[activeKey] ?? 0;
        return (
          <div
            role="dialog"
            aria-modal="true"
            aria-label={`Connect ${activeCfg.name}`}
            style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.35)", backdropFilter: "blur(4px)", WebkitBackdropFilter: "blur(4px)", zIndex: 200, display: "flex", alignItems: "center", justifyContent: "center", padding: 16 }}
            onClick={() => setActiveKey(null)}
          >
            <div
              className="aci-popup"
              onClick={(e) => e.stopPropagation()}
              style={{ background: "rgba(15,15,15,0.92)", backdropFilter: "blur(16px)", borderRadius: 14, width: 520, maxWidth: "100%", boxShadow: "0 20px 60px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.1)", overflow: "hidden", animation: "aci-popup 200ms cubic-bezier(0.22,1,0.36,1) forwards" }}
            >
              {/* Modal header */}
              <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "16px 20px", borderBottom: "1px solid rgba(255,255,255,0.1)" }}>
                {activeKey === "api-mcp" ? (
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" style={{ flexShrink: 0 }}><rect x="3" y="6" width="18" height="12" rx="2" stroke="rgba(237,236,234,0.7)" strokeWidth="1.5"/><path d="M7 9L4 12L7 15" stroke="rgba(188,155,255,0.60)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/><path d="M17 9L20 12L17 15" stroke="rgba(188,155,255,0.60)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/><line x1="13" y1="8" x2="11" y2="16" stroke="rgba(237,236,234,0.5)" strokeWidth="1.5" strokeLinecap="round"/></svg>
                ) : (
                  <img
                    src={activeKey === "claude-code" ? "/visuals/logos/claude.svg" : activeKey === "codex" ? "/visuals/logos/codex.svg" : "/visuals/logos/openclaw.svg"}
                    alt={activeCfg?.name}
                    style={{ width: 24, height: 24, objectFit: "contain", flexShrink: 0 }}
                  />
                )}
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 15, fontWeight: 700, color: "#EDECEA", lineHeight: "20px" }}>Connect {activeCfg.name}</div>
                  <div style={{ fontSize: 12, color: "rgba(237,236,234,0.45)", marginTop: 1 }}>Step {currentStep + 1} of {steps.length}</div>
                </div>
                <button
                  onClick={() => setActiveKey(null)}
                  aria-label="Close"
                  style={{ background: "none", border: "none", color: "rgba(237,236,234,0.65)", cursor: "pointer", padding: 4, borderRadius: 6, lineHeight: 1, flexShrink: 0 }}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>
                </button>
              </div>

              {/* Step rows — click any row to navigate to that step */}
              {steps.map((step, i) => {
                const isActive = currentStep === i;
                const isDone = i < currentStep;
                return (
                  <div
                    key={i}
                    className="aci-step-row"
                    data-active={isActive ? "true" : undefined}
                    onClick={() => goToStep(activeKey, i)}
                    style={{ borderBottom: i < steps.length - 1 ? "1px solid rgba(255,255,255,0.07)" : "none", cursor: isActive ? "default" : "pointer" }}
                  >
                    {/* Row header — always visible */}
                    <div style={{ display: "flex", alignItems: "center", gap: 12, padding: isActive ? "14px 20px 0" : "14px 20px" }}>
                      {/* Step indicator */}
                      <div style={{
                        width: 24, height: 24, borderRadius: "50%", flexShrink: 0,
                        display: "flex", alignItems: "center", justifyContent: "center",
                        background: isDone ? "rgba(34,197,94,0.18)" : isActive ? "#6510F4" : "rgba(255,255,255,0.1)",
                        transition: "background 200ms ease",
                      }}>
                        {isDone ? (
                          <svg width="10" height="10" viewBox="0 0 16 16" fill="none" style={{ animation: "aci-check 220ms cubic-bezier(0.22,1,0.36,1) forwards" }}>
                            <path d="M3 8.5L6.5 12L13 5" stroke="#22C55E" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
                          </svg>
                        ) : (
                          <span style={{ fontSize: 11, fontWeight: 700, color: isActive ? "#fff" : "rgba(237,236,234,0.65)", lineHeight: 1 }}>{i + 1}</span>
                        )}
                      </div>
                      {/* Step title */}
                      <span style={{ flex: 1, fontSize: 14, fontWeight: isActive ? 500 : 400, color: isDone ? "rgba(237,236,234,0.45)" : isActive ? "#EDECEA" : "rgba(237,236,234,0.30)" }}>
                        {step.title}
                      </span>
                      {isDone && (
                        <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase", background: "rgba(34,197,94,0.18)", color: "#22C55E", borderRadius: 100, padding: "2px 8px", flexShrink: 0, animation: "aci-check 200ms ease forwards" }}>
                          Done
                        </span>
                      )}
                    </div>

                    {/* Expanded content — grid-template-rows animates from exact content height,
                        keeping modal height stable when switching between steps */}
                    <div
                      className="aci-step-body"
                      style={{
                        display: "grid",
                        gridTemplateRows: isActive ? "1fr" : "0fr",
                        opacity: isActive ? 1 : 0,
                        transition: "grid-template-rows 260ms ease, opacity 200ms ease",
                      }}
                    >
                      <div style={{ overflow: "hidden" }}>
                      <div
                        onClick={(e) => e.stopPropagation()}
                        style={{ padding: "10px 20px 18px 56px" }}
                      >
                        {step.description && (
                          <p style={{ fontSize: 13, color: "rgba(237,236,234,0.65)", margin: "0 0 12px", lineHeight: 1.6 }}>{step.description}</p>
                        )}
                        {step.code && (
                          <InlineCodeBlock code={step.code} toCopy={step.codeToCopy} loading={step.loading} card={activeKey ?? undefined} block={step.title} />
                        )}
                        {step.codeBlocks && (
                          <div style={{ display: "flex", flexDirection: "column", gap: step.codeBlocks.some(cb => cb.label) ? 14 : 8 }}>
                            {step.codeBlocks.map((cb, j) => (
                              cb.label ? (
                                <div key={j}>
                                  <div style={{ fontSize: 13, fontWeight: 600, color: "#EDECEA", marginBottom: 6 }}>{cb.label}</div>
                                  <InlineCodeBlock code={cb.code} toCopy={cb.toCopy} card={activeKey ?? undefined} block={cb.label} />
                                </div>
                              ) : (
                                <InlineCodeBlock key={j} code={cb.code} toCopy={cb.toCopy} card={activeKey ?? undefined} block={step.title} />
                              )
                            ))}
                          </div>
                        )}
                        {step.skillPath && step.skillContent && (
                          <SkillCopyBlock path={step.skillPath} content={step.skillContent} card={activeKey ?? undefined} />
                        )}
                        {i < steps.length - 1 ? (
                          <p style={{ margin: "10px 0 0", fontSize: 12, color: "rgba(237,236,234,0.65)" }}>
                            Click step {i + 2} when ready ↓
                          </p>
                        ) : (
                          <button
                            onClick={(e) => { e.stopPropagation(); router.push("/sessions"); }}
                            style={{ marginTop: 12, display: "inline-flex", alignItems: "center", gap: 5, background: "none", border: "1px solid rgba(255,255,255,0.2)", borderRadius: 8, padding: "7px 14px", fontSize: 13, fontWeight: 500, color: "#EDECEA", fontFamily: "inherit", cursor: "pointer" }}
                          >
                            Go to Sessions →
                          </button>
                        )}
                      </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })()}
    </div>
  );
}
