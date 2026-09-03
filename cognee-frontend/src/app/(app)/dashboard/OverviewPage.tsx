"use client";

import React, { useRef, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { notifications } from "@mantine/notifications";
import { trackEvent } from "@/modules/analytics";
import { useCogniInstance, useTenant } from "@/modules/tenant/TenantProvider";
import { useCurrentUser } from "@/modules/users/useCurrentUser";
import type { SessionRow } from "@/modules/sessions/getSessions";
import { useTenantHourlyCosts } from "@/modules/billing/useTenantHourlyCosts";
import { useFilter, useRefreshDatasetsOnMount } from "@/ui/layout/FilterContext";
import type { PipelineRun, Range } from "@/ui/elements/AgentActivityTerminal";
import DashboardSkeleton from "./DashboardSkeleton";
import { useDashboardTelemetry } from "./hooks/useDashboardTelemetry";
import { useConnectedIntegrations } from "./hooks/useConnectedIntegrations";
import { useDataSourceStatuses } from "./hooks/useDataSourceStatuses";
import { DATA_SOURCE_CARDS } from "@/modules/integrations/dataSourceCards";
import { useCreditsBanner } from "./hooks/useCreditsBanner";
import { useAwaitingDataset } from "./hooks/useAwaitingDataset";
import { useOnboardingRedirect } from "./hooks/useOnboardingRedirect";
import { useDatasetUpload } from "./hooks/useDatasetUpload";
import { CreditBanners } from "./partials/CreditBanners";
import { AgentConnectionSection } from "./partials/AgentConnectionSection";
import { DatasetPickerModal } from "./partials/DatasetPickerModal";
import { UploadDoneModal } from "./partials/UploadDoneModal";
import PodUnreachableCard from "@/ui/elements/PodUnreachableCard";
// Terminal/ASCII dashboard redesign (Paper WO-0) — see partials/redesign.
import { GetStartedBar } from "./partials/redesign/GetStartedBar";
import { AsciiFrame } from "./partials/redesign/AsciiFrame";
import { MemoryFlowDiagram } from "./partials/redesign/MemoryFlowDiagram";
import type { FlowSource, FlowAgent, NodeStatus } from "./partials/redesign/MemoryFlowDiagram";
import { CostPanel } from "./partials/redesign/CostPanel";
import { PerformancePanel } from "./partials/redesign/PerformancePanel";
import type { TopicScore } from "./partials/redesign/PerformancePanel";
import { ActivityPanel } from "./partials/redesign/ActivityPanel";
import type { DashRange } from "./partials/redesign/RangeToggle";
import { FONT, T } from "./partials/redesign/mono";

const DATA_SOURCE_PROVIDERS = DATA_SOURCE_CARDS.map((card) => card.key);

interface AgentDef { name: string; logo: string; prefixes: string[] }

// Persistent agents — always listed in the memory graph. They read as "live"
// only while a matching session is actively running; with no active session
// they show disconnected. Keep prefixes in sync with the shipped plugins.
const PERSISTENT_AGENT_DEFS: AgentDef[] = [
  { name: "Claude Code", logo: "claude", prefixes: ["claude_", "cc_"] },
  { name: "Codex", logo: "codex", prefixes: ["codex_"] },
  { name: "Claude Desktop", logo: "claude", prefixes: ["claude_desktop_"] },
  { name: "OpenClaw", logo: "openclaw", prefixes: ["openclaw_"] },
];

// Dynamic agents — NOT shown by default. They appear only once they've
// registered (a session with their prefix has been seen), then follow the same
// active/inactive rule as the persistent ones.
const DYNAMIC_AGENT_DEFS: AgentDef[] = [
  { name: "Hermes Agent", logo: "hermes", prefixes: ["hermes_"] },
  { name: "VS Code", logo: "vscode", prefixes: ["vscode_"] },
  { name: "Cursor", logo: "cursor", prefixes: ["cursor_"] },
  { name: "Gemini CLI", logo: "gemini", prefixes: ["gemini_"] },
  { name: "Cline", logo: "cline", prefixes: ["cline_"] },
];

function matches(session: SessionRow, prefixes: string[]): boolean {
  return prefixes.some((p) => session.session_id.startsWith(p));
}

/** An agent has ever registered if any session (past or present) carries its prefix. */
function hasRegistered(sessions: SessionRow[], prefixes: string[]): boolean {
  return sessions.some((s) => matches(s, prefixes));
}

/** Active only while a matching session is running; otherwise disconnected. */
function agentStatus(sessions: SessionRow[], prefixes: string[]): NodeStatus {
  return sessions.some((s) => s.effective_status === "running" && matches(s, prefixes)) ? "live" : "disconnected";
}

export default function OverviewPage(): React.ReactElement {
  const { cogniInstance, isInitializing, serviceUrl, apiKey } = useCogniInstance();
  const { tenantReady, podUnreachable, tenant, isOwner } = useTenant();
  const { agents, datasets, selectedAgent, loading: filterLoading } = useFilter();
  // Greeting identity comes from the logged-in user, NOT the FilterContext
  // agent selection (which defaults to null and would leave the greeting nameless).
  const { data: currentUser } = useCurrentUser();
  useRefreshDatasetsOnMount();
  const router = useRouter();
  const uploadInputRef = useRef<HTMLInputElement>(null);

  // Overview time range matches the backend hourly cost window.
  const [range, setRange] = useState<DashRange>("7d");
  const telemetryRange: Range = range;

  // Clicking a disconnected agent node in the memory graph opens the same
  // onboarding connection card as the Integrations page, resolved by name
  // since the graph's agent defs and AGENT_CARDS share no slug convention.

  const awaitingDataset = useAwaitingDataset();
  const workspaceReady = !!cogniInstance && tenantReady && !awaitingDataset;

  // Core live/idle badge = the pod's /health endpoint, probed on a slow poll.
  const [memoryHealthy, setMemoryHealthy] = useState(false);
  useEffect(() => {
    if (!cogniInstance) { setMemoryHealthy(false); return; }
    let cancelled = false;
    const probe = async (): Promise<void> => {
      try {
        const res = await cogniInstance.fetch("/health");
        if (!cancelled) setMemoryHealthy(res.ok);
      } catch {
        if (!cancelled) setMemoryHealthy(false);
      }
    };
    void probe();
    const id = setInterval(() => { void probe(); }, 30_000);
    return () => { cancelled = true; clearInterval(id); };
  }, [cogniInstance]);
  const prevWorkspaceReady = useRef(workspaceReady);

  useEffect(() => {
    if (!prevWorkspaceReady.current && workspaceReady) {
      notifications.show({
        title: "Your workspace is ready",
        message: "All features are now available.",
        color: "teal",
        autoClose: 5000,
      });
      trackEvent({ pageName: "Dashboard", eventName: "workspace_active" });
    }
    prevWorkspaceReady.current = workspaceReady;
  }, [workspaceReady]);

  const { runs, sessions, loading } = useDashboardTelemetry(telemetryRange);
  const { data: hourlyCosts = null } = useTenantHourlyCosts(tenant?.tenant_id ?? null, range);
  const connectedIntegrations = useConnectedIntegrations(sessions, tenant?.tenant_id ?? null);
  const sourceStatuses = useDataSourceStatuses(DATA_SOURCE_PROVIDERS, tenant?.tenant_id ?? null);
  // Dashboard metrics always report workspace-wide totals — pass null so the
  // graph counts never inherit a dataset selection carried over from another page.
  const credits = useCreditsBanner();
  const upload = useDatasetUpload();
  useOnboardingRedirect();

  // Fires once per item, the moment it first becomes true — not on every
  // render where it's already true (prevChecklistDone tracks that boundary).
  const checklistDone = useMemo(() => ({
    // Only a real pipeline run (upload/cognify) counts here — operation rows
    // (recall/search/remember/etc., kind: "operation") mean the agent talked
    // to memory, not that anything was uploaded.
    upload_document: runs.some((r) => r.kind === "pipeline"),
    first_query: sessions.length > 0,
    connect_agent: agents.some((a) => a.is_agent && !a.is_default),
  }), [runs, sessions, agents]);
  const prevChecklistDone = useRef(checklistDone);
  useEffect(() => {
    if (checklistDone.upload_document && !prevChecklistDone.current.upload_document) {
      trackEvent({ pageName: "Dashboard", eventName: "checklist_item_completed", additionalProperties: { item: "upload_document" } });
    }
    if (checklistDone.first_query && !prevChecklistDone.current.first_query) {
      trackEvent({ pageName: "Dashboard", eventName: "checklist_item_completed", additionalProperties: { item: "first_query" } });
    }
    if (checklistDone.connect_agent && !prevChecklistDone.current.connect_agent) {
      trackEvent({ pageName: "Dashboard", eventName: "checklist_item_completed", additionalProperties: { item: "connect_agent" } });
    }
    prevChecklistDone.current = checklistDone;
  }, [checklistDone]);

  // podUnreachable is checked before workspaceReady: a genuinely dead pod is
  // a terminal state, not "still connecting" — showing the skeleton forever
  // against it would be the eternal-skeleton bug this replaces.
  if (podUnreachable) {
    return <PodUnreachableCard />;
  }

  if (!workspaceReady) {
    return <DashboardSkeleton />;
  }

  const dataLoading = loading || isInitializing || filterLoading;

  // Deduplicate runs by pipeline_run_id before rendering.
  const latestRuns: PipelineRun[] = [];
  const seenIds = new Set<string>();
  for (const r of runs) {
    const key = r.pipeline_run_id || r.id;
    if (!seenIds.has(key)) { seenIds.add(key); latestRuns.push(r); }
  }

  // ── Memory graph inputs ──

  // Persistent agents always render; dynamic agents only once they've registered
  // (a matching session exists), so the graph never advertises tools nobody uses.
  const visibleAgentDefs = [
    ...PERSISTENT_AGENT_DEFS,
    ...DYNAMIC_AGENT_DEFS.filter((d) => hasRegistered(sessions, d.prefixes)),
  ];
  const flowAgents: FlowAgent[] = visibleAgentDefs.map((d) => ({
    name: d.name,
    logo: d.logo,
    status: agentStatus(sessions, d.prefixes),
  }));

  // Company Brain reflects whether any data has been ingested (datasets
  // present); every other source is a live connector from DATA_SOURCE_CARDS,
  // carrying its real control-plane connection status. Sources that aren't
  // built yet live in the Integrations page's notify-me list, not here — the
  // graph never advertises a source nobody can connect.
  const flowSources: FlowSource[] = [
    { name: "Company Brain", logo: "company-brain", status: datasets.length > 0 ? "connected" : "disconnected" },
    ...DATA_SOURCE_CARDS.map((card) => ({
      name: card.name,
      // The card's glyph is optional; the graph node falls back to the key,
      // which is the filename convention under /visuals/logos/datasources.
      logo: card.logo ?? card.key,
      status: sourceStatuses[card.key] ?? "disconnected",
    })),
  ];

  // "Connectors" in the Get Started bar counts agents that have ever registered
  // (not just the currently-active ones), so it reflects lifetime connections.
  const connectedAgentCount = [...PERSISTENT_AGENT_DEFS, ...DYNAMIC_AGENT_DEFS]
    .filter((d) => hasRegistered(sessions, d.prefixes)).length;

  // Memory Coverage is a Cognee Cloud feature (see PerformancePanel) — no
  // local score to compute, so the panel gets an empty state directly.
  const recallPct: number | null = null;
  const topics: TopicScore[] = [];

  const greetingName = currentUser?.name?.trim() || (currentUser?.email ? currentUser.email.split("@")[0] : "");

  return (
    <div style={{ minHeight: "100%", flexShrink: 0 }}>
      {/* Hidden file input — triggered by connection cards & the Upload data action */}
      <input
        ref={uploadInputRef}
        type="file"
        multiple
        accept=".pdf,.csv,.txt,.md,.json,.docx"
        className="hidden"
        onChange={upload.handleDashboardUpload}
      />

      <DatasetPickerModal
        open={upload.showDatasetPicker}
        datasets={datasets}
        pendingFiles={upload.pendingFiles}
        onPick={upload.handlePickDataset}
        onClose={() => { upload.setShowDatasetPicker(false); upload.setPendingFiles([]); }}
      />

      <div style={{ padding: "24px 32px 32px", display: "flex", flexDirection: "column", gap: 28 }}>

        {/* Greeting — standard page-header type (matches every other page). */}
        <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
          <h1 style={{ ...FONT, margin: 0, fontSize: 20, fontWeight: 300, color: T.text, lineHeight: "28px" }}>
            {greetingForTime()}{greetingName ? `, ${greetingName}` : ""}
          </h1>
          {selectedAgent && (
            <span style={{ ...FONT, background: "var(--color-cognee-lavender-tint-10)", borderRadius: 100, padding: "2px 10px", fontSize: 11, fontWeight: 500, color: T.lavender }}>
              {selectedAgent.agent_type}
            </span>
          )}
        </div>

        <CreditBanners
          creditsSpentPct={credits.creditsSpentPct}
          creditsRemainingUsd={credits.creditsRemainingUsd}
          showCreditPctBanner={credits.showCreditPctBanner}
          showLowBalanceBanner={credits.showLowBalanceBanner}
          // Voucher promo banner removed from the dashboard; credit-warning
          // banners still surface.
          showVoucherBanner={false}
          onDismiss={credits.dismiss}
          isOwner={isOwner}
        />

        {/* Get started — collapsed connection strip */}
        <GetStartedBar
          subtitle="Connect your AI agents to give them persistent memory"
          connectors={connectedAgentCount}
        >
          <AgentConnectionSection
            onUploadClick={() => uploadInputRef.current?.click()}
            isUploading={upload.isUploading}
            serviceUrl={serviceUrl}
            apiKey={apiKey}
            isInitializing={isInitializing}
            hasDocuments={datasets.length > 0}
            sessions={sessions}
            integrationConnected={connectedIntegrations}
          />
        </GetStartedBar>

        {/* Overview header — the range toggle lives in the Balance card now, since
            it's the only panel whose figures are actually range-scoped. */}
        <div>
          <h2 style={{ ...FONT, margin: 0, fontSize: 19, fontWeight: 600, color: T.text, letterSpacing: "-0.01em" }}>Overview</h2>
          <p style={{ ...FONT, margin: "5px 0 0", fontSize: 13, color: T.muted }}>
            Balance, spend, and usage across your workspace
          </p>
        </div>

        {/* Memory graph — sources → cognee memory → agents, in a framed container */}
        <AsciiFrame label={null} style={{ background: "#000000" }}>
          <MemoryFlowDiagram
            sources={flowSources}
            healthy={memoryHealthy}
            agents={flowAgents.filter((a) => a.name !== "Claude Desktop")}
            onInvite={() => router.push("/members")}
            onCoreClick={() => router.push("/knowledge-graph")}
            onNodeNavigate={() => router.push("/integrations")}
            onTeamsClick={() => router.push("/datasets")}
          />
        </AsciiFrame>


        {/* Cost · Performance side by side; Activity spans full width below so the
            log's time/agent/dataset/action columns have room to breathe. */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 16 }}>
          <CostPanel
            sessions={sessions}
            runs={runs}
            balanceUsd={credits.creditsRemainingUsd}
            range={range}
            onRangeChange={setRange}
            hourlyCosts={hourlyCosts}
            onViewBreakdown={() => router.push("/analytics")}
          />
          <PerformancePanel
            recallPct={recallPct}
            topics={topics}
            onUpload={() => uploadInputRef.current?.click()}
            onViewAnalysis={() => router.push("/memory-gap-analysis")}
          />
        </div>

        <ActivityPanel
          runs={latestRuns}
          sessions={sessions}
          datasets={datasets}
          agents={agents}
          onViewFullLog={() => router.push("/activity")}
        />

        {dataLoading && (
          <div style={{ ...FONT, fontSize: 11, color: T.faint }}>refreshing telemetry…</div>
        )}

        {upload.showUploadDoneModal && (
          <UploadDoneModal
            datasetName={upload.showUploadDoneModal.datasetName}
            datasetId={upload.showUploadDoneModal.datasetId}
            onClose={() => upload.setShowUploadDoneModal(null)}
            onNavigate={(path) => { upload.setShowUploadDoneModal(null); router.push(path); }}
          />
        )}

      </div>
    </div>
  );
}

function greetingForTime(): string {
  const h = new Date().getHours();
  if (h < 12) return "Good morning";
  if (h < 18) return "Good afternoon";
  return "Good evening";
}
