"use client";

import React, { useRef, useEffect, useMemo } from "react";
import { useRouter } from "next/navigation";
import { notifications } from "@mantine/notifications";
import { trackEvent } from "@/modules/analytics";
import { useCogniInstance, useTenant } from "@/modules/tenant/TenantProvider";
import { useFilter, useRefreshDatasetsOnMount } from "@/ui/layout/FilterContext";
import { AgentActivityTerminal, ownerDisplayName } from "@/ui/elements/AgentActivityTerminal";
import type { PipelineRun, Range } from "@/ui/elements/AgentActivityTerminal";
import DashboardSkeleton from "./DashboardSkeleton";
import { useDashboardTelemetry } from "./hooks/useDashboardTelemetry";
import { useConnectedIntegrations } from "./hooks/useConnectedIntegrations";
import { useGraphSummary } from "./hooks/useGraphSummary";
import { useCreditsBanner } from "./hooks/useCreditsBanner";
import { useAwaitingDataset } from "./hooks/useAwaitingDataset";
import { useOnboardingRedirect } from "./hooks/useOnboardingRedirect";
import { useDatasetUpload } from "./hooks/useDatasetUpload";
import { DashboardKpiStrip } from "./partials/DashboardKpiStrip";
// import { GettingStartedChecklist } from "./partials/GettingStartedChecklist"; — hidden pending a tracked task, see below
import { CreditBanners } from "./partials/CreditBanners";
import { UseCaseSlider } from "./partials/UseCaseSlider";
import { AgentConnectionSection } from "./partials/AgentConnectionSection";
import { DatasetPickerModal } from "./partials/DatasetPickerModal";
import { UploadDoneModal } from "./partials/UploadDoneModal";
import PodUnreachableCard from "@/ui/elements/PodUnreachableCard";

const RANGE: Range = "24h";

export default function OverviewPage(): React.ReactElement {
  const { cogniInstance, isInitializing, serviceUrl, apiKey } = useCogniInstance();
  const { tenantReady, podUnreachable, tenant, isOwner } = useTenant();
  const { agents, datasets, selectedDataset, selectedAgent, loading: filterLoading } = useFilter();
  useRefreshDatasetsOnMount();
  const router = useRouter();
  const uploadInputRef = useRef<HTMLInputElement>(null);

  const awaitingDataset = useAwaitingDataset();
  const workspaceReady = !!cogniInstance && tenantReady && !awaitingDataset;
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

  const { runs, sessions, loading } = useDashboardTelemetry(RANGE);
  const connectedIntegrations = useConnectedIntegrations(sessions, tenant?.tenant_id ?? null);
  // Dashboard KPIs always report workspace-wide totals — pass null so the
  // graph counts never inherit a dataset selection carried over from another page.
  const { graphNodes, graphEdges } = useGraphSummary(null, datasets, runs);
  const credits = useCreditsBanner();
  const upload = useDatasetUpload();
  useOnboardingRedirect();

  // Fires once per item, the moment it first becomes true — not on every
  // render where it's already true (prevChecklistDone tracks that boundary).
  const checklistDone = useMemo(() => ({
    upload_document: runs.length > 0,
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

  const filteredDatasets = selectedDataset ? datasets.filter((d) => d.id === selectedDataset.id) : datasets;
  const connectedAgents = agents.filter((a) => a.is_agent && !a.is_default);
  const liveAgentIds = new Set(sessions.filter((s) => s.effective_status === "running").map((s) => s.user_id));
  const liveAgents = connectedAgents.filter((a) => liveAgentIds.has(a.id) || a.status === "LIVE");

  return (
    <div style={{ minHeight: "100%", flexShrink: 0 }}>
      {/* Hidden file input — triggered by AgentConnectionSection's upload card */}
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

      <div style={{ padding: "clamp(16px, 3vw, 32px)", display: "flex", flexDirection: "column", gap: 40 }}>

        {/* Greeting */}
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <h1 style={{ margin: 0, fontSize: 26, fontWeight: 700, color: "#EDECEA", letterSpacing: "-0.02em", lineHeight: "32px" }}>
            {greetingForTime()}{selectedAgent ? `, ${ownerDisplayName(selectedAgent.email)}` : ""}
          </h1>
          {selectedAgent && (
            <span style={{ background: "var(--color-cognee-selected)", borderRadius: 4, padding: "2px 8px", fontSize: 12, fontWeight: 500, color: "var(--color-cognee-purple)" }}>
              {selectedAgent.agent_type}
            </span>
          )}
        </div>

        {/* Hidden pending a tracked task for the redesigned progress-card version — component kept, just not rendered.
        <GettingStartedChecklist
          items={[
            { label: "Upload & process a document", done: checklistDone.upload_document },
            { label: "Run your first query", done: checklistDone.first_query },
            { label: "Connect an agent", done: checklistDone.connect_agent },
          ]}
        />
        */}

        <CreditBanners
          creditsSpentPct={credits.creditsSpentPct}
          showCreditPctBanner={credits.showCreditPctBanner}
          showLowBalanceBanner={credits.showLowBalanceBanner}
          showVoucherBanner={credits.showVoucherBanner}
          onDismiss={credits.dismiss}
          isOwner={isOwner}
        />

        <DashboardKpiStrip
          liveAgents={liveAgents.length}
          apiCalls={latestRuns.length}
          sessionCount={sessions.length}
          graphNodes={graphNodes}
          graphEdges={graphEdges}
          brains={datasets.length}
          dataLoading={dataLoading}
        />

        {/* Get started — agent & brain connection cards */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div>
            <div style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", lineHeight: "24px" }}>Get started</div>
            <div style={{ fontSize: 13, color: "rgba(237,236,234,0.65)", marginTop: 3 }}>
              Connect your AI agents to give them persistent memory
            </div>
          </div>
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
        </div>

        {/* Memory Activity terminal */}
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div>
            <h2 style={{ margin: 0, fontSize: 18, fontWeight: 700, color: "#EDECEA", letterSpacing: "-0.01em", lineHeight: "24px" }}>
              Memory Activity
            </h2>
            <p style={{ margin: "3px 0 0", fontSize: 13, color: "rgba(237,236,234,0.55)" }}>
              A live log of every search against your memory — by your agents and by you. Click any row to see what was searched and why it answered.
            </p>
          </div>
          <AgentActivityTerminal
            sessions={sessions}
            runs={latestRuns}
            agents={agents}
            datasets={filteredDatasets}
            selectedDataset={selectedDataset}
            cogniInstance={cogniInstance}
            dataLoading={dataLoading}
            range={RANGE}
            onNavigate={(path) => router.push(path)}
          />
        </div>

        <UseCaseSlider />

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
