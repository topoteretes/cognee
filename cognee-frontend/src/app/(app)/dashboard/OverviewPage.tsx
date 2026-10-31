"use client";

import React, { useRef, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { notifications } from "@mantine/notifications";
import { trackEvent } from "@/modules/analytics";
import { useChecklistCompletionTracking } from "@/app/(app)/dashboard/hooks/useChecklistCompletionTracking";
import { useCogniInstance, useTenant } from "@/modules/tenant/TenantProvider";
import { useCurrentUser } from "@/modules/users/useCurrentUser";
import {
  DYNAMIC_AGENT_DEFS,
  PERSISTENT_AGENT_DEFS,
  agentStatus,
  agentStatusTooltip,
  hasRegistered,
} from "@/app/(app)/dashboard/agentNodes";
import { agentSummary } from "@/app/(app)/dashboard/agentSummary";
import { dedupeByRunId } from "@/modules/sessions/getSessions";
import { keyForChannel, lastSeenByAgent, runningAgents } from "@/modules/agents/getAgentConnections";
import { AGENT_CARDS } from "@/modules/integrations/agentCards";
import { AgentSetupModal } from "@/ui/elements/AgentSetupModal";
import { AgentHoverCard } from "./partials/redesign/AgentHoverCard";
import { useTenantHourlyCosts } from "@/modules/billing/useTenantHourlyCosts";
import { useFilter, useRefreshDatasetsOnMount } from "@/ui/layout/FilterContext";
import { useCoverageRuns } from "@/app/(app)/memory-gap-analysis/hooks/useCoverageRuns";
import { deriveCoverageView } from "@/app/(app)/dashboard/coverageView";
import DashboardSkeleton from "./DashboardSkeleton";
import WorkspaceWaking from "@/ui/layout/WorkspaceWaking";
import { useDashboardTelemetry } from "./hooks/useDashboardTelemetry";
import { useConnectedIntegrations } from "./hooks/useConnectedIntegrations";
import { useDataSourceStatuses } from "./hooks/useDataSourceStatuses";
import { DATA_SOURCE_CARDS } from "@/modules/integrations/dataSourceCards";
import { useShownDataSourceCards } from "@/modules/integrations/useShownDataSourceCards";
import { useCreditsBanner } from "./hooks/useCreditsBanner";
import { useAwaitingDataset } from "./hooks/useAwaitingDataset";
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
import type { FlowSource, FlowAgent } from "./partials/redesign/MemoryFlowDiagram";
import { CostPanel } from "./partials/redesign/CostPanel";
import { PerformancePanel } from "./partials/redesign/PerformancePanel";
import { ActivityPanel } from "./partials/redesign/ActivityPanel";
import type { DashRange } from "./partials/redesign/RangeToggle";
import { FONT, T } from "./partials/redesign/mono";

/** How many datasets the Memory Gap panel lists at a glance. */
const MAX_COVERAGE_ROWS = 4;

const DATA_SOURCE_PROVIDERS = DATA_SOURCE_CARDS.map((card) => card.key);

export default function OverviewPage(): React.ReactElement {
  const { cogniInstance, isInitializing, serviceUrl, apiKey } = useCogniInstance();
  const { tenantReady, podUnreachable, wakeState, wakeStartedAt, tenant, isOwner, currentName } = useTenant();
  const { agents, datasets, selectedAgent, loading: filterLoading, agentsLoading, agentsError } = useFilter();
  // `loading` and `refresh` were being discarded. Without the first the panel
  // renders "not scored yet" over a live request, because its `unavailable`
  // flag needs a brain to have answered and there are none yet; without the
  // second it told the reader to open another page to retry a refetch this
  // page already has in hand.
  //
  // Bounded to MAX_COVERAGE_ROWS: this panel only ever lists that many
  // datasets, so fetching the rest of the workspace's coverage was a request
  // per dataset for rows that never rendered. Memory Gap Analysis calls the
  // same hook with no limit and keeps the full fan-out.
  const { brains: coverageBrains, loading: coverageLoading, refresh: refreshCoverage } = useCoverageRuns({
    limit: MAX_COVERAGE_ROWS,
  });
  // Greeting identity comes from the logged-in user, NOT the FilterContext
  // agent selection (which defaults to null and would leave the greeting nameless).
  const { data: currentUser } = useCurrentUser();
  useRefreshDatasetsOnMount();
  const router = useRouter();
  const uploadInputRef = useRef<HTMLInputElement>(null);

  // Owned by the Balance card. It scopes that card's figures and the hourly
  // cost window behind them, and nothing else.
  const [range, setRange] = useState<DashRange>("7d");

  // Clicking a claude-code/codex node in the memory graph opens the exact
  // modal a click on its Integrations card would (AgentSetupModal, shared
  // with SetupConnectorSection rather than a second copy of the choice
  // between the connected summary and the install wizard). Every other node
  // (data sources, non-plugin agents) still just navigates to Integrations.
  const [activeAgentCardKey, setActiveAgentCardKey] = useState<string | null>(null);

  const awaitingDataset = useAwaitingDataset();
  const workspaceReady = !!cogniInstance && tenantReady && !awaitingDataset;

  // Core live/idle badge = the pod's /health endpoint, probed on a slow poll.
  const [memoryHealthy, setMemoryHealthy] = useState(false);
  useEffect(() => {
    // cogniInstance is created before the pod is known to be reachable, and a
    // suspended workspace is scaled to zero, so probing on the instance alone
    // fired this at a dead pod every 30s for the whole wake (and kept firing
    // behind the terminal error card). Every other pod poll in the app gates
    // on tenantReady; this one has to as well, or the workspace can never be
    // woken without the browser hammering the host it is waiting on.
    if (!cogniInstance || !tenantReady || wakeState !== "idle") { setMemoryHealthy(false); return; }
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
  }, [cogniInstance, tenantReady, wakeState]);
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

  // Deliberately NOT `range`. The toggle is visually the Balance card's, but
  // feeding it here re-scoped every consumer of this feed behind the user's
  // back: the Get Started checklist (so a workspace that uploaded 8 days ago
  // read as never having uploaded on the 24h view), agent liveness in the
  // memory graph, the connected-integrations discovery, and the Activity
  // panel. Fetch the widest window once and let each widget slice it — which
  // CostPanel already does internally, in both its billing and session-priced
  // paths, so it needs nothing from this. One query key per tenant also means
  // flipping the toggle no longer refetches at all.
  const { runs, sessions, connections, loading, error: telemetryError, sessionsFailed, updatedAt, runsTruncated, sessionsTruncated } = useDashboardTelemetry();
  const { data: hourlyCosts = null } = useTenantHourlyCosts(tenant?.tenant_id ?? null, range);
  // A plugin registers with the pod at launch, seconds before a session or
  // memory write exists — reading connections too is what lets a card and
  // the memory graph both show "Connected" that early instead of waiting on
  // sessions alone. Registration only ever proves the agent installed, never
  // that it is running right now (the pod's "active" status sticks after a
  // crash), so this feeds the sticky registry below and nothing else:
  // agentStatus's "live" state is still session-only. Polled alongside
  // runs/sessions (useDashboardTelemetry) rather than fetched separately, so
  // this and a tutorial reading the same connections endpoint can never
  // disagree about whether the agent has ever registered.
  const connectedIntegrations = useConnectedIntegrations(sessions, tenant?.tenant_id ?? null, connections);
  // Same `connections` and `sessions` this page already has, read a third
  // way: "when did we last see it" rather than "have we ever seen it" or
  // "is a session open right now". Used only for the agent cards' and the
  // memory graph tooltip's "Last seen" line, never for agentStatus's "live"
  // state.
  const agentLastSeen = useMemo(() => lastSeenByAgent(connections, sessions), [connections, sessions]);
  const agentsRunning = useMemo(() => runningAgents(sessions), [sessions]);
  const sourceStatuses = useDataSourceStatuses(DATA_SOURCE_PROVIDERS, tenant?.tenant_id ?? null);
  const shownCards = useShownDataSourceCards();
  // Dashboard metrics always report workspace-wide totals — pass null so the
  // graph counts never inherit a dataset selection carried over from another page.
  const credits = useCreditsBanner();
  const upload = useDatasetUpload();

  // Gated per source, not by one shared flag — see useChecklistCompletionTracking.
  // upload_document/first_query have nothing to do with the agent roster, so
  // folding agentsError into their gate left the whole checklist permanently
  // dark on any pod image that hasn't allowlisted GET /v1/activity/agents (a
  // 404 there is never retried and never refetched — backgroundQueryRetry
  // drops non-429 4xx, and the query has no refetchInterval — so agentsError
  // sticks forever). Conversely, telemetryError was previously not part of
  // either gate at all: on a first-load telemetry failure, `loading` goes
  // false immediately (isPending only covers the pending state, not an error),
  // seeding off empty runs/sessions — a workspace that finished both items
  // long ago then fires two spurious completions on the next successful poll.
  useChecklistCompletionTracking({
    runs,
    sessions,
    agents,
    telemetryNotReady: loading || telemetryError,
    agentsNotReady: agentsLoading || agentsError,
  });

  // Deduplicate runs by pipeline_run_id before rendering — a run emits an
  // initiated/started row and a separate terminal row sharing one id, and
  // the terminal row is the one carrying real tokens/status.
  // Memoized because dedupeByRunId returns a fresh array every call, and that
  // array is the dep of every memo inside ActivityPanel. Unmemoized, the panel
  // rebuilt its whole row model — dedupe, index maps, tree walk over up to 500
  // runs — on every dashboard render, and re-rendered every row, which is the
  // exact opposite of what React.memo is there for.
  const latestRuns = useMemo(() => dedupeByRunId(runs), [runs]);

  // podUnreachable is checked before workspaceReady: a genuinely dead pod is
  // a terminal state, not "still connecting" — showing the skeleton forever
  // against it would be the eternal-skeleton bug this replaces.
  if (podUnreachable) {
    return <PodUnreachableCard wakeFailed={wakeState === "failed"} workspaceName={currentName} />;
  }

  // A wake is not a provisioning: the workspace already exists and its data is
  // intact, so it gets its own screen rather than the brand-new-account
  // skeleton, which is what /dashboard used to show for both.
  if (wakeState === "waking") {
    return <WorkspaceWaking workspaceName={currentName} startedAt={wakeStartedAt} />;
  }

  if (!workspaceReady) {
    return <DashboardSkeleton />;
  }

  const dataLoading = loading || isInitializing || filterLoading;

  // The clock the memory graph's per-agent 24h sparkline is bucketed against.
  // Same pattern as CostPanel: derived from the fetch itself rather than the
  // render clock, so react-query's structural sharing across an unchanged
  // poll can't quietly slide the window forward on an open tab.
  const now = new Date(updatedAt || Date.now());

  // ── Memory graph inputs ──

  // Persistent agents always render; dynamic agents only once they've registered
  // (a matching session exists), so the graph never advertises tools nobody uses.
  const visibleAgentDefs = [
    ...PERSISTENT_AGENT_DEFS,
    ...DYNAMIC_AGENT_DEFS.filter((d) => hasRegistered(sessions, connectedIntegrations, d.name)),
  ];
  const flowAgents: FlowAgent[] = visibleAgentDefs.map((d) => {
    const status = agentStatus(sessions, connectedIntegrations, d.name, loading);
    // Only the two plugin agents have a lastSeenByAgent entry (keyed by
    // integration key, not channel name); every other node's tooltip is
    // just its badge, with no last-seen phrase to add.
    const integrationKey = keyForChannel(d.name);
    const lastSeenAt = (integrationKey && agentLastSeen[integrationKey]) ?? null;
    if (!integrationKey) {
      return { name: d.name, logo: d.logo, status, tooltip: agentStatusTooltip(status, lastSeenAt, now) };
    }
    // The two plugin agents get the interactive hover card instead of a
    // plain tooltip — same agentSummary view model ConnectedAgentModal
    // builds its header from, so the two can't say something different
    // about the same agent.
    const summary = agentSummary(integrationKey, status !== "disconnected" && status !== "unknown", loading, connections, sessions, now, sessionsTruncated);
    const cfg = AGENT_CARDS.find((c) => c.key === integrationKey);
    return {
      name: d.name,
      logo: d.logo,
      status,
      hoverCard: (
        <AgentHoverCard
          agentName={d.name}
          icon={cfg?.icon ?? null}
          channel={d.name}
          summary={summary}
          onOpenConnection={() => setActiveAgentCardKey(integrationKey)}
        />
      ),
    };
  });

  // Company Dataset reflects whether any data has been ingested (datasets
  // present); every other source is a live connector from DATA_SOURCE_CARDS,
  // carrying its real control-plane connection status. Sources that aren't
  // built yet live in the Integrations page's notify-me list, not here — the
  // graph never advertises a source nobody can connect.
  const flowSources: FlowSource[] = [
    { name: "Company Dataset", logo: "company-brain", status: datasets.length > 0 ? "connected" : "disconnected" },
    ...shownCards.flatMap((card): FlowSource[] => {
      const status = sourceStatuses[card.key];
      // A connector this deployment does not ship draws no node at all: a
      // "Google Drive" that is disconnected forever is the graph advertising
      // a source nobody here can connect (review-reported).
      if (status === "absent") return [];
      return [
        {
          name: card.name,
          // The card's glyph is optional; the graph node falls back to the key,
          // which is the filename convention under /visuals/logos/datasources.
          logo: card.logo ?? card.key,
          status: status ?? "disconnected",
        },
      ];
    }),
  ];

  // "Connectors" in the Get Started bar counts agents that have ever registered
  // (not just the currently-active ones), so it reflects lifetime connections.
  const connectedAgentCount = [...PERSISTENT_AGENT_DEFS, ...DYNAMIC_AGENT_DEFS]
    .filter((d) => hasRegistered(sessions, connectedIntegrations, d.name)).length;
  // A real zero (telemetry answered and truly found nothing sticky) is fine
  // to show; "0 connectors" while the first poll is still in flight is the
  // same false claim as a node reading "Not connected yet" too early — the
  // sticky registry can still answer instantly, so only an unconfirmed zero
  // is held back.
  const connectorsLoading = loading && connectedAgentCount === 0;

  // coverageBrains is already the bounded, first-MAX_COVERAGE_ROWS set the
  // panel shows (see the useCoverageRuns call above), so the rows and the
  // headline below describe the same datasets in the same order. No second
  // slice of `datasets` and no id lookup to keep in sync with it.
  const coverage = deriveCoverageView(coverageBrains);

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
          creditsRemainingUsd={credits.creditsRemainingUsd}
          showLowBalanceBanner={credits.showLowBalanceBanner}
          // Voucher promo banner removed from the dashboard; the low-balance
          // banner still surfaces.
          showVoucherBanner={false}
          onDismiss={credits.dismiss}
          isOwner={isOwner}
        />

        {/* Get started — collapsed connection strip */}
        <GetStartedBar
          subtitle="Connect your AI agents to give them persistent memory"
          connectors={connectedAgentCount}
          connectorsLoading={connectorsLoading}
        >
          <AgentConnectionSection
            onUploadClick={() => uploadInputRef.current?.click()}
            onOpenAgentSetup={setActiveAgentCardKey}
            isUploading={upload.isUploading}
            serviceUrl={serviceUrl}
            apiKey={apiKey}
            isInitializing={isInitializing}
            hasDocuments={datasets.length > 0}
            integrationConnected={connectedIntegrations}
            lastSeenByAgent={agentLastSeen}
            runningAgents={agentsRunning}
            connectionsLoading={loading}
          />
        </GetStartedBar>

        {/* Overview header — the range toggle lives in the Balance card, which
            is the only panel whose figures it scopes. */}
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
            onNodeNavigate={(node) => {
              const integrationKey = keyForChannel(node.name);
              if (integrationKey) { setActiveAgentCardKey(integrationKey); return; }
              router.push("/integrations");
            }}
            onTeamsClick={() => router.push("/datasets")}
          />
        </AsciiFrame>

        {activeAgentCardKey && (
          <AgentSetupModal
            cards={AGENT_CARDS}
            connectedKeys={connectedIntegrations}
            activeKey={activeAgentCardKey}
            onClose={() => setActiveAgentCardKey(null)}
          />
        )}


        {/* Cost · Performance side by side; Activity spans full width below so the
            log's time/agent/dataset/action columns have room to breathe. */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 16 }}>
          <CostPanel
            sessions={sessions}
            runs={runs}
            balanceUsd={credits.creditsRemainingUsd}
            range={range}
            updatedAt={updatedAt}
            loading={loading}
            runsTruncated={runsTruncated}
            sessionsTruncated={sessionsTruncated}
            onRangeChange={setRange}
            hourlyCosts={hourlyCosts}
            // Carries the range. The affordance whose job is to explain this panel's
            // number landed on the Analytics page's own hardcoded default, so the
            // two surfaces showed different windows in the same click.
            onViewBreakdown={() => router.push(`/analytics?range=${range}`)}
          />
          <PerformancePanel
            recallPct={coverage.recallPct}
            unavailable={coverage.unavailable}
            loading={coverageLoading}
            datasetTotal={datasets.length}
            onRetry={refreshCoverage}
            onViewDataset={(datasetId) => router.push(`/memory-gap-analysis?brain=${encodeURIComponent(datasetId)}`)}
            topics={coverage.topics}
            onUpload={() => uploadInputRef.current?.click()}
            onViewAnalysis={() => router.push("/memory-gap-analysis")}
          />
        </div>

        <ActivityPanel
          runs={latestRuns}
          sessions={sessions}
          datasets={datasets}
          agents={agents}
          loading={dataLoading}
          error={telemetryError}
          partial={sessionsFailed}
          onViewFullLog={() => router.push("/activity")}
          onOpenSession={(sessionId) => router.push(`/sessions?session=${encodeURIComponent(sessionId)}`)}
        />

        {/* The feed's failure signals, read rather than discarded. This page
            destructured five of eight fields, so a pod that answered at mount
            and then failed left every panel committed to its empty state:
            "No activity in this range yet", every agent disconnected in the
            memory graph, "No spend priced for this range yet", and an unchecked
            checklist. `podUnreachable` above does not cover it — that is scoped
            to `waitForPodReady` giving up at mount, not to a 500 afterwards. */}
        {telemetryError && (
          <div role="alert" style={{ ...FONT, fontSize: 12, color: T.red }}>
            Couldn&apos;t load workspace activity, so the panels above may be
            incomplete or empty. We&apos;ll keep trying in the background.
          </div>
        )}

        {!telemetryError && sessionsFailed && (
          <div role="alert" style={{ ...FONT, fontSize: 12, color: T.amber }}>
            Sessions couldn&apos;t be loaded, so activity from before your workspace
            started recording operations is missing from these panels.
          </div>
        )}

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
