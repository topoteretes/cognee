"use client";

import { createContext, useContext, useState, useEffect, useCallback, useMemo, useRef, ReactNode } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useCogniInstance, useTenant } from "@/modules/tenant/TenantProvider";
import getDatasets from "@/modules/datasets/getDatasets";
import { BACKGROUND_QUERY_RETRY_COUNT, backgroundQueryRetryDelay } from "@/modules/query/backgroundQueryRetry";

// Local/loaded pods can take several seconds per request (see COG-5722) —
// a background poll shouldn't surface a false "error" at the default 10s
// GET timeout just because the pod is slow, so it gets more headroom than
// a user-initiated request would.
const BACKGROUND_POLL_TIMEOUT_MS = 25_000;
// Syncs the selected-dataset filter across same-browser tabs on the same
// tenant. Deliberately narrow — only a real user action (picking a dataset)
// broadcasts, never a poll tick, so this can't multiply background load the
// way COG-5721/5722 just fixed.
const DATASET_SYNC_CHANNEL = "cognee-selected-dataset-sync";

export interface Agent {
  id: string;
  email: string;
  agent_type: string;
  agent_short_id: string;
  is_agent: boolean;
  is_default: boolean;
  status: string;
}

export interface Dataset {
  id: string;
  name: string;
}

export interface Workspace {
  id: string;
  name: string;
  initial: string;
  color: string;
  type: "personal" | "organization";
}

interface FilterContextValue {
  // Workspace
  workspace: Workspace;
  workspaces: Workspace[];
  setWorkspace: (ws: Workspace) => void;

  // Selected filters (null = "All")
  selectedAgent: Agent | null;
  selectedDataset: Dataset | null;
  setSelectedAgent: (agent: Agent | null) => void;
  setSelectedDataset: (dataset: Dataset | null) => void;

  // Available data
  agents: Agent[];
  datasets: Dataset[];
  loading: boolean;

  // Refresh
  refreshDatasets: () => void;

}

const DEFAULT_WORKSPACES: Workspace[] = [
  { id: "personal", name: "Personal workspace", initial: "P", color: "#6510F4", type: "personal" },
];

const FilterContext = createContext<FilterContextValue>({
  workspace: DEFAULT_WORKSPACES[0],
  workspaces: DEFAULT_WORKSPACES,
  setWorkspace: () => {},
  selectedAgent: null,
  selectedDataset: null,
  setSelectedAgent: () => {},
  setSelectedDataset: () => {},
  agents: [],
  datasets: [],
  loading: true,
  refreshDatasets: () => {},
});

// Pick a deterministic color for a tenant based on its ID
const TENANT_COLORS = ["#6510F4", "#2563EB", "#059669", "#D97706", "#DC2626", "#7C3AED", "#0891B2", "#BE185D"];
function colorForTenant(id: string): string {
  let hash = 0;
  for (let i = 0; i < id.length; i++) hash = ((hash << 5) - hash + id.charCodeAt(i)) | 0;
  return TENANT_COLORS[Math.abs(hash) % TENANT_COLORS.length];
}

export function FilterProvider({ children }: { children: ReactNode }) {
  const { cogniInstance, isInitializing } = useCogniInstance();
  const { tenant, tenantReady, availableTenants, switchTenant } = useTenant();
  const queryClient = useQueryClient();
  const [agents, setAgents] = useState<Agent[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const tenantId = tenant?.tenant_id ?? null;

  // Single source of truth for the datasets list — shared via this query key
  // with any other hook/page that needs it (e.g. OverviewPage), so they
  // dedupe into one in-flight request instead of fetching independently.
  // The list rarely changes (only on user create/delete/upload), so there's
  // no background poll here — refreshDatasets() invalidates on mutation, and
  // react-query's default refetchOnWindowFocus keeps it from going stale
  // across tabs/sessions without hammering every page that mounts this
  // provider (most of which never even render the list).
  const datasetsQuery = useQuery({
    queryKey: ["datasets", tenantId],
    queryFn: ({ signal }) =>
      getDatasets(cogniInstance!, signal, BACKGROUND_POLL_TIMEOUT_MS).then((d: Dataset[]) => (Array.isArray(d) ? d : [])),
    // tenantReady, not just cogniInstance: a freshly-created workspace's pod
    // can still be unreachable while cogniInstance already exists (see
    // useDashboardTelemetry.ts / useGraphSummary.ts for the same fix).
    enabled: !!cogniInstance && !isInitializing && tenantReady,
    refetchInterval: false,
    retry: BACKGROUND_QUERY_RETRY_COUNT,
    retryDelay: backgroundQueryRetryDelay,
  });

  // Memoized so the fallback `[]` isn't a fresh reference on every render —
  // keeps the `value` memo below (and its consumers) stable when there's no
  // query data yet.
  const datasets = useMemo(() => datasetsQuery.data ?? [], [datasetsQuery.data]);
  const loading = datasetsQuery.isLoading;

  // Build workspaces from available tenants
  const tenantWorkspaces = useMemo<Workspace[]>(() => {
    if (availableTenants.length === 0) return DEFAULT_WORKSPACES;
    return availableTenants.map((t) => ({
      id: t.id,
      name: t.name,
      initial: t.name.charAt(0).toUpperCase(),
      color: colorForTenant(t.id),
      type: "organization" as const,
    }));
  }, [availableTenants]);

  // Snapshot of the persisted workspace selection, read synchronously during
  // the FIRST render via a lazy initializer — not in an effect. An effect
  // only runs after that first render commits, so it still paints the
  // hardcoded personal default for one frame before correcting itself; this
  // was visible as a brief flash even after switching to a non-personal
  // workspace. Guarded for SSR, where localStorage doesn't exist — this
  // component only ever runs client-side, so the guard is just to keep the
  // initializer from throwing during the server render pass.
  const [persistedSelection] = useState<Workspace | null>(() => {
    if (typeof window === "undefined") return null;
    const id = localStorage.getItem("cognee_selected_tenant");
    const name = localStorage.getItem("cognee_selected_tenant_name");
    return id && name
      ? { id, name, initial: name.charAt(0).toUpperCase(), color: colorForTenant(id), type: "organization" as const }
      : null;
  });

  // The workspace shown in the topbar — derived, not its own state. It used
  // to be mirrored into a separate useState synced by an effect, which added
  // an extra render round-trip (persistedSelection settles -> this recomputes
  // -> the effect fires -> the mirror updates) for the same flash this exists
  // to avoid.
  const workspace = useMemo(() => {
    if (tenant) return tenantWorkspaces.find((ws) => ws.id === tenant.tenant_id) ?? tenantWorkspaces[0];
    // tenant not resolved yet — prefer the persisted selection over the
    // hardcoded personal default so a user on another workspace never sees
    // "Personal workspace" flash first.
    return persistedSelection ?? tenantWorkspaces[0];
  }, [tenant, tenantWorkspaces, persistedSelection]);

  // Window-focus refetch, retry-with-backoff, and interval polling are all
  // handled by the query above (refetchOnWindowFocus defaults to true,
  // respecting staleTime so rapid alt-tabbing doesn't burst-refetch).
  const refreshDatasets = useCallback(() => {
    void queryClient.invalidateQueries({ queryKey: ["datasets", tenantId] });
  }, [queryClient, tenantId]);

  // Read inside the channel's onmessage below instead of depending on
  // `datasets` directly — that would tear down and recreate the channel on
  // every 15s datasets poll.
  const datasetsRef = useRef(datasets);
  useEffect(() => { datasetsRef.current = datasets; }, [datasets]);

  const channelRef = useRef<BroadcastChannel | null>(null);
  useEffect(() => {
    if (typeof BroadcastChannel === "undefined" || !tenantId) return;
    const channel = new BroadcastChannel(DATASET_SYNC_CHANNEL);
    channelRef.current = channel;
    channel.onmessage = (event: MessageEvent<{ tenantId: string; datasetId: string | null }>) => {
      if (event.data.tenantId !== tenantId) return;
      const dataset = event.data.datasetId
        ? datasetsRef.current.find((d) => d.id === event.data.datasetId) ?? null
        : null;
      setSelectedDataset(dataset);
    };
    return () => {
      channel.close();
      channelRef.current = null;
    };
  }, [tenantId]);

  const setSelectedDatasetSynced = useCallback((dataset: Dataset | null) => {
    setSelectedDataset(dataset);
    if (tenantId) channelRef.current?.postMessage({ tenantId, datasetId: dataset?.id ?? null });
  }, [tenantId]);

  const handleAgentChange = useCallback((agent: Agent | null) => {
    setSelectedAgent(agent);
    setSelectedDatasetSynced(null);
  }, [setSelectedDatasetSynced]);

  const handleWorkspaceChange = useCallback((ws: Workspace) => {
    // If selecting a different tenant, trigger a full tenant switch (sets cookie + reloads)
    if (tenant && ws.id !== tenant.tenant_id) {
      switchTenant(ws.id, ws.name);
      return;
    }
    // Re-selecting the current workspace: nothing to switch, just reset the
    // page-local filters (workspace itself is derived from `tenant`, above).
    setSelectedAgent(null);
    setSelectedDatasetSynced(null);
  }, [tenant, switchTenant, setSelectedDatasetSynced]);


  const value = useMemo(() => ({
    workspace,
    workspaces: tenantWorkspaces,
    setWorkspace: handleWorkspaceChange,
    selectedAgent,
    selectedDataset,
    setSelectedAgent: handleAgentChange,
    setSelectedDataset: setSelectedDatasetSynced,
    agents,
    datasets,
    loading,
    refreshDatasets,
  }), [workspace, tenantWorkspaces, selectedAgent, selectedDataset, agents, datasets, loading, handleAgentChange, handleWorkspaceChange, refreshDatasets, setSelectedDatasetSynced]);

  return (
    <FilterContext.Provider value={value}>
      {children}
    </FilterContext.Provider>
  );
}

export function useFilter() {
  return useContext(FilterContext);
}
