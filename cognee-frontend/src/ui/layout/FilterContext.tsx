"use client";

import { createContext, useContext, useState, useEffect, useCallback, useMemo, useRef, ReactNode } from "react";
import { useCogniInstance, useTenant } from "@/modules/tenant/TenantProvider";
import getDatasets from "@/modules/datasets/getDatasets";

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
  const { tenant, availableTenants, switchTenant } = useTenant();
  const [agents, setAgents] = useState<Agent[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [loading, setLoading] = useState(true);

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

  const currentWorkspace = useMemo(() => {
    if (!tenant) return tenantWorkspaces[0];
    return tenantWorkspaces.find((ws) => ws.id === tenant.tenant_id) ?? tenantWorkspaces[0];
  }, [tenant, tenantWorkspaces]);

  const [workspace, setWorkspaceState] = useState<Workspace>(DEFAULT_WORKSPACES[0]);

  // Sync workspace state when tenant data loads
  useEffect(() => {
    if (currentWorkspace) setWorkspaceState(currentWorkspace);
  }, [currentWorkspace]);

  const refreshDatasets = useCallback(() => {
    if (!cogniInstance) return;
    getDatasets(cogniInstance).then((d: Dataset[]) => {
      setDatasets(Array.isArray(d) ? d : []);
    }).catch((err) => {
      console.error("Failed to refresh datasets:", err);
      setDatasets([]);
    });
  }, [cogniInstance]);

  // Refresh all data (datasets)
  const refreshAll = useCallback(() => {
    if (!cogniInstance) return;
    getDatasets(cogniInstance)
      .then((d: Dataset[]) => { setDatasets(Array.isArray(d) ? d : []); })
      .catch((err) => {
        console.error("Failed to refresh all datasets:", err);
        setDatasets([]);
      });
  }, [cogniInstance]);

  // Initial fetch
  useEffect(() => {
    if (!cogniInstance || isInitializing) return;

    let cancelled = false;

    function fetchAll() {
      return getDatasets(cogniInstance!)
        .then((d: Dataset[]) => {
          if (!cancelled) setDatasets(Array.isArray(d) ? d : []);
        })
        .catch((err) => {
          console.error("Failed to fetch datasets:", err);
          if (!cancelled) setDatasets([]);
        });
    }

    fetchAll().finally(() => {
      if (!cancelled) setLoading(false);
    });

    const interval = setInterval(fetchAll, 15000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [cogniInstance, isInitializing]);

  // Refetch on window focus (debounced to prevent burst on rapid alt-tab)
  useEffect(() => {
    let timeout: ReturnType<typeof setTimeout>;
    const onFocus = () => {
      clearTimeout(timeout);
      timeout = setTimeout(refreshAll, 2000);
    };
    window.addEventListener("focus", onFocus);
    return () => { window.removeEventListener("focus", onFocus); clearTimeout(timeout); };
  }, [refreshAll]);

  // Retry with backoff if initial fetch returned empty (e.g. cold start 401s)
  const retryRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    if (loading || !cogniInstance || datasets.length > 0) {
      if (retryRef.current) clearTimeout(retryRef.current);
      return;
    }
    let attempt = 0;
    const maxAttempts = 3;
    const retry = () => {
      if (attempt >= maxAttempts) return;
      attempt++;
      retryRef.current = setTimeout(() => {
        refreshAll();
        retry();
      }, attempt * 2000);
    };
    retry();
    return () => { if (retryRef.current) clearTimeout(retryRef.current); };
  }, [loading, cogniInstance, datasets.length, refreshAll]);

  const handleAgentChange = useCallback((agent: Agent | null) => {
    setSelectedAgent(agent);
    setSelectedDataset(null);
  }, []);

  const handleWorkspaceChange = useCallback((ws: Workspace) => {
    // If selecting a different tenant, trigger a full tenant switch (sets cookie + reloads)
    if (tenant && ws.id !== tenant.tenant_id) {
      switchTenant(ws.id, ws.name);
      return;
    }
    setWorkspaceState(ws);
    setSelectedAgent(null);
    setSelectedDataset(null);
  }, [tenant, switchTenant]);


  const value = useMemo(() => ({
    workspace,
    workspaces: tenantWorkspaces,
    setWorkspace: handleWorkspaceChange,
    selectedAgent,
    selectedDataset,
    setSelectedAgent: handleAgentChange,
    setSelectedDataset,
    agents,
    datasets,
    loading,
    refreshDatasets,
  }), [workspace, tenantWorkspaces, selectedAgent, selectedDataset, agents, datasets, loading, handleAgentChange, handleWorkspaceChange, refreshDatasets]);

  return (
    <FilterContext.Provider value={value}>
      {children}
    </FilterContext.Provider>
  );
}

export function useFilter() {
  return useContext(FilterContext);
}
