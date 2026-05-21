"use client";

import { createContext, useContext, useState, useEffect, useCallback, useMemo, ReactNode } from "react";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import getDatasets from "@/modules/datasets/getDatasets";
import createDataset from "@/modules/datasets/createDataset";

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
  { id: "personal", name: "Personal", initial: "P", color: "#6510F4", type: "personal" },
  { id: "default", name: "Default", initial: "D", color: "#6510F4", type: "organization" },
];

const FilterContext = createContext<FilterContextValue>({
  workspace: DEFAULT_WORKSPACES[1],
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

export function FilterProvider({ children }: { children: ReactNode }) {
  const { cogniInstance, isInitializing } = useCogniInstance();
  const [agents, setAgents] = useState<Agent[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [workspace, setWorkspaceState] = useState<Workspace>(DEFAULT_WORKSPACES[1]);
  const [loading, setLoading] = useState(true);

  const refreshDatasets = useCallback(() => {
    if (!cogniInstance) return;
    getDatasets(cogniInstance).then((d: Dataset[]) => {
      setDatasets(Array.isArray(d) ? d : []);
    }).catch(() => {});
  }, [cogniInstance]);

  useEffect(() => {
    if (!cogniInstance || isInitializing) return;

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10000);

    Promise.all([
      cogniInstance.fetch("/v1/activity/agents", { signal: controller.signal })
        .then((r) => r.ok ? r.json() : [])
        .catch(() => []),
      getDatasets(cogniInstance)
        .then((d: Dataset[]) => (Array.isArray(d) ? d : []))
        .catch(() => []),
    ]).then(async ([agentData, datasetData]) => {
      setAgents(Array.isArray(agentData) ? agentData : []);
      // Auto-create a default dataset if none exist
      if (datasetData.length === 0 && cogniInstance) {
        try {
          const ds = await createDataset({ name: "default_dataset" }, cogniInstance);
          datasetData = [ds];
        } catch (e) {
          console.warn("[FilterContext] Failed to auto-create default dataset:", e);
        }
      }
      setDatasets(datasetData);
    }).catch(() => {}).finally(() => {
      clearTimeout(timeout);
      setLoading(false);
    });

    return () => {
      controller.abort();
      clearTimeout(timeout);
    };
  }, [cogniInstance, isInitializing]);

  const handleAgentChange = useCallback((agent: Agent | null) => {
    setSelectedAgent(agent);
    setSelectedDataset(null);
  }, []);

  const handleWorkspaceChange = useCallback((ws: Workspace) => {
    setWorkspaceState(ws);
    setSelectedAgent(null);
    setSelectedDataset(null);
  }, []);

  const value = useMemo(() => ({
    workspace,
    workspaces: DEFAULT_WORKSPACES,
    setWorkspace: handleWorkspaceChange,
    selectedAgent,
    selectedDataset,
    setSelectedAgent: handleAgentChange,
    setSelectedDataset,
    agents,
    datasets,
    loading,
    refreshDatasets,
  }), [workspace, selectedAgent, selectedDataset, agents, datasets, loading, handleAgentChange, handleWorkspaceChange, refreshDatasets]);

  return (
    <FilterContext.Provider value={value}>
      {children}
    </FilterContext.Provider>
  );
}

export function useFilter() {
  return useContext(FilterContext);
}
