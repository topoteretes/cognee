"use client";

import { useEffect, useState, useRef } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import { useFilter } from "@/ui/layout/FilterContext";
import { TrackPageView, trackEvent } from "@/modules/analytics";
import { notifications } from "@mantine/notifications";
import { Tooltip } from "@mantine/core";
import type { DatasetProcessingStatus } from "@/modules/datasets/pollDatasetStatus";
import pollDatasetStatus from "@/modules/datasets/pollDatasetStatus";
import type { GraphModel } from "@/modules/graphModels/types";
import { toCleanSchema } from "@/modules/graphModels/types";
import { toGraphModelSchema } from "@/modules/graphModels/toGraphModelSchema";
import {
  loadGraphModelsConfig,
  findModelForDataset,
  findPromptForDataset,
  findOntologyForDataset,
  assignGraphModelToDataset,
  assignPromptToDataset,
  assignOntologyToDataset,
  saveCustomPrompt,
  deleteCustomPrompt,
  type CustomPromptsMap,
} from "@/modules/configuration/userConfiguration";
import { listOntologies, uploadOntology, deleteOntology, type OntologyMeta } from "@/modules/ontologies/ontologyApi";
import { generateCustomPrompt } from "@/modules/llm/managementLlmApi";
import cognifyDataset from "@/modules/datasets/cognifyDataset";

type DisplayStatus = "pending" | "running" | "completed" | "failed" | "empty";

function mapProcessingStatus(raw: DatasetProcessingStatus | undefined): DisplayStatus {
  if (!raw) return "empty";
  if (raw === "DATASET_PROCESSING_COMPLETED") return "completed";
  if (raw === "DATASET_PROCESSING_ERRORED") return "failed";
  if (raw === "DATASET_PROCESSING_STARTED") return "running";
  if (raw === "DATASET_PROCESSING_INITIATED") return "pending";
  return "empty";
}

const STATUS_CONFIG: Record<DisplayStatus, { label: string; color: string }> = {
  pending:   { label: "Pending",    color: "#F59E0B" },
  running:   { label: "Processing", color: "#F59E0B" },
  completed: { label: "Ready",      color: "#22C55E" },
  failed:    { label: "Failed",     color: "#EF4444" },
  empty:     { label: "Empty",      color: "#A1A1AA" },
};

const DEFAULT_EXTRACTION_PROMPT = `You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph.
**Nodes** represent entities and concepts. They're akin to Wikipedia nodes.
**Edges** represent relationships between concepts. They're akin to Wikipedia links.

The aim is to achieve simplicity and clarity in the knowledge graph.

# 1. Labeling Nodes
**Consistency**: Ensure you use basic or elementary types for node labels.
  - For example, when you identify an entity representing a person, always label it as **"Person"**.
  - Avoid using more specific terms like "Mathematician" or "Scientist", keep those as "profession" property.
  - Don't use too generic terms like "Entity".
**Node IDs**: Never utilize integers as node IDs.
  - Node IDs should be names or human-readable identifiers found in the text.
**Node Names**: Every node MUST include a "name" field.
  - Use the most complete human-readable name for the entity (e.g., "Albert Einstein", "Python").

# 2. Handling Numerical Data and Dates
  - For example, when you identify an entity representing a date, make sure it has type **"Date"**.
  - Extract the date in the format "YYYY-MM-DD"
  - If not possible to extract the whole date, extract month or year, or both if available.
  - **Property Format**: Properties must be in a key-value format.
  - **Quotation Marks**: Never use escaped single or double quotes within property values.
  - **Naming Convention**: Use snake_case for relationship names, e.g., \`acted_in\`.

# 3. Coreference Resolution
  - **Maintain Entity Consistency**: When extracting entities, it's vital to ensure consistency.
  If an entity is mentioned multiple times in the text but is referred to by different names or pronouns,
  always use the most complete identifier for that entity throughout the knowledge graph.
Remember, the knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.

# 4. Strict Compliance
Adhere to the rules strictly. Non-compliance will result in termination.`;

export default function KnowledgeGraphPage() {
  const router = useRouter();
  const { cogniInstance, isInitializing } = useCogniInstance();
  const { datasets, selectedDataset } = useFilter();

  // Visualization state
  const [iframeSrc, setIframeSrc] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const blobRef = useRef<string | null>(null);
  const [datasetStatus, setDatasetStatus] = useState<DisplayStatus>("empty");
  const [pollKey, setPollKey] = useState(0);
  const [vizRefreshKey, setVizRefreshKey] = useState(0);
  const prevStatusRef = useRef<DisplayStatus>("empty");

  // Config state
  const [graphModels, setGraphModels] = useState<GraphModel[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [modelDropdownOpen, setModelDropdownOpen] = useState(false);
  const modelDropdownRef = useRef<HTMLDivElement>(null);
  const [customPrompts, setCustomPrompts] = useState<CustomPromptsMap>({});
  const [selectedPromptName, setSelectedPromptName] = useState<string | null>(null);
  const [promptDropdownOpen, setPromptDropdownOpen] = useState(false);
  const promptDropdownRef = useRef<HTMLDivElement>(null);
  const [ontologies, setOntologies] = useState<Record<string, OntologyMeta>>({});
  const [selectedOntologyKey, setSelectedOntologyKey] = useState<string | null>(null);
  const [ontologyDropdownOpen, setOntologyDropdownOpen] = useState(false);
  const ontologyDropdownRef = useRef<HTMLDivElement>(null);

  // Modal state
  const [showCreatePromptModal, setShowCreatePromptModal] = useState(false);
  const [inferringPrompt, setInferringPrompt] = useState(false);
  const [showPromptEditor, setShowPromptEditor] = useState(false);
  const [editingPromptName, setEditingPromptName] = useState("");
  const [editingPromptText, setEditingPromptText] = useState("");
  const [savingPrompt, setSavingPrompt] = useState(false);
  const [showUploadOntologyModal, setShowUploadOntologyModal] = useState(false);
  const [reprocessing, setReprocessing] = useState(false);

  const datasetId = selectedDataset?.id ?? null;
  const datasetName = selectedDataset?.name ?? null;

  // Load config whenever dataset changes
  useEffect(() => {
    if (!cogniInstance || isInitializing || !datasetId) return;
    loadGraphModelsConfig(cogniInstance).then((cfg) => {
      setGraphModels(cfg.models);
      setCustomPrompts(cfg.customPrompts ?? {});
      setSelectedModelId(findModelForDataset(cfg.models, datasetId)?.id ?? null);
      setSelectedPromptName(findPromptForDataset(cfg.promptAssignments ?? {}, datasetId));
      setSelectedOntologyKey(findOntologyForDataset(cfg.ontologyAssignments ?? {}, datasetId));
    }).catch(() => {});
    listOntologies(cogniInstance).then(setOntologies).catch(() => {});
  }, [datasetId, cogniInstance, isInitializing]);

  // Reset config state on dataset change
  useEffect(() => {
    setSelectedModelId(null);
    setSelectedPromptName(null);
    setSelectedOntologyKey(null);
    setGraphModels([]);
    setCustomPrompts({});
    setOntologies({});
    setDatasetStatus("empty");
    prevStatusRef.current = "empty";
  }, [datasetId]);

  // Poll dataset processing status
  useEffect(() => {
    if (!datasetId || !cogniInstance || isInitializing) return;

    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    async function checkStatus() {
      try {
        const resp = await cogniInstance!.fetch(`/v1/datasets/status?dataset=${datasetId}`);
        if (!resp.ok || cancelled) return;
        const data: Record<string, DatasetProcessingStatus> = await resp.json();
        const status = mapProcessingStatus(data[datasetId!]);
        if (!cancelled) setDatasetStatus(status);
        if (!cancelled && (status === "pending" || status === "running")) {
          timer = setTimeout(checkStatus, 5000);
        }
      } catch {
        // ignore
      }
    }

    checkStatus();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [datasetId, cogniInstance, isInitializing, pollKey]);

  // Reload visualization when status transitions to completed
  useEffect(() => {
    if (prevStatusRef.current !== datasetStatus) {
      if ((prevStatusRef.current === "running" || prevStatusRef.current === "pending") && datasetStatus === "completed") {
        setVizRefreshKey((k) => k + 1);
      }
      prevStatusRef.current = datasetStatus;
    }
  }, [datasetStatus]);

  // Fetch visualization
  useEffect(() => {
    if (!datasetId || isInitializing) {
      setLoading(false);
      return;
    }

    setLoading(true);
    setIframeSrc(null);
    setError(null);

    if (blobRef.current) {
      URL.revokeObjectURL(blobRef.current);
      blobRef.current = null;
    }

    const fetchVisualization = cogniInstance
      ? cogniInstance.fetch(`/v1/visualize?dataset_id=${datasetId}`)
      : global.fetch(`/api/visualize?dataset_id=${datasetId}`, { credentials: "include" });

    fetchVisualization
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.text();
      })
      .then((html) => {
        if (html && html.length > 100 && (html.includes("<!DOCTYPE") || html.includes("<html"))) {
          const blob = new Blob([html], { type: "text/html" });
          const url = URL.createObjectURL(blob);
          blobRef.current = url;
          setIframeSrc(url);
        } else {
          setError("No graph data in this brain yet.");
        }
      })
      .catch((err) => {
        setError(err.message || "Failed to load visualization");
      })
      .finally(() => setLoading(false));

    return () => {
      if (blobRef.current) {
        URL.revokeObjectURL(blobRef.current);
        blobRef.current = null;
      }
    };
  }, [datasetId, isInitializing, vizRefreshKey]);

  // Close dropdowns on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (modelDropdownRef.current && !modelDropdownRef.current.contains(e.target as Node)) setModelDropdownOpen(false);
    }
    if (modelDropdownOpen) { document.addEventListener("mousedown", handleClick); return () => document.removeEventListener("mousedown", handleClick); }
  }, [modelDropdownOpen]);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (promptDropdownRef.current && !promptDropdownRef.current.contains(e.target as Node)) setPromptDropdownOpen(false);
    }
    if (promptDropdownOpen) { document.addEventListener("mousedown", handleClick); return () => document.removeEventListener("mousedown", handleClick); }
  }, [promptDropdownOpen]);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ontologyDropdownRef.current && !ontologyDropdownRef.current.contains(e.target as Node)) setOntologyDropdownOpen(false);
    }
    if (ontologyDropdownOpen) { document.addEventListener("mousedown", handleClick); return () => document.removeEventListener("mousedown", handleClick); }
  }, [ontologyDropdownOpen]);

  // Config helpers
  function getCognifyOptions(): { graphModel?: object; customPrompt?: string; ontologyKey?: string[] } {
    const opts: { graphModel?: object; customPrompt?: string; ontologyKey?: string[] } = {};
    if (selectedModelId) {
      const model = graphModels.find((m) => m.id === selectedModelId);
      if (model) opts.graphModel = toGraphModelSchema(toCleanSchema(model.schema));
    }
    if (selectedPromptName && customPrompts[selectedPromptName]) {
      opts.customPrompt = customPrompts[selectedPromptName];
    }
    if (selectedOntologyKey) {
      opts.ontologyKey = [selectedOntologyKey];
    }
    return opts;
  }

  function handleSelectModel(modelId: string | null) {
    setSelectedModelId(modelId);
    setModelDropdownOpen(false);
    const name = modelId ? (graphModels.find((m) => m.id === modelId)?.name ?? "Unknown") : "Automatic";
    notifications.show({ title: "Graph model updated", message: `"${datasetName}" will use "${name}" on next run.`, color: "green", autoClose: 4000 });
    if (cogniInstance && datasetId) {
      assignGraphModelToDataset(cogniInstance, datasetId, modelId).catch(() => {});
    }
  }

  function handleSelectPrompt(name: string | null) {
    setSelectedPromptName(name);
    setPromptDropdownOpen(false);
    notifications.show({ title: "Prompt updated", message: `"${datasetName}" will use "${name ?? "Automatic"}" on next run.`, color: "green", autoClose: 4000 });
    if (cogniInstance && datasetId) {
      assignPromptToDataset(cogniInstance, datasetId, name).catch(() => {});
    }
  }

  function handleSelectOntology(key: string | null) {
    setSelectedOntologyKey(key);
    setOntologyDropdownOpen(false);
    notifications.show({ title: "Ontology updated", message: `"${datasetName}" will use "${key ?? "Automatic"}" on next run.`, color: "green", autoClose: 4000 });
    if (cogniInstance && datasetId) {
      assignOntologyToDataset(cogniInstance, datasetId, key).catch(() => {});
    }
  }

  async function handleInferPrompt() {
    if (!selectedModelId || !cogniInstance) return;
    setInferringPrompt(true);
    try {
      const model = graphModels.find((m) => m.id === selectedModelId);
      if (model) {
        const result = await generateCustomPrompt(cogniInstance, toGraphModelSchema(toCleanSchema(model.schema)));
        setEditingPromptName(`${datasetName} Prompt`);
        setEditingPromptText(result.customPrompt);
        setShowCreatePromptModal(false);
        setShowPromptEditor(true);
        notifications.show({ title: "Prompt generated", message: "Review and edit the prompt below.", color: "green", autoClose: 4000 });
      }
    } catch (err) {
      notifications.show({ title: "Generation failed", message: err instanceof Error ? err.message : String(err), color: "red" });
    } finally {
      setInferringPrompt(false);
    }
  }

  async function handleSavePrompt() {
    if (!cogniInstance) return;
    const name = editingPromptName.trim();
    if (!name) { notifications.show({ title: "Name required", message: "Please enter a prompt name.", color: "yellow" }); return; }
    setSavingPrompt(true);
    try {
      await saveCustomPrompt(cogniInstance, name, editingPromptText);
      setCustomPrompts((prev) => ({ ...prev, [name]: editingPromptText }));
      setSelectedPromptName(name);
      setShowPromptEditor(false);
      notifications.show({ title: "Prompt saved", message: `"${name}" saved.`, color: "green", autoClose: 4000 });
    } catch {
      notifications.show({ title: "Failed", message: "Could not save prompt.", color: "red" });
    } finally {
      setSavingPrompt(false);
    }
  }

  async function handleReprocess() {
    if (!cogniInstance || !datasetId || !datasetName) return;
    setReprocessing(true);
    setIframeSrc(null);
    setDatasetStatus("running");
    try {
      await cognifyDataset({ id: datasetId, name: datasetName, data: [], status: "processing" }, cogniInstance, getCognifyOptions());
      trackEvent({ pageName: "Knowledge Graph", eventName: "dataset_reprocessed", additionalProperties: { dataset_id: datasetId } });
      setPollKey((k) => k + 1);
      notifications.show({ title: "Re-processing started", message: "The knowledge graph is being rebuilt.", color: "blue", autoClose: 4000 });
    } catch (err) {
      setReprocessing(false);
      setDatasetStatus("failed");
      notifications.show({ title: "Re-process failed", message: err instanceof Error ? err.message : String(err), color: "red" });
    } finally {
      setReprocessing(false);
    }
  }

  if (isInitializing) {
    return <><TrackPageView page="Knowledge Graph" /><div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", background: "#FFFFFF" }}><video src="/videos/mascot-waiting.mp4" autoPlay loop muted playsInline style={{ width: 200, height: "auto" }} /></div></>;
  }

  if (!selectedDataset && datasets.length > 0) {
    return (
      <><TrackPageView page="Knowledge Graph" />
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", gap: 12, fontFamily: '"Inter", system-ui, sans-serif' }}>
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#A1A1AA" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="2" /><circle cx="12" cy="5" r="1.5" /><circle cx="19" cy="16" r="1.5" /><circle cx="5" cy="16" r="1.5" />
          <line x1="12" y1="7" x2="12" y2="10" /><line x1="13.7" y1="13.3" x2="17.8" y2="15.2" /><line x1="10.3" y1="13.3" x2="6.2" y2="15.2" />
        </svg>
        <span style={{ fontSize: 15, fontWeight: 500, color: "#18181B" }}>Select a brain</span>
        <span style={{ fontSize: 13, color: "#A1A1AA", textAlign: "center", maxWidth: 360, lineHeight: "20px" }}>
          Use the dataset selector in the breadcrumbs at the top to choose which knowledge graph to visualize.
        </span>
      </div></>
    );
  }

  if (datasets.length === 0) {
    return (
      <><TrackPageView page="Knowledge Graph" /><div style={{ padding: 32, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", gap: 12, fontFamily: '"Inter", system-ui, sans-serif' }}>
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#A1A1AA" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="2" /><circle cx="12" cy="5" r="1.5" /><circle cx="19" cy="16" r="1.5" /><circle cx="5" cy="16" r="1.5" />
          <line x1="12" y1="7" x2="12" y2="10" /><line x1="13.7" y1="13.3" x2="17.8" y2="15.2" /><line x1="10.3" y1="13.3" x2="6.2" y2="15.2" />
        </svg>
        <span style={{ fontSize: 15, fontWeight: 500, color: "#18181B" }}>No brains yet</span>
        <span style={{ fontSize: 13, color: "#A1A1AA", textAlign: "center", maxWidth: 360, lineHeight: "20px" }}>
          Create a brain and upload documents to visualize your knowledge graph.
        </span>
        <Link href="/datasets" style={{ background: "#6510F4", color: "#fff", border: "none", borderRadius: 8, padding: "8px 20px", fontSize: 13, fontWeight: 500, textDecoration: "none", marginTop: 4 }}>
          Go to Datasets
        </Link>
      </div></>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", fontFamily: '"Inter", system-ui, sans-serif' }}>
      <TrackPageView page="Knowledge Graph" />

      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "20px 24px 16px", flexShrink: 0, gap: 12, flexWrap: "wrap" }}>
        {/* Left: title + status badge */}
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <h1 style={{ fontSize: 20, fontWeight: 300, color: "#18181B", margin: 0, fontFamily: '"TWK Lausanne", system-ui, sans-serif' }}>Knowledge Graph</h1>
          {datasetName && (
            <div style={{ display: "flex", alignItems: "center", gap: 5, background: (datasetStatus === "pending" || datasetStatus === "running") ? "#FFFBEB" : datasetStatus === "failed" ? "#FEF2F2" : "#F0FDF4", borderRadius: 6, padding: "3px 8px" }}>
              <div style={{ width: 6, height: 6, borderRadius: "50%", background: STATUS_CONFIG[datasetStatus].color, ...(datasetStatus === "running" || datasetStatus === "pending" ? { animation: "pulse-dot 1.5s ease-in-out infinite" } : {}) }} />
              <span style={{ fontSize: 12, fontWeight: 500, color: STATUS_CONFIG[datasetStatus].color }}>{STATUS_CONFIG[datasetStatus].label}</span>
            </div>
          )}
          {(datasetStatus === "running" || datasetStatus === "pending") && (
            <style>{`@keyframes pulse-dot { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }`}</style>
          )}
        </div>

        {/* Right: config toolbar (only when a dataset is selected) */}
        {datasetId && (
          <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>

            {/* Graph Model dropdown */}
            <div ref={modelDropdownRef} style={{ position: "relative" }}>
              <Tooltip label="Define entity types and relationships to control how Cognee structures your knowledge graph." withArrow multiline w={240} position="bottom">
                <button
                  onClick={() => setModelDropdownOpen((v) => !v)}
                  className="cursor-pointer hover:bg-cognee-hover"
                  style={{ background: "#fff", color: "#3F3F46", border: "1px solid #E4E4E7", borderRadius: 6, padding: "6px 12px", fontSize: 12, fontWeight: 500, display: "flex", alignItems: "center", gap: 5, whiteSpace: "nowrap" }}
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#3F3F46" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="6" cy="6" r="3" /><circle cx="18" cy="6" r="3" /><circle cx="12" cy="18" r="3" /><line x1="8.5" y1="7.5" x2="10.5" y2="16" /><line x1="15.5" y1="7.5" x2="13.5" y2="16" /></svg>
                  <span>{selectedModelId ? (graphModels.find((m) => m.id === selectedModelId)?.name ?? "Automatic") : "Automatic"}</span>
                  <svg width="9" height="9" viewBox="0 0 10 10" fill="none"><path d="M2.5 4L5 6.5L7.5 4" stroke="#71717A" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
                </button>
              </Tooltip>
              {modelDropdownOpen && (
                <div style={{ position: "absolute", top: "calc(100% + 4px)", right: 0, background: "#fff", border: "1px solid #E4E4E7", borderRadius: 8, boxShadow: "0 8px 24px rgba(0,0,0,0.1)", minWidth: 220, zIndex: 50, overflow: "hidden" }}>
                  <div style={{ padding: "6px 10px 4px", fontSize: 11, fontWeight: 600, color: "#A1A1AA", letterSpacing: 0.3, textTransform: "uppercase" }}>Graph Model</div>
                  <button onClick={() => handleSelectModel(null)} className="cursor-pointer hover:bg-cognee-hover" style={{ width: "100%", background: "none", border: "none", padding: "8px 12px", fontSize: 13, color: "#18181B", display: "flex", alignItems: "center", gap: 8, textAlign: "left", fontFamily: "inherit" }}>
                    <span style={{ width: 16, textAlign: "center", fontSize: 13, color: "#6510F4" }}>{selectedModelId === null ? "✓" : ""}</span>
                    <span style={{ flex: 1 }}>Automatic</span>
                    <span style={{ fontSize: 11, color: "#A1A1AA" }}>Default</span>
                  </button>
                  {graphModels.length > 0 && <div style={{ height: 1, background: "#F4F4F5", margin: "4px 0" }} />}
                  {graphModels.map((model) => (
                    <div key={model.id} className="hover:bg-cognee-hover" style={{ display: "flex", alignItems: "center", padding: "8px 12px", gap: 8 }}>
                      <button onClick={() => handleSelectModel(model.id)} className="cursor-pointer" style={{ flex: 1, background: "none", border: "none", padding: 0, fontSize: 13, color: "#18181B", display: "flex", alignItems: "center", gap: 8, textAlign: "left", fontFamily: "inherit" }}>
                        <span style={{ width: 16, textAlign: "center", fontSize: 13, color: "#6510F4", flexShrink: 0 }}>{selectedModelId === model.id ? "✓" : ""}</span>
                        <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{model.name}</span>
                      </button>
                      <button onClick={(e) => { e.stopPropagation(); setModelDropdownOpen(false); router.push(`/graph-models/${model.id}`); }} className="cursor-pointer hover:opacity-100" style={{ background: "none", border: "none", padding: 2, opacity: 0.4, transition: "opacity 150ms", flexShrink: 0 }} title="Edit model">
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#3F3F46" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7" /><path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z" /></svg>
                      </button>
                    </div>
                  ))}
                  <div style={{ height: 1, background: "#F4F4F5", margin: "4px 0" }} />
                  <button onClick={() => { setModelDropdownOpen(false); router.push("/graph-models"); }} className="cursor-pointer hover:bg-cognee-hover" style={{ width: "100%", background: "none", border: "none", padding: "8px 12px", fontSize: 13, color: "#6510F4", fontWeight: 500, display: "flex", alignItems: "center", gap: 8, textAlign: "left", fontFamily: "inherit" }}>
                    <span style={{ width: 16, textAlign: "center" }}>+</span>
                    <span>Create new</span>
                  </button>
                </div>
              )}
            </div>

            {/* Prompt dropdown */}
            <div ref={promptDropdownRef} style={{ position: "relative" }}>
              <Tooltip label="Custom instructions that guide how Cognee extracts entities and relationships from your data." withArrow multiline w={240} position="bottom">
                <button
                  onClick={() => setPromptDropdownOpen((v) => !v)}
                  className="cursor-pointer hover:bg-cognee-hover"
                  style={{ background: "#fff", color: "#3F3F46", border: "1px solid #E4E4E7", borderRadius: 6, padding: "6px 12px", fontSize: 12, fontWeight: 500, display: "flex", alignItems: "center", gap: 5, whiteSpace: "nowrap" }}
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#3F3F46" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" /></svg>
                  <span>{selectedPromptName ?? "Automatic"}</span>
                  <svg width="9" height="9" viewBox="0 0 10 10" fill="none"><path d="M2.5 4L5 6.5L7.5 4" stroke="#71717A" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
                </button>
              </Tooltip>
              {promptDropdownOpen && (
                <div style={{ position: "absolute", top: "calc(100% + 4px)", right: 0, background: "#fff", border: "1px solid #E4E4E7", borderRadius: 8, boxShadow: "0 8px 24px rgba(0,0,0,0.1)", minWidth: 220, zIndex: 50, overflow: "hidden" }}>
                  <div style={{ padding: "6px 10px 4px", fontSize: 11, fontWeight: 600, color: "#A1A1AA", letterSpacing: 0.3, textTransform: "uppercase" }}>Custom Prompt</div>
                  <button onClick={() => handleSelectPrompt(null)} className="cursor-pointer hover:bg-cognee-hover" style={{ width: "100%", background: "none", border: "none", padding: "8px 12px", fontSize: 13, color: "#18181B", display: "flex", alignItems: "center", gap: 8, textAlign: "left", fontFamily: "inherit" }}>
                    <span style={{ width: 16, textAlign: "center", fontSize: 13, color: "#6510F4" }}>{selectedPromptName === null ? "✓" : ""}</span>
                    <span style={{ flex: 1 }}>Automatic</span>
                    <span style={{ fontSize: 11, color: "#A1A1AA" }}>Default</span>
                  </button>
                  {Object.keys(customPrompts).length > 0 && <div style={{ height: 1, background: "#F4F4F5", margin: "4px 0" }} />}
                  {Object.entries(customPrompts).map(([name, text]) => (
                    <div key={name} className="hover:bg-cognee-hover" style={{ display: "flex", alignItems: "center", padding: "8px 12px", gap: 8 }}>
                      <button onClick={() => handleSelectPrompt(name)} className="cursor-pointer" style={{ flex: 1, background: "none", border: "none", padding: 0, fontSize: 13, color: "#18181B", display: "flex", alignItems: "center", gap: 8, textAlign: "left", fontFamily: "inherit" }}>
                        <span style={{ width: 16, textAlign: "center", fontSize: 13, color: "#6510F4", flexShrink: 0 }}>{selectedPromptName === name ? "✓" : ""}</span>
                        <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{name}</span>
                      </button>
                      <button onClick={(e) => { e.stopPropagation(); setPromptDropdownOpen(false); setEditingPromptName(name); setEditingPromptText(text); setShowPromptEditor(true); }} className="cursor-pointer hover:opacity-100" style={{ background: "none", border: "none", padding: 2, opacity: 0.4, transition: "opacity 150ms", flexShrink: 0 }} title="Edit prompt">
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#3F3F46" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7" /><path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z" /></svg>
                      </button>
                    </div>
                  ))}
                  <div style={{ height: 1, background: "#F4F4F5", margin: "4px 0" }} />
                  <button onClick={() => { setPromptDropdownOpen(false); setShowCreatePromptModal(true); }} className="cursor-pointer hover:bg-cognee-hover" style={{ width: "100%", background: "none", border: "none", padding: "8px 12px", fontSize: 13, color: "#6510F4", fontWeight: 500, display: "flex", alignItems: "center", gap: 8, textAlign: "left", fontFamily: "inherit" }}>
                    <span style={{ width: 16, textAlign: "center" }}>+</span>
                    <span>Create new</span>
                  </button>
                </div>
              )}
            </div>

            {/* Ontology dropdown */}
            <div ref={ontologyDropdownRef} style={{ position: "relative" }}>
              <Tooltip label="Upload a formal ontology (OWL/RDF) to enforce domain-specific vocabulary and relationships in your knowledge graph." withArrow multiline w={240} position="bottom">
                <button
                  onClick={() => setOntologyDropdownOpen((v) => !v)}
                  className="cursor-pointer hover:bg-cognee-hover"
                  style={{ background: "#fff", color: "#3F3F46", border: "1px solid #E4E4E7", borderRadius: 6, padding: "6px 12px", fontSize: 12, fontWeight: 500, display: "flex", alignItems: "center", gap: 5, whiteSpace: "nowrap" }}
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#3F3F46" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 19.5A2.5 2.5 0 016.5 17H20" /><path d="M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z" /></svg>
                  <span>{selectedOntologyKey ?? "Automatic"}</span>
                  <svg width="9" height="9" viewBox="0 0 10 10" fill="none"><path d="M2.5 4L5 6.5L7.5 4" stroke="#71717A" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
                </button>
              </Tooltip>
              {ontologyDropdownOpen && (
                <div style={{ position: "absolute", top: "calc(100% + 4px)", right: 0, background: "#fff", border: "1px solid #E4E4E7", borderRadius: 8, boxShadow: "0 8px 24px rgba(0,0,0,0.1)", width: 260, zIndex: 50, overflow: "hidden" }}>
                  <div style={{ padding: "6px 10px 4px", fontSize: 11, fontWeight: 600, color: "#A1A1AA", letterSpacing: 0.3, textTransform: "uppercase" }}>Ontology</div>
                  <button onClick={() => handleSelectOntology(null)} className="cursor-pointer hover:bg-cognee-hover" style={{ width: "100%", background: "none", border: "none", padding: "8px 12px", fontSize: 13, color: "#18181B", display: "flex", alignItems: "center", gap: 8, textAlign: "left", fontFamily: "inherit" }}>
                    <span style={{ width: 16, textAlign: "center", fontSize: 13, color: "#6510F4", flexShrink: 0 }}>{selectedOntologyKey === null ? "✓" : ""}</span>
                    <span>Automatic</span>
                  </button>
                  {Object.keys(ontologies).length > 0 && <div style={{ height: 1, background: "#F4F4F5", margin: "4px 0" }} />}
                  {Object.entries(ontologies).map(([key, meta]) => (
                    <div key={key} className="hover:bg-cognee-hover" style={{ display: "flex", alignItems: "center", padding: "8px 12px", gap: 8 }}>
                      <button onClick={() => handleSelectOntology(key)} className="cursor-pointer" style={{ flex: 1, minWidth: 0, background: "none", border: "none", padding: 0, fontSize: 13, color: "#18181B", display: "flex", alignItems: "center", gap: 8, textAlign: "left", fontFamily: "inherit" }}>
                        <span style={{ width: 16, textAlign: "center", fontSize: 13, color: "#6510F4", flexShrink: 0 }}>{selectedOntologyKey === key ? "✓" : ""}</span>
                        <span title={meta.filename} style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{meta.filename}</span>
                      </button>
                      <button onClick={async (e) => {
                        e.stopPropagation();
                        if (!cogniInstance) return;
                        try {
                          await deleteOntology(cogniInstance, key);
                          setOntologies((prev) => { const next = { ...prev }; delete next[key]; return next; });
                          if (selectedOntologyKey === key) setSelectedOntologyKey(null);
                          notifications.show({ title: "Ontology deleted", message: `"${key}" removed.`, color: "green", autoClose: 4000 });
                        } catch (err) {
                          notifications.show({ title: "Delete failed", message: err instanceof Error ? err.message : String(err), color: "red" });
                        }
                      }} className="cursor-pointer hover:opacity-100" style={{ background: "none", border: "none", padding: 4, opacity: 0.5, transition: "opacity 150ms", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center" }} title="Delete ontology">
                        <svg width="14" height="14" viewBox="0 0 16 16" fill="none" style={{ flexShrink: 0 }}><path d="M3 4h10M6 4V3h4v1M5 4v8.5a.5.5 0 00.5.5h5a.5.5 0 00.5-.5V4" stroke="#EF4444" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" /></svg>
                      </button>
                    </div>
                  ))}
                  <div style={{ height: 1, background: "#F4F4F5", margin: "4px 0" }} />
                  <button onClick={() => { setOntologyDropdownOpen(false); setShowUploadOntologyModal(true); }} className="cursor-pointer hover:bg-cognee-hover" style={{ width: "100%", background: "none", border: "none", padding: "8px 12px", fontSize: 13, color: "#6510F4", fontWeight: 500, display: "flex", alignItems: "center", gap: 8, textAlign: "left", fontFamily: "inherit" }}>
                    <span style={{ width: 16, textAlign: "center", flexShrink: 0 }}>+</span>
                    <span>Upload new</span>
                  </button>
                </div>
              )}
            </div>

            {/* Re-process button */}
            <button
              onClick={handleReprocess}
              disabled={reprocessing || datasetStatus === "running" || datasetStatus === "pending"}
              className="cursor-pointer hover:bg-cognee-purple-hover"
              style={{ background: "#6510F4", color: "#fff", border: "none", borderRadius: 6, padding: "6px 12px", fontSize: 12, fontWeight: 500, display: "flex", alignItems: "center", gap: 5, whiteSpace: "nowrap", opacity: (reprocessing || datasetStatus === "running" || datasetStatus === "pending") ? 0.6 : 1 }}
            >
              {(reprocessing || datasetStatus === "running" || datasetStatus === "pending") ? (
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ animation: "spin 1s linear infinite" }}><path d="M21 12a9 9 0 11-6.219-8.56" /></svg>
              ) : (
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="1 4 1 10 7 10" /><path d="M3.51 15a9 9 0 102.13-9.36L1 10" /></svg>
              )}
              <span>{(reprocessing || datasetStatus === "running" || datasetStatus === "pending") ? "Processing…" : "Re-process"}</span>
            </button>
            <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
          </div>
        )}
      </div>

      {/* Graph */}
      <div style={{ flex: 1, position: "relative", overflow: "hidden", borderTop: "1px solid #E4E4E7" }}>
        {loading && (
          <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", background: "#FFFFFF", zIndex: 1 }}>
            <video src="/videos/mascot-waiting.mp4" autoPlay loop muted playsInline style={{ width: 200, height: "auto" }} />
          </div>
        )}
        {iframeSrc ? (
          <iframe
            key={datasetId}
            src={iframeSrc}
            style={{ width: "100%", height: "100%", border: "none" }}
            title="Knowledge Graph Visualization"
          />
        ) : !loading ? (
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", flexDirection: "column", gap: 12 }}>
            {(datasetStatus === "pending" || datasetStatus === "running") ? (
              <>
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#6510F4" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ animation: "spin 2s linear infinite" }}>
                  <path d="M21 12a9 9 0 11-6.219-8.56" />
                </svg>
                <span style={{ fontSize: 15, fontWeight: 500, color: "#18181B" }}>Graph is currently being generated</span>
                <span style={{ fontSize: 13, color: "#A1A1AA", textAlign: "center", maxWidth: 400, lineHeight: "20px" }}>
                  Your data is being processed. The knowledge graph will appear here once it&apos;s ready.
                </span>
              </>
            ) : (
              <>
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#A1A1AA" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="2" /><circle cx="12" cy="5" r="1.5" /><circle cx="19" cy="16" r="1.5" /><circle cx="5" cy="16" r="1.5" />
                  <line x1="12" y1="7" x2="12" y2="10" /><line x1="13.7" y1="13.3" x2="17.8" y2="15.2" /><line x1="10.3" y1="13.3" x2="6.2" y2="15.2" />
                </svg>
                <span style={{ fontSize: 15, fontWeight: 500, color: "#18181B" }}>No graph data for this dataset yet</span>
                <span style={{ fontSize: 13, color: "#A1A1AA", textAlign: "center", maxWidth: 400, lineHeight: "20px" }}>
                  Upload documents to a brain first, then the knowledge graph will be built automatically. You can upload files from the <Link href="/datasets" style={{ color: "#6510F4", textDecoration: "underline" }}>Brains</Link> page.
                </span>
                {error && error !== "No graph data in this brain yet." && (
                  <span style={{ fontSize: 11, color: "#EF4444", textAlign: "center", maxWidth: 400, fontFamily: "monospace", wordBreak: "break-all" }}>{error}</span>
                )}
              </>
            )}
          </div>
        ) : null}
      </div>

      {/* Create custom prompt modal */}
      {showCreatePromptModal && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.3)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={() => !inferringPrompt && setShowCreatePromptModal(false)}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "#fff", borderRadius: 12, padding: 24, width: 440, display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 16px 48px rgba(0,0,0,0.12)" }}>
            <h2 style={{ fontSize: 18, fontWeight: 600, color: "#18181B", margin: 0 }}>Create Custom Prompt</h2>
            {inferringPrompt ? (
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 12, padding: "24px 0" }}>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#6510F4" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ animation: "spin 1s linear infinite" }}><path d="M21 12a9 9 0 11-6.219-8.56" /></svg>
                <span style={{ fontSize: 14, color: "#6510F4", fontWeight: 500 }}>Generating prompt from &ldquo;{graphModels.find((m) => m.id === selectedModelId)?.name ?? "graph model"}&rdquo;...</span>
                <span style={{ fontSize: 12, color: "#71717A" }}>This may take a moment</span>
              </div>
            ) : (
              <>
                <p style={{ fontSize: 13, color: "#71717A", margin: 0, lineHeight: "20px" }}>
                  A custom prompt guides how Cognee extracts entities and relationships from your data.
                </p>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  <button
                    onClick={handleInferPrompt}
                    disabled={!selectedModelId}
                    className="cursor-pointer hover:bg-cognee-hover"
                    style={{ display: "flex", alignItems: "center", gap: 12, background: "#fff", border: "1px solid #E4E4E7", borderRadius: 8, padding: "14px 16px", textAlign: "left", fontFamily: "inherit", opacity: !selectedModelId ? 0.5 : 1 }}
                  >
                    <div style={{ width: 36, height: 36, background: "#F0EDFF", borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#6510F4" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><path d="M9.09 9a3 3 0 015.83 1c0 2-3 3-3 3" /><line x1="12" y1="17" x2="12.01" y2="17" /></svg>
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                      <span style={{ fontSize: 14, fontWeight: 500, color: "#18181B" }}>Generate from graph model</span>
                      <span style={{ fontSize: 12, color: "#71717A" }}>
                        {!selectedModelId ? "Select a graph model first" : `Using "${graphModels.find((m) => m.id === selectedModelId)?.name ?? "Unknown"}"`}
                      </span>
                    </div>
                  </button>
                  <button
                    onClick={() => { setEditingPromptName(`${datasetName} Prompt`); setEditingPromptText(DEFAULT_EXTRACTION_PROMPT); setShowCreatePromptModal(false); setShowPromptEditor(true); }}
                    className="cursor-pointer hover:bg-cognee-hover"
                    style={{ display: "flex", alignItems: "center", gap: 12, background: "#fff", border: "1px solid #E4E4E7", borderRadius: 8, padding: "14px 16px", textAlign: "left", fontFamily: "inherit" }}
                  >
                    <div style={{ width: 36, height: 36, background: "#F4F4F5", borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#71717A" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" /></svg>
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                      <span style={{ fontSize: 14, fontWeight: 500, color: "#18181B" }}>Start blank</span>
                      <span style={{ fontSize: 12, color: "#71717A" }}>Write your own extraction prompt</span>
                    </div>
                  </button>
                </div>
                <button onClick={() => setShowCreatePromptModal(false)} className="cursor-pointer" style={{ background: "none", border: "none", fontSize: 13, color: "#71717A", fontFamily: "inherit", padding: "4px 0" }}>Cancel</button>
              </>
            )}
          </div>
        </div>
      )}

      {/* Prompt editor modal */}
      {showPromptEditor && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.3)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={() => !savingPrompt && setShowPromptEditor(false)}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "#fff", borderRadius: 12, padding: 24, width: 600, maxHeight: "80vh", display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 16px 48px rgba(0,0,0,0.12)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <h2 style={{ fontSize: 18, fontWeight: 600, color: "#18181B", margin: 0 }}>Edit Prompt</h2>
              <button onClick={() => setShowPromptEditor(false)} className="cursor-pointer" style={{ background: "none", border: "none", color: "#A1A1AA", fontSize: 18 }}>&#10005;</button>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <label style={{ fontSize: 12, fontWeight: 600, color: "#71717A", textTransform: "uppercase", letterSpacing: 0.3 }}>Name</label>
              <input type="text" value={editingPromptName} onChange={(e) => setEditingPromptName(e.target.value)} placeholder="Prompt name" style={{ border: "1px solid #E4E4E7", borderRadius: 8, padding: "8px 12px", fontSize: 14, fontFamily: "inherit", color: "#18181B", outline: "none" }} />
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4, flex: 1, minHeight: 0 }}>
              <label style={{ fontSize: 12, fontWeight: 600, color: "#71717A", textTransform: "uppercase", letterSpacing: 0.3 }}>Prompt</label>
              <textarea value={editingPromptText} onChange={(e) => setEditingPromptText(e.target.value)} placeholder="Write your extraction prompt here..." style={{ border: "1px solid #E4E4E7", borderRadius: 8, padding: "10px 12px", fontSize: 13, fontFamily: '"Inter", system-ui, sans-serif', color: "#18181B", outline: "none", resize: "vertical", minHeight: 200, maxHeight: 400, lineHeight: "20px" }} />
            </div>
            <div style={{ display: "flex", gap: 8, justifyContent: "space-between" }}>
              <button
                onClick={async () => {
                  const name = editingPromptName.trim();
                  if (!name || !cogniInstance) return;
                  if (!window.confirm(`Delete prompt "${name}"?`)) return;
                  try {
                    await deleteCustomPrompt(cogniInstance, name);
                    setCustomPrompts((prev) => { const next = { ...prev }; delete next[name]; return next; });
                    if (selectedPromptName === name) setSelectedPromptName(null);
                    setShowPromptEditor(false);
                    notifications.show({ title: "Prompt deleted", message: `"${name}" removed.`, color: "green", autoClose: 4000 });
                  } catch (err) {
                    notifications.show({ title: "Delete failed", message: err instanceof Error ? err.message : String(err), color: "red" });
                  }
                }}
                className="cursor-pointer hover:opacity-100"
                style={{ background: "none", border: "none", padding: 4, opacity: 0.5, transition: "opacity 150ms", display: "flex", alignItems: "center" }}
                title="Delete prompt"
              >
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M3 4h10M6 4V3h4v1M5 4v8.5a.5.5 0 00.5.5h5a.5.5 0 00.5-.5V4" stroke="#EF4444" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" /></svg>
              </button>
              <div style={{ display: "flex", gap: 8 }}>
                <button onClick={() => setShowPromptEditor(false)} className="cursor-pointer" style={{ background: "#fff", border: "1px solid #E4E4E7", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "#3F3F46", fontFamily: "inherit" }}>Cancel</button>
                <button onClick={handleSavePrompt} disabled={savingPrompt} className="cursor-pointer" style={{ background: "#6510F4", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "#fff", fontFamily: "inherit" }}>
                  {savingPrompt ? "Saving..." : "Save prompt"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Upload ontology modal */}
      {showUploadOntologyModal && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.3)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={() => setShowUploadOntologyModal(false)}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "#fff", borderRadius: 12, padding: 24, width: 440, display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 16px 48px rgba(0,0,0,0.12)" }}>
            <h2 style={{ fontSize: 18, fontWeight: 600, color: "#18181B", margin: 0 }}>Upload Ontology</h2>
            <p style={{ fontSize: 13, color: "#71717A", margin: 0, lineHeight: "20px" }}>
              Upload an OWL ontology file to guide how Cognee structures your knowledge graph.
            </p>
            <form onSubmit={async (e) => {
              e.preventDefault();
              if (!cogniInstance) return;
              const form = e.currentTarget;
              const keyInput = form.elements.namedItem("ontologyKey") as HTMLInputElement;
              const fileInput = form.elements.namedItem("ontologyFile") as HTMLInputElement;
              const descInput = form.elements.namedItem("description") as HTMLInputElement;
              const key = keyInput.value.trim();
              const file = fileInput.files?.[0];
              if (!key || !file) return;
              const submitBtn = form.querySelector("button[type=submit]") as HTMLButtonElement;
              submitBtn.disabled = true;
              submitBtn.textContent = "Uploading...";
              try {
                await uploadOntology(cogniInstance, key, file, descInput.value.trim() || undefined);
                const updated = await listOntologies(cogniInstance);
                setOntologies(updated);
                setSelectedOntologyKey(key);
                setShowUploadOntologyModal(false);
                if (cogniInstance && datasetId) {
                  assignOntologyToDataset(cogniInstance, datasetId, key).catch(() => {});
                }
                notifications.show({ title: "Ontology uploaded", message: `"${key}" is ready to use.`, color: "green", autoClose: 4000 });
              } catch (err) {
                notifications.show({ title: "Upload failed", message: err instanceof Error ? err.message : String(err), color: "red" });
                submitBtn.disabled = false;
                submitBtn.textContent = "Upload";
              }
            }}>
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                  <label style={{ fontSize: 12, fontWeight: 600, color: "#71717A", textTransform: "uppercase", letterSpacing: 0.3 }}>Key</label>
                  <input name="ontologyKey" type="text" required placeholder="e.g. biomedical-ontology" style={{ border: "1px solid #E4E4E7", borderRadius: 8, padding: "8px 12px", fontSize: 14, fontFamily: "inherit", color: "#18181B", outline: "none" }} />
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                  <label style={{ fontSize: 12, fontWeight: 600, color: "#71717A", textTransform: "uppercase", letterSpacing: 0.3 }}>OWL File</label>
                  <label className="cursor-pointer" style={{ display: "flex", alignItems: "center", gap: 8, border: "1px solid #E4E4E7", borderRadius: 8, padding: "8px 12px", fontSize: 13, fontFamily: "inherit", color: "#71717A" }}>
                    <svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M8 1v10M4 5l4-4 4 4" stroke="#A1A1AA" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" /><path d="M1 11v2.5A1.5 1.5 0 002.5 15h11a1.5 1.5 0 001.5-1.5V11" stroke="#A1A1AA" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" /></svg>
                    <span data-file-label="true" style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>Choose a .owl file…</span>
                    <input name="ontologyFile" type="file" required accept=".owl" style={{ display: "none" }} onChange={(e) => {
                      const label = e.currentTarget.parentElement?.querySelector("[data-file-label]");
                      if (label) label.textContent = e.currentTarget.files?.[0]?.name ?? "Choose a .owl file…";
                    }} />
                  </label>
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                  <label style={{ fontSize: 12, fontWeight: 600, color: "#71717A", textTransform: "uppercase", letterSpacing: 0.3 }}>Description <span style={{ fontWeight: 400, textTransform: "none" }}>(optional)</span></label>
                  <input name="description" type="text" placeholder="What does this ontology define?" style={{ border: "1px solid #E4E4E7", borderRadius: 8, padding: "8px 12px", fontSize: 14, fontFamily: "inherit", color: "#18181B", outline: "none" }} />
                </div>
              </div>
              <div style={{ display: "flex", gap: 8, justifyContent: "flex-end", marginTop: 8 }}>
                <button type="button" onClick={() => setShowUploadOntologyModal(false)} className="cursor-pointer" style={{ background: "#fff", border: "1px solid #E4E4E7", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "#3F3F46", fontFamily: "inherit" }}>Cancel</button>
                <button type="submit" className="cursor-pointer" style={{ background: "#6510F4", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "#fff", fontFamily: "inherit" }}>Upload</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
