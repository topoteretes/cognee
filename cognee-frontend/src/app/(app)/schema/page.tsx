"use client";

import { useEffect, useState, useRef } from "react";
import { useRouter } from "next/navigation";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import PageLoading from "@/ui/elements/PageLoading";
import { useFilter, useRefreshDatasetsOnMount } from "@/ui/layout/FilterContext";
import { TrackPageView, trackEvent } from "@/modules/analytics";
import BrainSelector from "@/ui/elements/BrainSelector";
import { notifications } from "@mantine/notifications";
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
import { captureException } from "@/utils/monitoring";

// Visualize renders the full graph synchronously with no caching, so large
// graphs can take well past the default 10s GET timeout.
const VISUALIZE_TIMEOUT_MS = 90_000;

// ── Shared button style (black) ───────────────────────────────────────────

const BTN: React.CSSProperties = {
  background: "rgba(0,0,0,0.75)",
  border: "1px solid rgba(255,255,255,0.18)",
  color: "#EDECEA",
  borderRadius: 7,
  padding: "6px 12px",
  fontSize: 12,
  fontWeight: 500,
  cursor: "pointer",
  display: "flex",
  alignItems: "center",
  gap: 5,
  whiteSpace: "nowrap" as const,
  fontFamily: "inherit",
  transition: "border-color 150ms, background 150ms",
};

const DEFAULT_EXTRACTION_PROMPT = `You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph.
**Nodes** represent entities and concepts.
**Edges** represent relationships between concepts.

# 1. Labeling Nodes
- Use basic types (e.g. "Person", not "Mathematician")
- Node IDs should be human-readable names found in the text
- Every node MUST include a "name" field

# 2. Numerical Data and Dates
- Use "Date" type for date entities, format "YYYY-MM-DD"
- Properties must be in key-value format
- Use snake_case for relationship names

# 3. Coreference Resolution
- Always use the most complete identifier for each entity throughout the knowledge graph

# 4. Strict Compliance
Adhere to the rules strictly.`;

function EmptyState({ onGoToBrains }: { onGoToBrains: () => void }) {
  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", paddingInline: 32, paddingBottom: 32 }}>
      <div style={{ flex: 1, background: "rgba(255,255,255,0.06)", backdropFilter: "blur(12px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 16, padding: 48 }}>
        <div style={{ width: 56, height: 56, background: "rgba(188,155,255,0.20)", border: "1px solid rgba(188,155,255,0.35)", borderRadius: 12, display: "flex", alignItems: "center", justifyContent: "center" }}>
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#6510F4" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="6" cy="6" r="3" /><circle cx="18" cy="6" r="3" /><circle cx="12" cy="18" r="3" /><line x1="8.5" y1="7.5" x2="10.5" y2="16" /><line x1="15.5" y1="7.5" x2="13.5" y2="16" /></svg>
        </div>
        <span style={{ fontSize: 16, fontWeight: 700, color: "#EDECEA" }}>No memory schema yet</span>
        <p style={{ fontSize: 14, color: "rgba(237,236,234,0.35)", margin: 0, maxWidth: 360, textAlign: "center" }}>
          Upload documents to a brain and let cognee process them. Once a knowledge graph is built, its type structure will appear here.
        </p>
        <button onClick={onGoToBrains} style={{ background: "#6510F4", color: "#fff", border: "none", borderRadius: 8, padding: "8px 20px", fontSize: 14, fontWeight: 500, marginTop: 8, cursor: "pointer" }}>
          Go to Brains
        </button>
      </div>
    </div>
  );
}

export default function SchemaPage() {
  const router = useRouter();
  const { cogniInstance, isInitializing } = useCogniInstance();
  const { datasets, selectedDataset } = useFilter();
  useRefreshDatasetsOnMount();

  // ── Viz state ────────────────────────────────────────────────────────────
  const [vizSrc, setVizSrc] = useState<string | null>(null);
  const [vizLoading, setVizLoading] = useState(true);
  const [vizError, setVizError] = useState<string | null>(null);
  // The iframe stays hidden behind the loading overlay until its content has
  // settled (dark theme applied, schema tab selected) — without this the
  // visualization visibly boots through light-mode/graph-view states.
  const [vizReady, setVizReady] = useState(false);
  const vizBlobRef = useRef<string | null>(null);
  const [vizRefreshKey, setVizRefreshKey] = useState(0);

  // ── Config state (moved from Mindmap) ─────────────────────────
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

  const [showCreatePromptModal, setShowCreatePromptModal] = useState(false);
  const [inferringPrompt, setInferringPrompt] = useState(false);
  const [showPromptEditor, setShowPromptEditor] = useState(false);
  const [editingPromptName, setEditingPromptName] = useState("");
  const [editingPromptText, setEditingPromptText] = useState("");
  const [savingPrompt, setSavingPrompt] = useState(false);
  const [showUploadOntologyModal, setShowUploadOntologyModal] = useState(false);
  const [reprocessing, setReprocessing] = useState(false);

  const activeDataset = selectedDataset ?? datasets[0] ?? null;
  const datasetId = activeDataset?.id ?? null;
  const datasetName = activeDataset?.name ?? null;

  // ── Reset on dataset change ───────────────────────────────────────────
  useEffect(() => {
    setVizSrc(null);
    setVizError(null);
    setVizReady(false);
    if (vizBlobRef.current) { URL.revokeObjectURL(vizBlobRef.current); vizBlobRef.current = null; }
    setSelectedModelId(null);
    setSelectedPromptName(null);
    setSelectedOntologyKey(null);
    setGraphModels([]);
    setCustomPrompts({});
    setOntologies({});
  }, [datasetId]);

  // ── Load config ───────────────────────────────────────────────────────
  useEffect(() => {
    if (!cogniInstance || isInitializing || !datasetId) return;
    loadGraphModelsConfig(cogniInstance).then((cfg) => {
      setGraphModels(cfg.models);
      setCustomPrompts(cfg.customPrompts ?? {});
      setSelectedModelId(findModelForDataset(cfg.models, datasetId)?.id ?? null);
      setSelectedPromptName(findPromptForDataset(cfg.promptAssignments ?? {}, datasetId));
      setSelectedOntologyKey(findOntologyForDataset(cfg.ontologyAssignments ?? {}, datasetId));
    }).catch((err) => captureException(err, { context: "schema-page.load-graph-config" }));
    listOntologies(cogniInstance).then(setOntologies).catch((err) =>
      captureException(err, { context: "schema-page.list-ontologies" }));
  }, [datasetId, cogniInstance, isInitializing]);

  // ── Close dropdowns on outside click ─────────────────────────────────
  useEffect(() => {
    function h(e: MouseEvent) { if (modelDropdownRef.current && !modelDropdownRef.current.contains(e.target as Node)) setModelDropdownOpen(false); }
    if (modelDropdownOpen) { document.addEventListener("mousedown", h); return () => document.removeEventListener("mousedown", h); }
  }, [modelDropdownOpen]);
  useEffect(() => {
    function h(e: MouseEvent) { if (promptDropdownRef.current && !promptDropdownRef.current.contains(e.target as Node)) setPromptDropdownOpen(false); }
    if (promptDropdownOpen) { document.addEventListener("mousedown", h); return () => document.removeEventListener("mousedown", h); }
  }, [promptDropdownOpen]);
  useEffect(() => {
    function h(e: MouseEvent) { if (ontologyDropdownRef.current && !ontologyDropdownRef.current.contains(e.target as Node)) setOntologyDropdownOpen(false); }
    if (ontologyDropdownOpen) { document.addEventListener("mousedown", h); return () => document.removeEventListener("mousedown", h); }
  }, [ontologyDropdownOpen]);

  // ── Fetch viz ─────────────────────────────────────────────────────────
  useEffect(() => {
    if (!activeDataset || isInitializing) { setVizLoading(false); return; }
    setVizLoading(true);
    setVizError(null);
    setVizReady(false);
    if (vizBlobRef.current) { URL.revokeObjectURL(vizBlobRef.current); vizBlobRef.current = null; }

    const fetchViz = cogniInstance
      ? cogniInstance.fetch(`/v1/visualize?dataset_id=${activeDataset.id}`, { timeoutMs: VISUALIZE_TIMEOUT_MS })
      : global.fetch(`/api/visualize?dataset_id=${activeDataset.id}`, { credentials: "include" });

    fetchViz
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.text(); })
      .then((html) => {
        if (html && html.length > 100 && (html.includes("<!DOCTYPE") || html.includes("<html"))) {
          const closeScript = "<" + "/script>";
          // Dark background CSS beats the first paint (no white flash); the
          // theme switch runs synchronously, and the schema-tab click retries
          // until the viz scripts have actually rendered the tab button.
          const inject =
            '<style>html{background:#0A0A0A!important;color-scheme:dark}#view-tabs{display:none!important}</style>' +
            "<script>(function(){" +
            "document.documentElement.classList.remove('light');window._isLightMode=false;" +
            "var t=document.getElementById('theme-toggle');if(t)t.textContent='Light mode';" +
            "var tries=0;var iv=setInterval(function(){tries++;" +
            "var btn=document.querySelector('.tab-btn[data-view=\"schema\"]');" +
            "if(btn){btn.click();clearInterval(iv);}else if(tries>40){clearInterval(iv);}" +
            "},50);})()" + closeScript;
          const blob = new Blob([html.replace("</body>", inject + "</body>")], { type: "text/html" });
          const url = URL.createObjectURL(blob);
          vizBlobRef.current = url;
          setVizSrc(url);
        } else {
          setVizError("No schema data in this brain yet.");
        }
      })
      .catch((err) => setVizError(err.message || "Failed to load schema"))
      .finally(() => setVizLoading(false));

    return () => { if (vizBlobRef.current) { URL.revokeObjectURL(vizBlobRef.current); vizBlobRef.current = null; } };
  }, [datasetId, isInitializing, cogniInstance, vizRefreshKey]);

  // ── Config helpers ────────────────────────────────────────────────────
  function getCognifyOptions() {
    const opts: { graphModel?: object; customPrompt?: string; ontologyKey?: string[] } = {};
    if (selectedModelId) {
      const model = graphModels.find((m) => m.id === selectedModelId);
      if (model) opts.graphModel = toGraphModelSchema(toCleanSchema(model.schema));
    }
    if (selectedPromptName && customPrompts[selectedPromptName]) opts.customPrompt = customPrompts[selectedPromptName];
    if (selectedOntologyKey) opts.ontologyKey = [selectedOntologyKey];
    return opts;
  }

  function handleSelectModel(id: string | null) {
    setSelectedModelId(id);
    setModelDropdownOpen(false);
    const name = id ? (graphModels.find((m) => m.id === id)?.name ?? "Unknown") : "Automatic";
    notifications.show({ title: "Graph model updated", message: `"${datasetName}" will use "${name}" on next run.`, color: "green", autoClose: 4000 });
    if (cogniInstance && datasetId) {
      assignGraphModelToDataset(cogniInstance, datasetId, id).catch((err) => {
        captureException(err, { context: "schema-page.assign-graph-model", datasetId, id });
        notifications.show({ title: "Graph model choice not saved", message: "Please try selecting it again.", color: "orange" });
      });
    }
  }

  function handleSelectPrompt(name: string | null) {
    setSelectedPromptName(name);
    setPromptDropdownOpen(false);
    notifications.show({ title: "Prompt updated", message: `"${datasetName}" will use "${name ?? "Automatic"}" on next run.`, color: "green", autoClose: 4000 });
    if (cogniInstance && datasetId) {
      assignPromptToDataset(cogniInstance, datasetId, name).catch((err) => {
        captureException(err, { context: "schema-page.assign-prompt", datasetId, name });
        notifications.show({ title: "Prompt choice not saved", message: "Please try selecting it again.", color: "orange" });
      });
    }
  }

  function handleSelectOntology(key: string | null) {
    setSelectedOntologyKey(key);
    setOntologyDropdownOpen(false);
    notifications.show({ title: "Ontology updated", message: `"${datasetName}" will use "${key ?? "Automatic"}" on next run.`, color: "green", autoClose: 4000 });
    if (cogniInstance && datasetId) {
      assignOntologyToDataset(cogniInstance, datasetId, key).catch((err) => {
        captureException(err, { context: "schema-page.assign-ontology", datasetId, key });
        notifications.show({ title: "Ontology choice not saved", message: "Please try selecting it again.", color: "orange" });
      });
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
    setVizSrc(null);
    try {
      await cognifyDataset({ id: datasetId, name: datasetName, data: [], status: "processing" }, cogniInstance, getCognifyOptions());
      trackEvent({ pageName: "Memory Schema", eventName: "dataset_reprocessed", additionalProperties: { dataset_id: datasetId } });
      notifications.show({ title: "Re-processing started", message: "The memory schema is being rebuilt.", color: "blue", autoClose: 4000 });
      setTimeout(() => setVizRefreshKey((k) => k + 1), 8000);
    } catch (err) {
      notifications.show({ title: "Re-process failed", message: err instanceof Error ? err.message : String(err), color: "red" });
    } finally {
      setReprocessing(false);
    }
  }

  // ── Dropdown panel style ─────────────────────────────────────────────
  const PANEL: React.CSSProperties = {
    position: "absolute", top: "calc(100% + 4px)", left: 0,
    background: "#1a1a1a", border: "1px solid rgba(255,255,255,0.1)",
    backdropFilter: "blur(16px)", borderRadius: 8,
    boxShadow: "0 8px 24px rgba(0,0,0,0.5)", minWidth: 220, zIndex: 50, overflow: "hidden",
  };

  if (!activeDataset) {
    return (
      <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
        <TrackPageView page="Memory Schema" />
        <div style={{ padding: "24px 32px 16px" }}>
          <h1 style={{ fontSize: 20, fontWeight: 300, color: "#EDECEA", margin: "0 0 4px", fontFamily: '"TWKLausanne", sans-serif' }}>Memory Schema</h1>
          <p style={{ fontSize: 14, color: "rgba(237,236,234,0.55)", margin: 0 }}>Define how Cognee extracts knowledge from your brain — set the extraction model, prompt, and ontology, then re-process to build memory.</p>
        </div>
        <EmptyState onGoToBrains={() => router.push("/datasets")} />
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      <TrackPageView page="Memory Schema" />

      {/* ── Header row 1: title + brain selector ── */}
      <div style={{ padding: "16px 32px 0", display: "flex", justifyContent: "space-between", alignItems: "center", flexShrink: 0 }}>
        <div>
          <h1 style={{ fontSize: 20, fontWeight: 300, color: "#EDECEA", margin: "0 0 2px", fontFamily: '"TWKLausanne", sans-serif' }}>Memory Schema</h1>
          <p style={{ fontSize: 13, color: "rgba(237,236,234,0.55)", margin: 0 }}>Define extraction rules and re-process to rebuild the memory for <strong style={{ color: "rgba(237,236,234,0.8)", fontWeight: 500 }}>{activeDataset.name}</strong></p>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <BrainSelector allowAll={false} align="right" />
          <button
            onClick={() => setVizRefreshKey((k) => k + 1)}
            style={{ ...BTN, padding: "7px 10px" }}
            title="Refresh visualization"
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.7)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 2v6h-6" /><path d="M3 12a9 9 0 0115.36-6.36L21 8" /><path d="M3 22v-6h6" /><path d="M21 12a9 9 0 01-15.36 6.36L3 16" /></svg>
          </button>
        </div>
      </div>

      {/* ── Header row 2: config toolbar ── */}
      {datasetId && (
        <div style={{ padding: "10px 32px 12px", display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", borderBottom: "1px solid rgba(255,255,255,0.08)", flexShrink: 0 }}>

          {/* Graph Model dropdown */}
          <div ref={modelDropdownRef} style={{ position: "relative" }}>
            <button onClick={() => setModelDropdownOpen((v) => !v)} style={BTN}>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.6)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="6" cy="6" r="3" /><circle cx="18" cy="6" r="3" /><circle cx="12" cy="18" r="3" /><line x1="8.5" y1="7.5" x2="10.5" y2="16" /><line x1="15.5" y1="7.5" x2="13.5" y2="16" /></svg>
              Model: {selectedModelId ? (graphModels.find((m) => m.id === selectedModelId)?.name ?? "Custom") : "Automatic"}
              <svg width="9" height="9" viewBox="0 0 10 10" fill="none"><path d="M2.5 4L5 6.5L7.5 4" stroke="rgba(237,236,234,0.4)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
            </button>
            {modelDropdownOpen && (
              <div style={PANEL}>
                <div style={{ padding: "6px 10px 4px", fontSize: 10, fontWeight: 700, color: "rgba(237,236,234,0.35)", letterSpacing: 0.4, textTransform: "uppercase" }}>Graph Model</div>
                <button onClick={() => handleSelectModel(null)} style={{ width: "100%", background: "none", border: "none", padding: "8px 12px", fontSize: 13, color: "#EDECEA", display: "flex", alignItems: "center", gap: 8, textAlign: "left", fontFamily: "inherit", cursor: "pointer" }}>
                  <span style={{ width: 16, color: "#BC9BFF" }}>{selectedModelId === null ? "✓" : ""}</span>
                  <span style={{ flex: 1 }}>Automatic</span>
                  <span style={{ fontSize: 10, color: "rgba(237,236,234,0.35)" }}>Default</span>
                </button>
                {graphModels.length > 0 && <div style={{ height: 1, background: "rgba(255,255,255,0.07)", margin: "2px 0" }} />}
                {graphModels.map((model) => (
                  <div key={model.id} style={{ display: "flex", alignItems: "center", padding: "6px 12px", gap: 8 }}>
                    <button onClick={() => handleSelectModel(model.id)} style={{ flex: 1, background: "none", border: "none", padding: 0, fontSize: 13, color: "#EDECEA", display: "flex", alignItems: "center", gap: 8, textAlign: "left", fontFamily: "inherit", cursor: "pointer" }}>
                      <span style={{ width: 16, color: "#BC9BFF" }}>{selectedModelId === model.id ? "✓" : ""}</span>
                      <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{model.name}</span>
                    </button>
                    <button onClick={() => { setModelDropdownOpen(false); router.push(`/graph-models/${model.id}`); }} style={{ background: "none", border: "none", padding: 2, opacity: 0.5, cursor: "pointer" }} title="Edit">
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.7)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7" /><path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z" /></svg>
                    </button>
                  </div>
                ))}
                <div style={{ height: 1, background: "rgba(255,255,255,0.07)", margin: "2px 0" }} />
                <button onClick={() => { setModelDropdownOpen(false); router.push("/graph-models/new"); }} style={{ width: "100%", background: "none", border: "none", padding: "8px 12px", fontSize: 13, color: "#BC9BFF", fontWeight: 500, display: "flex", alignItems: "center", gap: 8, textAlign: "left", fontFamily: "inherit", cursor: "pointer" }}>
                  <span style={{ width: 16 }}>+</span><span>Create new</span>
                </button>
              </div>
            )}
          </div>

          {/* Prompt dropdown */}
          <div ref={promptDropdownRef} style={{ position: "relative" }}>
            <button onClick={() => setPromptDropdownOpen((v) => !v)} style={BTN}>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.6)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" /></svg>
              Prompt: {selectedPromptName ?? "Automatic"}
              <svg width="9" height="9" viewBox="0 0 10 10" fill="none"><path d="M2.5 4L5 6.5L7.5 4" stroke="rgba(237,236,234,0.4)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
            </button>
            {promptDropdownOpen && (
              <div style={PANEL}>
                <div style={{ padding: "6px 10px 4px", fontSize: 10, fontWeight: 700, color: "rgba(237,236,234,0.35)", letterSpacing: 0.4, textTransform: "uppercase" }}>Custom Prompt</div>
                <button onClick={() => handleSelectPrompt(null)} style={{ width: "100%", background: "none", border: "none", padding: "8px 12px", fontSize: 13, color: "#EDECEA", display: "flex", alignItems: "center", gap: 8, textAlign: "left", fontFamily: "inherit", cursor: "pointer" }}>
                  <span style={{ width: 16, color: "#BC9BFF" }}>{selectedPromptName === null ? "✓" : ""}</span>
                  <span style={{ flex: 1 }}>Automatic</span>
                  <span style={{ fontSize: 10, color: "rgba(237,236,234,0.35)" }}>Default</span>
                </button>
                {Object.keys(customPrompts).length > 0 && <div style={{ height: 1, background: "rgba(255,255,255,0.07)", margin: "2px 0" }} />}
                {Object.entries(customPrompts).map(([name, text]) => (
                  <div key={name} style={{ display: "flex", alignItems: "center", padding: "6px 12px", gap: 8 }}>
                    <button onClick={() => handleSelectPrompt(name)} style={{ flex: 1, background: "none", border: "none", padding: 0, fontSize: 13, color: "#EDECEA", display: "flex", alignItems: "center", gap: 8, textAlign: "left", fontFamily: "inherit", cursor: "pointer" }}>
                      <span style={{ width: 16, color: "#BC9BFF" }}>{selectedPromptName === name ? "✓" : ""}</span>
                      <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{name}</span>
                    </button>
                    <button onClick={() => { setPromptDropdownOpen(false); setEditingPromptName(name); setEditingPromptText(text); setShowPromptEditor(true); }} style={{ background: "none", border: "none", padding: 2, opacity: 0.5, cursor: "pointer" }} title="Edit">
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.7)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7" /><path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z" /></svg>
                    </button>
                  </div>
                ))}
                <div style={{ height: 1, background: "rgba(255,255,255,0.07)", margin: "2px 0" }} />
                <button onClick={() => { setPromptDropdownOpen(false); setShowCreatePromptModal(true); }} style={{ width: "100%", background: "none", border: "none", padding: "8px 12px", fontSize: 13, color: "#BC9BFF", fontWeight: 500, display: "flex", alignItems: "center", gap: 8, textAlign: "left", fontFamily: "inherit", cursor: "pointer" }}>
                  <span style={{ width: 16 }}>+</span><span>Create new</span>
                </button>
              </div>
            )}
          </div>

          {/* Ontology dropdown */}
          <div ref={ontologyDropdownRef} style={{ position: "relative" }}>
            <button onClick={() => setOntologyDropdownOpen((v) => !v)} style={BTN}>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.6)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 19.5A2.5 2.5 0 016.5 17H20" /><path d="M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z" /></svg>
              Ontology: {selectedOntologyKey ?? "Automatic"}
              <svg width="9" height="9" viewBox="0 0 10 10" fill="none"><path d="M2.5 4L5 6.5L7.5 4" stroke="rgba(237,236,234,0.4)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
            </button>
            {ontologyDropdownOpen && (
              <div style={{ ...PANEL, width: 260 }}>
                <div style={{ padding: "6px 10px 4px", fontSize: 10, fontWeight: 700, color: "rgba(237,236,234,0.35)", letterSpacing: 0.4, textTransform: "uppercase" }}>Ontology</div>
                <button onClick={() => handleSelectOntology(null)} style={{ width: "100%", background: "none", border: "none", padding: "8px 12px", fontSize: 13, color: "#EDECEA", display: "flex", alignItems: "center", gap: 8, textAlign: "left", fontFamily: "inherit", cursor: "pointer" }}>
                  <span style={{ width: 16, color: "#BC9BFF" }}>{selectedOntologyKey === null ? "✓" : ""}</span>
                  <span>Automatic</span>
                </button>
                {Object.keys(ontologies).length > 0 && <div style={{ height: 1, background: "rgba(255,255,255,0.07)", margin: "2px 0" }} />}
                {Object.entries(ontologies).map(([key, meta]) => (
                  <div key={key} style={{ display: "flex", alignItems: "center", padding: "6px 12px", gap: 8 }}>
                    <button onClick={() => handleSelectOntology(key)} style={{ flex: 1, minWidth: 0, background: "none", border: "none", padding: 0, fontSize: 13, color: "#EDECEA", display: "flex", alignItems: "center", gap: 8, textAlign: "left", fontFamily: "inherit", cursor: "pointer" }}>
                      <span style={{ width: 16, color: "#BC9BFF" }}>{selectedOntologyKey === key ? "✓" : ""}</span>
                      <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={meta.filename}>{meta.filename}</span>
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
                    }} style={{ background: "none", border: "none", padding: 4, opacity: 0.6, cursor: "pointer" }} title="Delete">
                      <svg width="13" height="13" viewBox="0 0 16 16" fill="none"><path d="M3 4h10M6 4V3h4v1M5 4v8.5a.5.5 0 00.5.5h5a.5.5 0 00.5-.5V4" stroke="#EF4444" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" /></svg>
                    </button>
                  </div>
                ))}
                <div style={{ height: 1, background: "rgba(255,255,255,0.07)", margin: "2px 0" }} />
                <button onClick={() => { setOntologyDropdownOpen(false); setShowUploadOntologyModal(true); }} style={{ width: "100%", background: "none", border: "none", padding: "8px 12px", fontSize: 13, color: "#BC9BFF", fontWeight: 500, display: "flex", alignItems: "center", gap: 8, textAlign: "left", fontFamily: "inherit", cursor: "pointer" }}>
                  <span style={{ width: 16 }}>+</span><span>Upload new</span>
                </button>
              </div>
            )}
          </div>

          {/* Spacer */}
          <div style={{ flex: 1 }} />

          {/* Re-process */}
          <button
            onClick={handleReprocess}
            disabled={reprocessing}
            style={{
              ...BTN,
              background: reprocessing ? "rgba(0,0,0,0.4)" : "rgba(0,0,0,0.75)",
              border: "1px solid rgba(255,255,255,0.18)",
              opacity: reprocessing ? 0.6 : 1,
              cursor: reprocessing ? "not-allowed" : "pointer",
            }}
          >
            {reprocessing
              ? <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.7)" strokeWidth="2.5" strokeLinecap="round" style={{ animation: "spin 1s linear infinite" }}><path d="M21 12a9 9 0 11-6.219-8.56" /></svg>
              : <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.7)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="1 4 1 10 7 10" /><path d="M3.51 15a9 9 0 102.13-9.36L1 10" /></svg>}
            {reprocessing ? "Processing…" : "Re-process"}
          </button>

          <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
        </div>
      )}

      {/* ── Visualization ── */}
      <div style={{ flex: 1, position: "relative", overflow: "hidden" }}>
        {(vizLoading || (vizSrc && !vizReady)) && (
          <div style={{ position: "absolute", inset: 0, zIndex: 1 }}>
            <PageLoading name="Memory Schema" />
          </div>
        )}
        {vizError && !vizLoading && (
          <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1 }}>
            <span style={{ fontSize: 14, color: "#EF4444" }}>{vizError}</span>
          </div>
        )}
        {vizSrc && (
          <iframe
            key={`${datasetId}-${vizRefreshKey}`}
            src={vizSrc}
            onLoad={() => setTimeout(() => setVizReady(true), 250)}
            style={{ width: "100%", height: "100%", border: "none", opacity: vizReady ? 1 : 0, transition: "opacity 200ms ease" }}
            title="Memory Schema"
          />
        )}
      </div>

      {/* ── Create custom prompt modal ── */}
      {showCreatePromptModal && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.5)", backdropFilter: "blur(4px)", WebkitBackdropFilter: "blur(4px)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={() => !inferringPrompt && setShowCreatePromptModal(false)}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "rgba(15,15,15,0.92)", backdropFilter: "blur(16px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, padding: 24, width: 440, display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 16px 48px rgba(0,0,0,0.5)" }}>
            <h2 style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", margin: 0 }}>Create Custom Prompt</h2>
            {inferringPrompt ? (
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 12, padding: "24px 0" }}>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#6510F4" strokeWidth="2.5" strokeLinecap="round" style={{ animation: "spin 1s linear infinite" }}><path d="M21 12a9 9 0 11-6.219-8.56" /></svg>
                <span style={{ fontSize: 14, color: "#BC9BFF", fontWeight: 500 }}>Generating from "{graphModels.find((m) => m.id === selectedModelId)?.name ?? "graph model"}"…</span>
              </div>
            ) : (
              <>
                <p style={{ fontSize: 13, color: "rgba(237,236,234,0.55)", margin: 0, lineHeight: "20px" }}>A custom prompt guides how Cognee extracts entities and relationships.</p>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  <button onClick={handleInferPrompt} disabled={!selectedModelId} style={{ display: "flex", alignItems: "center", gap: 12, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: "14px 16px", textAlign: "left", fontFamily: "inherit", cursor: !selectedModelId ? "not-allowed" : "pointer", opacity: !selectedModelId ? 0.5 : 1 }}>
                    <div style={{ width: 36, height: 36, background: "rgba(188,155,255,0.20)", border: "1px solid rgba(188,155,255,0.35)", borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#6510F4" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><path d="M9.09 9a3 3 0 015.83 1c0 2-3 3-3 3" /><line x1="12" y1="17" x2="12.01" y2="17" /></svg>
                    </div>
                    <div>
                      <div style={{ fontSize: 14, fontWeight: 500, color: "#EDECEA" }}>Generate from graph model</div>
                      <div style={{ fontSize: 12, color: "rgba(237,236,234,0.55)" }}>{!selectedModelId ? "Select a graph model first" : `Using "${graphModels.find((m) => m.id === selectedModelId)?.name ?? "Unknown"}"`}</div>
                    </div>
                  </button>
                  <button onClick={() => { setEditingPromptName(`${datasetName} Prompt`); setEditingPromptText(DEFAULT_EXTRACTION_PROMPT); setShowCreatePromptModal(false); setShowPromptEditor(true); }} style={{ display: "flex", alignItems: "center", gap: 12, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: "14px 16px", textAlign: "left", fontFamily: "inherit", cursor: "pointer" }}>
                    <div style={{ width: 36, height: 36, background: "rgba(255,255,255,0.06)", borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.55)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" /></svg>
                    </div>
                    <div>
                      <div style={{ fontSize: 14, fontWeight: 500, color: "#EDECEA" }}>Start blank</div>
                      <div style={{ fontSize: 12, color: "rgba(237,236,234,0.55)" }}>Write your own extraction prompt</div>
                    </div>
                  </button>
                </div>
                <button onClick={() => setShowCreatePromptModal(false)} style={{ background: "none", border: "none", fontSize: 13, color: "rgba(237,236,234,0.55)", fontFamily: "inherit", padding: "4px 0", cursor: "pointer" }}>Cancel</button>
              </>
            )}
          </div>
        </div>
      )}

      {/* ── Prompt editor modal ── */}
      {showPromptEditor && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.5)", backdropFilter: "blur(4px)", WebkitBackdropFilter: "blur(4px)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={() => !savingPrompt && setShowPromptEditor(false)}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "rgba(15,15,15,0.92)", backdropFilter: "blur(16px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, padding: 24, width: 600, maxHeight: "80vh", display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 16px 48px rgba(0,0,0,0.5)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <h2 style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", margin: 0 }}>Edit Prompt</h2>
              <button onClick={() => setShowPromptEditor(false)} style={{ background: "none", border: "none", color: "rgba(237,236,234,0.35)", fontSize: 18, cursor: "pointer" }}>&#10005;</button>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <label style={{ fontSize: 12, fontWeight: 700, color: "rgba(237,236,234,0.55)", textTransform: "uppercase", letterSpacing: 0.3 }}>Name</label>
              <input type="text" value={editingPromptName} onChange={(e) => setEditingPromptName(e.target.value)} style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 8, padding: "8px 12px", fontSize: 14, fontFamily: "inherit", color: "#EDECEA", outline: "none" }} />
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4, flex: 1, minHeight: 0 }}>
              <label style={{ fontSize: 12, fontWeight: 700, color: "rgba(237,236,234,0.55)", textTransform: "uppercase", letterSpacing: 0.3 }}>Prompt</label>
              <textarea value={editingPromptText} onChange={(e) => setEditingPromptText(e.target.value)} style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 8, padding: "10px 12px", fontSize: 13, fontFamily: "inherit", color: "#EDECEA", outline: "none", resize: "vertical", minHeight: 200, maxHeight: 400, lineHeight: "20px" }} />
            </div>
            <div style={{ display: "flex", gap: 8, justifyContent: "space-between" }}>
              <button onClick={async () => {
                const name = editingPromptName.trim();
                if (!name || !cogniInstance) return;
                if (!window.confirm(`Delete prompt "${name}"?`)) return;
                try {
                  await deleteCustomPrompt(cogniInstance, name);
                  setCustomPrompts((prev) => { const next = { ...prev }; delete next[name]; return next; });
                  if (selectedPromptName === name) setSelectedPromptName(null);
                  setShowPromptEditor(false);
                } catch (err) {
                  notifications.show({ title: "Delete failed", message: err instanceof Error ? err.message : String(err), color: "red" });
                }
              }} style={{ background: "none", border: "none", padding: 4, opacity: 0.5, cursor: "pointer" }}>
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M3 4h10M6 4V3h4v1M5 4v8.5a.5.5 0 00.5.5h5a.5.5 0 00.5-.5V4" stroke="#EF4444" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" /></svg>
              </button>
              <div style={{ display: "flex", gap: 8 }}>
                <button onClick={() => setShowPromptEditor(false)} style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "rgba(237,236,234,0.7)", fontFamily: "inherit", cursor: "pointer" }}>Cancel</button>
                <button onClick={handleSavePrompt} disabled={savingPrompt} style={{ background: "#6510F4", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "#fff", fontFamily: "inherit", cursor: "pointer" }}>{savingPrompt ? "Saving…" : "Save prompt"}</button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Upload ontology modal ── */}
      {showUploadOntologyModal && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.5)", backdropFilter: "blur(4px)", WebkitBackdropFilter: "blur(4px)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={() => setShowUploadOntologyModal(false)}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "rgba(15,15,15,0.92)", backdropFilter: "blur(16px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, padding: 24, width: 440, display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 16px 48px rgba(0,0,0,0.5)" }}>
            <h2 style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", margin: 0 }}>Upload Ontology</h2>
            <p style={{ fontSize: 13, color: "rgba(237,236,234,0.55)", margin: 0, lineHeight: "20px" }}>Upload an OWL ontology file to guide how Cognee structures your memory schema.</p>
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
              submitBtn.disabled = true; submitBtn.textContent = "Uploading…";
              try {
                await uploadOntology(cogniInstance, key, file, descInput.value.trim() || undefined);
                setOntologies(await listOntologies(cogniInstance));
                setSelectedOntologyKey(key);
                setShowUploadOntologyModal(false);
                if (datasetId) {
                  assignOntologyToDataset(cogniInstance, datasetId, key).catch((err) => {
                    captureException(err, { context: "schema-page.assign-ontology-after-upload", datasetId, key });
                    notifications.show({
                      title: "Ontology uploaded, but not assigned",
                      message: `"${key}" was uploaded but couldn't be assigned to this dataset automatically. Assign it manually from the dropdown.`,
                      color: "orange",
                    });
                  });
                }
                notifications.show({ title: "Ontology uploaded", message: `"${key}" is ready.`, color: "green", autoClose: 4000 });
              } catch (err) {
                notifications.show({ title: "Upload failed", message: err instanceof Error ? err.message : String(err), color: "red" });
                submitBtn.disabled = false; submitBtn.textContent = "Upload";
              }
            }}>
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                {[{ name: "ontologyKey", label: "Key", placeholder: "e.g. biomedical-ontology", type: "text" }, { name: "description", label: "Description (optional)", placeholder: "What does this ontology define?", type: "text" }].map((f) => (
                  <div key={f.name} style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                    <label style={{ fontSize: 12, fontWeight: 700, color: "rgba(237,236,234,0.55)", textTransform: "uppercase", letterSpacing: 0.3 }}>{f.label}</label>
                    <input name={f.name} type={f.type} required={f.name === "ontologyKey"} placeholder={f.placeholder} style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 8, padding: "8px 12px", fontSize: 14, fontFamily: "inherit", color: "#EDECEA", outline: "none" }} />
                  </div>
                ))}
                <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                  <label style={{ fontSize: 12, fontWeight: 700, color: "rgba(237,236,234,0.55)", textTransform: "uppercase", letterSpacing: 0.3 }}>OWL File</label>
                  <label style={{ display: "flex", alignItems: "center", gap: 8, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 8, padding: "8px 12px", fontSize: 13, color: "rgba(237,236,234,0.55)", cursor: "pointer" }}>
                    <svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M8 1v10M4 5l4-4 4 4" stroke="#A1A1AA" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" /><path d="M1 11v2.5A1.5 1.5 0 002.5 15h11a1.5 1.5 0 001.5-1.5V11" stroke="#A1A1AA" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" /></svg>
                    <span data-file-label="true" style={{ flex: 1 }}>Choose a .owl file…</span>
                    <input name="ontologyFile" type="file" required accept=".owl" style={{ display: "none" }} onChange={(e) => { const l = e.currentTarget.parentElement?.querySelector("[data-file-label]"); if (l) l.textContent = e.currentTarget.files?.[0]?.name ?? "Choose a .owl file…"; }} />
                  </label>
                </div>
              </div>
              <div style={{ display: "flex", gap: 8, justifyContent: "flex-end", marginTop: 4 }}>
                <button type="button" onClick={() => setShowUploadOntologyModal(false)} style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "rgba(237,236,234,0.7)", fontFamily: "inherit", cursor: "pointer" }}>Cancel</button>
                <button type="submit" style={{ background: "#6510F4", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "#fff", fontFamily: "inherit", cursor: "pointer" }}>Upload</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
