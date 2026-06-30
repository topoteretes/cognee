"use client";

import { captureException, recordUploadSuccess, recordUploadFailure } from "@/utils/monitoring";
import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import { useFilter } from "@/ui/layout/FilterContext";
import getDatasetData from "@/modules/datasets/getDatasetData";
import deleteDatasetData from "@/modules/datasets/deleteDatasetData";
import deleteDataset from "@/modules/datasets/deleteDataset";
import rememberData from "@/modules/ingestion/rememberData";
import cognifyDataset from "@/modules/datasets/cognifyDataset";
import pollDatasetStatus, { type DatasetProcessingStatus } from "@/modules/datasets/pollDatasetStatus";
import { notifications } from "@mantine/notifications";
import { Tooltip } from "@mantine/core";
import { TrackPageView, trackEvent } from "@/modules/analytics";
import type { GraphModel } from "@/modules/graphModels/types";
import { toCleanSchema } from "@/modules/graphModels/types";
import { toGraphModelSchema } from "@/modules/graphModels/toGraphModelSchema";
import { loadGraphModelsConfig, syncGraphModels, assignGraphModelToDataset, assignPromptToDataset, assignOntologyToDataset, clearDatasetOutdated, findModelForDataset, findPromptForDataset, findOntologyForDataset, saveCustomPrompt, deleteCustomPrompt, type GraphModelsConfig, type CustomPromptsMap } from "@/modules/configuration/userConfiguration";
import { inferSchema, generateCustomPrompt } from "@/modules/llm/managementLlmApi";
import { listOntologies, uploadOntology, deleteOntology, type OntologyMeta } from "@/modules/ontologies/ontologyApi";
import ShareDatasetModal from "@/ui/elements/ShareDatasetModal";
import { v4 as uuid } from "uuid";

interface FileEntry {
  id: string;
  name: string;
  extension?: string;
  mimeType?: string;
  createdAt?: string;
}

// ── File type colors and labels ──

const EXT_META: Record<string, { fill: string; stroke: string; text: string; label: string }> = {
  pdf:  { fill: "#FEE2E2", stroke: "#EF4444", text: "#DC2626", label: "PDF" },
  docx: { fill: "#DBEAFE", stroke: "#3B82F6", text: "#2563EB", label: "DOC" },
  doc:  { fill: "#DBEAFE", stroke: "#3B82F6", text: "#2563EB", label: "DOC" },
  md:   { fill: "#F3F4F6", stroke: "#6B7280", text: "#374151", label: "MD" },
  txt:  { fill: "#F3F4F6", stroke: "#9CA3AF", text: "#6B7280", label: "TXT" },
  csv:  { fill: "#DCFCE7", stroke: "#22C55E", text: "#16A34A", label: "CSV" },
  json: { fill: "#FEF3C7", stroke: "#D97706", text: "#B45309", label: "JSON" },
};

function getExtMeta(name: string, ext?: string) {
  const e = (ext || name.split(".").pop() || "").toLowerCase();
  return EXT_META[e] || { fill: "#F3F4F6", stroke: "#9CA3AF", text: "#6B7280", label: e.toUpperCase().slice(0, 4) || "FILE" };
}

function getTypeName(name: string, ext?: string) {
  const e = (ext || name.split(".").pop() || "").toLowerCase();
  const names: Record<string, string> = { pdf: "PDF", docx: "DOCX", doc: "DOC", md: "Markdown", txt: "Text", csv: "CSV", json: "JSON" };
  return names[e] || e.toUpperCase();
}

function formatDate(dateStr?: string): string {
  if (!dateStr) return "—";
  const d = new Date(dateStr);
  const date = d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
  const time = d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
  return `${date}, ${time}`;
}

// Text remembered via the memory API is stored as "text_<md5>.txt" — a
// meaningless hash filename. Detect those so we can render a proper title.
function isMemoryBlobName(name: string): boolean {
  return /^text_[0-9a-f]{16,}(\.txt)?$/i.test(name);
}

// ── SVG document icon matching Paper reference ──

function FileIcon({ fill, stroke, text, label }: { fill: string; stroke: string; text: string; label: string }) {
  const fontSize = label.length > 3 ? 4.5 : label.length > 2 ? 5 : 5.5;
  return (
    <svg width="16" height="20" viewBox="0 0 16 20" fill="none" style={{ flexShrink: 0 }}>
      <path d="M10 1H3a2 2 0 00-2 2v14a2 2 0 002 2h10a2 2 0 002-2V6l-5-5z" fill={fill} stroke={stroke} strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M10 1v5h5" stroke={stroke} strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
      <text x="8" y="14.5" textAnchor="middle" fontSize={fontSize} fontWeight="700" fill={text}>{label}</text>
    </svg>
  );
}

// Default extraction prompt from cognee OSS (generate_graph_prompt.txt)
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

function TrashIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" style={{ flexShrink: 0 }}>
      <path d="M3 4h10M6 4V3h4v1M5 4v8.5a.5.5 0 00.5.5h5a.5.5 0 00.5-.5V4" stroke="#A1A1AA" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// ── Main Page ──


export default function DatasetDetailPage({ datasetId }: { datasetId: string }) {
  const router = useRouter();
  const { cogniInstance, isInitializing } = useCogniInstance();
  const { datasets: contextDatasets } = useFilter();
  const [datasetName, setDatasetName] = useState<string>(datasetId);
  const [lastSynced, setLastSynced] = useState<string | null>(null);
  const [files, setFiles] = useState<FileEntry[]>([]);
  const [loading, setLoading] = useState(true);
  // data id → session id parsed from the memory blob ("Session ID: <id>"
  // header written by the session→graph bridge), or null when none found.
  const [memorySessionIds, setMemorySessionIds] = useState<Record<string, string | null>>({});

  useEffect(() => {
    if (!cogniInstance) return;
    const targets = files.filter(f => isMemoryBlobName(f.name) && !(f.id in memorySessionIds));
    if (targets.length === 0) return;
    let cancelled = false;
    (async () => {
      const entries = await Promise.all(targets.map(async (f) => {
        try {
          const res = await cogniInstance.fetch(`/v1/datasets/${datasetId}/data/${f.id}/raw`);
          if (!res.ok) return [f.id, null] as const;
          const text = await res.text();
          const m = text.match(/^Session ID:\s*(\S+)/);
          return [f.id, m ? m[1] : null] as const;
        } catch {
          return [f.id, null] as const;
        }
      }));
      if (cancelled) return;
      setMemorySessionIds(prev => {
        const next = { ...prev };
        for (const [id, sid] of entries) next[id] = sid;
        return next;
      });
    })();
    return () => { cancelled = true; };
  }, [cogniInstance, datasetId, files, memorySessionIds]);
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [isConnectedSource, setIsConnectedSource] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [search, setSearch] = useState("");
  const [showShareModal, setShowShareModal] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Graph model selection
  const [graphModels, setGraphModels] = useState<GraphModel[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [modelDropdownOpen, setModelDropdownOpen] = useState(false);
  const modelDropdownRef = useRef<HTMLDivElement>(null);
  const [graphOutdated, setGraphOutdated] = useState(false);
  const [datasetStatus, setDatasetStatus] = useState<"ready" | "pending" | "processing" | "failed" | "outdated" | "empty">("empty");
  const statusPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [showCreateModelModal, setShowCreateModelModal] = useState(false);
  const [inferring, setInferring] = useState(false);

  // Custom prompt selection (simple dict: { name: text })
  const [customPrompts, setCustomPrompts] = useState<CustomPromptsMap>({});
  const [selectedPromptName, setSelectedPromptName] = useState<string | null>(null);
  const [promptDropdownOpen, setPromptDropdownOpen] = useState(false);
  const promptDropdownRef = useRef<HTMLDivElement>(null);
  const [showCreatePromptModal, setShowCreatePromptModal] = useState(false);
  const [inferringPrompt, setInferringPrompt] = useState(false);

  // Ontology selection
  const [ontologies, setOntologies] = useState<Record<string, OntologyMeta>>({});
  const [selectedOntologyKey, setSelectedOntologyKey] = useState<string | null>(null);
  const [ontologyDropdownOpen, setOntologyDropdownOpen] = useState(false);
  const ontologyDropdownRef = useRef<HTMLDivElement>(null);
  const [showUploadOntologyModal, setShowUploadOntologyModal] = useState(false);

  // Load graph models and assignment from backend config
  useEffect(() => {
    if (!cogniInstance || isInitializing) return;
    loadGraphModelsConfig(cogniInstance).then((cfg) => {
      setGraphModels(cfg.models);
      setCustomPrompts(cfg.customPrompts ?? {});
      const assignedModel = findModelForDataset(cfg.models, datasetId);
      setSelectedModelId(assignedModel?.id ?? null);
      const assignedPrompt = findPromptForDataset(cfg.promptAssignments ?? {}, datasetId);
      setSelectedPromptName(assignedPrompt);
      const assignedOntology = findOntologyForDataset(cfg.ontologyAssignments ?? {}, datasetId);
      setSelectedOntologyKey(assignedOntology);
      if (cfg.outdatedDatasets?.includes(datasetId)) setGraphOutdated(true);
    }).catch(() => {});
    listOntologies(cogniInstance).then(setOntologies).catch(() => {});
  }, [datasetId, cogniInstance, isInitializing]);

  // Close dropdown on outside click
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (modelDropdownRef.current && !modelDropdownRef.current.contains(e.target as Node)) {
        setModelDropdownOpen(false);
      }
    }
    if (modelDropdownOpen) {
      document.addEventListener("mousedown", handleClickOutside);
      return () => document.removeEventListener("mousedown", handleClickOutside);
    }
  }, [modelDropdownOpen]);

  // Close prompt dropdown on outside click
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (promptDropdownRef.current && !promptDropdownRef.current.contains(e.target as Node)) {
        setPromptDropdownOpen(false);
      }
    }
    if (promptDropdownOpen) {
      document.addEventListener("mousedown", handleClickOutside);
      return () => document.removeEventListener("mousedown", handleClickOutside);
    }
  }, [promptDropdownOpen]);

  // Close ontology dropdown on outside click
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (ontologyDropdownRef.current && !ontologyDropdownRef.current.contains(e.target as Node)) {
        setOntologyDropdownOpen(false);
      }
    }
    if (ontologyDropdownOpen) {
      document.addEventListener("mousedown", handleClickOutside);
      return () => document.removeEventListener("mousedown", handleClickOutside);
    }
  }, [ontologyDropdownOpen]);

  function handleSelectOntology(key: string | null) {
    const prev = selectedOntologyKey;
    setSelectedOntologyKey(key);
    setOntologyDropdownOpen(false);
    notifications.show({ title: "Ontology updated", message: `"${datasetName}" now uses "${key ?? "Automatic"}".`, color: "green", autoClose: 4000 });
    if (prev !== key && files.length > 0) {
      setGraphOutdated(true);
    }
    if (cogniInstance) {
      assignOntologyToDataset(cogniInstance, datasetId, key).catch((err) =>
        console.error("Failed to save ontology assignment:", err)
      );
    }
  }

  function handleSelectPrompt(name: string | null) {
    const prev = selectedPromptName;
    setSelectedPromptName(name);
    setPromptDropdownOpen(false);
    notifications.show({ title: "Prompt updated", message: `"${datasetName}" now uses "${name ?? "Automatic"}".`, color: "green", autoClose: 4000 });
    if (prev !== name && files.length > 0) {
      setGraphOutdated(true);
    }
    if (cogniInstance) {
      assignPromptToDataset(cogniInstance, datasetId, name).catch((err) =>
        console.error("Failed to save prompt assignment:", err)
      );
    }
  }

  // Prompt editor state
  const [editingPromptText, setEditingPromptText] = useState("");
  const [editingPromptName, setEditingPromptName] = useState("");
  const [showPromptEditor, setShowPromptEditor] = useState(false);
  const [savingPrompt, setSavingPrompt] = useState(false);

  async function handleInferPrompt() {
    if (!selectedModelId) return;
    setInferringPrompt(true);
    try {
      const model = graphModels.find((m) => m.id === selectedModelId);
      if (model) {
        const cleanSchema = toCleanSchema(model.schema);
        const graphModelSchema = toGraphModelSchema(cleanSchema);
        const result = await generateCustomPrompt(cogniInstance!, graphModelSchema);
        setEditingPromptName(`${datasetName} Prompt`);
        setEditingPromptText(result.customPrompt);
        setShowCreatePromptModal(false);
        setShowPromptEditor(true);
        notifications.show({ title: "Prompt generated", message: "Review and edit the prompt below.", color: "green", autoClose: 4000 });
      }
    } catch (err) {
      console.error("Generate prompt failed:", err);
      notifications.show({ title: "Generation failed", message: err instanceof Error ? err.message : String(err), color: "red" });
    } finally {
      setInferringPrompt(false);
    }
  }

  function handleStartBlankPrompt() {
    setEditingPromptName(`${datasetName} Prompt`);
    setEditingPromptText(DEFAULT_EXTRACTION_PROMPT);
    setShowCreatePromptModal(false);
    setShowPromptEditor(true);
  }

  async function handleSavePrompt() {
    if (!cogniInstance) return;
    const name = editingPromptName.trim();
    if (!name) {
      notifications.show({ title: "Name required", message: "Please enter a prompt name.", color: "yellow" });
      return;
    }
    setSavingPrompt(true);
    try {
      await saveCustomPrompt(cogniInstance, name, editingPromptText);
      setCustomPrompts((prev) => ({ ...prev, [name]: editingPromptText }));
      setSelectedPromptName(name);
      setShowPromptEditor(false);
      notifications.show({ title: "Prompt saved", message: `"${name}" saved.`, color: "green", autoClose: 4000 });
    } catch (err) {
      console.error("Failed to save prompt:", err);
      notifications.show({ title: "Failed", message: "Could not save prompt.", color: "red" });
    } finally {
      setSavingPrompt(false);
    }
  }

  function handleSelectModel(modelId: string | null) {
    const prevModelId = selectedModelId;
    setSelectedModelId(modelId);
    setModelDropdownOpen(false);
    const modelName = modelId ? graphModels.find((m) => m.id === modelId)?.name ?? "Unknown" : "Automatic";
    trackEvent({ pageName: "Dataset Detail", eventName: "graph_model_selected", additionalProperties: { dataset_id: datasetId, model_id: modelId ?? "automatic" } });
    notifications.show({ title: "Graph model updated", message: `"${datasetName}" now uses "${modelName}".`, color: "green", autoClose: 4000 });
    // Mark graph as outdated if the model actually changed and there are files
    if (prevModelId !== modelId && files.length > 0) {
      setGraphOutdated(true);
    }
    if (cogniInstance) {
      assignGraphModelToDataset(cogniInstance, datasetId, modelId).catch((err) =>
        console.error("Failed to save graph model assignment:", err)
      );
    }
  }

  function getCognifyOptions(): { graphModel?: object; customPrompt?: string; ontologyKey?: string[] } {
    const opts: { graphModel?: object; customPrompt?: string; ontologyKey?: string[] } = {};
    if (selectedModelId) {
      const model = graphModels.find((m) => m.id === selectedModelId);
      if (model) {
        const cleanSchema = toCleanSchema(model.schema);
        opts.graphModel = toGraphModelSchema(cleanSchema);
      }
    }
    if (selectedPromptName && customPrompts[selectedPromptName]) {
      opts.customPrompt = customPrompts[selectedPromptName];
    }
    if (selectedOntologyKey) {
      opts.ontologyKey = [selectedOntologyKey];
    }
    return opts;
  }

  async function handleCreateModel(infer: boolean) {
    if (!cogniInstance) return;
    const newModelId = uuid();
    const now = new Date().toISOString();
    let modelSchema = { options: {}, entities: [] } as GraphModel["schema"];

    if (infer && files.length > 0) {
      setInferring(true);
      try {
        const sampleText = `Dataset: ${datasetName}. Files: ${files.map(f => f.name).join(", ")}`;
        const result = await inferSchema(cogniInstance, sampleText);
        if (result.graphSchema) {
            // Convert the JSON Schema from the LLM into our internal format
            // Store as-is for now — the editor can display it
            modelSchema = {
              options: {},
              entities: Object.entries(result.graphSchema.$defs || {})
                .filter(([key]) => !key.endsWith("Type"))
                .map(([name, def]: [string, any]) => ({
                  _id: uuid(),
                  name: def.title || name,
                  description: def.description || "",
                  fields: Object.entries(def.properties || {})
                    .filter(([fieldName]) => fieldName !== "is_type" && fieldName !== "metadata")
                    .map(([fieldName, fieldDef]: [string, any]) => {
                      if (fieldDef.$ref) {
                        const target = fieldDef.$ref.replace("#/$defs/", "");
                        return { _id: uuid(), name: fieldName, kind: "relation" as const, relation: { targetEntityName: target, cardinality: "one" as const } };
                      }
                      if (fieldDef.type === "array" && fieldDef.items?.$ref) {
                        const target = fieldDef.items.$ref.replace("#/$defs/", "");
                        return { _id: uuid(), name: fieldName, kind: "relation" as const, relation: { targetEntityName: target, cardinality: "many" as const } };
                      }
                      const primitiveType = (fieldDef.type === "number" || fieldDef.type === "integer") ? "number" : fieldDef.type === "boolean" ? "boolean" : "string";
                      return { _id: uuid(), name: fieldName, kind: "primitive" as const, primitiveType: primitiveType as "string" | "number" | "boolean" | "date", required: (def.required || []).includes(fieldName) };
                    }),
                  indexFields: [],
                })),
            };
            notifications.show({ title: "Schema inferred", message: `Detected ${modelSchema.entities.length} entity types from your data.`, color: "green", autoClose: 4000 });
          }
      } catch (err) {
        console.error("Infer schema failed:", err);
        notifications.show({ title: "Inference failed", message: "Could not infer schema. Starting with blank model.", color: "yellow", autoClose: 4000 });
      } finally {
        setInferring(false);
      }
    }

    // Save the new model to backend config (read-modify-write)
    try {
      const cfg = await loadGraphModelsConfig(cogniInstance);
      const newModel: GraphModel = {
        id: newModelId,
        name: `${datasetName} Schema`,
        schema: modelSchema,
        createdAt: now,
        updatedAt: now,
        status: "draft",
        assignedDatasets: [datasetId],
      };
      await syncGraphModels(cogniInstance, [...cfg.models, newModel]);
      // Store in sessionStorage so the editor can load immediately
      sessionStorage.setItem(`graph-model-${newModelId}`, JSON.stringify(newModel));
      setShowCreateModelModal(false);
      router.push(`/graph-models/${newModelId}`);
    } catch (err) {
      console.error("Failed to create model:", err);
      notifications.show({ title: "Failed", message: "Could not create graph model.", color: "red" });
    }
  }

  useEffect(() => {
    if (!cogniInstance || isInitializing) return;
    // Resolve dataset name from FilterContext
    const ds = contextDatasets.find((d) => d.id === datasetId) as { id: string; name: string; updatedAt?: string; connection_id?: string } | undefined;
    if (ds) {
      setDatasetName(ds.name);
      if (ds.updatedAt) setLastSynced(ds.updatedAt);
      if (ds.connection_id) setIsConnectedSource(true);
    }
    loadFiles();
  }, [cogniInstance, isInitializing]);

  // Fetch dataset status once on load; only poll while processing
  useEffect(() => {
    if (!cogniInstance || isInitializing) return;
    async function fetchStatus() {
      try {
        const resp = await cogniInstance!.fetch(`/v1/datasets/status?dataset=${datasetId}`);
        if (!resp.ok) return;
        const data: Record<string, DatasetProcessingStatus> = await resp.json();
        const raw = data[datasetId] ?? Object.values(data)[0];
        if (!raw) {
          if (files.length === 0) setDatasetStatus("empty");
          else if (graphOutdated) setDatasetStatus("outdated");
          else setDatasetStatus("ready");
          return;
        }
        if (raw === "DATASET_PROCESSING_COMPLETED") {
          setDatasetStatus(graphOutdated ? "outdated" : "ready");
          setProcessing(false);
        } else if (raw === "DATASET_PROCESSING_ERRORED") {
          setDatasetStatus("failed");
          setProcessing(false);
        } else if (raw === "DATASET_PROCESSING_STARTED" || raw === "DATASET_PROCESSING_INITIATED") {
          setDatasetStatus("processing");
          setProcessing(true);
        }
      } catch {}
    }
    fetchStatus();
    // Only poll while actively processing
    if (processing) {
      statusPollRef.current = setInterval(fetchStatus, 5000);
      return () => { if (statusPollRef.current) clearInterval(statusPollRef.current); };
    }
  }, [cogniInstance, isInitializing, datasetId, graphOutdated, files.length, processing]);

  async function loadFiles() {
    if (!cogniInstance) return;
    try {
      const data = await getDatasetData(datasetId, cogniInstance);
      setFiles(Array.isArray(data) ? data.map((d: FileEntry & { rawDataLocation?: string; originalExtension?: string; original_extension?: string; originalMimeType?: string; original_mime_type?: string }) => ({
        id: d.id,
        name: d.name || d.rawDataLocation?.split("/").pop() || d.id,
        extension: d.originalExtension || d.original_extension || d.extension,
        mimeType: d.originalMimeType || d.original_mime_type || d.mimeType,
        createdAt: d.createdAt,
      })) : []);
    } catch {
      setFiles([]);
    } finally {
      setLoading(false);
    }
  }

  async function handleUpload(newFiles: FileList | File[]) {
    if (!cogniInstance) return;
    const filesArray = Array.from(newFiles);
    const totalBytes = filesArray.reduce((sum, f) => sum + f.size, 0);
    const fileTypes = filesArray.map((f) => f.type || "unknown");
    const uploadStartedAt = Date.now();

    setUploading(true);
    trackEvent({
      pageName: "Dataset Detail",
      eventName: "dataset_upload_started",
      additionalProperties: {
        dataset_id: datasetId,
        file_count: String(filesArray.length),
        total_bytes: String(totalBytes),
        file_types: fileTypes.join(","),
      },
    });

    try {
      // run_in_background=false — blocks until ingestion and graph building complete.
      // No polling needed; when this resolves the docs are already saved.
      await rememberData({ id: datasetId }, filesArray, cogniInstance, getCognifyOptions());
      await loadFiles();
      setUploading(false);
      setLastSynced(new Date().toISOString());

      trackEvent({
        pageName: "Dataset Detail",
        eventName: "dataset_files_uploaded",
        additionalProperties: {
          dataset_id: datasetId,
          file_count: String(filesArray.length),
          total_bytes: String(totalBytes),
          duration_ms: String(Date.now() - uploadStartedAt),
        },
      });
      recordUploadSuccess(Date.now() - uploadStartedAt, totalBytes, filesArray.length);
    } catch (err) {
      setUploading(false);

      const durationMs = Date.now() - uploadStartedAt;
      const errorName = err instanceof Error ? err.name : "UnknownError";
      const errorMessage = err instanceof Error ? err.message : String(err);

      recordUploadFailure(errorName, durationMs);
      trackEvent({
        pageName: "Dataset Detail",
        eventName: "dataset_upload_failed",
        additionalProperties: {
          dataset_id: datasetId,
          file_count: String(filesArray.length),
          total_bytes: String(totalBytes),
          file_types: fileTypes.join(","),
          duration_ms: String(durationMs),
          error_name: errorName,
          error_message: errorMessage,
        },
      });

      if (errorName === "UploadTimeoutError") {
        notifications.show({
          title: "Upload timed out",
          message: "The file took too long to process. Please try again with a smaller file.",
          color: "red",
          autoClose: false,
        });
      } else {
        captureException(err, { datasetId, fileCount: filesArray.length, totalBytes, durationMs });
        notifications.show({
          title: "Upload failed",
          message: errorMessage,
          color: "red",
        });
      }
    }
  }

  async function handleDelete(fileId: string) {
    if (!cogniInstance) return;
    try {
      await deleteDatasetData(datasetId, fileId, cogniInstance);
      const deletedFile = files.find((f) => f.id === fileId);
      trackEvent({ pageName: "Dataset Detail", eventName: "dataset_file_deleted", additionalProperties: { dataset_id: datasetId, file_name: deletedFile?.name ?? fileId } });
      setFiles((prev) => prev.filter((f) => f.id !== fileId));
    } catch (err) {
      console.error("Delete failed:", err);
    }
  }

  async function handleSync() {
    if (!cogniInstance) return;
    setSyncing(true);
    try {
      await cognifyDataset({ id: datasetId, name: datasetName, data: [], status: "processing" }, cogniInstance, getCognifyOptions());
      const finalStatus = await pollDatasetStatus(datasetId, cogniInstance, { intervalMs: 5000 });
      trackEvent({ pageName: "Dataset Detail", eventName: "dataset_synced", additionalProperties: { dataset_id: datasetId, status: finalStatus } });
      setLastSynced(new Date().toISOString());
    } catch (err) {
      console.error("Sync failed:", err);
    } finally {
      setSyncing(false);
    }
  }

  async function handleDeleteDataset() {
    if (!cogniInstance) return;
    setDeleting(true);
    try {
      await deleteDataset(datasetId, cogniInstance);
      trackEvent({ pageName: "Dataset Detail", eventName: "dataset_deleted", additionalProperties: { dataset_id: datasetId } });
      window.location.href = "/datasets";
    } catch (err) {
      console.error("Delete brain failed:", err);
      setDeleting(false);
      setShowDeleteConfirm(false);
    }
  }

  const filtered = search ? files.filter((f) => f.name.toLowerCase().includes(search.toLowerCase())) : files;

  if (loading || isInitializing) {
    return <><TrackPageView page="Dataset Detail" additionalProperties={{ dataset_id: datasetId }} /><div style={{ padding: 32, display: "flex", alignItems: "center", justifyContent: "center", height: "100%" }}><span style={{ fontSize: 14, color: "rgba(237,236,234,0.55)" }}>Loading files...</span></div></>;
  }

  return (
    <div
      style={{ padding: 32, display: "flex", flexDirection: "column", gap: 24, height: "100%" }}
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
      onDragLeave={(e) => { if (e.currentTarget === e.target || !e.currentTarget.contains(e.relatedTarget as Node)) setIsDragging(false); }}
      onDrop={(e) => { e.preventDefault(); setIsDragging(false); if (e.dataTransfer.files.length) handleUpload(e.dataTransfer.files); }}
    >
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".pdf,.csv,.txt,.md,.json,.docx"
        className="hidden"
        onChange={(e) => { if (e.target.files) handleUpload(e.target.files); e.target.value = ""; }}
      />

      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontSize: 22, fontWeight: 700, color: "#EDECEA" }}>{datasetName}</span>
            {datasetName === "default_dataset" && (
              <span style={{ background: "rgba(188,155,255,0.20)", color: "#BC9BFF", border: "1px solid rgba(188,155,255,0.35)", fontSize: 11, fontWeight: 500, padding: "2px 8px", borderRadius: 4 }}>Default</span>
            )}
          </div>
          <span style={{ fontSize: 14, color: "rgba(237,236,234,0.55)", display: "flex", alignItems: "center", gap: 6 }}>
            {files.length} documents
            {datasetStatus === "processing" || processing ? (
              <span style={{ display: "inline-flex", alignItems: "center", gap: 4, color: "#6510F4", fontWeight: 500 }}>
                · <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#6510F4" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ animation: "spin 1s linear infinite" }}><path d="M21 12a9 9 0 11-6.219-8.56" /></svg>
                Processing
              </span>
            ) : datasetStatus === "failed" ? (
              <span style={{ color: "#EF4444", fontWeight: 500 }}>· Failed</span>
            ) : graphOutdated || datasetStatus === "outdated" ? (
              <span style={{ display: "inline-flex", alignItems: "center", gap: 4, color: "#D97706", fontWeight: 500 }}>
                · <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#F59E0B", display: "inline-block" }} /> Outdated
              </span>
            ) : datasetStatus === "ready" ? (
              <span style={{ display: "inline-flex", alignItems: "center", gap: 4, color: "#22C55E", fontWeight: 500 }}>
                · <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#22C55E", display: "inline-block" }} /> Ready
              </span>
            ) : files.length === 0 ? (
              <span style={{ color: "rgba(237,236,234,0.35)" }}>· Empty</span>
            ) : null}
          </span>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {isConnectedSource && (
            <button
              onClick={handleSync}
              disabled={syncing}
              className="cursor-pointer hover:bg-white/10"
              style={{ background: "rgba(255,255,255,0.06)", backdropFilter: "blur(12px)", color: "rgba(237,236,234,0.7)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 6, padding: "8px 16px", fontSize: 13, fontWeight: 500, display: "flex", alignItems: "center", gap: 6 }}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.7)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={syncing ? { animation: "spin 1s linear infinite" } : undefined}><path d="M21 2v6h-6" /><path d="M3 12a9 9 0 0115.36-6.36L21 8" /><path d="M3 22v-6h6" /><path d="M21 12a9 9 0 01-15.36 6.36L3 16" /></svg>
              {syncing ? "Syncing..." : "Sync"}
            </button>
          )}
          {datasetName !== "default_dataset" && (
            <button
              onClick={() => setShowDeleteConfirm(true)}
              className="cursor-pointer hover:bg-red-500/10"
              style={{ background: "rgba(255,255,255,0.06)", backdropFilter: "blur(12px)", color: "#EF4444", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 6, padding: "8px 16px", fontSize: 13, fontWeight: 500, display: "flex", alignItems: "center", gap: 6 }}
            >
              <TrashIcon />
              Delete
            </button>
          )}
          <button
            onClick={() => setShowShareModal(true)}
            className="cursor-pointer hover:bg-white/10"
            style={{ background: "rgba(255,255,255,0.06)", backdropFilter: "blur(12px)", color: "rgba(237,236,234,0.7)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 6, padding: "8px 16px", fontSize: 13, fontWeight: 500, display: "flex", alignItems: "center", gap: 6 }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.7)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="18" cy="5" r="3" /><circle cx="6" cy="12" r="3" /><circle cx="18" cy="19" r="3" /><line x1="8.59" y1="13.51" x2="15.42" y2="17.49" /><line x1="15.41" y1="6.51" x2="8.59" y2="10.49" /></svg>
            Share
          </button>
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            className="cursor-pointer hover:bg-[#5A0ED6]"
            style={{ background: "#6510F4", color: "#fff", border: "none", borderRadius: 6, padding: "8px 16px", fontSize: 13, fontWeight: 500, display: "flex", alignItems: "center", gap: 6 }}
          >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" /></svg>
          {uploading ? "Uploading..." : "Upload files"}
          </button>
        </div>
      </div>

      {/* Share modal */}
      {showShareModal && (
        <ShareDatasetModal
          datasetId={datasetId}
          datasetName={datasetName}
          pageName="Dataset Detail"
          onClose={() => setShowShareModal(false)}
        />
      )}

      {/* Delete confirmation modal */}
      {showDeleteConfirm && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.5)", backdropFilter: "blur(4px)", WebkitBackdropFilter: "blur(4px)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={() => setShowDeleteConfirm(false)}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "rgba(15,15,15,0.92)", backdropFilter: "blur(16px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, padding: 24, width: 420, display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 16px 48px rgba(0,0,0,0.12)" }}>
            <h2 style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", margin: 0 }}>Delete brain</h2>
            <p style={{ fontSize: 13, color: "rgba(237,236,234,0.55)", margin: 0 }}>
              Are you sure you want to delete <strong>{datasetName}</strong>? This will permanently remove the dataset and all its files. This action cannot be undone.
            </p>
            <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
              <button onClick={() => setShowDeleteConfirm(false)} className="cursor-pointer" style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "rgba(237,236,234,0.7)", fontFamily: "inherit" }}>Cancel</button>
              <button onClick={handleDeleteDataset} disabled={deleting} className="cursor-pointer" style={{ background: "#EF4444", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "#fff", fontFamily: "inherit" }}>
                {deleting ? "Deleting..." : "Delete"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Create graph model modal */}
      {showCreateModelModal && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.5)", backdropFilter: "blur(4px)", WebkitBackdropFilter: "blur(4px)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={() => !inferring && setShowCreateModelModal(false)}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "rgba(15,15,15,0.92)", backdropFilter: "blur(16px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, padding: 24, width: 440, display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 16px 48px rgba(0,0,0,0.12)" }}>
            <h2 style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", margin: 0 }}>Create Graph Model</h2>
            {inferring ? (
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 12, padding: "24px 0" }}>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#6510F4" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ animation: "spin 1s linear infinite" }}><path d="M21 12a9 9 0 11-6.219-8.56" /></svg>
                <span style={{ fontSize: 14, color: "#6510F4", fontWeight: 500 }}>Inferring schema from your data...</span>
                <span style={{ fontSize: 12, color: "rgba(237,236,234,0.55)" }}>This may take a moment</span>
              </div>
            ) : (
              <>
                <p style={{ fontSize: 13, color: "rgba(237,236,234,0.55)", margin: 0, lineHeight: "20px" }}>
                  Would you like to infer a graph model from your existing data? Cognee will analyze your files and suggest entity types and relationships.
                </p>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  <button
                    onClick={() => handleCreateModel(true)}
                    disabled={files.length === 0}
                    className="cursor-pointer hover:bg-white/10"
                    style={{ display: "flex", alignItems: "center", gap: 12, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: "14px 16px", textAlign: "left", fontFamily: "inherit", opacity: files.length === 0 ? 0.5 : 1 }}
                  >
                    <div style={{ width: 36, height: 36, background: "rgba(188,155,255,0.20)", border: "1px solid rgba(188,155,255,0.35)", borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#6510F4" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><path d="M9.09 9a3 3 0 015.83 1c0 2-3 3-3 3" /><line x1="12" y1="17" x2="12.01" y2="17" /></svg>
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                      <span style={{ fontSize: 14, fontWeight: 500, color: "#EDECEA" }}>Infer from data</span>
                      <span style={{ fontSize: 12, color: "rgba(237,236,234,0.55)" }}>
                        {files.length === 0 ? "No files in this brain yet" : `Analyze ${files.length} file${files.length !== 1 ? "s" : ""} to suggest a schema`}
                      </span>
                    </div>
                  </button>
                  <button
                    onClick={() => handleCreateModel(false)}
                    className="cursor-pointer hover:bg-white/10"
                    style={{ display: "flex", alignItems: "center", gap: 12, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: "14px 16px", textAlign: "left", fontFamily: "inherit" }}
                  >
                    <div style={{ width: 36, height: 36, background: "rgba(255,255,255,0.06)", borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.55)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" /></svg>
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                      <span style={{ fontSize: 14, fontWeight: 500, color: "#EDECEA" }}>Start blank</span>
                      <span style={{ fontSize: 12, color: "rgba(237,236,234,0.55)" }}>Define entity types and relationships manually</span>
                    </div>
                  </button>
                </div>
                <button
                  onClick={() => setShowCreateModelModal(false)}
                  className="cursor-pointer"
                  style={{ background: "none", border: "none", fontSize: 13, color: "rgba(237,236,234,0.55)", fontFamily: "inherit", padding: "4px 0" }}
                >
                  Cancel
                </button>
              </>
            )}
          </div>
        </div>
      )}

      {/* Create custom prompt modal */}
      {showCreatePromptModal && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.5)", backdropFilter: "blur(4px)", WebkitBackdropFilter: "blur(4px)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={() => !inferringPrompt && setShowCreatePromptModal(false)}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "rgba(15,15,15,0.92)", backdropFilter: "blur(16px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, padding: 24, width: 440, display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 16px 48px rgba(0,0,0,0.12)" }}>
            <h2 style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", margin: 0 }}>Create Custom Prompt</h2>
            {inferringPrompt ? (
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 12, padding: "24px 0" }}>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#6510F4" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ animation: "spin 1s linear infinite" }}><path d="M21 12a9 9 0 11-6.219-8.56" /></svg>
                <span style={{ fontSize: 14, color: "#6510F4", fontWeight: 500 }}>Generating prompt from &ldquo;{graphModels.find((m) => m.id === selectedModelId)?.name ?? "graph model"}&rdquo;...</span>
                <span style={{ fontSize: 12, color: "rgba(237,236,234,0.55)" }}>This may take a moment</span>
              </div>
            ) : (
              <>
                <p style={{ fontSize: 13, color: "rgba(237,236,234,0.55)", margin: 0, lineHeight: "20px" }}>
                  A custom prompt guides how Cognee extracts entities and relationships from your data.
                </p>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  <button
                    onClick={handleInferPrompt}
                    disabled={!selectedModelId}
                    className="cursor-pointer hover:bg-white/10"
                    style={{ display: "flex", alignItems: "center", gap: 12, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: "14px 16px", textAlign: "left", fontFamily: "inherit", opacity: !selectedModelId ? 0.5 : 1 }}
                  >
                    <div style={{ width: 36, height: 36, background: "rgba(188,155,255,0.20)", border: "1px solid rgba(188,155,255,0.35)", borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#6510F4" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><path d="M9.09 9a3 3 0 015.83 1c0 2-3 3-3 3" /><line x1="12" y1="17" x2="12.01" y2="17" /></svg>
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                      <span style={{ fontSize: 14, fontWeight: 500, color: "#EDECEA" }}>Generate from graph model</span>
                      <span style={{ fontSize: 12, color: "rgba(237,236,234,0.55)" }}>
                        {!selectedModelId
                          ? "Select a graph model first"
                          : `Using "${graphModels.find((m) => m.id === selectedModelId)?.name ?? "Unknown"}"`}
                      </span>
                    </div>
                  </button>
                  <button
                    onClick={handleStartBlankPrompt}
                    className="cursor-pointer hover:bg-white/10"
                    style={{ display: "flex", alignItems: "center", gap: 12, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: "14px 16px", textAlign: "left", fontFamily: "inherit" }}
                  >
                    <div style={{ width: 36, height: 36, background: "rgba(255,255,255,0.06)", borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.55)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" /></svg>
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                      <span style={{ fontSize: 14, fontWeight: 500, color: "#EDECEA" }}>Start blank</span>
                      <span style={{ fontSize: 12, color: "rgba(237,236,234,0.55)" }}>Write your own extraction prompt</span>
                    </div>
                  </button>
                </div>
                <button
                  onClick={() => setShowCreatePromptModal(false)}
                  className="cursor-pointer"
                  style={{ background: "none", border: "none", fontSize: 13, color: "rgba(237,236,234,0.55)", fontFamily: "inherit", padding: "4px 0" }}
                >
                  Cancel
                </button>
              </>
            )}
          </div>
        </div>
      )}

      {/* Prompt editor modal */}
      {showPromptEditor && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.5)", backdropFilter: "blur(4px)", WebkitBackdropFilter: "blur(4px)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={() => !savingPrompt && setShowPromptEditor(false)}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "rgba(15,15,15,0.92)", backdropFilter: "blur(16px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, padding: 24, width: 600, maxHeight: "80vh", display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 16px 48px rgba(0,0,0,0.12)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <h2 style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", margin: 0 }}>Edit Prompt</h2>
              <button onClick={() => setShowPromptEditor(false)} className="cursor-pointer" style={{ background: "none", border: "none", color: "rgba(237,236,234,0.35)", fontSize: 18 }}>&#10005;</button>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <label style={{ fontSize: 12, fontWeight: 700, color: "rgba(237,236,234,0.55)", textTransform: "uppercase", letterSpacing: 0.3 }}>Name</label>
              <input
                type="text"
                value={editingPromptName}
                onChange={(e) => setEditingPromptName(e.target.value)}
                placeholder="Prompt name"
                style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 8, padding: "8px 12px", fontSize: 14, fontFamily: "inherit", color: "#EDECEA", outline: "none" }}
              />
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 4, flex: 1, minHeight: 0 }}>
              <label style={{ fontSize: 12, fontWeight: 700, color: "rgba(237,236,234,0.55)", textTransform: "uppercase", letterSpacing: 0.3 }}>Prompt</label>
              <textarea
                value={editingPromptText}
                onChange={(e) => setEditingPromptText(e.target.value)}
                placeholder="Write your extraction prompt here. This prompt will be used by Cognee when extracting entities and relationships from your data.&#10;&#10;Example: Extract all companies, people, and their relationships from the text. Focus on ownership, employment, and partnership relations."
                style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 8, padding: "10px 12px", fontSize: 13, color: "#EDECEA", outline: "none", resize: "vertical", minHeight: 200, maxHeight: 400, lineHeight: "20px" }}
              />
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
                style={{ background: "none", border: "none", padding: 4, opacity: 0.5, transition: "opacity 150ms", display: "flex", alignItems: "center", justifyContent: "center" }}
                title="Delete prompt"
              >
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M3 4h10M6 4V3h4v1M5 4v8.5a.5.5 0 00.5.5h5a.5.5 0 00.5-.5V4" stroke="#EF4444" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" /></svg>
              </button>
              <div style={{ display: "flex", gap: 8 }}>
                <button
                  onClick={() => setShowPromptEditor(false)}
                  className="cursor-pointer"
                  style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "rgba(237,236,234,0.7)", fontFamily: "inherit" }}
                >
                  Cancel
                </button>
                <button
                  onClick={handleSavePrompt}
                  disabled={savingPrompt}
                  className="cursor-pointer"
                  style={{ background: "#6510F4", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "#fff", fontFamily: "inherit" }}
                >
                  {savingPrompt ? "Saving..." : "Save prompt"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Search */}
      <div style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 8, display: "flex", alignItems: "center", gap: 10, height: 40, paddingInline: 14 }}>
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><circle cx="7" cy="7" r="4.5" stroke="rgba(237,236,234,0.35)" strokeWidth="1.5" /><path d="M10.5 10.5L14 14" stroke="rgba(237,236,234,0.35)" strokeWidth="1.5" strokeLinecap="round" /></svg>
        <input
          type="text" value={search} onChange={(e) => setSearch(e.target.value)}
          placeholder="Search files..."
          style={{ flex: 1, border: "none", outline: "none", fontSize: 14, color: "#EDECEA", background: "transparent", fontFamily: "inherit" }}
        />
        {search && <button onClick={() => setSearch("")} className="cursor-pointer" style={{ background: "none", border: "none", color: "rgba(237,236,234,0.35)", fontSize: 14 }}>&#10005;</button>}
      </div>

      {/* Memory customization */}
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <span style={{ fontSize: 12, fontWeight: 700, color: "rgba(237,236,234,0.35)", letterSpacing: 0.3, textTransform: "uppercase" }}>Memory customization</span>
        <div style={{ display: "flex", gap: 16 }}>
          {/* Graph model dropdown */}
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <span style={{ fontSize: 11, fontWeight: 700, color: "rgba(237,236,234,0.55)", letterSpacing: 0.2, display: "flex", alignItems: "center", gap: 4 }}>Graph Model <Tooltip label="Define entity types and relationships to control how Cognee structures your knowledge graph." withArrow multiline w={240} position="top"><svg width="12" height="12" viewBox="0 0 16 16" fill="none" style={{ flexShrink: 0 }}><circle cx="8" cy="8" r="7" stroke="#A1A1AA" strokeWidth="1.5" /><text x="8" y="12" textAnchor="middle" fontSize="10" fontWeight="700" fill="#A1A1AA">i</text></svg></Tooltip></span>
          <div ref={modelDropdownRef} style={{ position: "relative" }}>
            <button
              onClick={() => setModelDropdownOpen((v) => !v)}
              className="cursor-pointer hover:bg-white/10"
              style={{ background: "rgba(255,255,255,0.06)", backdropFilter: "blur(12px)", color: "rgba(237,236,234,0.7)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 6, padding: "8px 16px", fontSize: 13, fontWeight: 500, display: "flex", alignItems: "center", gap: 6, whiteSpace: "nowrap" }}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.7)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="6" cy="6" r="3" /><circle cx="18" cy="6" r="3" /><circle cx="12" cy="18" r="3" /><line x1="8.5" y1="7.5" x2="10.5" y2="16" /><line x1="15.5" y1="7.5" x2="13.5" y2="16" /></svg>
              {selectedModelId ? (graphModels.find((m) => m.id === selectedModelId)?.name ?? "Automatic") : "Automatic"}
              <svg width="10" height="10" viewBox="0 0 10 10" fill="none"><path d="M2.5 4L5 6.5L7.5 4" stroke="rgba(237,236,234,0.55)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
            </button>
            {modelDropdownOpen && (
              <div style={{ position: "absolute", top: "calc(100% + 4px)", left: 0, background: "#1a1a1a", border: "1px solid rgba(255,255,255,0.08)", backdropFilter: "blur(16px)", borderRadius: 8, boxShadow: "0 8px 24px rgba(0,0,0,0.4)", minWidth: 220, zIndex: 50, overflow: "hidden" }}>
                <div style={{ padding: "6px 10px 4px", fontSize: 11, fontWeight: 700, color: "rgba(237,236,234,0.35)", letterSpacing: 0.3, textTransform: "uppercase" }}>Graph Model</div>
                <button onClick={() => handleSelectModel(null)} className="cursor-pointer hover:bg-white/10" style={{ width: "100%", background: "none", border: "none", padding: "8px 12px", fontSize: 13, color: "#EDECEA", display: "flex", alignItems: "center", gap: 8, textAlign: "left", fontFamily: "inherit" }}>
                  <span style={{ width: 16, textAlign: "center", fontSize: 13, color: "#BC9BFF" }}>{selectedModelId === null ? "✓" : ""}</span>
                  <span style={{ flex: 1 }}>Automatic</span>
                  <span style={{ fontSize: 11, color: "rgba(237,236,234,0.35)" }}>Default</span>
                </button>
                {graphModels.length > 0 && <div style={{ height: 1, background: "rgba(255,255,255,0.08)", margin: "4px 0" }} />}
                {graphModels.map((model) => (
                  <div key={model.id} className="hover:bg-white/10" style={{ display: "flex", alignItems: "center", padding: "8px 12px", gap: 8 }}>
                    <button onClick={() => handleSelectModel(model.id)} className="cursor-pointer" style={{ flex: 1, background: "none", border: "none", padding: 0, fontSize: 13, color: "#EDECEA", display: "flex", alignItems: "center", gap: 8, textAlign: "left", fontFamily: "inherit" }}>
                      <span style={{ width: 16, textAlign: "center", fontSize: 13, color: "#BC9BFF", flexShrink: 0 }}>{selectedModelId === model.id ? "✓" : ""}</span>
                      <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{model.name}</span>
                    </button>
                    <button onClick={(e) => { e.stopPropagation(); setModelDropdownOpen(false); router.push(`/graph-models/${model.id}`); }} className="cursor-pointer hover:opacity-100" style={{ background: "none", border: "none", padding: 2, opacity: 0.4, transition: "opacity 150ms", flexShrink: 0 }} title="Edit model">
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.7)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7" /><path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z" /></svg>
                    </button>
                  </div>
                ))}
                <div style={{ height: 1, background: "rgba(255,255,255,0.08)", margin: "4px 0" }} />
                <button onClick={() => { setModelDropdownOpen(false); setShowCreateModelModal(true); }} className="cursor-pointer hover:bg-white/10" style={{ width: "100%", background: "none", border: "none", padding: "8px 12px", fontSize: 13, color: "#6510F4", fontWeight: 500, display: "flex", alignItems: "center", gap: 8, textAlign: "left", fontFamily: "inherit" }}>
                  <span style={{ width: 16, textAlign: "center" }}>+</span>
                  <span>Create new</span>
                </button>
              </div>
            )}
          </div>
          </div>
          {/* Custom prompt dropdown */}
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <span style={{ fontSize: 11, fontWeight: 700, color: "rgba(237,236,234,0.55)", letterSpacing: 0.2, display: "flex", alignItems: "center", gap: 4 }}>Prompt <Tooltip label="Custom instructions that guide how Cognee extracts entities and relationships from your data." withArrow multiline w={240} position="top"><svg width="12" height="12" viewBox="0 0 16 16" fill="none" style={{ flexShrink: 0 }}><circle cx="8" cy="8" r="7" stroke="#A1A1AA" strokeWidth="1.5" /><text x="8" y="12" textAnchor="middle" fontSize="10" fontWeight="700" fill="#A1A1AA">i</text></svg></Tooltip></span>
          <div ref={promptDropdownRef} style={{ position: "relative" }}>
            <button
              onClick={() => setPromptDropdownOpen((v) => !v)}
              className="cursor-pointer hover:bg-white/10"
              style={{ background: "rgba(255,255,255,0.06)", backdropFilter: "blur(12px)", color: "rgba(237,236,234,0.7)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 6, padding: "8px 16px", fontSize: 13, fontWeight: 500, display: "flex", alignItems: "center", gap: 6, whiteSpace: "nowrap" }}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.7)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" /></svg>
              {selectedPromptName ?? "Automatic"}
              <svg width="10" height="10" viewBox="0 0 10 10" fill="none"><path d="M2.5 4L5 6.5L7.5 4" stroke="rgba(237,236,234,0.55)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
            </button>
            {promptDropdownOpen && (
              <div style={{ position: "absolute", top: "calc(100% + 4px)", left: 0, background: "#1a1a1a", border: "1px solid rgba(255,255,255,0.08)", backdropFilter: "blur(16px)", borderRadius: 8, boxShadow: "0 8px 24px rgba(0,0,0,0.4)", minWidth: 220, zIndex: 50, overflow: "hidden" }}>
                <div style={{ padding: "6px 10px 4px", fontSize: 11, fontWeight: 700, color: "rgba(237,236,234,0.35)", letterSpacing: 0.3, textTransform: "uppercase" }}>Custom Prompt</div>
                <button onClick={() => handleSelectPrompt(null)} className="cursor-pointer hover:bg-white/10" style={{ width: "100%", background: "none", border: "none", padding: "8px 12px", fontSize: 13, color: "#EDECEA", display: "flex", alignItems: "center", gap: 8, textAlign: "left", fontFamily: "inherit" }}>
                  <span style={{ width: 16, textAlign: "center", fontSize: 13, color: "#BC9BFF" }}>{selectedPromptName === null ? "✓" : ""}</span>
                  <span style={{ flex: 1 }}>Automatic</span>
                  <span style={{ fontSize: 11, color: "rgba(237,236,234,0.35)" }}>Default</span>
                </button>
                {Object.keys(customPrompts).length > 0 && <div style={{ height: 1, background: "rgba(255,255,255,0.08)", margin: "4px 0" }} />}
                {Object.entries(customPrompts).map(([name, text]) => (
                  <div key={name} className="hover:bg-white/10" style={{ display: "flex", alignItems: "center", padding: "8px 12px", gap: 8 }}>
                    <button onClick={() => handleSelectPrompt(name)} className="cursor-pointer" style={{ flex: 1, background: "none", border: "none", padding: 0, fontSize: 13, color: "#EDECEA", display: "flex", alignItems: "center", gap: 8, textAlign: "left", fontFamily: "inherit" }}>
                      <span style={{ width: 16, textAlign: "center", fontSize: 13, color: "#BC9BFF", flexShrink: 0 }}>{selectedPromptName === name ? "✓" : ""}</span>
                      <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{name}</span>
                    </button>
                    <button onClick={(e) => { e.stopPropagation(); setPromptDropdownOpen(false); setEditingPromptName(name); setEditingPromptText(text); setShowPromptEditor(true); }} className="cursor-pointer hover:opacity-100" style={{ background: "none", border: "none", padding: 2, opacity: 0.4, transition: "opacity 150ms", flexShrink: 0 }} title="Edit prompt">
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.7)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7" /><path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z" /></svg>
                    </button>
                  </div>
                ))}
                <div style={{ height: 1, background: "rgba(255,255,255,0.08)", margin: "4px 0" }} />
                <button onClick={() => { setPromptDropdownOpen(false); setShowCreatePromptModal(true); }} className="cursor-pointer hover:bg-white/10" style={{ width: "100%", background: "none", border: "none", padding: "8px 12px", fontSize: 13, color: "#6510F4", fontWeight: 500, display: "flex", alignItems: "center", gap: 8, textAlign: "left", fontFamily: "inherit" }}>
                  <span style={{ width: 16, textAlign: "center" }}>+</span>
                  <span>Create new</span>
                </button>
              </div>
            )}
          </div>
          </div>
          {/* Ontology dropdown */}
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <span style={{ fontSize: 11, fontWeight: 700, color: "rgba(237,236,234,0.55)", letterSpacing: 0.2, display: "flex", alignItems: "center", gap: 4 }}>Ontology <Tooltip label="Upload a formal ontology (OWL/RDF) to enforce domain-specific vocabulary and relationships in your knowledge graph." withArrow multiline w={240} position="top"><svg width="12" height="12" viewBox="0 0 16 16" fill="none" style={{ flexShrink: 0 }}><circle cx="8" cy="8" r="7" stroke="#A1A1AA" strokeWidth="1.5" /><text x="8" y="12" textAnchor="middle" fontSize="10" fontWeight="700" fill="#A1A1AA">i</text></svg></Tooltip></span>
          <div ref={ontologyDropdownRef} style={{ position: "relative" }}>
            <button
              onClick={() => setOntologyDropdownOpen((v) => !v)}
              className="cursor-pointer hover:bg-white/10"
              style={{ background: "rgba(255,255,255,0.06)", backdropFilter: "blur(12px)", color: "rgba(237,236,234,0.7)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 6, padding: "8px 16px", fontSize: 13, fontWeight: 500, display: "flex", alignItems: "center", gap: 6, whiteSpace: "nowrap" }}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.7)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 19.5A2.5 2.5 0 016.5 17H20" /><path d="M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z" /></svg>
              {selectedOntologyKey ?? "Automatic"}
              <svg width="10" height="10" viewBox="0 0 10 10" fill="none"><path d="M2.5 4L5 6.5L7.5 4" stroke="rgba(237,236,234,0.55)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
            </button>
            {ontologyDropdownOpen && (
              <div style={{ position: "absolute", top: "calc(100% + 4px)", left: 0, background: "#1a1a1a", border: "1px solid rgba(255,255,255,0.08)", backdropFilter: "blur(16px)", borderRadius: 8, boxShadow: "0 8px 24px rgba(0,0,0,0.4)", width: 260, zIndex: 50, overflow: "hidden" }}>
                <div style={{ padding: "6px 10px 4px", fontSize: 11, fontWeight: 700, color: "rgba(237,236,234,0.35)", letterSpacing: 0.3, textTransform: "uppercase" }}>Ontology</div>
                <button onClick={() => handleSelectOntology(null)} className="cursor-pointer hover:bg-white/10" style={{ width: "100%", background: "none", border: "none", padding: "8px 12px", fontSize: 13, color: "#EDECEA", display: "flex", alignItems: "center", gap: 8, textAlign: "left", fontFamily: "inherit" }}>
                  <span style={{ width: 16, textAlign: "center", fontSize: 13, color: "#BC9BFF", flexShrink: 0 }}>{selectedOntologyKey === null ? "✓" : ""}</span>
                  <span>Automatic</span>
                </button>
                {Object.keys(ontologies).length > 0 && <div style={{ height: 1, background: "rgba(255,255,255,0.08)", margin: "4px 0" }} />}
                {Object.entries(ontologies).map(([key, meta]) => (
                  <div key={key} className="hover:bg-white/10" style={{ display: "flex", alignItems: "center", padding: "8px 12px", gap: 8 }}>
                    <button onClick={() => handleSelectOntology(key)} className="cursor-pointer" style={{ flex: 1, minWidth: 0, background: "none", border: "none", padding: 0, fontSize: 13, color: "#EDECEA", display: "flex", alignItems: "center", gap: 8, textAlign: "left", fontFamily: "inherit" }}>
                      <span style={{ width: 16, textAlign: "center", fontSize: 13, color: "#BC9BFF", flexShrink: 0 }}>{selectedOntologyKey === key ? "✓" : ""}</span>
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
                    }} className="cursor-pointer hover:opacity-100" style={{ background: "none", border: "none", padding: 4, opacity: 0.5, transition: "opacity 150ms", flexShrink: 0, minWidth: 20, minHeight: 20, display: "flex", alignItems: "center", justifyContent: "center" }} title="Delete ontology">
                      <svg width="14" height="14" viewBox="0 0 16 16" fill="none" style={{ flexShrink: 0 }}><path d="M3 4h10M6 4V3h4v1M5 4v8.5a.5.5 0 00.5.5h5a.5.5 0 00.5-.5V4" stroke="#EF4444" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" /></svg>
                    </button>
                  </div>
                ))}
                <div style={{ height: 1, background: "rgba(255,255,255,0.08)", margin: "4px 0" }} />
                <button onClick={() => { setOntologyDropdownOpen(false); setShowUploadOntologyModal(true); }} className="cursor-pointer hover:bg-white/10" style={{ width: "100%", background: "none", border: "none", padding: "8px 12px", fontSize: 13, color: "#6510F4", fontWeight: 500, display: "flex", alignItems: "center", gap: 8, textAlign: "left", fontFamily: "inherit" }}>
                  <span style={{ width: 16, textAlign: "center", flexShrink: 0 }}>+</span>
                  <span>Upload new</span>
                </button>
              </div>
            )}
          </div>
          </div>
        </div>
      </div>

      {/* Upload ontology modal */}
      {showUploadOntologyModal && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.5)", backdropFilter: "blur(4px)", WebkitBackdropFilter: "blur(4px)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={() => setShowUploadOntologyModal(false)}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "rgba(15,15,15,0.92)", backdropFilter: "blur(16px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, padding: 24, width: 440, display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 16px 48px rgba(0,0,0,0.12)" }}>
            <h2 style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", margin: 0 }}>Upload Ontology</h2>
            <p style={{ fontSize: 13, color: "rgba(237,236,234,0.55)", margin: 0, lineHeight: "20px" }}>
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
                if (cogniInstance) {
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
                  <label style={{ fontSize: 12, fontWeight: 700, color: "rgba(237,236,234,0.55)", textTransform: "uppercase", letterSpacing: 0.3 }}>Key</label>
                  <input name="ontologyKey" type="text" required placeholder="e.g. biomedical-ontology" style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 8, padding: "8px 12px", fontSize: 14, fontFamily: "inherit", color: "#EDECEA", outline: "none" }} />
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                  <label style={{ fontSize: 12, fontWeight: 700, color: "rgba(237,236,234,0.55)", textTransform: "uppercase", letterSpacing: 0.3 }}>OWL File</label>
                  <label className="cursor-pointer" style={{ display: "flex", alignItems: "center", gap: 8, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 8, padding: "8px 12px", fontSize: 13, fontFamily: "inherit", color: "rgba(237,236,234,0.55)" }}>
                    <svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M8 1v10M4 5l4-4 4 4" stroke="#A1A1AA" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" /><path d="M1 11v2.5A1.5 1.5 0 002.5 15h11a1.5 1.5 0 001.5-1.5V11" stroke="#A1A1AA" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" /></svg>
                    <span data-file-label="true" style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>Choose a .owl file…</span>
                    <input name="ontologyFile" type="file" required accept=".owl" style={{ display: "none" }} onChange={(e) => {
                      const label = e.currentTarget.parentElement?.querySelector("[data-file-label]");
                      if (label) label.textContent = e.currentTarget.files?.[0]?.name ?? "Choose a .owl file…";
                    }} />
                  </label>
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                  <label style={{ fontSize: 12, fontWeight: 700, color: "rgba(237,236,234,0.55)", textTransform: "uppercase", letterSpacing: 0.3 }}>Description <span style={{ fontWeight: 400, textTransform: "none" }}>(optional)</span></label>
                  <input name="description" type="text" placeholder="What does this ontology define?" style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 8, padding: "8px 12px", fontSize: 14, fontFamily: "inherit", color: "#EDECEA", outline: "none" }} />
                </div>
              </div>
              <div style={{ display: "flex", gap: 8, justifyContent: "flex-end", marginTop: 8 }}>
                <button type="button" onClick={() => setShowUploadOntologyModal(false)} className="cursor-pointer" style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "rgba(237,236,234,0.7)", fontFamily: "inherit" }}>Cancel</button>
                <button type="submit" className="cursor-pointer" style={{ background: "#6510F4", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "#fff", fontFamily: "inherit" }}>Upload</button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Outdated graph banner */}
      {graphOutdated && !processing && datasetStatus !== "processing" && (
        <div style={{ background: "rgba(245,158,11,0.1)", border: "1px solid rgba(245,158,11,0.3)", borderRadius: 8, padding: "12px 16px", display: "flex", alignItems: "center", gap: 12 }}>
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" style={{ flexShrink: 0 }}><path d="M8 1L1 14h14L8 1z" fill="rgba(245,158,11,0.25)" stroke="#F59E0B" strokeWidth="1" /><text x="8" y="12" textAnchor="middle" fontSize="9" fontWeight="700" fill="#FBBF24">!</text></svg>
          <span style={{ flex: 1, fontSize: 13, color: "#FBBF24" }}>
            Knowledge graph is outdated. The graph model was changed since the last build.
          </span>
          <button
            onClick={async () => {
              if (!cogniInstance) return;
              setGraphOutdated(false);
              setDatasetStatus("processing");
              try {
                const cognifyPayload = getCognifyOptions();
                const result = await cognifyDataset({ id: datasetId, name: datasetName, data: [], status: "processing" }, cogniInstance, cognifyPayload);
                trackEvent({ pageName: "Dataset Detail", eventName: "dataset_recognified", additionalProperties: { dataset_id: datasetId } });
                clearDatasetOutdated(cogniInstance, datasetId).catch(() => {});
                // Status polling will pick up the real state
              } catch (err) {
                console.error("Re-cognify failed:", err);
                notifications.show({ title: "Rebuild failed", message: err instanceof Error ? err.message : String(err), color: "red" });
                setGraphOutdated(true);
                setDatasetStatus("outdated");
              }
            }}
            className="cursor-pointer hover:bg-yellow-500/20"
            style={{ background: "rgba(245,158,11,0.2)", border: "1px solid rgba(245,158,11,0.35)", borderRadius: 6, padding: "6px 14px", fontSize: 13, fontWeight: 500, color: "#FBBF24", whiteSpace: "nowrap", fontFamily: "inherit" }}
          >
            Rebuild graph
          </button>
        </div>
      )}

      {/* Drag overlay */}
      {isDragging && (
        <div style={{ position: "fixed", inset: 0, zIndex: 50, background: "rgba(101,16,244,0.04)", border: "2px dashed #6510F4", borderRadius: 12, display: "flex", alignItems: "center", justifyContent: "center", pointerEvents: "none" }}>
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
            <div style={{ width: 52, height: 52, background: "rgba(188,155,255,0.20)", border: "1px solid rgba(188,155,255,0.35)", borderRadius: 12, display: "flex", alignItems: "center", justifyContent: "center" }}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M12 17V7M12 7L7 12M12 7L17 12" stroke="#6510F4" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>
            </div>
            <span style={{ fontSize: 15, fontWeight: 700, color: "#BC9BFF" }}>Drop files to upload</span>
          </div>
        </div>
      )}

      {/* Files table */}
      {filtered.length > 0 ? (
        <div style={{ background: "rgba(255,255,255,0.06)", backdropFilter: "blur(12px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, overflow: "hidden" }}>
          <div style={{ display: "flex", alignItems: "center", background: "rgba(255,255,255,0.04)", borderBottom: "1px solid rgba(255,255,255,0.1)", padding: "12px 20px" }}>
            <span style={{ flex: 1, fontSize: 12, fontWeight: 700, color: "rgba(237,236,234,0.55)" }}>Name</span>
            <span style={{ width: 100, fontSize: 12, fontWeight: 700, color: "rgba(237,236,234,0.55)", flexShrink: 0 }}>Type</span>
            <span style={{ width: 80, fontSize: 12, fontWeight: 700, color: "rgba(237,236,234,0.55)", flexShrink: 0 }}>Size</span>
            <span style={{ width: 170, fontSize: 12, fontWeight: 700, color: "rgba(237,236,234,0.55)", flexShrink: 0 }}>Added</span>
            <span style={{ width: 40, flexShrink: 0 }} />
          </div>
          {filtered.map((file, i) => {
            const isMemory = isMemoryBlobName(file.name);
            const memorySession = memorySessionIds[file.id];
            const meta = isMemory
              ? { fill: "rgba(188,155,255,0.20)", stroke: "#BC9BFF", text: "#BC9BFF", label: "MEM" }
              : getExtMeta(file.name, file.extension);
            const typeName = isMemory ? "Memory" : getTypeName(file.name, file.extension);
            const displayName = isMemory
              ? (memorySession ? `Memory · ${memorySession}` : "Memory")
              : decodeURIComponent(file.name);
            return (
              <div
                key={file.id}
                className="hover:bg-white/10"
                style={{ display: "flex", alignItems: "center", padding: "14px 20px", borderBottom: i < filtered.length - 1 ? "1px solid rgba(255,255,255,0.07)" : "none", transition: "background 150ms" }}
              >
                <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 10 }}>
                  <FileIcon fill={meta.fill} stroke={meta.stroke} text={meta.text} label={meta.label} />
                  <span style={{ fontSize: 13, fontWeight: 500, color: "#EDECEA" }}>{displayName}</span>
                </div>
                <span style={{ width: 100, fontSize: 13, color: "rgba(237,236,234,0.55)", flexShrink: 0 }}>{typeName}</span>
                <span style={{ width: 80, fontSize: 13, color: "rgba(237,236,234,0.55)", flexShrink: 0 }}>—</span>
                <span style={{ width: 170, fontSize: 13, color: "rgba(237,236,234,0.35)", flexShrink: 0 }}>{formatDate(file.createdAt)}</span>
                <div style={{ width: 40, display: "flex", justifyContent: "flex-end", flexShrink: 0 }}>
                  <button
                    onClick={() => handleDelete(file.id)}
                    className="cursor-pointer hover:opacity-100 rounded p-1"
                    style={{ background: "none", border: "none", opacity: 0.5, transition: "opacity 150ms" }}
                    onMouseEnter={(e) => { e.currentTarget.style.opacity = "1"; }}
                    onMouseLeave={(e) => { e.currentTarget.style.opacity = "0.5"; }}
                    title="Delete file"
                  >
                    <TrashIcon />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div style={{ flex: 1, background: "rgba(255,255,255,0.06)", backdropFilter: "blur(12px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 12, padding: 48 }}>
          <span style={{ fontSize: 15, color: "rgba(237,236,234,0.35)" }}>{search ? "No files match your search" : "No files yet"}</span>
          <button onClick={() => fileInputRef.current?.click()} className="cursor-pointer hover:bg-[#5A0ED6]" style={{ background: "#6510F4", color: "#fff", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500 }}>Upload files</button>
        </div>
      )}

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
