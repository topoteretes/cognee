"use client";

import { captureException, recordUploadSuccess, recordUploadFailure } from "@/utils/monitoring";
import { useState, useEffect, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import { useFilter } from "@/ui/layout/FilterContext";
import PageLoading from "@/ui/elements/PageLoading";
import getDatasetData from "@/modules/datasets/getDatasetData";
import deleteDatasetData from "@/modules/datasets/deleteDatasetData";
import deleteDataset from "@/modules/datasets/deleteDataset";
import { useBrainUpload } from "@/modules/ingestion/useBrainUpload";
import { MAX_FILES_PER_UPLOAD } from "@/modules/ingestion/uploadLimits";
import DeleteConfirmModal from "@/ui/elements/DeleteConfirmModal";
import CreateModelModal from "./partials/CreateModelModal";
import CreatePromptModal from "./partials/CreatePromptModal";
import PromptEditorModal from "./partials/PromptEditorModal";
import UploadOntologyModal from "./partials/UploadOntologyModal";
import MemoryCustomizationBar from "./partials/MemoryCustomizationBar";
import FilesTable from "./partials/FilesTable";
import { decodeFilename } from "@/utils/fileFormat";
import mapInferredSchema from "@/modules/graphModels/mapInferredSchema";
import isMemoryBlobName from "@/modules/datasets/isMemoryBlobName";
import TrashIcon from "@/ui/elements/TrashIcon";
import cognifyDataset from "@/modules/datasets/cognifyDataset";
import pollDatasetStatus from "@/modules/datasets/pollDatasetStatus";
import { useDatasetStatuses } from "@/modules/datasets/useDatasetStatuses";
import { notifications } from "@mantine/notifications";
import { Loader } from "@mantine/core";
import { TrackPageView, trackEvent } from "@/modules/analytics";
import type { GraphModel } from "@/modules/graphModels/types";
import { toCleanSchema } from "@/modules/graphModels/types";
import { toGraphModelSchema } from "@/modules/graphModels/toGraphModelSchema";
import { loadGraphModelsConfig, syncGraphModels, assignGraphModelToDataset, assignPromptToDataset, assignOntologyToDataset, clearDatasetOutdated, findModelForDataset, findPromptForDataset, findOntologyForDataset, saveCustomPrompt, deleteCustomPrompt, type CustomPromptsMap } from "@/modules/configuration/userConfiguration";
import { inferSchema, generateCustomPrompt } from "@/modules/llm/managementLlmApi";
import { listOntologies, uploadOntology, deleteOntology, type OntologyMeta } from "@/modules/ontologies/ontologyApi";
import ShareDatasetModal from "@/ui/elements/ShareDatasetModal";
import { describeProcessingError } from "../processingErrorMessage";
import { v4 as uuid } from "uuid";

interface FileEntry {
  id: string;
  name: string;
  extension?: string;
  mimeType?: string;
  size?: number;
  createdAt?: string;
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


// ── Main Page ──


export default function DatasetDetailPage({ datasetId }: { datasetId: string }) {
  const router = useRouter();
  const { cogniInstance, isInitializing } = useCogniInstance();
  const { datasets: contextDatasets } = useFilter();
  const [datasetName, setDatasetName] = useState<string>(datasetId);
  const [, setLastSynced] = useState<string | null>(null);
  const [files, setFiles] = useState<FileEntry[]>([]);
  const [filesError, setFilesError] = useState(false);
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
  const { isUploading, stage: uploadStage, upload } = useBrainUpload(cogniInstance);
  const [processing, setProcessing] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [isConnectedSource, setIsConnectedSource] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [deletingFileId, setDeletingFileId] = useState<string | null>(null);
  const [deleteFileTarget, setDeleteFileTarget] = useState<FileEntry | null>(null);
  const [search, setSearch] = useState("");
  const [showShareModal, setShowShareModal] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [confirmDeletePrompt, setConfirmDeletePrompt] = useState(false);
  const [deletingPrompt, setDeletingPrompt] = useState(false);
  const [confirmDeleteOntologyKey, setConfirmDeleteOntologyKey] = useState<string | null>(null);
  const [deletingOntology, setDeletingOntology] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Graph model selection
  const [graphModels, setGraphModels] = useState<GraphModel[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [graphOutdated, setGraphOutdated] = useState(false);
  const [datasetStatus, setDatasetStatus] = useState<"ready" | "pending" | "processing" | "failed" | "outdated" | "empty">("empty");
  const [showCreateModelModal, setShowCreateModelModal] = useState(false);
  const [inferring, setInferring] = useState(false);

  // Custom prompt selection (simple dict: { name: text })
  const [customPrompts, setCustomPrompts] = useState<CustomPromptsMap>({});
  const [selectedPromptName, setSelectedPromptName] = useState<string | null>(null);
  const [showCreatePromptModal, setShowCreatePromptModal] = useState(false);
  const [inferringPrompt, setInferringPrompt] = useState(false);

  // Ontology selection
  const [ontologies, setOntologies] = useState<Record<string, OntologyMeta>>({});
  const [selectedOntologyKey, setSelectedOntologyKey] = useState<string | null>(null);
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
    }).catch((err) => captureException(err, { context: "dataset-detail.load-graph-config" }));
    listOntologies(cogniInstance).then(setOntologies).catch((err) =>
      captureException(err, { context: "dataset-detail.list-ontologies" }));
  }, [datasetId, cogniInstance, isInitializing]);

  // Dropdown open/close + outside-click handling lives inside
  // MemoryCustomizationBar; the page only owns selection + persistence.

  async function handleDeleteOntology(key: string): Promise<void> {
    if (!cogniInstance) return;
    setDeletingOntology(true);
    try {
      await deleteOntology(cogniInstance, key);
      setOntologies((prev) => { const next = { ...prev }; delete next[key]; return next; });
      if (selectedOntologyKey === key) setSelectedOntologyKey(null);
      setConfirmDeleteOntologyKey(null);
      notifications.show({ title: "Ontology deleted", message: `"${key}" removed.`, color: "green", autoClose: 4000 });
    } catch (err) {
      notifications.show({ title: "Delete failed", message: err instanceof Error ? err.message : String(err), color: "red" });
    } finally {
      setDeletingOntology(false);
    }
  }

  function handleSelectOntology(key: string | null) {
    const prev = selectedOntologyKey;
    setSelectedOntologyKey(key);
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
    if (!cogniInstance || !selectedModelId) return;
    setInferringPrompt(true);
    try {
      const model = graphModels.find((m) => m.id === selectedModelId);
      if (model) {
        const cleanSchema = toCleanSchema(model.schema);
        const graphModelSchema = toGraphModelSchema(cleanSchema);
        const result = await generateCustomPrompt(cogniInstance, graphModelSchema);
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

  async function handleConfirmDeletePrompt(): Promise<void> {
    const name = editingPromptName.trim();
    if (!name || !cogniInstance) return;
    setDeletingPrompt(true);
    try {
      await deleteCustomPrompt(cogniInstance, name);
      setCustomPrompts((prev) => { const next = { ...prev }; delete next[name]; return next; });
      if (selectedPromptName === name) setSelectedPromptName(null);
      setConfirmDeletePrompt(false);
      setShowPromptEditor(false);
      notifications.show({ title: "Prompt deleted", message: `"${name}" removed.`, color: "green", autoClose: 4000 });
    } catch (err) {
      notifications.show({ title: "Delete failed", message: err instanceof Error ? err.message : String(err), color: "red" });
    } finally {
      setDeletingPrompt(false);
    }
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

  // Rejects on failure so UploadOntologyModal re-enables its form; resolves on
  // success after closing the modal itself.
  async function handleUploadOntology(key: string, file: File, description?: string): Promise<void> {
    if (!cogniInstance) return;
    try {
      await uploadOntology(cogniInstance, key, file, description);
      const updated = await listOntologies(cogniInstance);
      setOntologies(updated);
      setSelectedOntologyKey(key);
      setShowUploadOntologyModal(false);
      assignOntologyToDataset(cogniInstance, datasetId, key).catch((err) => {
        captureException(err, { context: "dataset-detail.assign-ontology-after-upload", datasetId, key });
        notifications.show({
          title: "Ontology uploaded, but not assigned",
          message: `"${key}" was uploaded but couldn't be assigned to this dataset automatically. Assign it manually from the dropdown.`,
          color: "orange",
        });
      });
      notifications.show({ title: "Ontology uploaded", message: `"${key}" is ready to use.`, color: "green", autoClose: 4000 });
    } catch (err) {
      notifications.show({ title: "Upload failed", message: err instanceof Error ? err.message : String(err), color: "red" });
      throw err;
    }
  }

  function handleSelectModel(modelId: string | null) {
    const prevModelId = selectedModelId;
    setSelectedModelId(modelId);
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
            modelSchema = mapInferredSchema(result.graphSchema);
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

  const loadFiles = useCallback(async () => {
    if (!cogniInstance) return;
    try {
      const data = await getDatasetData(datasetId, cogniInstance);
      setFiles(Array.isArray(data) ? data.map((d: FileEntry & { rawDataLocation?: string; originalExtension?: string; original_extension?: string; originalMimeType?: string; original_mime_type?: string; size_bytes?: number; file_size?: number }) => ({
        id: d.id,
        name: d.name || d.rawDataLocation?.split("/").pop() || d.id,
        extension: d.originalExtension || d.original_extension || d.extension,
        mimeType: d.originalMimeType || d.original_mime_type || d.mimeType,
        size: d.size ?? d.size_bytes ?? d.file_size,
        createdAt: d.createdAt,
      })) : []);
      setFilesError(false);
    } catch {
      // Don't blank the list into a fake "empty" state — surface the load
      // failure so the user knows their files aren't gone, just unreachable.
      setFilesError(true);
    } finally {
      setLoading(false);
    }
  }, [cogniInstance, datasetId]);

  // Resolves the dataset's display name from FilterContext's shared datasets
  // list, which loads asynchronously and may still be empty on the first
  // render of this effect (e.g. a hard refresh landing directly on this page)
  // — depending on contextDatasets lets this retry once it arrives, instead
  // of leaving datasetName stuck on the raw id fallback forever.
  useEffect(() => {
    const ds = contextDatasets.find((d) => d.id === datasetId) as { id: string; name: string; updatedAt?: string; connection_id?: string } | undefined;
    if (ds) {
      setDatasetName(ds.name);
      if (ds.updatedAt) setLastSynced(ds.updatedAt);
      if (ds.connection_id) setIsConnectedSource(true);
    }
  }, [contextDatasets, datasetId]);

  useEffect(() => {
    if (!cogniInstance || isInitializing) return;
    loadFiles();
  }, [cogniInstance, isInitializing, loadFiles]);

  const { statuses, refetch: refetchStatuses } = useDatasetStatuses(!isInitializing);

  useEffect(() => {
    const raw = statuses[datasetId];
    if (!raw) {
      if (files.length === 0) setDatasetStatus("empty");
      else if (graphOutdated) setDatasetStatus("outdated");
      else setDatasetStatus("ready");
      setProcessing(false);
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
  }, [statuses, datasetId, graphOutdated, files.length]);

  async function handleUpload(newFiles: FileList | File[]) {
    if (!cogniInstance) return;
    const filesArray = Array.from(newFiles);

    if (filesArray.length > MAX_FILES_PER_UPLOAD) {
      notifications.show({
        title: "Too many files",
        message: `You selected ${filesArray.length} files. Please upload ${MAX_FILES_PER_UPLOAD} or fewer at a time.`,
        color: "red",
      });
      return;
    }

    const totalBytes = filesArray.reduce((sum, f) => sum + f.size, 0);
    const fileTypes = filesArray.map((f) => f.type || "unknown");

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

    await upload({
      datasetId,
      files: filesArray,
      options: getCognifyOptions(),
      // Belt-and-suspenders: the check above already blocks this case, but if
      // that check is ever removed/changed the hook's own guard must still
      // surface an error instead of silently no-opping.
      onLimitExceeded: (selected) => {
        notifications.show({
          title: "Too many files",
          message: `You selected ${selected.length} files. Please upload ${MAX_FILES_PER_UPLOAD} or fewer at a time.`,
          color: "red",
        });
      },
      onUploadError: (error, ctx) => {
        const errorName = error instanceof Error ? error.name : "UnknownError";
        const errorMessage = error instanceof Error ? error.message : String(error);
        recordUploadFailure(errorName, ctx.durationMs);
        trackEvent({
          pageName: "Dataset Detail",
          eventName: "dataset_upload_failed",
          additionalProperties: {
            dataset_id: datasetId,
            file_count: String(filesArray.length),
            total_bytes: String(totalBytes),
            file_types: fileTypes.join(","),
            duration_ms: String(ctx.durationMs),
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
          captureException(error, { datasetId, fileCount: filesArray.length, totalBytes, durationMs: ctx.durationMs });
          notifications.show({
            title: "Upload failed",
            message: errorMessage,
            color: "red",
          });
        }
      },
      onProcessed: async (ctx) => {
        await loadFiles();
        setLastSynced(new Date().toISOString());
        trackEvent({
          pageName: "Dataset Detail",
          eventName: "dataset_files_uploaded",
          additionalProperties: {
            dataset_id: datasetId,
            file_count: String(filesArray.length),
            total_bytes: String(totalBytes),
            duration_ms: String(ctx.durationMs),
          },
        });
        recordUploadSuccess(ctx.durationMs, totalBytes, filesArray.length);
      },
      onProcessingError: async (error, ctx) => {
        await loadFiles();
        const errorMessage = error instanceof Error ? error.message : String(error);
        captureException(error, { datasetId, fileCount: filesArray.length, totalBytes, durationMs: ctx.durationMs, stage: "processing" });
        trackEvent({
          pageName: "Dataset Detail",
          eventName: "dataset_processing_failed",
          additionalProperties: {
            dataset_id: datasetId,
            file_count: String(filesArray.length),
            total_bytes: String(totalBytes),
            duration_ms: String(ctx.durationMs),
            error_message: errorMessage,
          },
        });
        const { title, message, isTimeout } = describeProcessingError(error);
        notifications.show({ title, message, color: isTimeout ? "yellow" : "red" });
      },
    });
  }

  async function handleDelete(fileId: string) {
    if (!cogniInstance || deletingFileId) return;
    setDeletingFileId(fileId);
    try {
      await deleteDatasetData(datasetId, fileId, cogniInstance);
      const deletedFile = files.find((f) => f.id === fileId);
      trackEvent({ pageName: "Dataset Detail", eventName: "dataset_file_deleted", additionalProperties: { dataset_id: datasetId, file_name: deletedFile?.name ?? fileId } });
      setFiles((prev) => prev.filter((f) => f.id !== fileId));
      setDeleteFileTarget(null);
    } catch (err) {
      console.error("Delete failed:", err);
    } finally {
      setDeletingFileId(null);
    }
  }

  async function handleSync() {
    if (!cogniInstance) return;
    setSyncing(true);
    try {
      await cognifyDataset({ id: datasetId, name: datasetName, data: [], status: "processing" }, cogniInstance, getCognifyOptions());
      refetchStatuses();
      // Track completion in the background instead of awaiting it here — the
      // build can take minutes, and the header's Processing indicator (driven
      // by the shared status poller) already reflects progress. Blocking the
      // Sync button for that whole duration reads as "stuck", same issue as
      // the upload button (CLO-292).
      pollDatasetStatus(datasetId, cogniInstance, { intervalMs: 5000 })
        .then((finalStatus) => {
          trackEvent({ pageName: "Dataset Detail", eventName: "dataset_synced", additionalProperties: { dataset_id: datasetId, status: finalStatus } });
          setLastSynced(new Date().toISOString());
          refetchStatuses();
        })
        .catch((err) => console.error("Sync build failed:", err));
    } catch (err) {
      console.error("Sync failed:", err);
    } finally {
      setSyncing(false);
    }
  }

  // Re-run cognify for the current dataset (used by both the "outdated" and
  // "failed" banners). onError restores the pre-rebuild banner state.
  async function rebuildGraph(onError: () => void): Promise<void> {
    if (!cogniInstance) return;
    setGraphOutdated(false);
    setDatasetStatus("processing");
    try {
      await cognifyDataset({ id: datasetId, name: datasetName, data: [], status: "processing" }, cogniInstance, getCognifyOptions());
      trackEvent({ pageName: "Dataset Detail", eventName: "dataset_recognified", additionalProperties: { dataset_id: datasetId } });
      clearDatasetOutdated(cogniInstance, datasetId).catch((err) =>
        captureException(err, { context: "dataset-detail.clear-outdated-flag", datasetId }));
      // Force an immediate status re-fetch so the shared poller picks up the
      // new in-progress state right away instead of waiting for the next tick.
      refetchStatuses();
    } catch (err) {
      console.error("Re-cognify failed:", err);
      notifications.show({ title: "Rebuild failed", message: err instanceof Error ? err.message : String(err), color: "red" });
      onError();
    }
  }

  async function handleDeleteDataset() {
    if (!cogniInstance) return;
    setDeleting(true);
    try {
      await deleteDataset(datasetId, cogniInstance);
      trackEvent({ pageName: "Dataset Detail", eventName: "dataset_deleted", additionalProperties: { dataset_id: datasetId } });
      router.push("/datasets");
    } catch (err) {
      console.error("Delete brain failed:", err);
      setDeleting(false);
      setShowDeleteConfirm(false);
    }
  }

  const filtered = search ? files.filter((f) => f.name.toLowerCase().includes(search.toLowerCase())) : files;

  if (loading || isInitializing) {
    return <><TrackPageView page="Dataset Detail" additionalProperties={{ dataset_id: datasetId }} /><PageLoading name="Files" /></>;
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
              {syncing ? (
                <Loader size={14} color="rgba(237,236,234,0.7)" />
              ) : (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.7)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 2v6h-6" /><path d="M3 12a9 9 0 0115.36-6.36L21 8" /><path d="M3 22v-6h6" /><path d="M21 12a9 9 0 01-15.36 6.36L3 16" /></svg>
              )}
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
            disabled={isUploading}
            className="cursor-pointer hover:bg-[#5A0ED6]"
            style={{ background: "#6510F4", color: "#fff", border: "none", borderRadius: 6, padding: "8px 16px", fontSize: 13, fontWeight: 500, display: "flex", alignItems: "center", gap: 6 }}
          >
          {isUploading ? (
            <Loader size={14} color="#fff" />
          ) : (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" /></svg>
          )}
          {isUploading ? (uploadStage === "processing" ? "Building knowledge graph..." : "Uploading...") : "Upload files"}
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
        <DeleteConfirmModal
          title="Delete brain"
          message={<>Are you sure you want to delete <strong>{datasetName}</strong>? This will permanently remove the dataset and all its files. This action cannot be undone.</>}
          onConfirm={handleDeleteDataset}
          onCancel={() => setShowDeleteConfirm(false)}
          busy={deleting}
        />
      )}

      {/* Create graph model modal */}
      {showCreateModelModal && (
        <CreateModelModal
          inferring={inferring}
          filesCount={files.length}
          onInfer={() => handleCreateModel(true)}
          onBlank={() => handleCreateModel(false)}
          onCancel={() => setShowCreateModelModal(false)}
        />
      )}

      {/* Create custom prompt modal */}
      {showCreatePromptModal && (
        <CreatePromptModal
          inferringPrompt={inferringPrompt}
          modelName={graphModels.find((m) => m.id === selectedModelId)?.name ?? null}
          onGenerate={handleInferPrompt}
          onBlank={handleStartBlankPrompt}
          onCancel={() => setShowCreatePromptModal(false)}
        />
      )}

      {/* Prompt editor modal */}
      {showPromptEditor && (
        <PromptEditorModal
          name={editingPromptName}
          text={editingPromptText}
          saving={savingPrompt}
          onNameChange={setEditingPromptName}
          onTextChange={setEditingPromptText}
          onSave={handleSavePrompt}
          onClose={() => setShowPromptEditor(false)}
          onDelete={() => setConfirmDeletePrompt(true)}
        />
      )}

      {/* Delete prompt confirmation modal */}
      {confirmDeletePrompt && (
        <DeleteConfirmModal
          title="Delete prompt"
          message={<>Are you sure you want to delete <strong>{editingPromptName.trim() || "this prompt"}</strong>? This action cannot be undone.</>}
          onConfirm={handleConfirmDeletePrompt}
          onCancel={() => setConfirmDeletePrompt(false)}
          busy={deletingPrompt}
        />
      )}

      {/* Delete ontology confirmation modal */}
      {confirmDeleteOntologyKey && (
        <DeleteConfirmModal
          title="Delete ontology"
          message={<>Are you sure you want to delete <strong>{confirmDeleteOntologyKey}</strong>? This action cannot be undone.</>}
          onConfirm={() => handleDeleteOntology(confirmDeleteOntologyKey)}
          onCancel={() => setConfirmDeleteOntologyKey(null)}
          busy={deletingOntology}
        />
      )}

      {/* Delete file confirmation modal */}
      {deleteFileTarget && (
        <DeleteConfirmModal
          title="Delete file"
          message={<>Are you sure you want to delete <strong>{decodeFilename(deleteFileTarget.name)}</strong>? This action cannot be undone.</>}
          onConfirm={() => handleDelete(deleteFileTarget.id)}
          onCancel={() => setDeleteFileTarget(null)}
          busy={deletingFileId === deleteFileTarget.id}
        />
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
      <MemoryCustomizationBar
        graphModels={graphModels}
        selectedModelId={selectedModelId}
        onSelectModel={handleSelectModel}
        onEditModel={(id) => router.push(`/graph-models/${id}`)}
        onCreateModel={() => setShowCreateModelModal(true)}
        customPrompts={customPrompts}
        selectedPromptName={selectedPromptName}
        onSelectPrompt={handleSelectPrompt}
        onEditPrompt={(name, text) => { setEditingPromptName(name); setEditingPromptText(text); setShowPromptEditor(true); }}
        onCreatePrompt={() => setShowCreatePromptModal(true)}
        ontologies={ontologies}
        selectedOntologyKey={selectedOntologyKey}
        onSelectOntology={handleSelectOntology}
        onDeleteOntology={setConfirmDeleteOntologyKey}
        onUploadOntology={() => setShowUploadOntologyModal(true)}
      />

      {/* Upload ontology modal */}
      {showUploadOntologyModal && (
        <UploadOntologyModal
          onClose={() => setShowUploadOntologyModal(false)}
          onSubmit={handleUploadOntology}
        />
      )}

      {/* Outdated graph banner */}
      {graphOutdated && !processing && datasetStatus !== "processing" && (
        <div style={{ background: "rgba(245,158,11,0.1)", border: "1px solid rgba(245,158,11,0.3)", borderRadius: 8, padding: "12px 16px", display: "flex", alignItems: "center", gap: 12 }}>
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" style={{ flexShrink: 0 }}><path d="M8 1L1 14h14L8 1z" fill="rgba(245,158,11,0.25)" stroke="#F59E0B" strokeWidth="1" /><text x="8" y="12" textAnchor="middle" fontSize="9" fontWeight="700" fill="#FBBF24">!</text></svg>
          <span style={{ flex: 1, fontSize: 13, color: "#FBBF24" }}>
            Knowledge graph is outdated. The graph model was changed since the last build.
          </span>
          <button
            onClick={() => rebuildGraph(() => { setGraphOutdated(true); setDatasetStatus("outdated"); })}
            className="cursor-pointer hover:bg-yellow-500/20"
            style={{ background: "rgba(245,158,11,0.2)", border: "1px solid rgba(245,158,11,0.35)", borderRadius: 6, padding: "6px 14px", fontSize: 13, fontWeight: 500, color: "#FBBF24", whiteSpace: "nowrap", fontFamily: "inherit" }}
          >
            Rebuild graph
          </button>
        </div>
      )}

      {/* Failed build banner */}
      {datasetStatus === "failed" && !processing && (
        <div style={{ background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: 8, padding: "12px 16px", display: "flex", alignItems: "center", gap: 12 }}>
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" style={{ flexShrink: 0 }}><path d="M8 1L1 14h14L8 1z" fill="rgba(239,68,68,0.25)" stroke="#EF4444" strokeWidth="1" /><text x="8" y="12" textAnchor="middle" fontSize="9" fontWeight="700" fill="#F87171">!</text></svg>
          <span style={{ flex: 1, fontSize: 13, color: "#F87171" }}>
            Building the knowledge graph failed. Your files are still here — you can retry the build.
          </span>
          <button
            onClick={() => rebuildGraph(() => setDatasetStatus("failed"))}
            className="cursor-pointer hover:bg-red-500/20"
            style={{ background: "rgba(239,68,68,0.2)", border: "1px solid rgba(239,68,68,0.35)", borderRadius: 6, padding: "6px 14px", fontSize: 13, fontWeight: 500, color: "#F87171", whiteSpace: "nowrap", fontFamily: "inherit" }}
          >
            Retry build
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
      <FilesTable
        files={filtered}
        memorySessionIds={memorySessionIds}
        search={search}
        loadError={filesError}
        onDelete={(id) => setDeleteFileTarget(filtered.find((f) => f.id === id) ?? null)}
        onUploadClick={() => fileInputRef.current?.click()}
        onRetry={loadFiles}
        deletingId={deletingFileId}
      />

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
