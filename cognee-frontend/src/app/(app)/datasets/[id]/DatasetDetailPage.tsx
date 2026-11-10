"use client";

import { captureException, recordUploadSuccess, recordUploadFailure } from "@/utils/monitoring";
import { isInsufficientCreditsError } from "@/utils/insufficientCredits";
import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import { useFilter } from "@/ui/layout/FilterContext";
import PageLoading from "@/ui/elements/PageLoading";
import { MAX_RENDERED_ROWS } from "@/modules/datasets/maxRenderedRows";
import useDatasetDataPages from "@/modules/datasets/useDatasetDataPages";
import deleteDatasetData from "@/modules/datasets/deleteDatasetData";
import deleteDataset from "@/modules/datasets/deleteDataset";
import { useBrainUpload } from "@/modules/ingestion/useBrainUpload";
import { withEstimateStage } from "@/modules/ingestion/uploadProgress";
import UploadProgressBar from "../partials/UploadProgressBar";
import ScrollLoader from "../partials/ScrollLoader";
import ProcessingSummary from "../partials/ProcessingSummary";
import { useLowBalanceUploadWarning } from "@/modules/billing/useLowBalanceUploadWarning";
import LowBalanceWarningModal from "@/ui/elements/LowBalanceWarningModal";
import { MAX_FILES_PER_UPLOAD } from "@/modules/ingestion/uploadLimits";
import DeleteConfirmModal from "@/ui/elements/DeleteConfirmModal";
import CreateModelModal from "./partials/CreateModelModal";
import CreatePromptModal from "./partials/CreatePromptModal";
import PromptEditorModal from "./partials/PromptEditorModal";
import UploadOntologyModal from "./partials/UploadOntologyModal";
import MemoryCustomizationBar from "./partials/MemoryCustomizationBar";
import FilesTable from "./partials/FilesTable";
import { decodeFilename } from "@/utils/fileFormat";
import { fmtCount } from "@/utils/formatCount";
import { useIsMounted } from "@/utils/useIsMounted";
import { useAssignmentChoice } from "./useAssignmentChoice";
import { joinDistinct } from "@/utils/joinDistinct";
import mapInferredSchema from "@/modules/graphModels/mapInferredSchema";
import describeTextBlob, { blobsToRead, connectorRepository, type TextBlobDescription } from "@/modules/datasets/describeTextBlob";
import forEachWithConcurrency from "@/utils/forEachWithConcurrency";
import readTextPrefix from "@/utils/readTextPrefix";
import TrashIcon from "@/ui/elements/TrashIcon";
import cognifyDataset from "@/modules/datasets/cognifyDataset";
import pollDatasetStatus from "@/modules/datasets/pollDatasetStatus";
import { useDatasetStatuses } from "@/modules/datasets/useDatasetStatuses";
import { useDatasetProcessing } from "@/modules/datasets/useDatasetProcessing";
import { INSUFFICIENT_CREDITS_REASON } from "@/modules/datasets/datasetStatusDetail";
import { notifications, type NotificationData } from "@mantine/notifications";
import { Loader } from "@mantine/core";
import { TrackPageView, trackEvent } from "@/modules/analytics";
import type { CogneeInstance } from "@/modules/instances/types";
import type { GraphModel } from "@/modules/graphModels/types";
import { toCleanSchema } from "@/modules/graphModels/types";
import { toGraphModelSchema } from "@/modules/graphModels/toGraphModelSchema";
import { fetchGraphModelsConfig, addGraphModelForDataset, assignGraphModelToDataset, assignPromptToDataset, assignOntologyToDataset, clearDatasetOutdated, removeOntologyAssignments, flagDatasetsUsingOntology, datasetsAssignedTo, withoutValue, findModelForDataset, findPromptForDataset, findOntologyForDataset, saveCustomPrompt, deleteCustomPrompt, type CustomPromptsMap } from "@/modules/configuration/userConfiguration";
import { inferSchema, generateCustomPrompt } from "@/modules/llm/managementLlmApi";
import { listOntologies, uploadOntology, deleteOntology, type OntologyMeta } from "@/modules/ontologies/ontologyApi";
import ShareDatasetModal from "@/ui/elements/ShareDatasetModal";
import { describeProcessingError } from "../processingErrorMessage";
import { v4 as uuid } from "uuid";

interface FileEntry {
  id: string;
  name: string;
  label?: string;
  extension?: string;
  mimeType?: string;
  size?: number;
  createdAt?: string;
}


// How many hashed blobs to read at once to title their rows. A dataset fed by
// a connector holds one per synced file, so the unbounded version of this
// opened hundreds of simultaneous reads against the tenant pod — the pattern
// CLO-976 already had to protect the graph endpoint from (review-reported).
const BLOB_READ_CONCURRENCY = 6;
// `describeTextBlob` reads a session header or the first meaningful line, so
// nothing past the head of the file can change its answer. Downloading a whole
// synced document to render a row title is pure waste (review-reported).
const BLOB_READ_PREFIX_BYTES = 8 * 1024;

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

/**
 * What a deleted ontology leaves behind. When the unassign failed, what the page
 * knows decides the wording: with no dataset known to use the key, the failure
 * was reading the settings, so the user is only told it could not be checked.
 */
function ontologyDeletedNotice(key: string, unassigned: boolean, knownInUse: boolean): NotificationData {
  if (unassigned) return { title: "Ontology deleted", message: `"${key}" removed.`, color: "green", autoClose: 4000 };
  if (knownInUse) {
    return { title: "Ontology deleted, not unassigned", message: `"${key}" is gone, but the datasets that used it could not be updated and still have it selected. Choose another ontology for each of them.`, color: "yellow" };
  }
  return { title: "Ontology deleted", message: `"${key}" removed, but the dataset settings could not be checked for it. If a dataset still has it selected, choose another ontology there.`, color: "yellow" };
}

// ── Main Page ──


export default function DatasetDetailPage({ datasetId }: { datasetId: string }) {
  const router = useRouter();
  const isMounted = useIsMounted();
  const { cogniInstance, isInitializing } = useCogniInstance();
  const { datasets: contextDatasets } = useFilter();
  const [datasetName, setDatasetName] = useState<string>(datasetId);
  const [, setLastSynced] = useState<string | null>(null);
  // The list is one bounded page at a time; the dataset-wide total comes from
  // the count endpoint, because the loaded rows cannot supply it.
  const {
    data: rawFiles, setData: setFiles, error: filesError, loading,
    total: filesTotal, hasMore, load: loadFilePage, loadMore,
  } = useDatasetDataPages<FileEntry & {
    rawDataLocation?: string; originalExtension?: string; original_extension?: string;
    originalMimeType?: string; original_mime_type?: string; dataSize?: number; size_bytes?: number; file_size?: number;
  }>(cogniInstance, MAX_RENDERED_ROWS, true);
  const files = useMemo(() => rawFiles.map((d) => ({
    id: d.id,
    name: d.name || d.rawDataLocation?.split("/").pop() || d.id,
    label: typeof d.label === "string" ? d.label : undefined,
    extension: d.originalExtension || d.original_extension || d.extension,
    mimeType: d.originalMimeType || d.original_mime_type || d.mimeType,
    size: d.dataSize ?? d.size ?? d.size_bytes ?? d.file_size,
    createdAt: d.createdAt,
  })), [rawFiles]);
  // data id → what that hashed text blob turned out to be, read from its own
  // content, or null when it could not be read. A blob is only a memory if it
  // says so; connector content arrives under the same hashed name.
  const [blobDescriptions, setBlobDescriptions] = useState<Record<string, TextBlobDescription | null>>({});

  useEffect(() => {
    if (!cogniInstance) return;
    const targets = blobsToRead(files, blobDescriptions);
    if (targets.length === 0) return;
    let cancelled = false;
    void forEachWithConcurrency(
      targets,
      BLOB_READ_CONCURRENCY,
      async (f) => {
        try {
          const res = await cogniInstance.fetch(`/v1/datasets/${datasetId}/data/${f.id}/raw`);
          if (!res.ok) return null;
          return describeTextBlob(await readTextPrefix(res, BLOB_READ_PREFIX_BYTES));
        } catch {
          return null;
        }
      },
      (f, description) => {
        // Per result rather than per batch: with a cap, waiting for the whole
        // list would hold every row's skeleton until the slowest read in the
        // last round returns.
        if (cancelled) return;
        setBlobDescriptions(prev => (f.id in prev ? prev : { ...prev, [f.id]: description }));
      },
    );
    return () => { cancelled = true; };
  }, [cogniInstance, datasetId, files, blobDescriptions]);
  const [isDragging, setIsDragging] = useState(false);
  const { stage: uploadStage, progress: uploadProgress, upload } = useBrainUpload(cogniInstance);
  const { pendingWarning: pendingLowBalanceWarning, isEstimating, confirmUpload, cancel: cancelLowBalanceUpload } = useLowBalanceUploadWarning();
  // Estimate → upload → build report through one progress stream; the estimate
  // runs before upload() starts, so fold it in here (see withEstimateStage).
  const combinedUploadProgress = withEstimateStage(uploadProgress, isEstimating);
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
  const [graphOutdated, setGraphOutdated] = useState(false);
  const [datasetStatus, setDatasetStatus] = useState<"ready" | "pending" | "processing" | "failed" | "failed_insufficient_credits" | "outdated" | "empty">("empty");
  const [showCreateModelModal, setShowCreateModelModal] = useState(false);
  const [inferring, setInferring] = useState(false);

  // Custom prompt selection (simple dict: { name: text })
  const [customPrompts, setCustomPrompts] = useState<CustomPromptsMap>({});
  const [showCreatePromptModal, setShowCreatePromptModal] = useState(false);
  const [inferringPrompt, setInferringPrompt] = useState(false);

  // Ontology selection
  const [ontologies, setOntologies] = useState<Record<string, OntologyMeta>>({});
  // Which datasets use which prompt and ontology, as loaded, so a delete can say
  // what else it affects: deleting one unassigns it from every dataset.
  const [storedAssignments, setStoredAssignments] = useState<{ prompts: Record<string, string>; ontologies: Record<string, string> }>({ prompts: {}, ontologies: {} });
  const [showUploadOntologyModal, setShowUploadOntologyModal] = useState(false);
  const [configStatus, setConfigStatus] = useState<"loading" | "loaded" | "failed">("loading");
  const [configAttempt, setConfigAttempt] = useState(0);

  // A stored change to what this dataset builds with (its model, prompt or
  // ontology, or the wording of its prompt) makes a graph that already has files
  // outdated. Each choice is stored as it is made (see useAssignmentChoice).
  // The changes are counted: a rebuild started before one finished building with
  // the old input, so it must not clear the flag that change raised (see rebuildGraph).
  const storedChanges = useRef(0);
  const noteStoredChange = () => {
    storedChanges.current += 1;
    if (files.length > 0) setGraphOutdated(true);
  };
  const assignmentOptions = { datasetId, datasetName, onChanged: noteStoredChange };
  const storeWith = (assign: (instance: CogneeInstance, datasetId: string, choice: string | null) => Promise<void>) =>
    async (choice: string | null): Promise<void> => {
      if (!cogniInstance) throw new Error("Not connected to a workspace");
      await assign(cogniInstance, datasetId, choice);
    };
  const modelChoice = useAssignmentChoice({
    ...assignmentOptions,
    label: "Graph model",
    describe: (id) => (id ? graphModels.find((m) => m.id === id)?.name ?? "Unknown" : "Automatic"),
    save: storeWith(assignGraphModelToDataset),
  });
  const promptChoice = useAssignmentChoice({ ...assignmentOptions, label: "Prompt", save: storeWith(assignPromptToDataset) });
  const ontologyChoice = useAssignmentChoice({ ...assignmentOptions, label: "Ontology", save: storeWith(assignOntologyToDataset) });
  const selectedModelId = modelChoice.selected;
  const selectedPromptName = promptChoice.selected;
  const selectedOntologyKey = ontologyChoice.selected;
  // Builds and uploads read the three choices, so they are blocked until the
  // stored configuration has been read (before that the choices are unknown) and
  // until none is still being stored. Blocked means the buttons are disabled and
  // files dropped meanwhile are refused with a message; nothing is queued.
  const choicesSaving = modelChoice.saving || promptChoice.saving || ontologyChoice.saving;
  const choicesNotReady = choicesSaving || configStatus !== "loaded";
  const alsoUsedElsewhere = (assignments: Record<string, string>, value: string, what: string): string => {
    const others = datasetsAssignedTo(assignments, value).filter((id) => id !== datasetId).length;
    if (others === 0) return "";
    return ` ${others} other ${others === 1 ? "dataset uses" : "datasets use"} this ${what}; they go back to the default ${what} and their graphs need a rebuild.`;
  };
  const { load: loadModel } = modelChoice;
  const { load: loadPrompt } = promptChoice;
  const { load: loadOntology } = ontologyChoice;
  const { setConfirmed: setStoredPrompt } = promptChoice;
  const { setConfirmed: setStoredOntology } = ontologyChoice;

  // Load graph models and assignment from backend config. A read that fails is
  // shown as a failure: taking it for "nothing is assigned" would show Automatic
  // and let a build run without the dataset's model, prompt and ontology.
  useEffect(() => {
    if (!cogniInstance || isInitializing) return;
    // A read for a dataset or workspace the page has since left must not land.
    let cancelled = false;
    setConfigStatus("loading");
    fetchGraphModelsConfig(cogniInstance).then((cfg) => {
      if (cancelled) return;
      setGraphModels(cfg.models);
      setCustomPrompts(cfg.customPrompts ?? {});
      const assignedModel = findModelForDataset(cfg.models, datasetId);
      loadModel(assignedModel?.id ?? null);
      const assignedPrompt = findPromptForDataset(cfg.promptAssignments ?? {}, datasetId);
      loadPrompt(assignedPrompt);
      const assignedOntology = findOntologyForDataset(cfg.ontologyAssignments ?? {}, datasetId);
      loadOntology(assignedOntology);
      if (cfg.outdatedDatasets?.includes(datasetId)) setGraphOutdated(true);
      setStoredAssignments({ prompts: cfg.promptAssignments ?? {}, ontologies: cfg.ontologyAssignments ?? {} });
      setConfigStatus("loaded");
    }).catch((err) => {
      captureException(err, { context: "dataset-detail.load-graph-config" });
      if (!cancelled) setConfigStatus("failed");
    });
    listOntologies(cogniInstance).then((list) => { if (!cancelled) setOntologies(list); }).catch((err) =>
      captureException(err, { context: "dataset-detail.list-ontologies" }));
    return () => { cancelled = true; };
  }, [datasetId, cogniInstance, isInitializing, configAttempt, loadModel, loadPrompt, loadOntology]);

  // Dropdown open/close + outside-click handling lives inside
  // MemoryCustomizationBar; the page only owns selection + persistence.

  async function handleDeleteOntology(key: string): Promise<void> {
    if (!cogniInstance) return;
    setDeletingOntology(true);
    try {
      await deleteOntology(cogniInstance, key);
      // The ontology is gone either way. If the datasets that pointed at it cannot
      // be updated, it is deleted but still assigned there, this one included, so
      // the selection keeps showing it: picking another one then stores a change.
      const unassigned = await removeOntologyAssignments(cogniInstance, key).then(() => true, (err: unknown) => {
        captureException(err, { context: "dataset-detail.remove-ontology-assignments", datasetId, key });
        return false;
      });
      setOntologies((prev) => { const next = { ...prev }; delete next[key]; return next; });
      if (unassigned) {
        setStoredAssignments((current) => ({ ...current, ontologies: withoutValue(current.ontologies, key) }));
        if (selectedOntologyKey === key) {
          setStoredOntology(null);
          noteStoredChange();
        }
      }
      setConfirmDeleteOntologyKey(null);
      // storedAssignments is what was loaded; a pick made on this page since is only in the selection.
      const knownInUse = selectedOntologyKey === key || datasetsAssignedTo(storedAssignments.ontologies, key).length > 0;
      notifications.show(ontologyDeletedNotice(key, unassigned, knownInUse));
    } catch (err) {
      notifications.show({ title: "Delete failed", message: err instanceof Error ? err.message : String(err), color: "red" });
    } finally {
      setDeletingOntology(false);
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
      setStoredAssignments((current) => ({ ...current, prompts: withoutValue(current.prompts, name) }));
      setCustomPrompts((prev) => { const next = { ...prev }; delete next[name]; return next; });
      if (selectedPromptName === name) {
        setStoredPrompt(null);
        noteStoredChange();
      }
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
      const wordingChanged = name in customPrompts && customPrompts[name] !== editingPromptText;
      await saveCustomPrompt(cogniInstance, name, editingPromptText);
      setCustomPrompts((prev) => ({ ...prev, [name]: editingPromptText }));
      setShowPromptEditor(false);
      // The stored configuration already flags this dataset; this is the banner,
      // and it keeps a rebuild that started before from clearing that flag.
      if (wordingChanged && name === selectedPromptName) noteStoredChange();
      // Saving the text does not assign it to this dataset; selecting does.
      promptChoice.select(name);
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
      // Datasets that still point at this key (a delete that could not unassign
      // it) build from the new file now. This one is told here: re-selecting the
      // key it already has would store nothing.
      flagDatasetsUsingOntology(cogniInstance, key).catch((err) =>
        captureException(err, { context: "dataset-detail.flag-ontology-users", datasetId, key }));
      if (selectedOntologyKey === key) noteStoredChange();
      else ontologyChoice.select(key);
      const updated = await listOntologies(cogniInstance);
      setOntologies(updated);
      setShowUploadOntologyModal(false);
      notifications.show({ title: "Ontology uploaded", message: `"${key}" is ready to use.`, color: "green", autoClose: 4000 });
    } catch (err) {
      notifications.show({ title: "Upload failed", message: err instanceof Error ? err.message : String(err), color: "red" });
      throw err;
    }
  }

  function handleSelectModel(modelId: string | null) {
    trackEvent({ pageName: "Dataset Detail", eventName: "graph_model_selected", additionalProperties: { dataset_id: datasetId, model_id: modelId ?? "automatic" } });
    modelChoice.select(modelId);
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
      const newModel: GraphModel = {
        id: newModelId,
        name: `${datasetName} Schema`,
        schema: modelSchema,
        createdAt: now,
        updatedAt: now,
        status: "draft",
        assignedDatasets: [datasetId],
      };
      await addGraphModelForDataset(cogniInstance, newModel, datasetId);
      noteStoredChange();
      // Store in sessionStorage so the editor can load immediately
      sessionStorage.setItem(`graph-model-${newModelId}`, JSON.stringify(newModel));
      setShowCreateModelModal(false);
      if (isMounted()) router.push(`/graph-models/${newModelId}`);
    } catch (err) {
      console.error("Failed to create model:", err);
      notifications.show({ title: "Failed", message: "Could not create graph model.", color: "red" });
    }
  }

  const loadFiles = useCallback(async () => {
    await loadFilePage(datasetId);
  }, [loadFilePage, datasetId]);

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

  const { statusDetails, refetch: refetchStatuses } = useDatasetStatuses(!isInitializing);
  const processingCounts = useDatasetProcessing(datasetId);
  const completionById = useMemo(
    () => new Map(processingCounts.data?.items.map((item) => [item.id, item.completed])),
    [processingCounts.data],
  );

  useEffect(() => {
    const detail = statusDetails[datasetId];
    if (!detail) {
      if (files.length === 0) setDatasetStatus("empty");
      else if (graphOutdated) setDatasetStatus("outdated");
      else setDatasetStatus("ready");
      setProcessing(false);
      return;
    }
    if (detail.status === "DATASET_PROCESSING_COMPLETED") {
      setDatasetStatus(graphOutdated ? "outdated" : "ready");
      setProcessing(false);
    } else if (detail.status === "DATASET_PROCESSING_ERRORED") {
      setDatasetStatus(detail.reason === INSUFFICIENT_CREDITS_REASON ? "failed_insufficient_credits" : "failed");
      setProcessing(false);
    } else if (detail.status === "DATASET_PROCESSING_STARTED" || detail.status === "DATASET_PROCESSING_INITIATED") {
      setDatasetStatus("processing");
      setProcessing(true);
    }
  }, [statusDetails, datasetId, graphOutdated, files.length]);

  async function handleUpload(newFiles: FileList | File[]) {
    if (!cogniInstance) return;
    if (choicesNotReady) {
      notifications.show(configStatus === "failed"
        ? { title: "Settings not loaded", message: "This dataset's model, prompt and ontology could not be read, so nothing is built or uploaded with them. Try again.", color: "red" }
        : { title: "Saving your settings", message: "Try again in a moment.", color: "yellow" });
      return;
    }
    const filesArray = Array.from(newFiles);

    if (filesArray.length > MAX_FILES_PER_UPLOAD) {
      notifications.show({
        title: "Too many files",
        message: `You selected ${filesArray.length} files. Please upload ${MAX_FILES_PER_UPLOAD} or fewer at a time.`,
        color: "red",
      });
      return;
    }

    if (!(await confirmUpload(filesArray))) return;

    const totalBytes = filesArray.reduce((sum, f) => sum + f.size, 0);
    const fileTypes = filesArray.map((f) => f.type || "unknown");

    trackEvent({
      pageName: "Dataset Detail",
      eventName: "dataset_upload_started",
      additionalProperties: {
        dataset_id: datasetId,
        file_count: String(filesArray.length),
        total_bytes: String(totalBytes),
        file_types: joinDistinct(fileTypes),
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
            file_types: joinDistinct(fileTypes),
            duration_ms: String(ctx.durationMs),
            error_name: errorName,
            error_message: errorMessage,
          },
        });
        if (isInsufficientCreditsError(error)) {
          // The pod interceptor (services/http/pod.ts) already opened the
          // global insufficient-credits modal — avoid a second, conflicting
          // "upload failed" toast for the same event.
          return;
        }
        if (errorName === "UploadTimeoutError") {
          // Same contract as the brains list (useBrainsData): name elapsed
          // time, never imply the files were lost, and never imply this page
          // is still working on the remainder — upload() has already returned.
          // The pending files live in the upload session, so a reload is what
          // actually finishes them.
          notifications.show({
            title: "Upload stalled",
            message:
              ctx.filesUploaded > 0
                ? `Uploaded ${ctx.filesUploaded} of ${filesArray.length} files before the connection stalled. The rest haven't been sent — reload the page to pick up where this left off.`
                : "The upload stalled before any files were sent. Reload the page to resume it, or try again.",
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
      trackEvent({ pageName: "Dataset Detail", eventName: "dataset_file_deleted", additionalProperties: { dataset_id: datasetId, file_name: deletedFile ? (connectorRepository(deletedFile) ?? deletedFile.name) : fileId } });
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
    const changesAtStart = storedChanges.current;
    setGraphOutdated(false);
    setDatasetStatus("processing");
    try {
      await cognifyDataset({ id: datasetId, name: datasetName, data: [], status: "processing" }, cogniInstance, getCognifyOptions());
      trackEvent({ pageName: "Dataset Detail", eventName: "dataset_recognified", additionalProperties: { dataset_id: datasetId } });
      // A change stored while the build was starting raised the flag again, and
      // this build did not use it: leave the flag for the next rebuild.
      if (storedChanges.current === changesAtStart) {
        clearDatasetOutdated(cogniInstance, datasetId).catch((err) =>
          captureException(err, { context: "dataset-detail.clear-outdated-flag", datasetId }));
      }
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
      if (isMounted()) router.push("/datasets");
    } catch (err) {
      console.error("Delete dataset failed:", err);
      setDeleting(false);
      setShowDeleteConfirm(false);
    }
  }

  const filtered = search
    ? files.filter((f) => `${f.name} ${connectorRepository(f) ?? ""}`.toLowerCase().includes(search.toLowerCase()))
    : files;

  if ((loading && files.length === 0) || isInitializing) {
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
            {filesTotal === null ? `${fmtCount(files.length)} loaded` : fmtCount(filesTotal)} documents
            {datasetStatus === "processing" || processing ? (
              <span style={{ display: "inline-flex", alignItems: "center", gap: 4, color: "#6510F4", fontWeight: 500 }}>
                · <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#6510F4" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ animation: "spin 1s linear infinite" }}><path d="M21 12a9 9 0 11-6.219-8.56" /></svg>
                Processing
              </span>
            ) : datasetStatus === "failed_insufficient_credits" ? (
              <Link href="/billing" style={{ color: "#BC9BFF", fontWeight: 500, textDecoration: "underline", textUnderlineOffset: 2 }}>· Failed — insufficient credits</Link>
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
              disabled={syncing || choicesNotReady}
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
            disabled={combinedUploadProgress.stage !== "idle" || choicesNotReady}
            className="cursor-pointer hover:bg-[#5A0ED6]"
            style={{ background: "#6510F4", color: "#fff", border: "none", borderRadius: 6, padding: "8px 16px", fontSize: 13, fontWeight: 500, display: "flex", alignItems: "center", gap: 6 }}
          >
          {combinedUploadProgress.stage !== "idle" ? (
            <Loader size={14} color="#fff" />
          ) : (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" /></svg>
          )}
          {combinedUploadProgress.stage === "idle"
            ? "Upload files"
            : combinedUploadProgress.stage === "estimating"
              ? "Estimating cost..."
              : uploadStage === "processing"
                ? "Building knowledge graph..."
                : `Uploading ${uploadProgress.filesTotal} ${uploadProgress.filesTotal === 1 ? "file" : "files"}...`}
          </button>
        </div>
      </div>

      {combinedUploadProgress.stage !== "idle" && (
        <div style={{ margin: "0 0 12px" }}>
          <UploadProgressBar progress={combinedUploadProgress} />
        </div>
      )}

      <LowBalanceWarningModal
        warning={pendingLowBalanceWarning}
        onCancel={cancelLowBalanceUpload}
      />

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
          title="Delete dataset"
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
          message={<>Are you sure you want to delete <strong>{editingPromptName.trim() || "this prompt"}</strong>?{alsoUsedElsewhere(storedAssignments.prompts, editingPromptName.trim(), "prompt")} This action cannot be undone.</>}
          onConfirm={handleConfirmDeletePrompt}
          onCancel={() => setConfirmDeletePrompt(false)}
          busy={deletingPrompt}
        />
      )}

      {/* Delete ontology confirmation modal */}
      {confirmDeleteOntologyKey && (
        <DeleteConfirmModal
          title="Delete ontology"
          message={<>Are you sure you want to delete <strong>{confirmDeleteOntologyKey}</strong>?{alsoUsedElsewhere(storedAssignments.ontologies, confirmDeleteOntologyKey, "ontology")} This action cannot be undone.</>}
          onConfirm={() => handleDeleteOntology(confirmDeleteOntologyKey)}
          onCancel={() => setConfirmDeleteOntologyKey(null)}
          busy={deletingOntology}
        />
      )}

      {/* Delete file confirmation modal */}
      {deleteFileTarget && (
        <DeleteConfirmModal
          title="Delete file"
          message={<>Are you sure you want to delete <strong>{connectorRepository(deleteFileTarget) ?? decodeFilename(deleteFileTarget.name)}</strong>? This action cannot be undone.</>}
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
          placeholder={hasMore ? "Search loaded files..." : "Search files..."}
          style={{ flex: 1, border: "none", outline: "none", fontSize: 14, color: "#EDECEA", background: "transparent", fontFamily: "inherit" }}
        />
        {search && <button onClick={() => setSearch("")} className="cursor-pointer" style={{ background: "none", border: "none", color: "rgba(237,236,234,0.35)", fontSize: 14 }}>&#10005;</button>}
      </div>

      {/* Memory customization */}
      {configStatus === "failed" && (
        <div role="alert" style={{ background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: 8, padding: "12px 16px", display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ flex: 1, fontSize: 13, color: "#F87171" }}>
            This dataset&apos;s graph model, prompt and ontology could not be loaded. Uploads and builds are paused so they do not run without them.
          </span>
          <button
            onClick={() => setConfigAttempt((attempt) => attempt + 1)}
            className="cursor-pointer"
            style={{ background: "rgba(239,68,68,0.2)", border: "1px solid rgba(239,68,68,0.35)", borderRadius: 6, padding: "6px 14px", fontSize: 13, fontWeight: 500, color: "#F87171", whiteSpace: "nowrap", fontFamily: "inherit" }}
          >
            Try again
          </button>
        </div>
      )}
      <MemoryCustomizationBar
        graphModels={graphModels}
        selectedModelId={selectedModelId}
        onSelectModel={handleSelectModel}
        onEditModel={(id) => router.push(`/graph-models/${id}`)}
        onCreateModel={() => setShowCreateModelModal(true)}
        customPrompts={customPrompts}
        selectedPromptName={selectedPromptName}
        onSelectPrompt={promptChoice.select}
        onEditPrompt={(name, text) => { setEditingPromptName(name); setEditingPromptText(text); setShowPromptEditor(true); }}
        onCreatePrompt={() => setShowCreatePromptModal(true)}
        ontologies={ontologies}
        selectedOntologyKey={selectedOntologyKey}
        onSelectOntology={ontologyChoice.select}
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

      <ProcessingSummary
        data={processingCounts.data}
        error={processingCounts.isError}
        running={processing || datasetStatus === "processing" || datasetStatus === "pending"}
        failed={datasetStatus === "failed" || datasetStatus === "failed_insufficient_credits"}
        onRefresh={() => { void processingCounts.refetch(); void refetchStatuses(); }}
      />

      {/* Outdated graph banner */}
      {graphOutdated && files.length > 0 && !processing && datasetStatus !== "processing" && (
        <div style={{ background: "rgba(245,158,11,0.1)", border: "1px solid rgba(245,158,11,0.3)", borderRadius: 8, padding: "12px 16px", display: "flex", alignItems: "center", gap: 12 }}>
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" style={{ flexShrink: 0 }}><path d="M8 1L1 14h14L8 1z" fill="rgba(245,158,11,0.25)" stroke="#F59E0B" strokeWidth="1" /><text x="8" y="12" textAnchor="middle" fontSize="9" fontWeight="700" fill="#FBBF24">!</text></svg>
          <span style={{ flex: 1, fontSize: 13, color: "#FBBF24" }}>
            Knowledge graph is outdated. The graph model, prompt or ontology was changed since the last build.
          </span>
          <button
            onClick={() => rebuildGraph(() => { setGraphOutdated(true); setDatasetStatus("outdated"); })}
            disabled={choicesNotReady}
            className="cursor-pointer hover:bg-yellow-500/20"
            style={{ background: "rgba(245,158,11,0.2)", border: "1px solid rgba(245,158,11,0.35)", borderRadius: 6, padding: "6px 14px", fontSize: 13, fontWeight: 500, color: "#FBBF24", whiteSpace: "nowrap", fontFamily: "inherit" }}
          >
            Rebuild graph
          </button>
        </div>
      )}

      {/* Failed build banner — insufficient credits (retrying without topping up would just fail the same way) */}
      {datasetStatus === "failed_insufficient_credits" && !processing && (
        <div style={{ background: "rgba(101,16,244,0.1)", border: "1px solid rgba(188,155,255,0.35)", borderRadius: 8, padding: "12px 16px", display: "flex", alignItems: "center", gap: 12 }}>
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" style={{ flexShrink: 0 }}><path d="M8 1L1 14h14L8 1z" fill="rgba(188,155,255,0.25)" stroke="#BC9BFF" strokeWidth="1" /><text x="8" y="12" textAnchor="middle" fontSize="9" fontWeight="700" fill="#BC9BFF">!</text></svg>
          <span style={{ flex: 1, fontSize: 13, color: "#BC9BFF" }}>
            Building the knowledge graph failed — your workspace ran out of credits mid-run. Your files are still here.
          </span>
          <Link
            href="/billing"
            className="cursor-pointer hover:bg-[#6510F4]/20"
            style={{ background: "rgba(101,16,244,0.2)", border: "1px solid rgba(188,155,255,0.35)", borderRadius: 6, padding: "6px 14px", fontSize: 13, fontWeight: 500, color: "#BC9BFF", whiteSpace: "nowrap", textDecoration: "none" }}
          >
            Go to billing
          </Link>
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
            disabled={choicesNotReady}
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
        files={filtered.map((file) => ({ ...file, completed: completionById.get(file.id) }))}
        blobDescriptions={blobDescriptions}
        search={search}
        loadError={filesError}
        onDelete={(id) => setDeleteFileTarget(filtered.find((f) => f.id === id) ?? null)}
        onUploadClick={() => fileInputRef.current?.click()}
        onRetry={loadFiles}
        deletingId={deletingFileId}
      />
      {files.length > 0 && (
        <ScrollLoader loaded={files.length} total={filesTotal} hasMore={hasMore}
          maxLoaded={MAX_RENDERED_ROWS} busy={loading} error={filesError}
          onLoadMore={loadMore} noun="files" autoLoad={!search} />
      )}

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
