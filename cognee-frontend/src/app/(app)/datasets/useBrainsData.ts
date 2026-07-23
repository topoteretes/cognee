"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { notifications } from "@mantine/notifications";
import { captureException, recordUploadSuccess, recordUploadFailure } from "@/utils/monitoring";
import { useCogniInstance, useTenant } from "@/modules/tenant/TenantProvider";
import { useFilter } from "@/ui/layout/FilterContext";
import getDatasets from "@/modules/datasets/getDatasets";
import getDatasetData from "@/modules/datasets/getDatasetData";
import createDataset from "@/modules/datasets/createDataset";
import deleteDataset from "@/modules/datasets/deleteDataset";
import deleteDatasetData from "@/modules/datasets/deleteDatasetData";
import cognifyDataset from "@/modules/datasets/cognifyDataset";
import { type DatasetProcessingStatus } from "@/modules/datasets/pollDatasetStatus";
import { useDatasetStatuses } from "@/modules/datasets/useDatasetStatuses";
import { trackEvent } from "@/modules/analytics";
import { loadGraphModelsConfig } from "@/modules/configuration/userConfiguration";
import { buildCognifyOptionsForDataset } from "@/modules/configuration/buildCognifyOptionsForDataset";
import { useBrainUpload } from "@/modules/ingestion/useBrainUpload";
import { MAX_FILES_PER_UPLOAD } from "@/modules/ingestion/uploadLimits";
import { trackUploadStarted, trackUploadFailed, trackFilesUploaded, trackProcessingFailed } from "./brainUploadAnalytics";
import { describeProcessingError } from "./processingErrorMessage";
import { applyCreateBrainTemplate, type CreateBrainTemplateKey } from "./createBrainTemplates";
import { mapProcessingStatus, type DatasetRaw, type FileEntry, type DisplayStatus, type Dataset, type UseBrainsDataResult } from "./brainsTypes";

export type { FileEntry, DisplayStatus, Dataset, UseBrainsDataResult } from "./brainsTypes";

// Owns all data and interaction state for the brains (datasets) finder:
// loading the dataset list + per-dataset doc counts, live status polling,
// selection, upload/paste/delete flows, and the create/delete/share modal
// state. The page component is a pure view over what this returns.
export function useBrainsData(): UseBrainsDataResult {
  const { cogniInstance, isInitializing } = useCogniInstance();
  const { tenant } = useTenant();
  const { datasets: contextDatasets, refreshDatasets: refreshFilterDatasets } = useFilter();
  // Read as a one-shot fallback inside loadDatasets when getDatasets() itself
  // fails, not as a live data source — a ref keeps that read fresh without
  // making loadDatasets depend on (and re-run for) every FilterContext change.
  const contextDatasetsRef = useRef(contextDatasets);
  contextDatasetsRef.current = contextDatasets;

  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [datasetsError, setDatasetsError] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [outdatedDatasets, setOutdated] = useState<Set<string>>(new Set());

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedDocs, setSelectedDocs] = useState<FileEntry[]>([]);
  const [docsLoading, setDocsLoading] = useState(false);
  const [docsError, setDocsError] = useState(false);

  const { isUploading, stage: uploadStage, upload } = useBrainUpload(cogniInstance);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [canRetryBuild, setCanRetryBuild] = useState(false);

  const [showCreateModal, setShowCreate] = useState(false);
  const [newName, setNewName] = useState("");
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState("");
  const [deleteTarget, setDeleteTarget] = useState<Dataset | null>(null);
  const [shareTarget, setShareTarget] = useState<Dataset | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [deleteDocTarget, setDeleteDocTarget] = useState<FileEntry | null>(null);
  const [deletingDocId, setDeletingDocId] = useState<string | null>(null);
  const [showPasteModal, setShowPasteModal] = useState(false);
  const [pasteText, setPasteText] = useState("");
  const [pasting, setPasting] = useState(false);

  const { statuses } = useDatasetStatuses(datasets.length > 0);

  const loadDatasets = useCallback(async (): Promise<void> => {
    if (!cogniInstance) return;
    try {
      let list: DatasetRaw[];
      try {
        const fetched = await getDatasets(cogniInstance);
        // A non-array body is just as much a failed fetch as a thrown error —
        // treat it the same way instead of silently rendering an empty list.
        if (!Array.isArray(fetched)) throw new Error("Unexpected /v1/datasets response shape");
        list = fetched;
        setDatasetsError(false);
      } catch (err) {
        captureException(err, { stage: "load_datasets" });
        // Fall back to FilterContext's list (which can itself be empty or
        // stale) rather than leaving the page with nothing — but flag the
        // failure so the caller can distinguish this from a dataset-free
        // account instead of rendering a false "no brains yet" empty state.
        list = contextDatasetsRef.current as DatasetRaw[];
        setDatasetsError(true);
        notifications.show({
          title: list.length > 0 ? "Couldn't refresh your brains" : "Couldn't load your brains",
          message: list.length > 0
            ? "Showing the last known list — it may be out of date."
            : "This can happen while a large upload is still processing. Try refreshing in a moment.",
          color: "yellow",
        });
      }
      const initial = list.map((ds) => ({ ...ds, documents: -1, status: "loading" as DisplayStatus }));
      setDatasets(initial);
      setLoading(false);

      const statusResp = await cogniInstance.fetch("/v1/datasets/status").catch((err) => {
        console.error("Failed to fetch dataset processing status:", err);
        return null;
      });
      const statusData: Record<string, DatasetProcessingStatus> = statusResp?.ok ? await statusResp.json() : {};

      for (const ds of list) {
        getDatasetData(ds.id, cogniInstance)
          .then((data) => {
            const count = Array.isArray(data) ? data.length : 0;
            setDatasets((prev) => prev.map((d) => d.id === ds.id ? { ...d, documents: count, status: mapProcessingStatus(statusData[ds.id], count) } : d));
          })
          .catch(() => {
            setDatasets((prev) => prev.map((d) => d.id === ds.id ? { ...d, documents: 0, status: mapProcessingStatus(statusData[ds.id], 0) } : d));
          });
      }
    } catch (err) {
      captureException(err, { stage: "load_datasets_unexpected" });
      setDatasets([]);
      setDatasetsError(true);
      setLoading(false);
    }
  }, [cogniInstance]);

  useEffect(() => {
    if (!cogniInstance || isInitializing) return;
    loadDatasets();
    loadGraphModelsConfig(cogniInstance)
      .then((cfg) => setOutdated(new Set(cfg.outdatedDatasets ?? [])))
      .catch((err) => { console.error("Failed to load graph models config:", err); });
  }, [cogniInstance, isInitializing, loadDatasets]);

  useEffect(() => {
    if (!cogniInstance || Object.keys(statuses).length === 0) return;
    let completedSelectedId: string | null = null;
    setDatasets((prev) =>
      prev.map((d) => {
        const raw = statuses[d.id];
        if (!raw) return d;
        const newStatus = mapProcessingStatus(raw, d.documents);
        if (newStatus === d.status) return d;
        // When the selected brain finishes, schedule a file list refresh
        if (d.id === selectedId && (d.status === "pending" || d.status === "running") && newStatus === "completed") {
          completedSelectedId = d.id;
        }
        return { ...d, status: newStatus };
      }),
    );
    if (completedSelectedId) {
      getDatasetData(completedSelectedId, cogniInstance)
        .then((docs) => {
          setSelectedDocs(Array.isArray(docs) ? docs : []);
          setDatasets((prev) => prev.map((d) => d.id === completedSelectedId ? { ...d, documents: Array.isArray(docs) ? docs.length : d.documents } : d));
        })
        .catch((err) => {
          console.error("Failed to fetch dataset documents:", err);
          setSelectedDocs([]);
        });
    }
  }, [statuses, cogniInstance, selectedId]);

  async function refreshSelectedDocs(id: string): Promise<void> {
    if (!cogniInstance) return;
    setDocsLoading(true);
    try {
      const data = await getDatasetData(id, cogniInstance);
      setSelectedDocs(Array.isArray(data) ? data : []);
      setDocsError(false);
    } catch {
      // Surface the fetch failure instead of rendering a false "no documents"
      // empty state.
      setDocsError(true);
    } finally {
      setDocsLoading(false);
    }
  }

  async function handleRefresh(): Promise<void> {
    setRefreshing(true);
    await Promise.all([loadDatasets(), selectedId ? refreshSelectedDocs(selectedId) : Promise.resolve()]);
    setRefreshing(false);
  }

  async function handleSelectDataset(id: string): Promise<void> {
    if (selectedId === id) return;
    setSelectedId(id);
    setSelectedDocs([]);
    setDocsError(false);
    await refreshSelectedDocs(id);
  }

  async function handleUploadFiles(files: File[]): Promise<void> {
    if (!cogniInstance || !selectedId || !files.length) return;
    const ds = datasets.find((d) => d.id === selectedId);
    if (!ds) return;

    if (files.length > MAX_FILES_PER_UPLOAD) {
      setUploadError(`You selected ${files.length} files. Please upload ${MAX_FILES_PER_UPLOAD} or fewer at a time.`);
      return;
    }

    const totalBytes = files.reduce((sum, f) => sum + f.size, 0);
    const fileTypes = files.map((f) => f.type || "unknown");

    setUploadError(null);
    setCanRetryBuild(false);
    trackUploadStarted({ datasetId: ds.id, fileCount: files.length, totalBytes, fileTypes });

    // Load the dataset's saved graph model/prompt/ontology so uploads from
    // this page respect the same customization as the detail page, instead
    // of always building with defaults (CLO-292).
    const options = await buildCognifyOptionsForDataset(cogniInstance, ds.id).catch((err) => {
      console.error("Failed to load graph settings for upload, using defaults:", err);
      return undefined;
    });

    // Shared by the success and processing-error paths below; only the
    // success path also flips the dataset to "running" (a processing error
    // means the build did NOT start running, even though the files landed).
    const fetchSelectedDocs = async (): Promise<FileEntry[]> => {
      const data = (await getDatasetData(ds.id, cogniInstance)) as FileEntry[];
      const list = Array.isArray(data) ? data : [];
      setSelectedDocs(list);
      return list;
    };

    const refreshDocs = async (): Promise<void> => {
      const list = await fetchSelectedDocs();
      setDatasets((prev) =>
        prev.map((d) =>
          d.id === ds.id ? { ...d, documents: list.length, status: "running" } : d,
        ),
      );
    };

    await upload({
      datasetId: ds.id,
      datasetName: ds.name,
      files,
      options,
      // Belt-and-suspenders: the check above already blocks this case, but if
      // that check is ever removed/changed the hook's own guard must still
      // surface an error instead of silently no-opping.
      onLimitExceeded: (selected) => {
        setUploadError(`You selected ${selected.length} files. Please upload ${MAX_FILES_PER_UPLOAD} or fewer at a time.`);
      },
      onUploadError: (error, ctx) => {
        const errorName = error instanceof Error ? error.name : "UnknownError";
        const errorMessage = error instanceof Error ? error.message : String(error);
        recordUploadFailure(errorName, ctx.durationMs);
        trackUploadFailed({ datasetId: ds.id, fileCount: files.length, totalBytes, fileTypes, durationMs: ctx.durationMs, errorName, errorMessage });
        if (errorName === "UploadTimeoutError") {
          setUploadError("The file took too long to process. Please try again with a smaller file.");
        } else {
          captureException(error, { datasetId: ds.id, fileCount: files.length, totalBytes, durationMs: ctx.durationMs });
          setUploadError(errorMessage || "Upload failed. Please try again.");
        }
      },
      onProcessed: async (ctx) => {
        await refreshDocs();
        recordUploadSuccess(ctx.durationMs, totalBytes, files.length);
        trackFilesUploaded({ datasetId: ds.id, fileCount: files.length, totalBytes, durationMs: ctx.durationMs });
      },
      onProcessingError: async (error, ctx) => {
        const errorMessage = error instanceof Error ? error.message : String(error);
        captureException(error, { datasetId: ds.id, fileCount: files.length, totalBytes, durationMs: ctx.durationMs, stage: "processing" });
        trackProcessingFailed({ datasetId: ds.id, fileCount: files.length, totalBytes, durationMs: ctx.durationMs, errorMessage });
        const { message, isTimeout } = describeProcessingError(error);
        setUploadError(message);
        setCanRetryBuild(!isTimeout);
        // Refresh the doc list anyway — the files are there even though the
        // build errored. Don't flip the dataset to "running" here (it isn't).
        try {
          await fetchSelectedDocs();
        } catch {
          // best-effort refresh only
        }
      },
    });
  }

  // Re-kicks the knowledge-graph build for the selected dataset after a
  // processing failure, without re-uploading files. Fire-and-forget like
  // DatasetDetailPage's rebuildGraph — the shared status poller (statuses)
  // picks up the new in-progress state on its own, so this doesn't need to
  // await the build to completion.
  async function handleRetryBuild(): Promise<void> {
    if (!cogniInstance || !selectedId) return;
    const ds = datasets.find((d) => d.id === selectedId);
    if (!ds) return;
    setUploadError(null);
    setCanRetryBuild(false);
    setDatasets((prev) => prev.map((d) => d.id === selectedId ? { ...d, status: "running" } : d));
    try {
      const options = await buildCognifyOptionsForDataset(cogniInstance, ds.id).catch(() => undefined);
      await cognifyDataset({ id: ds.id, name: ds.name, data: [], status: "processing" }, cogniInstance, options);
    } catch (err) {
      console.error("Failed to retry build:", err);
      setUploadError("Retrying the build failed. Please try again.");
      setCanRetryBuild(true);
    }
  }

  async function handleDeleteFile(docId: string): Promise<void> {
    if (!cogniInstance || !selectedId) return;
    setDeletingDocId(docId);
    try {
      await deleteDatasetData(selectedId, docId, cogniInstance);
      setSelectedDocs((prev) => prev.filter((d) => d.id !== docId));
      setDatasets((prev) => prev.map((d) => d.id === selectedId ? { ...d, documents: Math.max(0, d.documents - 1) } : d));
      setDeleteDocTarget(null);
    } catch (err) {
      console.error("Failed to delete file:", err);
    } finally {
      setDeletingDocId(null);
    }
  }

  async function handleDelete(ds: Dataset): Promise<void> {
    if (!cogniInstance) return;
    setDeletingId(ds.id);
    try {
      await deleteDataset(ds.id, cogniInstance);
      trackEvent({ pageName: "Brains", eventName: "dataset_deleted", additionalProperties: { dataset_id: ds.id, dataset_name: ds.name } });
      setDatasets((prev) => prev.filter((d) => d.id !== ds.id));
      if (selectedId === ds.id) { setSelectedId(null); setSelectedDocs([]); }
      setDeleteTarget(null);
      refreshFilterDatasets();
    } catch (err) {
      console.error("Failed to delete brain:", err);
    } finally {
      setDeletingId(null);
    }
  }

  async function handleCreate(templateKey: CreateBrainTemplateKey | null): Promise<void> {
    const trimmed = newName.trim();
    if (!trimmed || !cogniInstance) return;
    setCreateError("");
    if (trimmed.includes(".")) {
      setCreateError("Dataset name cannot contain periods.");
      return;
    }
    setCreating(true);
    try {
      // The input masks spaces to hyphens as the user types (CreateBrainModal)
      // for a readable name while typing; the backend gets the same name in
      // snake_case, matching the naming convention datasets are stored under.
      const backendName = trimmed.toLowerCase().replace(/[\s-]+/g, "_");
      const ds = await createDataset({ name: backendName }, cogniInstance, tenant?.tenant_id);
      trackEvent({ pageName: "Brains", eventName: "dataset_created", additionalProperties: { dataset_name: ds.name, template: templateKey ?? "blank" } });
      setDatasets((prev) => [...prev, { ...ds, documents: 0, status: "empty" as DisplayStatus }]);
      setSelectedId(ds.id);
      setSelectedDocs([]);
      setNewName(""); setCreateError(""); setShowCreate(false);
      refreshFilterDatasets();
      if (templateKey) {
        applyCreateBrainTemplate(cogniInstance, ds.id, templateKey).catch((err) => {
          console.error("Failed to apply brain template:", err);
        });
      }
    } catch (err) {
      console.error("Failed to create dataset:", err);
      setCreateError("Failed to create brain. Please try again.");
    } finally {
      setCreating(false);
    }
  }

  async function handlePasteText(): Promise<void> {
    if (!pasteText.trim() || !selectedId) return;
    setPasting(true);
    try {
      const blob = new Blob([pasteText], { type: "text/plain" });
      const file = new File([blob], `pasted-text-${Date.now()}.txt`, { type: "text/plain" });
      setShowPasteModal(false);
      setPasteText("");
      await handleUploadFiles([file]);
    } finally {
      setPasting(false);
    }
  }

  const selectedDataset = datasets.find((d) => d.id === selectedId) ?? null;

  return {
    isLoading: loading || isInitializing,
    datasets,
    datasetsError,
    selectedId,
    selectedDataset,
    selectedDocs,
    docsLoading,
    docsError,
    retryDocs: () => { if (selectedId) refreshSelectedDocs(selectedId); },
    outdatedDatasets,
    refreshing,
    isUploading,
    uploadStage,
    uploadError,
    canRetryBuild,
    setUploadError,
    showCreateModal,
    setShowCreate,
    newName,
    setNewName,
    creating,
    createError,
    setCreateError,
    deleteTarget,
    setDeleteTarget,
    shareTarget,
    setShareTarget,
    deletingId,
    deleteDocTarget,
    setDeleteDocTarget,
    deletingDocId,
    showPasteModal,
    setShowPasteModal,
    pasteText,
    setPasteText,
    pasting,
    handleRefresh,
    handleSelectDataset,
    handleUploadFiles,
    handleRetryBuild,
    handleDeleteFile,
    handleDelete,
    handleCreate,
    handlePasteText,
  };
}
