"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { notifications } from "@mantine/notifications";
import { captureException, recordUploadSuccess, recordUploadFailure } from "@/utils/monitoring";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import { SLOW_READ_TIMEOUT_MS } from "@/services/http";
import { useFilter } from "@/ui/layout/FilterContext";
import getDatasets from "@/modules/datasets/getDatasets";
import getDatasetDataCount from "@/modules/datasets/getDatasetDataCount";
import { MAX_RENDERED_ROWS } from "@/modules/datasets/maxRenderedRows";
import useDatasetDataPages from "@/modules/datasets/useDatasetDataPages";
import { useDatasetProcessing } from "@/modules/datasets/useDatasetProcessing";
import createDataset from "@/modules/datasets/createDataset";
import deleteDataset from "@/modules/datasets/deleteDataset";
import deleteDatasetData from "@/modules/datasets/deleteDatasetData";
import cognifyDataset from "@/modules/datasets/cognifyDataset";
import { useDatasetStatuses } from "@/modules/datasets/useDatasetStatuses";
import { normalizeDatasetStatusResponse, type DatasetStatusDetail } from "@/modules/datasets/datasetStatusDetail";
import { trackEvent } from "@/modules/analytics";
import { loadGraphModelsConfig } from "@/modules/configuration/userConfiguration";
import { buildCognifyOptionsForDataset } from "@/modules/configuration/buildCognifyOptionsForDataset";
import { useBrainUpload } from "@/modules/ingestion/useBrainUpload";
import { withEstimateStage } from "@/modules/ingestion/uploadProgress";
import { MAX_FILES_PER_UPLOAD } from "@/modules/ingestion/uploadLimits";
import { trackUploadStarted, trackUploadFailed, trackFilesUploaded, trackProcessingFailed } from "./brainUploadAnalytics";
import { isInsufficientCreditsError } from "@/utils/insufficientCredits";
import { useLowBalanceUploadWarning } from "@/modules/billing/useLowBalanceUploadWarning";
import { describeProcessingError } from "./processingErrorMessage";
import { applyCreateBrainTemplate, type CreateBrainTemplateKey } from "./createBrainTemplates";
import {
  PENDING_DATASET_PREFIX,
  isPendingDataset,
  mapProcessingStatus,
  dropResolvedPlaceholders,
  dedupeDatasetsById,
  type DatasetRaw,
  type FileEntry,
  type DisplayStatus,
  type Dataset,
  type UseBrainsDataResult,
} from "./brainsTypes";

export type { FileEntry, DisplayStatus, Dataset, UseBrainsDataResult } from "./brainsTypes";

// Owns all data and interaction state for the datasets finder:
// loading the dataset list + per-dataset doc counts, live status polling,
// selection, upload/paste/delete flows, and the create/delete/share modal
// state. The page component is a pure view over what this returns.
export function useBrainsData(): UseBrainsDataResult {
  const { cogniInstance, isInitializing } = useCogniInstance();
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
  const [sessionCreatedIds, setSessionCreatedIds] = useState<Set<string>>(new Set());

  // A fencing token, bumped at the start of every loadDatasets call and every
  // confirmed create. A reload's own GET response reflects pod state no later
  // than when that reload started — so a create confirmed at a later fence
  // value couldn't possibly be in it, and the reload must not use its own
  // (earlier) result to erase a dataset the create already confirmed. Once a
  // later reload starts (a fence value past the create's), that reload's own
  // result speaks for the dataset again, including a legitimate deletion.
  const fenceRef = useRef(0);
  const confirmedAtRef = useRef(new Map<string, number>());

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const selectedIdRef = useRef(selectedId);
  selectedIdRef.current = selectedId;
  const processingCounts = useDatasetProcessing(selectedId);
  const {
    data: selectedDocs, setData: setSelectedDocs, loading: docsLoading, total: docsTotal,
    error: docsError, hasMore: hasMoreDocs, load: loadDocs, loadMore: loadMoreDocs, reset: resetDocs,
  } = useDatasetDataPages<FileEntry>(cogniInstance, MAX_RENDERED_ROWS);

  const { isUploading, stage: uploadStage, progress: uploadProgress, upload } = useBrainUpload(cogniInstance);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [canRetryBuild, setCanRetryBuild] = useState(false);
  const { pendingWarning: pendingLowBalanceWarning, isEstimating, confirmUpload, cancel: cancelLowBalanceUpload } = useLowBalanceUploadWarning();

  // The cost estimate runs before upload() starts, so fold it into the same
  // progress the bar reads: estimate → upload → build all show in one place.
  const combinedUploadProgress = withEstimateStage(uploadProgress, isEstimating);

  const [showCreateModal, setShowCreate] = useState(false);
  const [newName, setNewName] = useState("");
  const [createError, setCreateError] = useState("");
  const [deleteTarget, setDeleteTarget] = useState<Dataset | null>(null);
  const [deleteDocTarget, setDeleteDocTarget] = useState<FileEntry | null>(null);
  const [deletingDocId, setDeletingDocId] = useState<string | null>(null);
  const [showPasteModal, setShowPasteModal] = useState(false);
  const [pasteText, setPasteText] = useState("");
  const [pasting, setPasting] = useState(false);

  const { statusDetails } = useDatasetStatuses(datasets.length > 0);

  const loadDatasets = useCallback(async (): Promise<void> => {
    if (!cogniInstance) return;
    // Captured before the request goes out: any create that confirms after
    // this point couldn't be reflected in what this call is about to fetch.
    const loadFence = ++fenceRef.current;
    try {
      let list: DatasetRaw[];
      try {
        // The list resolves access for every dataset, which is a build, not a lookup:
        // on a slow database (or an instance with many datasets) it outruns the default
        // read budget, and the page then said its datasets could not be reached.
        const fetched = await getDatasets(cogniInstance, undefined, SLOW_READ_TIMEOUT_MS);
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
        // account instead of rendering a false "no datasets yet" empty state.
        list = contextDatasetsRef.current as DatasetRaw[];
        setDatasetsError(true);
        notifications.show({
          title: list.length > 0 ? "Couldn't refresh your datasets" : "Couldn't load your datasets",
          message: list.length > 0
            ? "Showing the last known list — it may be out of date."
            : "This can happen while a large upload is still processing. Try refreshing in a moment.",
          color: "yellow",
        });
      }
      const initial = list.map((ds) => ({ ...ds, documents: -1, status: "loading" as DisplayStatus }));
      // A create in flight when a reload lands (handleRefresh, or the mount
      // load racing a fast first create) isn't in `list` yet — the pod hasn't
      // answered for it — so replacing wholesale would flicker the row away
      // until handleCreate's own update arrives. Carry the placeholder over,
      // and the same for a create that confirmed after this reload started
      // (the fence check): this reload's `list` was captured too early to
      // possibly contain it.
      setDatasets((prev) => {
        const notYetReflected = prev.filter((d) => {
          if (isPendingDataset(d.id)) return true;
          const confirmedAt = confirmedAtRef.current.get(d.id);
          return confirmedAt !== undefined && confirmedAt > loadFence;
        });
        return dedupeDatasetsById(dropResolvedPlaceholders([...notYetReflected, ...initial]));
      });
      setLoading(false);

      // The shared client's default GET timeout races a caller's own signal
      // via AbortSignal.any and wins if shorter — this status poll can cover
      // many datasets at once, so give it more headroom than the default.
      const statusResp = await cogniInstance
        .fetch("/v1/datasets/status?include_error_detail=true", { timeoutMs: 60_000 })
        .catch((err) => {
          console.error("Failed to fetch dataset processing status:", err);
          return null;
        });
      const statusData: Record<string, DatasetStatusDetail> = statusResp?.ok
        ? normalizeDatasetStatusResponse(await statusResp.json())
        : {};

      for (const ds of list) {
        getDatasetDataCount(ds.id, cogniInstance)
          .then((count) => {
            // count is null when getDatasetDataCount's page-through fallback
            // got capped — a resolved-but-unknown state, distinct from -1
            // ("still loading", see the initial state below). BrainList
            // renders each one differently.
            setDatasets((prev) => prev.map((d) => d.id === ds.id ? { ...d, documents: count, status: mapProcessingStatus(statusData[ds.id]?.status, count, statusData[ds.id]?.reason ?? null) } : d));
          })
          .catch(() => {
            // A failed count is unknown, not zero: 0 would render as a real
            // count and mark the dataset "empty".
            setDatasets((prev) => prev.map((d) => d.id === ds.id ? { ...d, documents: null, status: mapProcessingStatus(statusData[ds.id]?.status, null, statusData[ds.id]?.reason ?? null) } : d));
          });
      }
    } catch (err) {
      captureException(err, { stage: "load_datasets_unexpected" });
      setDatasets((prev) => prev.filter((d) => isPendingDataset(d.id)));
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
    if (!cogniInstance || Object.keys(statusDetails).length === 0) return;
    let completedSelectedId: string | null = null;
    setDatasets((prev) =>
      prev.map((d) => {
        const detail = statusDetails[d.id];
        if (!detail) return d;
        const newStatus = mapProcessingStatus(detail.status, d.documents, detail.reason);
        if (newStatus === d.status) return d;
        // When the selected dataset finishes, schedule a file list refresh
        if (d.id === selectedId && (d.status === "pending" || d.status === "running") && newStatus === "completed") {
          completedSelectedId = d.id;
        }
        return { ...d, status: newStatus };
      }),
    );
    if (completedSelectedId) {
      // The paged loader fetches the dataset total along with the first page.
      void loadDocs(completedSelectedId);
    }
  }, [statusDetails, cogniInstance, selectedId, loadDocs]);

  // The list is a page, so the badge's total comes from the loader's own count
  // of the selected dataset. Null is "not known yet": keep what the badge has.
  useEffect(() => {
    if (selectedId && docsTotal !== null) {
      setDatasets((prev) => prev.map((d) => d.id === selectedId ? { ...d, documents: docsTotal } : d));
    }
  }, [selectedId, docsTotal]);

  async function refreshSelectedDocs(id: string): Promise<void> {
    await loadDocs(id);
  }

  // A connector can import more rows while this page stays open. Reload the
  // visible page when the persisted total changes, which is not every cognify
  // completion, so paging is not reset on each one.
  const importedTotal = processingCounts.data?.total;
  const lastImportSnapshot = useRef<{ id: string | null; total: number | undefined }>({ id: null, total: undefined });
  useEffect(() => {
    const previous = lastImportSnapshot.current;
    lastImportSnapshot.current = { id: selectedId, total: importedTotal };
    const sameDataset = selectedId !== null && previous.id === selectedId;
    if (sameDataset && previous.total !== undefined && importedTotal !== undefined && previous.total !== importedTotal) {
      void loadDocs(selectedId);
    }
  }, [selectedId, importedTotal, loadDocs]);

  async function handleRefresh(): Promise<void> {
    setRefreshing(true);
    await Promise.all([
      loadDatasets(),
      selectedId ? refreshSelectedDocs(selectedId) : Promise.resolve(),
      selectedId ? processingCounts.refetch() : Promise.resolve(),
    ]);
    setRefreshing(false);
  }

  async function handleSelectDataset(id: string): Promise<void> {
    if (selectedId === id) return;
    setSelectedId(id);
    setSelectedDocs([]);
    await refreshSelectedDocs(id);
  }

  async function handleUploadFiles(files: File[]): Promise<void> {
    if (!files.length) return;
    if (!cogniInstance) {
      setUploadError("Your workspace isn't ready yet. Please wait a moment and try again.");
      return;
    }
    if (!selectedId) {
      setUploadError("Select a dataset before uploading files.");
      return;
    }
    const ds = datasets.find((d) => d.id === selectedId);
    if (!ds) return;

    if (files.length > MAX_FILES_PER_UPLOAD) {
      setUploadError(`You selected ${files.length} files. Please upload ${MAX_FILES_PER_UPLOAD} or fewer at a time.`);
      return;
    }

    if (!(await confirmUpload(files))) return;

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

    // The upload can outlive the selection: only a dataset that is still
    // selected has its list (and, with it, its total) loaded here.
    const isStillSelected = (): boolean => selectedIdRef.current === ds.id;

    // Shared by the success and processing-error paths below; only the
    // success path also updates the dataset's status.
    const fetchSelectedDocs = async (): Promise<void> => {
      if (isStillSelected()) await loadDocs(ds.id);
    };

    const refreshDocs = async (): Promise<void> => {
      // Any other dataset is counted separately; a failed count stays unknown
      // (undefined) and keeps the last known value, where a capped one (null)
      // is itself the answer.
      let count: number | null | undefined;
      if (isStillSelected()) await fetchSelectedDocs();
      else count = await getDatasetDataCount(ds.id, cogniInstance).catch(() => undefined);
      // "completed", NOT "running": onProcessed fires only after the status
      // poll reached COMPLETED (useBrainUpload.ts). Writing "running" here
      // left the row stuck on "Processing" forever when the shared status
      // query had already read COMPLETED — react-query's structural sharing
      // keeps the data's identity unchanged on identical refetches, so the
      // statusDetails effect never fired again to correct it.
      setDatasets((prev) =>
        prev.map((d) =>
          d.id === ds.id
            ? { ...d, documents: count === undefined ? d.documents : count, status: "completed" }
            : d,
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
        if (isInsufficientCreditsError(error)) {
          // The pod interceptor (services/http/pod.ts) already opened the
          // global insufficient-credits modal — avoid a second, conflicting
          // "upload failed" banner for the same event.
          return;
        }
        if (errorName === "UploadTimeoutError") {
          // Names elapsed time and never claims data loss (CLO-492): the abort
          // is client-side only, so the files may well have landed. Telling
          // someone to "try a smaller file" was wrong on both counts.
          //
          // It must not overclaim in the other direction either: upload()
          // returns straight after this callback, so nothing on THIS page is
          // still working on the remainder. The pending files are held in the
          // upload session, which is picked up on the next page load — so
          // reloading is the action that actually finishes them.
          setUploadError(
            ctx.filesUploaded > 0
              ? `Uploaded ${ctx.filesUploaded} of ${files.length} files before the connection stalled. The rest haven't been sent — reload the page to pick up where this left off.`
              : "The upload stalled before any files were sent. Reload the page to resume it, or try again.",
          );
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
    const datasetId = selectedId;
    setDeletingDocId(docId);
    try {
      await deleteDatasetData(datasetId, docId, cogniInstance);
      // The user may have moved to another dataset while the delete ran. Editing
      // the list now would also cancel that dataset's page request, because
      // every list edit supersedes whatever load is in flight.
      if (selectedIdRef.current === datasetId) {
        setSelectedDocs((prev) => prev.filter((d) => d.id !== docId));
      }
      // An unknown (null) count can't be decremented into a real number —
      // leave it unknown rather than fabricate one.
      setDatasets((prev) => prev.map((d) => d.id === datasetId ? { ...d, documents: d.documents === null ? null : Math.max(0, d.documents - 1) } : d));
      setDeleteDocTarget(null);
    } catch (err) {
      console.error("Failed to delete file:", err);
    } finally {
      setDeletingDocId(null);
    }
  }

  function handleDelete(ds: Dataset): void {
    if (!cogniInstance) return;
    // Optimistic delete: drop the dataset from the UI right away and let the
    // backend request complete in the background.
    setDatasets((prev) => prev.filter((d) => d.id !== ds.id));
    if (selectedId === ds.id) { setSelectedId(null); resetDocs(); }
    setDeleteTarget(null);
    refreshFilterDatasets();
    trackEvent({ pageName: "Brains", eventName: "dataset_deleted", additionalProperties: { dataset_id: ds.id, dataset_name: ds.name } });
    deleteDataset(ds.id, cogniInstance).catch((err) => {
      console.error("Failed to delete dataset:", err);
    });
  }

  async function handleCreate(templateKey: CreateBrainTemplateKey | null): Promise<boolean> {
    const trimmed = newName.trim();
    if (!trimmed || !cogniInstance) return false;
    setCreateError("");
    if (trimmed.includes(".")) {
      setCreateError("Dataset name cannot contain periods.");
      return false;
    }
    // The input masks spaces to hyphens as the user types (CreateBrainModal)
    // for a readable name while typing; the backend gets the same name in
    // snake_case, matching the naming convention datasets are stored under.
    const backendName = trimmed.toLowerCase().replace(/[\s-]+/g, "_");
    if (datasets.some((d) => d.name.toLowerCase().replace(/[\s-]+/g, "_") === backendName)) {
      setCreateError("A dataset with this name already exists.");
      return false;
    }
    // The dataset is listed at once, as one being created, and the dialog closes: the
    // pod takes seconds to make it, and the checks that can fail on the name are done.
    const placeholderId = `${PENDING_DATASET_PREFIX}${crypto.randomUUID()}`;
    setDatasets((prev) => [...prev, { id: placeholderId, name: backendName, documents: -1, status: "loading" as DisplayStatus }]);
    setNewName(""); setCreateError(""); setShowCreate(false);
    try {
      const ds = await createDataset({ name: backendName }, cogniInstance);
      trackEvent({ pageName: "Brains", eventName: "dataset_created", additionalProperties: { dataset_name: ds.name, template: templateKey ?? "blank" } });
      const created = { ...ds, documents: 0, status: "empty" as DisplayStatus };
      // Confirmed before the dataset is folded into the list, so a reload that
      // is already in flight (started earlier, resolving later) knows not to
      // erase it — see loadDatasets' fence check.
      confirmedAtRef.current.set(ds.id, ++fenceRef.current);
      setSessionCreatedIds((prev) => new Set(prev).add(ds.id));
      setDatasets((prev) => dedupeDatasetsById(dropResolvedPlaceholders([...prev, created])));
      setSelectedId(ds.id);
      resetDocs();
      refreshFilterDatasets();
      // Private until its owner shares it: nobody else is given access here.
      if (templateKey) {
        applyCreateBrainTemplate(cogniInstance, ds.id, templateKey).catch((err) => {
          console.error("Failed to apply dataset template:", err);
        });
      }
      return true;
    } catch (err) {
      console.error("Failed to create dataset:", err);
      // Back to the dialog with the name, so it can be tried again.
      setDatasets((prev) => prev.filter((d) => d.id !== placeholderId));
      setNewName(trimmed);
      setCreateError("Failed to create dataset. Please try again.");
      setShowCreate(true);
      return false;
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

  const datasetsWithProcessing = useMemo(
    () => datasets.map((d) => d.id === selectedId && processingCounts.data ? { ...d, processingCounts: processingCounts.data } : d),
    [datasets, selectedId, processingCounts.data],
  );
  const docsWithProcessing = useMemo(() => {
    const completionById = new Map(processingCounts.data?.items.map((item) => [item.id, item.completed]));
    return selectedDocs.map((doc) => ({ ...doc, completed: completionById.get(doc.id) }));
  }, [selectedDocs, processingCounts.data]);

  return {
    processingCounts: processingCounts.data,
    processingCountsError: processingCounts.isError,
    refreshProcessingCounts: () => { void processingCounts.refetch(); },
    isLoading: loading || isInitializing,
    datasets: datasetsWithProcessing,
    datasetsError,
    sessionCreatedIds,
    selectedId,
    selectedDataset,
    selectedDocs: docsWithProcessing,
    docsLoading,
    docsError,
    hasMoreDocs,
    loadMoreDocs,
    retryDocs: () => { if (selectedId) refreshSelectedDocs(selectedId); },
    outdatedDatasets,
    refreshing,
    isUploading,
    uploadStage,
    uploadProgress: combinedUploadProgress,
    uploadError,
    canRetryBuild,
    setUploadError,
    pendingLowBalanceWarning,
    cancelLowBalanceUpload,
    showCreateModal,
    setShowCreate,
    newName,
    setNewName,
    // The dialog closes as soon as a dataset is accepted; it is never left waiting.
    creating: false,
    createError,
    setCreateError,
    deleteTarget,
    setDeleteTarget,
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
