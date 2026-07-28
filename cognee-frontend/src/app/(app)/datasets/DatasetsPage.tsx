"use client";

import { captureException, recordUploadSuccess, recordUploadFailure } from "@/utils/monitoring";
import { useState, useEffect, useRef } from "react";
import { useCogniInstance, useTenant } from "@/modules/tenant/TenantProvider";
import PageLoading from "@/ui/elements/PageLoading";
import { useFilter } from "@/ui/layout/FilterContext";
import getDatasets from "@/modules/datasets/getDatasets";
import getDatasetData from "@/modules/datasets/getDatasetData";
import createDataset from "@/modules/datasets/createDataset";
import deleteDataset from "@/modules/datasets/deleteDataset";
import pollDatasetStatus, { type DatasetProcessingStatus } from "@/modules/datasets/pollDatasetStatus";
import { useDatasetStatuses } from "@/modules/datasets/useDatasetStatuses";
import { TrackPageView, trackEvent } from "@/modules/analytics";
import { loadGraphModelsConfig } from "@/modules/configuration/userConfiguration";
import rememberData from "@/modules/ingestion/rememberData";
import { MAX_FILES_PER_UPLOAD } from "@/modules/ingestion/uploadLimits";
import deleteDatasetData from "@/modules/datasets/deleteDatasetData";
import ShareDatasetModal from "@/ui/elements/ShareDatasetModal";
import SkeletonBar from "@/ui/elements/SkeletonBar";

interface DatasetRaw {
  id: string;
  name: string;
  createdAt?: string;
  ownerId?: string;
}

interface FileEntry {
  id: string;
  name: string;
  extension?: string;
  mimeType?: string;
  createdAt?: string;
  size?: number;
}

type DisplayStatus = "pending" | "running" | "completed" | "failed" | "empty" | "loading";

interface Dataset extends DatasetRaw {
  documents: number;
  status: DisplayStatus;
}

function mapProcessingStatus(raw: DatasetProcessingStatus | undefined, docCount: number): DisplayStatus {
  if (!raw) return docCount > 0 ? "completed" : "empty";
  if (raw === "DATASET_PROCESSING_COMPLETED") return "completed";
  if (raw === "DATASET_PROCESSING_ERRORED") return "failed";
  if (raw === "DATASET_PROCESSING_STARTED") return "running";
  if (raw === "DATASET_PROCESSING_INITIATED") return "pending";
  return docCount > 0 ? "completed" : "empty";
}

const STATUS_DOT: Record<DisplayStatus, string> = {
  pending:   "#F59E0B",
  running:   "#F59E0B",
  completed: "#22C55E",
  failed:    "#EF4444",
  empty:     "#D4D4D8",
  loading:   "#D4D4D8",
};

const EXT_META: Record<string, { fill: string; stroke: string; text: string; label: string }> = {
  pdf:  { fill: "#FEE2E2", stroke: "#EF4444", text: "#DC2626", label: "PDF" },
  docx: { fill: "#DBEAFE", stroke: "#3B82F6", text: "#2563EB", label: "DOC" },
  doc:  { fill: "#DBEAFE", stroke: "#3B82F6", text: "#2563EB", label: "DOC" },
  md:   { fill: "#F3F4F6", stroke: "#6B7280", text: "#374151", label: "MD"  },
  txt:  { fill: "#F3F4F6", stroke: "#9CA3AF", text: "#6B7280", label: "TXT" },
  csv:  { fill: "#DCFCE7", stroke: "#22C55E", text: "#16A34A", label: "CSV" },
  json: { fill: "#FEF3C7", stroke: "#D97706", text: "#B45309", label: "JSON"},
};

function decodeFilename(name: string): string {
  try {
    let decoded = name;
    let prev: string;
    do { prev = decoded; decoded = decodeURIComponent(decoded); } while (decoded !== prev);
    return decoded;
  } catch {
    return name;
  }
}

function getExtMeta(name: string, ext?: string) {
  const e = (ext || name.split(".").pop() || "").toLowerCase();
  return EXT_META[e] || { fill: "#F3F4F6", stroke: "#9CA3AF", text: "#6B7280", label: e.toUpperCase().slice(0, 4) || "FILE" };
}

function FileIcon({ fill, stroke, text, label }: { fill: string; stroke: string; text: string; label: string }) {
  const fs = label.length > 3 ? 4.5 : label.length > 2 ? 5 : 5.5;
  return (
    <svg width="16" height="20" viewBox="0 0 16 20" fill="none" style={{ flexShrink: 0 }}>
      <path d="M10 1H3a2 2 0 00-2 2v14a2 2 0 002 2h10a2 2 0 002-2V6l-5-5z" fill={fill} stroke={stroke} strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M10 1v5h5" stroke={stroke} strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
      <text x="8" y="14.5" textAnchor="middle" fontSize={fs} fontWeight="700" fill={text}>{label}</text>
    </svg>
  );
}

function FolderIcon() {
  return (
    <svg width="15" height="13" viewBox="0 0 15 13" fill="none" style={{ flexShrink: 0 }}>
      <path d="M1 3a1 1 0 011-1h3.5L7 4h7a1 1 0 011 1v6a1 1 0 01-1 1H2a1 1 0 01-1-1V3z"
        fill="#D4D4D8" fillOpacity="0.5"
        stroke="#A1A1AA"
        strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"
      />
    </svg>
  );
}

function formatSize(bytes?: number): string {
  if (bytes == null) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function PlusIcon() {
  return <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M7 1v12M1 7h12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" /></svg>;
}

function EmptyStateIcon() {
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
      <path d="M6 4a2 2 0 012-2h8l6 6v16a2 2 0 01-2 2H8a2 2 0 01-2-2V4z" stroke="#6C5CE7" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M14 2v6h6" stroke="#6C5CE7" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M14 16v-4M12 14h4" stroke="#6C5CE7" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

function formatDate(dateStr?: string): string {
  if (!dateStr) return "—";
  return new Date(dateStr).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

export default function DatasetsPage() {
  const { cogniInstance, isInitializing } = useCogniInstance();
  const { tenant } = useTenant();
  const { datasets: contextDatasets, refreshDatasets: refreshFilterDatasets } = useFilter();

  const [datasets, setDatasets]           = useState<Dataset[]>([]);
  const [loading, setLoading]             = useState(true);
  const [refreshing, setRefreshing]       = useState(false);
  const [outdatedDatasets, setOutdated]   = useState<Set<string>>(new Set());

  // Finder selection
  const [selectedId, setSelectedId]       = useState<string | null>(null);
  const [selectedDocs, setSelectedDocs]   = useState<FileEntry[]>([]);
  const [docsLoading, setDocsLoading]     = useState(false);

  // Upload
  const [isDragOver, setIsDragOver]       = useState(false);
  const [isUploading, setIsUploading]     = useState(false);
  const [uploadError, setUploadError]     = useState<string | null>(null);
  const fileInputRef                      = useRef<HTMLInputElement>(null);
  const dragCounter                       = useRef(0);

  // Modals
  const [showCreateModal, setShowCreate]  = useState(false);
  const [newName, setNewName]             = useState("");
  const [creating, setCreating]           = useState(false);
  const [createError, setCreateError]     = useState("");
  const [deleteTarget, setDeleteTarget]   = useState<Dataset | null>(null);
  const [shareTarget, setShareTarget]     = useState<Dataset | null>(null);
  const [deletingId, setDeletingId]       = useState<string | null>(null);
  const [deleteDocTarget, setDeleteDocTarget] = useState<FileEntry | null>(null);
  const [showPasteModal, setShowPasteModal] = useState(false);
  const [pasteText, setPasteText]           = useState("");
  const [pasting, setPasting]               = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const { statuses } = useDatasetStatuses(datasets.length > 0);

  useEffect(() => {
    if (!cogniInstance || isInitializing) return;
    loadDatasets();
    loadGraphModelsConfig(cogniInstance)
      .then((cfg) => setOutdated(new Set(cfg.outdatedDatasets ?? [])))
      .catch((err) => { console.error("Failed to load graph models config:", err); });
  }, [cogniInstance, isInitializing]);

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

  async function loadDatasets() {
    if (!cogniInstance) return;
    try {
      let list: DatasetRaw[];
      try {
        const fetched = await getDatasets(cogniInstance);
        list = Array.isArray(fetched) ? fetched : [];
      } catch {
        list = contextDatasets as DatasetRaw[];
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
    } catch {
      setDatasets([]);
      setLoading(false);
    }
  }

  async function refreshSelectedDocs(id: string) {
    setDocsLoading(true);
    try {
      const data = await getDatasetData(id, cogniInstance!);
      setSelectedDocs(Array.isArray(data) ? data : []);
    } catch {
      setSelectedDocs([]);
    } finally {
      setDocsLoading(false);
    }
  }

  async function handleSelectDataset(id: string) {
    if (selectedId === id) return;
    setSelectedId(id);
    setSelectedDocs([]);
    await refreshSelectedDocs(id);
  }

  async function handleUploadFiles(files: File[]) {
    if (!cogniInstance || !selectedId || !files.length) return;
    const ds = datasets.find((d) => d.id === selectedId);
    if (!ds) return;

    if (files.length > MAX_FILES_PER_UPLOAD) {
      setUploadError(`You selected ${files.length} files. Please upload ${MAX_FILES_PER_UPLOAD} or fewer at a time.`);
      return;
    }

    const totalBytes = files.reduce((sum, f) => sum + f.size, 0);
    const fileTypes = files.map((f) => f.type || "unknown");
    const uploadStartedAt = Date.now();

    setIsUploading(true);
    setUploadError(null);

    trackEvent({
      pageName: "Brains",
      eventName: "dataset_upload_started",
      additionalProperties: {
        dataset_id: ds.id,
        file_count: String(files.length),
        total_bytes: String(totalBytes),
        file_types: fileTypes.join(","),
      },
    });

    try {
      // Kick off ingestion in the background so the upload POST returns immediately.
      // The add itself is done once this call returns — anything that fails after
      // this point (status polling) must not be reported as an upload failure.
      await rememberData({ id: ds.id, name: ds.name }, files, cogniInstance, { runInBackground: true });
    } catch (err) {
      setIsUploading(false);

      const durationMs = Date.now() - uploadStartedAt;
      const errorName = err instanceof Error ? err.name : "UnknownError";
      const errorMessage = err instanceof Error ? err.message : String(err);

      recordUploadFailure(errorName, durationMs);
      trackEvent({
        pageName: "Brains",
        eventName: "dataset_upload_failed",
        additionalProperties: {
          dataset_id: ds.id,
          file_count: String(files.length),
          total_bytes: String(totalBytes),
          file_types: fileTypes.join(","),
          duration_ms: String(durationMs),
          error_name: errorName,
          error_message: errorMessage,
        },
      });

      if (errorName === "UploadTimeoutError") {
        setUploadError("The file took too long to process. Please try again with a smaller file.");
      } else {
        captureException(err, { datasetId: ds.id, fileCount: files.length, totalBytes, durationMs });
        setUploadError(errorMessage || "Upload failed. Please try again.");
      }
      return;
    }

    try {
      // Files were already added successfully by this point — this only tracks
      // knowledge-graph build progress, so its failure is reported separately.
      await pollDatasetStatus(ds.id, cogniInstance, { intervalMs: 5000 });
      const data = await getDatasetData(ds.id, cogniInstance) as FileEntry[];
      setSelectedDocs(Array.isArray(data) ? data : []);
      setDatasets((prev) => prev.map((d) => d.id === ds.id ? { ...d, documents: Array.isArray(data) ? data.length : d.documents, status: "running" } : d));

      const durationMs = Date.now() - uploadStartedAt;
      recordUploadSuccess(durationMs, totalBytes, files.length);
      trackEvent({
        pageName: "Brains",
        eventName: "dataset_files_uploaded",
        additionalProperties: {
          dataset_id: ds.id,
          file_count: String(files.length),
          total_bytes: String(totalBytes),
          duration_ms: String(durationMs),
        },
      });
    } catch (err) {
      const durationMs = Date.now() - uploadStartedAt;
      const errorMessage = err instanceof Error ? err.message : String(err);

      captureException(err, { datasetId: ds.id, fileCount: files.length, totalBytes, durationMs, stage: "processing" });
      trackEvent({
        pageName: "Brains",
        eventName: "dataset_processing_failed",
        additionalProperties: {
          dataset_id: ds.id,
          file_count: String(files.length),
          total_bytes: String(totalBytes),
          duration_ms: String(durationMs),
          error_message: errorMessage,
        },
      });
      setUploadError(`Files were added, but building the knowledge graph failed: ${errorMessage}`);

      // Refresh doc list anyway — the files are there even though processing errored.
      try {
        const data = await getDatasetData(ds.id, cogniInstance) as FileEntry[];
        setSelectedDocs(Array.isArray(data) ? data : []);
      } catch {
        // best-effort refresh only
      }
    } finally {
      setIsUploading(false);
    }
  }

  async function handleDeleteFile(docId: string) {
    if (!cogniInstance || !selectedId) return;
    try {
      await deleteDatasetData(selectedId, docId, cogniInstance);
      setSelectedDocs((prev) => prev.filter((d) => d.id !== docId));
      setDatasets((prev) => prev.map((d) => d.id === selectedId ? { ...d, documents: Math.max(0, d.documents - 1) } : d));
    } catch (err) {
      console.error("Failed to delete file:", err);
    }
  }

  const handleDelete = async (ds: Dataset) => {
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
  };

  const handleCreate = async () => {
    const trimmed = newName.trim();
    if (!trimmed || !cogniInstance) return;
    setCreateError("");
    if (trimmed.includes(" ") || trimmed.includes(".")) {
      setCreateError("Dataset name cannot contain spaces or periods.");
      return;
    }
    setCreating(true);
    try {
      const ds = await createDataset({ name: trimmed.toLowerCase() }, cogniInstance, tenant?.tenant_id);
      trackEvent({ pageName: "Brains", eventName: "dataset_created", additionalProperties: { dataset_name: ds.name } });
      setDatasets((prev) => [...prev, { ...ds, documents: 0, status: "empty" as DisplayStatus }]);
      setNewName(""); setCreateError(""); setShowCreate(false);
      refreshFilterDatasets();
    } catch (err) {
      console.error("Failed to create dataset:", err);
      setCreateError("Failed to create brain. Please try again.");
    } finally {
      setCreating(false);
    }
  };

  async function handlePasteText() {
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

  if (loading || isInitializing) {
    return (
      <><TrackPageView page="Brains" />
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%" }}>
          <span style={{ fontSize: 14, color: "rgba(237,236,234,0.55)" }}>Loading datasets…</span>
        </div>
      </>
    );
  }

  const selectedDataset = datasets.find((d) => d.id === selectedId) ?? null;

  return (
    <div style={{ display: "flex", flexDirection: "column", flex: 1, minHeight: 0, overflow: "hidden" }}>
      <TrackPageView page="Brains" />

      {/* ── Create modal ── */}
      {showCreateModal && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.5)", backdropFilter: "blur(4px)", WebkitBackdropFilter: "blur(4px)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }}
          onClick={() => { setShowCreate(false); setCreateError(""); }}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "rgba(15,15,15,0.92)", backdropFilter: "blur(16px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, padding: 24, width: 420, display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 16px 48px rgba(0,0,0,0.12)" }}>
            <h2 style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", margin: 0 }}>Create brain</h2>
            <p style={{ fontSize: 13, color: "rgba(237,236,234,0.55)", margin: 0 }}>Give your brain a name. You can upload documents after creation.</p>
            <input ref={inputRef} autoFocus type="text" value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") handleCreate(); }}
              placeholder="e.g. product-docs, sec-filings..."
              style={{ width: "100%", height: 40, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 8, paddingInline: 14, fontSize: 14, color: "#EDECEA", fontFamily: "inherit", outline: "none", boxSizing: "border-box" }}
              onFocus={(e) => { e.target.style.borderColor = "#6510F4"; e.target.style.boxShadow = "0 0 0 3px rgba(188,155,255,0.10)"; }}
              onBlur={(e)  => { e.target.style.borderColor = "rgba(255,255,255,0.12)"; e.target.style.boxShadow = "none"; }}
            />
            {createError && <p style={{ fontSize: 13, color: "#EF4444", margin: 0 }}>{createError}</p>}
            <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
              <button onClick={() => { setShowCreate(false); setNewName(""); setCreateError(""); }} className="cursor-pointer"
                style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "rgba(237,236,234,0.7)", fontFamily: "inherit" }}>Cancel</button>
              <button onClick={handleCreate} disabled={creating} className="cursor-pointer"
                style={{ background: newName.trim() ? "#6510F4" : "rgba(255,255,255,0.06)", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: newName.trim() ? "#fff" : "rgba(237,236,234,0.35)", fontFamily: "inherit" }}>
                {creating ? "Creating…" : "Create"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Delete document modal ── */}
      {deleteDocTarget && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.5)", backdropFilter: "blur(4px)", WebkitBackdropFilter: "blur(4px)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }}
          onClick={() => setDeleteDocTarget(null)}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "rgba(15,15,15,0.92)", backdropFilter: "blur(16px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, padding: 24, width: 420, display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 16px 48px rgba(0,0,0,0.12)" }}>
            <h2 style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", margin: 0 }}>Delete document</h2>
            <p style={{ fontSize: 13, color: "rgba(237,236,234,0.55)", margin: 0 }}>
              Are you sure you want to delete <strong>{decodeFilename(deleteDocTarget.name)}</strong>? This action cannot be undone.
            </p>
            <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
              <button onClick={() => setDeleteDocTarget(null)} className="cursor-pointer"
                style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "rgba(237,236,234,0.7)", fontFamily: "inherit" }}>Cancel</button>
              <button onClick={() => { handleDeleteFile(deleteDocTarget.id); setDeleteDocTarget(null); }} className="cursor-pointer"
                style={{ background: "#EF4444", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "#fff", fontFamily: "inherit" }}>
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Paste text modal ── */}
      {showPasteModal && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.5)", backdropFilter: "blur(4px)", WebkitBackdropFilter: "blur(4px)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }}
          onClick={() => { setShowPasteModal(false); setPasteText(""); }}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "rgba(15,15,15,0.92)", backdropFilter: "blur(16px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, padding: 24, width: 420, display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 16px 48px rgba(0,0,0,0.12)" }}>
            <h2 style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", margin: 0 }}>Paste text</h2>
            <p style={{ fontSize: 13, color: "rgba(237,236,234,0.55)", margin: 0 }}>Paste your text below. It will be added as a document to the selected brain.</p>
            <textarea
              autoFocus
              value={pasteText}
              onChange={(e) => setPasteText(e.target.value)}
              placeholder="Paste your text here…"
              rows={8}
              style={{ width: "100%", background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 8, padding: "10px 14px", fontSize: 14, color: "#EDECEA", fontFamily: "inherit", outline: "none", resize: "vertical", boxSizing: "border-box" }}
              onFocus={(e) => { e.target.style.borderColor = "#6510F4"; e.target.style.boxShadow = "0 0 0 3px rgba(188,155,255,0.10)"; }}
              onBlur={(e)  => { e.target.style.borderColor = "rgba(255,255,255,0.12)"; e.target.style.boxShadow = "none"; }}
            />
            <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
              <button onClick={() => { setShowPasteModal(false); setPasteText(""); }} className="cursor-pointer"
                style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "rgba(237,236,234,0.7)", fontFamily: "inherit" }}>Cancel</button>
              <button onClick={handlePasteText} disabled={!pasteText.trim() || pasting} className="cursor-pointer"
                style={{ background: pasteText.trim() ? "#6510F4" : "rgba(255,255,255,0.06)", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: pasteText.trim() ? "#fff" : "rgba(237,236,234,0.35)", fontFamily: "inherit" }}>
                {pasting ? "Adding…" : "Add"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Delete brain modal ── */}
      {deleteTarget && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.5)", backdropFilter: "blur(4px)", WebkitBackdropFilter: "blur(4px)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }}
          onClick={() => setDeleteTarget(null)}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "rgba(15,15,15,0.92)", backdropFilter: "blur(16px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, padding: 24, width: 420, display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 16px 48px rgba(0,0,0,0.12)" }}>
            <h2 style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", margin: 0 }}>Delete brain</h2>
            <p style={{ fontSize: 13, color: "rgba(237,236,234,0.55)", margin: 0 }}>
              Are you sure you want to delete <strong>{deleteTarget.name}</strong>? This will permanently remove the dataset and all its files.
            </p>
            <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
              <button onClick={() => setDeleteTarget(null)} className="cursor-pointer"
                style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "rgba(237,236,234,0.7)", fontFamily: "inherit" }}>Cancel</button>
              <button onClick={() => handleDelete(deleteTarget)} disabled={deletingId === deleteTarget.id} className="cursor-pointer"
                style={{ background: "#EF4444", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "#fff", fontFamily: "inherit" }}>
                {deletingId === deleteTarget.id ? "Deleting…" : "Delete"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Share brain modal ── */}
      {shareTarget && (
        <ShareDatasetModal
          datasetId={shareTarget.id}
          datasetName={shareTarget.name}
          pageName="Brains"
          onClose={() => setShareTarget(null)}
        />
      )}

      {/* ── Header ── */}
      <div style={{ padding: "24px 32px 16px", display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexShrink: 0 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <h1 style={{ fontSize: 20, fontWeight: 300, color: "#EDECEA", margin: 0, fontFamily: '"TWKLausanne", sans-serif' }}>Brain</h1>
          <p style={{ fontSize: 14, color: "rgba(237,236,234,0.55)", margin: 0 }}>Upload documents to build searchable knowledge graphs.</p>
        </div>
        <button onClick={async () => { setRefreshing(true); await Promise.all([loadDatasets(), selectedId ? refreshSelectedDocs(selectedId) : Promise.resolve()]); setRefreshing(false); }} disabled={refreshing}
          className="hover:bg-white/10 cursor-pointer"
          style={{ background: "rgba(255,255,255,0.06)", color: "rgba(237,236,234,0.7)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: "8px 12px", fontSize: 13, fontWeight: 500, display: "flex", alignItems: "center", gap: 4 }}
          title="Refresh">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.7)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
            style={refreshing ? { animation: "spin 1s linear infinite" } : undefined}>
            <path d="M21 2v6h-6" /><path d="M3 12a9 9 0 0115.36-6.36L21 8" /><path d="M3 22v-6h6" /><path d="M21 12a9 9 0 01-15.36 6.36L3 16" />
          </svg>
        </button>
      </div>

      {/* ── Finder body ── */}
      {datasets.length > 0 ? (
        <div style={{ flex: 1, minHeight: 0, display: "flex", overflow: "hidden", marginInline: 32, marginBottom: 32, border: "1px solid rgba(255,255,255,0.12)", borderRadius: 12, background: "rgba(0,0,0,0.82)", backdropFilter: "blur(20px)" }}>

          {/* Column 1 — Datasets */}
          <div style={{ width: 312, flexShrink: 0, borderRight: "1px solid rgba(255,255,255,0.1)", display: "flex", flexDirection: "column", overflow: "hidden" }}>
            <div style={{ height: 44, padding: "0 14px", borderBottom: "1px solid rgba(255,255,255,0.1)", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <span style={{ fontSize: 11, fontWeight: 700, color: "rgba(237,236,234,0.55)", letterSpacing: "0.08em", textTransform: "uppercase" }}>Brain</span>
              <button onClick={() => { trackEvent({ pageName: "Brains", eventName: "dataset_create_modal_opened" }); setShowCreate(true); }}
                className="hover:bg-[#5A0ED6] cursor-pointer"
                style={{ background: "#6510F4", color: "#fff", border: "none", borderRadius: 6, padding: "3px 10px", fontSize: 11, fontWeight: 500, display: "flex", alignItems: "center", gap: 4 }}>
                <PlusIcon /> New brain
              </button>
            </div>
            <div style={{ flex: 1, overflowY: "auto" }}>
              {datasets.map((ds, i) => {
                const active = ds.id === selectedId;
                const statusLoading = ds.status === "loading";
                const docsLoadingRow = ds.documents < 0;
                const dotColor = outdatedDatasets.has(ds.id) ? "#F59E0B" : STATUS_DOT[ds.status];
                return (
                  <div key={ds.id} onClick={() => handleSelectDataset(ds.id)}
                    style={{
                      display: "flex", alignItems: "center", gap: 8,
                      padding: "8px 14px",
                      borderBottom: i < datasets.length - 1 ? "1px solid rgba(255,255,255,0.07)" : "none",
                      cursor: "pointer",
                      background: active ? "rgba(188,155,255,0.20)" : "transparent",
                      userSelect: "none",
                    }}
                    onMouseEnter={(e) => { if (!active) e.currentTarget.style.background = "rgba(255,255,255,0.06)"; }}
                    onMouseLeave={(e) => { if (!active) e.currentTarget.style.background = "transparent"; }}
                  >
                    {statusLoading ? (
                      <SkeletonBar width={7} height={7} />
                    ) : (
                      <span style={{ width: 7, height: 7, borderRadius: "50%", background: dotColor, flexShrink: 0 }} />
                    )}
                    <span style={{ flex: 1, fontSize: 13, fontWeight: 500, color: "#EDECEA", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {ds.name}
                    </span>
                    <span style={{ fontSize: 11, color: "rgba(237,236,234,0.35)", flexShrink: 0, minWidth: 16, textAlign: "right" }}>
                      {docsLoadingRow ? <SkeletonBar width={14} height={8} /> : ds.documents}
                    </span>
                    <button
                      onClick={(e) => { e.stopPropagation(); trackEvent({ pageName: "Brains", eventName: "dataset_share_modal_opened", additionalProperties: { dataset_id: ds.id } }); setShareTarget(ds); }}
                      className="hover:bg-white/10"
                      style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 6, padding: "3px 9px", display: "flex", alignItems: "center", gap: 4, fontSize: 11, fontWeight: 500, color: "rgba(237,236,234,0.7)", cursor: "pointer", flexShrink: 0 }}
                      title="Share brain"
                    >
                      <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.7)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="18" cy="5" r="3" /><circle cx="6" cy="12" r="3" /><circle cx="18" cy="19" r="3" /><line x1="8.59" y1="13.51" x2="15.42" y2="17.49" /><line x1="15.41" y1="6.51" x2="8.59" y2="10.49" /></svg>
                      Share
                    </button>
                    {ds.name !== "default_dataset" && (
                      <button
                        onClick={(e) => { e.stopPropagation(); setDeleteTarget(ds); }}
                        style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 6, padding: "3px 10px", fontSize: 11, fontWeight: 500, color: "rgba(237,236,234,0.7)", cursor: "pointer", flexShrink: 0 }}
                        title="Delete brain"
                      >
                        Delete
                      </button>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Column 2 — Documents */}
          <input
            ref={fileInputRef}
            type="file"
            multiple
            style={{ display: "none" }}
            onChange={(e) => { if (e.target.files?.length) { handleUploadFiles(Array.from(e.target.files)); e.target.value = ""; } }}
          />
          <div
            style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", position: "relative" }}
            onDragEnter={(e) => { e.preventDefault(); if (!selectedId) return; dragCounter.current++; setIsDragOver(true); }}
            onDragOver={(e) => { e.preventDefault(); }}
            onDragLeave={(e) => { e.preventDefault(); dragCounter.current--; if (dragCounter.current === 0) setIsDragOver(false); }}
            onDrop={(e) => { e.preventDefault(); dragCounter.current = 0; setIsDragOver(false); if (!selectedId) return; const files = Array.from(e.dataTransfer.files); if (files.length) handleUploadFiles(files); }}
          >
            {/* Drop overlay */}
            {isDragOver && selectedId && (
              <div style={{ position: "absolute", inset: 0, zIndex: 10, background: "rgba(101,16,244,0.06)", border: "2px dashed #6510F4", borderRadius: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 8, pointerEvents: "none" }}>
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none"><path d="M12 15V3m0 0L8 7m4-4l4 4" stroke="#6510F4" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /><path d="M3 15v4a2 2 0 002 2h14a2 2 0 002-2v-4" stroke="#6510F4" strokeWidth="1.5" strokeLinecap="round" /></svg>
                <span style={{ fontSize: 13, fontWeight: 500, color: "#6510F4" }}>Drop to upload</span>
              </div>
            )}

            {/* Header */}
            <div style={{ height: 44, padding: "0 16px", borderBottom: "1px solid rgba(255,255,255,0.1)", flexShrink: 0, display: "flex", alignItems: "center", gap: 6 }}>
              {selectedDataset ? (
                <>
                  <span style={{ fontSize: 11, fontWeight: 700, color: "rgba(237,236,234,0.55)", letterSpacing: "0.08em", textTransform: "uppercase" }}>{selectedDataset.name}</span>
                  <span style={{ fontSize: 11, color: "rgba(255,255,255,0.2)" }}>·</span>
                  <span style={{ fontSize: 11, color: "rgba(237,236,234,0.35)", display: "inline-flex", alignItems: "center", gap: 4 }}>
                    {docsLoading ? <SkeletonBar width={36} height={8} /> : <>{selectedDocs.length} doc{selectedDocs.length !== 1 ? "s" : ""}</>}
                  </span>
                  <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="hover:bg-[#5A0ED6] cursor-pointer"
                      style={{ background: "#6510F4", color: "#fff", border: "none", borderRadius: 6, padding: "3px 10px", fontSize: 11, fontWeight: 500, cursor: "pointer" }}
                    >
                      Add files
                    </button>
                    <button
                      onClick={() => setShowPasteModal(true)}
                      className="hover:bg-[#5A0ED6] cursor-pointer"
                      style={{ background: "#6510F4", color: "#fff", border: "none", borderRadius: 6, padding: "3px 10px", fontSize: 11, fontWeight: 500, cursor: "pointer" }}
                    >
                      Paste text
                    </button>
                  </div>
                </>
              ) : (
                <span style={{ fontSize: 11, fontWeight: 700, color: "rgba(237,236,234,0.55)", letterSpacing: "0.08em", textTransform: "uppercase" }}>Documents</span>
              )}
            </div>

            {/* Upload progress */}
            {isUploading && (
              <div style={{ padding: "8px 16px", borderBottom: "1px solid rgba(255,255,255,0.07)", background: "rgba(255,255,255,0.04)", display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#6510F4" strokeWidth="2" strokeLinecap="round" style={{ animation: "spin 1s linear infinite", flexShrink: 0 }}><path d="M21 12a9 9 0 11-6.219-8.56" /></svg>
                <span style={{ fontSize: 12, color: "#6510F4" }}>Uploading…</span>
              </div>
            )}
            {uploadError && (
              <div style={{ padding: "8px 16px", borderBottom: "1px solid rgba(239,68,68,0.3)", background: "rgba(239,68,68,0.1)", display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
                <span style={{ fontSize: 12, color: "#EF4444" }}>{uploadError}</span>
                <button onClick={() => setUploadError(null)} style={{ marginLeft: "auto", background: "none", border: "none", color: "rgba(237,236,234,0.35)", fontSize: 12, cursor: "pointer" }}>✕</button>
              </div>
            )}

            {/* Content */}
            <div style={{ flex: 1, overflowY: "auto" }}>
              {!selectedId ? (
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", gap: 8 }}>
                  <svg width="32" height="32" viewBox="0 0 32 32" fill="none"><path d="M4 8a2 2 0 012-2h6l2 3h12a2 2 0 012 2v13a2 2 0 01-2 2H6a2 2 0 01-2-2V8z" stroke="rgba(237,236,234,0.2)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
                  <span style={{ fontSize: 13, color: "rgba(237,236,234,0.35)" }}>Select a brain</span>
                </div>
              ) : docsLoading ? (
                <PageLoading name="Files" />
              ) : selectedDocs.length === 0 ? (
                <div
                  style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", gap: 10, cursor: "pointer" }}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <div style={{ width: 44, height: 44, background: "rgba(188,155,255,0.20)", border: "1px solid rgba(188,155,255,0.35)", borderRadius: 10, display: "flex", alignItems: "center", justifyContent: "center" }}>
                    <EmptyStateIcon />
                  </div>
                  <span style={{ fontSize: 13, color: "rgba(237,236,234,0.55)", fontWeight: 500 }}>No documents yet</span>
                  <span style={{ fontSize: 12, color: "rgba(237,236,234,0.35)", textAlign: "center", maxWidth: 220 }}>
                    Drag &amp; drop files here, or <span style={{ color: "#6510F4", textDecoration: "underline" }}>browse</span>
                  </span>
                </div>
              ) : (
                <>
                  {selectedDocs.map((doc, i) => {
                    const displayName = decodeFilename(doc.name);
                    const meta = getExtMeta(displayName, doc.extension);
                    return (
                      <div key={doc.id} style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 16px", borderBottom: i < selectedDocs.length - 1 ? "1px solid rgba(255,255,255,0.07)" : "none" }}>
                        <FileIcon {...meta} />
                        <span style={{ flex: 1, fontSize: 13, color: "#EDECEA", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{displayName}</span>
                        <div style={{ display: "flex", alignItems: "center", gap: 12, flexShrink: 0 }}>
                          <span style={{ fontSize: 11, color: "rgba(237,236,234,0.55)", fontWeight: 500, minWidth: 32, textAlign: "right" }}>{meta.label}</span>
                          <span style={{ fontSize: 11, color: "rgba(237,236,234,0.35)", minWidth: 52, textAlign: "right" }}>{formatSize(doc.size)}</span>
                          <span style={{ fontSize: 11, color: "rgba(237,236,234,0.35)", minWidth: 80, textAlign: "right", whiteSpace: "nowrap" }}>{formatDate(doc.createdAt)}</span>
                          <button
                            onClick={() => setDeleteDocTarget(doc)}
                            style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 6, padding: "3px 10px", fontSize: 11, fontWeight: 500, color: "rgba(237,236,234,0.7)", cursor: "pointer", flexShrink: 0 }}
                            title="Delete file"
                          >
                            Delete
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </>
              )}
            </div>
          </div>

        </div>
      ) : (
        /* ── Empty state ── */
        <div style={{ flex: 1, display: "flex", flexDirection: "column", paddingInline: 32, paddingBottom: 32 }}>
          <div style={{ flex: 1, background: "rgba(255,255,255,0.06)", backdropFilter: "blur(12px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 16, padding: 48 }}>
            <div style={{ width: 56, height: 56, background: "rgba(188,155,255,0.20)", border: "1px solid rgba(188,155,255,0.35)", borderRadius: 12, display: "flex", alignItems: "center", justifyContent: "center" }}>
              <EmptyStateIcon />
            </div>
            <span style={{ fontSize: 16, fontWeight: 700, color: "#EDECEA" }}>No brains yet</span>
            <p style={{ fontSize: 14, color: "rgba(237,236,234,0.35)", margin: 0, maxWidth: 340, textAlign: "center" }}>
              Create your first brain to start uploading documents and building knowledge graphs.
            </p>
            <button onClick={() => { trackEvent({ pageName: "Brains", eventName: "dataset_create_modal_opened" }); setShowCreate(true); }}
              className="hover:bg-[#5A0ED6] cursor-pointer"
              style={{ background: "#6510F4", color: "#fff", border: "none", borderRadius: 8, padding: "8px 20px", fontSize: 14, fontWeight: 500, display: "flex", alignItems: "center", gap: 6, marginTop: 12 }}>
              <PlusIcon /> Create brain
            </button>
          </div>
        </div>
      )}

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
