"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useCogniInstance, useTenant } from "@/modules/tenant/TenantProvider";
import { useFilter } from "@/ui/layout/FilterContext";
import getDatasets from "@/modules/datasets/getDatasets";
import getDatasetData from "@/modules/datasets/getDatasetData";
import createDataset from "@/modules/datasets/createDataset";
import deleteDataset from "@/modules/datasets/deleteDataset";
import { type DatasetProcessingStatus } from "@/modules/datasets/pollDatasetStatus";
import { TrackPageView, trackEvent } from "@/modules/analytics";
import { loadGraphModelsConfig } from "@/modules/configuration/userConfiguration";
import addData from "@/modules/ingestion/addData";
import cognifyDataset from "@/modules/datasets/cognifyDataset";
import deleteDatasetData from "@/modules/datasets/deleteDatasetData";

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

type PreviewTab = "CHUNKS" | "SUMMARIES";

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
      <text x="8" y="14.5" textAnchor="middle" fontFamily="Inter,system-ui,sans-serif" fontSize={fs} fontWeight="700" fill={text}>{label}</text>
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
  const [docSearch, setDocSearch]         = useState("");
  const [docSortKey, setDocSortKey]       = useState<"name" | "type" | "added">("added");
  const [docSortDir, setDocSortDir]       = useState<"asc" | "desc">("desc");

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
  const [deletingId, setDeletingId]       = useState<string | null>(null);
  const [deleteDocTarget, setDeleteDocTarget] = useState<FileEntry | null>(null);
  const [showPasteModal, setShowPasteModal] = useState(false);
  const [pasteText, setPasteText]           = useState("");
  const [pasting, setPasting]               = useState(false);

  // Content preview
  const [previewDoc, setPreviewDoc]         = useState<FileEntry | null>(null);
  const [previewTab, setPreviewTab]         = useState<PreviewTab>("CHUNKS");
  const [previewContent, setPreviewContent] = useState<string>("");
  const [previewLoading, setPreviewLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const pollRef  = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchPreview = useCallback(async (doc: FileEntry, tab: PreviewTab) => {
    if (!cogniInstance || !selectedId) return;
    setPreviewLoading(true);
    setPreviewContent("");
    try {
      const graphResp = await cogniInstance.fetch(`/v1/datasets/${selectedId}/graph`);
      if (graphResp.ok) {
        const graph = await graphResp.json();
        const nodes = graph.nodes || [];

        const docNode = nodes.find((n: any) =>
          n.type === "TextDocument" && (n.label?.includes(doc.name) || n.id === doc.id)
        );
        const contentHash = docNode?.properties?.source_content_hash;

        if (contentHash) {
          if (tab === "CHUNKS") {
            const chunks = nodes.filter((n: any) => n.type === "DocumentChunk" && n.properties?.source_content_hash === contentHash);
            setPreviewContent(chunks.map((c: any) => c.properties?.text || "").join("\n\n---\n\n") || "No chunks found");
          } else {
            const summaries = nodes.filter((n: any) => n.type === "TextSummary" && n.properties?.source_content_hash === contentHash);
            setPreviewContent(summaries.map((s: any) => s.properties?.text || "").join("\n\n---\n\n") || "No summaries found");
          }
        } else {
          setPreviewContent("File not yet processed (run cognify first)");
        }
      } else {
        setPreviewContent("Failed to load graph");
      }
    } catch {
      setPreviewContent("Error loading content");
    } finally {
      setPreviewLoading(false);
    }
  }, [cogniInstance, selectedId]);

  const fetchStatuses = useCallback(async (ids: string[]) => {
    if (!cogniInstance || !ids.length) return;
    try {
      const resp = await cogniInstance.fetch("/v1/datasets/status");
      if (!resp.ok) return;
      const data: Record<string, DatasetProcessingStatus> = await resp.json();
      let completedSelectedId: string | null = null;
      setDatasets((prev) => {
        return prev.map((d) => {
          const raw = data[d.id];
          if (!raw) return d;
          const newStatus = mapProcessingStatus(raw, d.documents);
          if (newStatus !== d.status) {
            // When the selected brain finishes, schedule a file list refresh
            if (d.id === selectedId && (d.status === "pending" || d.status === "running") && newStatus === "completed") {
              completedSelectedId = d.id;
            }
            return { ...d, status: newStatus };
          }
          return d;
        });
      });
      if (completedSelectedId) {
        getDatasetData(completedSelectedId, cogniInstance)
          .then((docs) => {
            setSelectedDocs(Array.isArray(docs) ? docs : []);
            setDatasets((prev) => prev.map((d) => d.id === completedSelectedId ? { ...d, documents: Array.isArray(docs) ? docs.length : d.documents } : d));
          })
          .catch(() => {});
      }
    } catch { /* graceful */ }
  }, [cogniInstance, selectedId]);

  useEffect(() => {
    if (!cogniInstance || isInitializing) return;
    loadDatasets();
    loadGraphModelsConfig(cogniInstance)
      .then((cfg) => setOutdated(new Set(cfg.outdatedDatasets ?? [])))
      .catch(() => {});
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [cogniInstance, isInitializing]);

  useEffect(() => {
    if (pollRef.current) clearInterval(pollRef.current);
    const hasActive = datasets.some((d) => d.status === "pending" || d.status === "running");
    if (!hasActive || !cogniInstance) return;
    pollRef.current = setInterval(() => fetchStatuses(datasets.map((d) => d.id)), 5000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [datasets, cogniInstance, fetchStatuses]);

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

      const statusResp = await cogniInstance.fetch("/v1/datasets/status").catch(() => null);
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

  function handleDocSort(key: "name" | "type" | "added") {
    if (key === docSortKey) {
      setDocSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setDocSortKey(key);
      setDocSortDir(key === "added" ? "desc" : "asc");
    }
  }

  async function handleSelectDataset(id: string) {
    if (selectedId === id) return;
    setSelectedId(id);
    setSelectedDocs([]);
    setDocSearch("");
    setDocSortKey("added");
    setDocSortDir("desc");
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

  async function handleUploadFiles(files: File[]) {
    if (!cogniInstance || !selectedId || !files.length) return;
    const ds = datasets.find((d) => d.id === selectedId);
    if (!ds) return;
    setIsUploading(true);
    setUploadError(null);
    try {
      await addData({ id: ds.id, name: ds.name }, files, cogniInstance);
      // Reload the file list immediately so files appear without a page refresh
      const data = await getDatasetData(ds.id, cogniInstance);
      setSelectedDocs(Array.isArray(data) ? data : []);
      setDatasets((prev) => prev.map((d) => d.id === selectedId ? { ...d, documents: Array.isArray(data) ? data.length : d.documents, status: "running" } : d));
      cognifyDataset({ id: ds.id, name: ds.name, data: [], status: "" }, cogniInstance);
    } catch {
      setUploadError("Upload failed. Please try again.");
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
          <span style={{ fontSize: 14, color: "#71717A" }}>Loading datasets…</span>
        </div>
      </>
    );
  }

  const selectedDataset = datasets.find((d) => d.id === selectedId) ?? null;

  const visibleDocs = [...(docSearch
    ? selectedDocs.filter((d) => decodeFilename(d.name).toLowerCase().includes(docSearch.toLowerCase()))
    : selectedDocs)].sort((a, b) => {
    let cmp = 0;
    if (docSortKey === "name") {
      cmp = (a.name || "").localeCompare(b.name || "");
    } else if (docSortKey === "type") {
      const extA = (a.extension || a.name?.split(".").pop() || "").toLowerCase();
      const extB = (b.extension || b.name?.split(".").pop() || "").toLowerCase();
      cmp = extA.localeCompare(extB);
    } else {
      if (!a.createdAt && !b.createdAt) cmp = 0;
      else if (!a.createdAt) return 1;
      else if (!b.createdAt) return -1;
      else cmp = a.createdAt.localeCompare(b.createdAt);
    }
    return docSortDir === "asc" ? cmp : -cmp;
  });

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", fontFamily: '"Inter", system-ui, sans-serif', overflow: "hidden" }}>
      <TrackPageView page="Brains" />

      {/* ── Create modal ── */}
      {showCreateModal && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.3)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }}
          onClick={() => { setShowCreate(false); setCreateError(""); }}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "#fff", borderRadius: 12, padding: 24, width: 420, display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 16px 48px rgba(0,0,0,0.12)" }}>
            <h2 style={{ fontSize: 18, fontWeight: 600, color: "#18181B", margin: 0 }}>Create brain</h2>
            <p style={{ fontSize: 13, color: "#71717A", margin: 0 }}>Give your brain a name. You can upload documents after creation.</p>
            <input ref={inputRef} autoFocus type="text" value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") handleCreate(); }}
              placeholder="e.g. product-docs, sec-filings..."
              style={{ width: "100%", height: 40, border: "1px solid #E4E4E7", borderRadius: 8, paddingInline: 14, fontSize: 14, color: "#18181B", fontFamily: "inherit", outline: "none", boxSizing: "border-box" }}
              onFocus={(e) => { e.target.style.borderColor = "#6510F4"; e.target.style.boxShadow = "0 0 0 3px rgba(101,16,244,0.1)"; }}
              onBlur={(e)  => { e.target.style.borderColor = "#E4E4E7"; e.target.style.boxShadow = "none"; }}
            />
            {createError && <p style={{ fontSize: 13, color: "#EF4444", margin: 0 }}>{createError}</p>}
            <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
              <button onClick={() => { setShowCreate(false); setNewName(""); setCreateError(""); }} className="cursor-pointer"
                style={{ background: "#fff", border: "1px solid #E4E4E7", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "#3F3F46", fontFamily: "inherit" }}>Cancel</button>
              <button onClick={handleCreate} disabled={creating} className="cursor-pointer"
                style={{ background: newName.trim() ? "#6510F4" : "#E4E4E7", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: newName.trim() ? "#fff" : "#A1A1AA", fontFamily: "inherit" }}>
                {creating ? "Creating…" : "Create"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Delete document modal ── */}
      {deleteDocTarget && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.3)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }}
          onClick={() => setDeleteDocTarget(null)}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "#fff", borderRadius: 12, padding: 24, width: 420, display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 16px 48px rgba(0,0,0,0.12)" }}>
            <h2 style={{ fontSize: 18, fontWeight: 600, color: "#18181B", margin: 0 }}>Delete document</h2>
            <p style={{ fontSize: 13, color: "#71717A", margin: 0 }}>
              Are you sure you want to delete <strong>{decodeFilename(deleteDocTarget.name)}</strong>? This action cannot be undone.
            </p>
            <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
              <button onClick={() => setDeleteDocTarget(null)} className="cursor-pointer"
                style={{ background: "#fff", border: "1px solid #E4E4E7", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "#3F3F46", fontFamily: "inherit" }}>Cancel</button>
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
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.3)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }}
          onClick={() => { setShowPasteModal(false); setPasteText(""); }}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "#fff", borderRadius: 12, padding: 24, width: 420, display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 16px 48px rgba(0,0,0,0.12)" }}>
            <h2 style={{ fontSize: 18, fontWeight: 600, color: "#18181B", margin: 0 }}>Paste text</h2>
            <p style={{ fontSize: 13, color: "#71717A", margin: 0 }}>Paste your text below. It will be added as a document to the selected brain.</p>
            <textarea
              autoFocus
              value={pasteText}
              onChange={(e) => setPasteText(e.target.value)}
              placeholder="Paste your text here…"
              rows={8}
              style={{ width: "100%", border: "1px solid #E4E4E7", borderRadius: 8, padding: "10px 14px", fontSize: 14, color: "#18181B", fontFamily: "inherit", outline: "none", resize: "vertical", boxSizing: "border-box" }}
              onFocus={(e) => { e.target.style.borderColor = "#6510F4"; e.target.style.boxShadow = "0 0 0 3px rgba(101,16,244,0.1)"; }}
              onBlur={(e)  => { e.target.style.borderColor = "#E4E4E7"; e.target.style.boxShadow = "none"; }}
            />
            <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
              <button onClick={() => { setShowPasteModal(false); setPasteText(""); }} className="cursor-pointer"
                style={{ background: "#fff", border: "1px solid #E4E4E7", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "#3F3F46", fontFamily: "inherit" }}>Cancel</button>
              <button onClick={handlePasteText} disabled={!pasteText.trim() || pasting} className="cursor-pointer"
                style={{ background: pasteText.trim() ? "#6510F4" : "#E4E4E7", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: pasteText.trim() ? "#fff" : "#A1A1AA", fontFamily: "inherit" }}>
                {pasting ? "Adding…" : "Add"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Delete brain modal ── */}
      {deleteTarget && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.3)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }}
          onClick={() => setDeleteTarget(null)}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "#fff", borderRadius: 12, padding: 24, width: 420, display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 16px 48px rgba(0,0,0,0.12)" }}>
            <h2 style={{ fontSize: 18, fontWeight: 600, color: "#18181B", margin: 0 }}>Delete brain</h2>
            <p style={{ fontSize: 13, color: "#71717A", margin: 0 }}>
              Are you sure you want to delete <strong>{deleteTarget.name}</strong>? This will permanently remove the dataset and all its files.
            </p>
            <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
              <button onClick={() => setDeleteTarget(null)} className="cursor-pointer"
                style={{ background: "#fff", border: "1px solid #E4E4E7", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "#3F3F46", fontFamily: "inherit" }}>Cancel</button>
              <button onClick={() => handleDelete(deleteTarget)} disabled={deletingId === deleteTarget.id} className="cursor-pointer"
                style={{ background: "#EF4444", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "#fff", fontFamily: "inherit" }}>
                {deletingId === deleteTarget.id ? "Deleting…" : "Delete"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Content Preview Modal */}
      {previewDoc && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.3)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }}
          onClick={() => setPreviewDoc(null)}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "#fff", borderRadius: 12, padding: 24, width: 600, maxHeight: "80vh", display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 16px 48px rgba(0,0,0,0.12)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <h2 style={{ fontSize: 16, fontWeight: 600, color: "#18181B", margin: 0 }}>{decodeFilename(previewDoc.name)}</h2>
              <button onClick={() => setPreviewDoc(null)} style={{ background: "none", border: "none", fontSize: 18, color: "#A1A1AA", cursor: "pointer" }}>✕</button>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <div style={{ display: "flex", gap: 4 }}>
                {([
                  { key: "CHUNKS", label: "Chunks", hint: "Text chunks extracted from this file" },
                  { key: "SUMMARIES", label: "Summaries", hint: "AI-generated summaries of this file's content" },
                ] as { key: PreviewTab; label: string; hint: string }[]).map(({ key, label, hint }) => (
                  <button
                    key={key}
                    onClick={() => { setPreviewTab(key); fetchPreview(previewDoc, key); }}
                    style={{
                      padding: "6px 12px", fontSize: 11, fontWeight: 500, fontFamily: "inherit",
                      border: "1px solid #E4E4E7", borderRadius: 6, cursor: "pointer",
                      background: previewTab === key ? "#6510F4" : "#fff",
                      color: previewTab === key ? "#fff" : "#52525B",
                    }}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <span style={{ fontSize: 11, color: "#A1A1AA" }}>
                {previewTab === "CHUNKS" && "Text chunks extracted from this file"}
                {previewTab === "SUMMARIES" && "AI-generated summaries of this file's content"}
              </span>
            </div>
            <div style={{ flex: 1, overflowY: "auto", background: "#FAFAFA", borderRadius: 8, padding: 16, fontSize: 13, color: "#18181B", whiteSpace: "pre-wrap", lineHeight: 1.6, minHeight: 200 }}>
              {previewLoading ? "Loading…" : previewContent || "No content"}
            </div>
          </div>
        </div>
      )}

      {/* ── Header ── */}
      <div style={{ padding: "24px 32px 16px", display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexShrink: 0 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <h1 style={{ fontSize: 20, fontWeight: 300, color: "#18181B", margin: 0, fontFamily: '"TWK Lausanne", system-ui, sans-serif' }}>Brains</h1>
          <p style={{ fontSize: 14, color: "#71717A", margin: 0 }}>Upload documents to build searchable knowledge graphs.</p>
        </div>
        <button onClick={async () => { setRefreshing(true); await loadDatasets(); setRefreshing(false); }} disabled={refreshing}
          className="hover:bg-cognee-hover cursor-pointer"
          style={{ background: "#fff", color: "#3F3F46", border: "1px solid #E4E4E7", borderRadius: 8, padding: "8px 12px", fontSize: 13, fontWeight: 500, display: "flex", alignItems: "center", gap: 4 }}
          title="Refresh">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#3F3F46" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
            style={refreshing ? { animation: "spin 1s linear infinite" } : undefined}>
            <path d="M21 2v6h-6" /><path d="M3 12a9 9 0 0115.36-6.36L21 8" /><path d="M3 22v-6h6" /><path d="M21 12a9 9 0 01-15.36 6.36L3 16" />
          </svg>
        </button>
      </div>

      {/* ── Finder body ── */}
      {datasets.length > 0 ? (
        <div style={{ flex: 1, display: "flex", overflow: "hidden", marginInline: 32, marginBottom: 32, border: "1px solid #E4E4E7", borderRadius: 12 }}>

          {/* Column 1 — Datasets */}
          <div style={{ width: 312, flexShrink: 0, borderRight: "1px solid #E4E4E7", display: "flex", flexDirection: "column", overflow: "hidden" }}>
            <div style={{ height: 44, padding: "0 14px", borderBottom: "1px solid #E4E4E7", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <span style={{ fontSize: 11, fontWeight: 600, color: "#71717A", letterSpacing: "0.08em", textTransform: "uppercase" }}>Brains</span>
              <button onClick={() => { trackEvent({ pageName: "Brains", eventName: "dataset_create_modal_opened" }); setShowCreate(true); }}
                className="hover:bg-cognee-purple-hover cursor-pointer"
                style={{ background: "#6510F4", color: "#fff", border: "none", borderRadius: 6, padding: "3px 10px", fontSize: 11, fontWeight: 500, display: "flex", alignItems: "center", gap: 4 }}>
                <PlusIcon /> New brain
              </button>
            </div>
            <div style={{ flex: 1, overflowY: "auto" }}>
              {datasets.map((ds, i) => {
                const active = ds.id === selectedId;
                const dotColor = outdatedDatasets.has(ds.id) ? "#F59E0B" : STATUS_DOT[ds.status];
                return (
                  <div key={ds.id} onClick={() => handleSelectDataset(ds.id)}
                    style={{
                      display: "flex", alignItems: "center", gap: 8,
                      padding: "8px 14px",
                      borderBottom: i < datasets.length - 1 ? "1px solid #F4F4F5" : "none",
                      cursor: "pointer",
                      background: active ? "#F4F4F5" : "transparent",
                      userSelect: "none",
                    }}
                    onMouseEnter={(e) => { if (!active) e.currentTarget.style.background = "#F4F4F5"; }}
                    onMouseLeave={(e) => { if (!active) e.currentTarget.style.background = "transparent"; }}
                  >
                    <span style={{ width: 7, height: 7, borderRadius: "50%", background: dotColor, flexShrink: 0 }} />
                    <span style={{ flex: 1, fontSize: 13, fontWeight: 500, color: "#18181B", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {ds.name}
                    </span>
                    <span style={{ fontSize: 11, color: "#A1A1AA", flexShrink: 0, minWidth: 16, textAlign: "right" }}>
                      {ds.documents < 0 ? "…" : ds.documents}
                    </span>
                    {ds.name !== "default_dataset" && (
                      <button
                        onClick={(e) => { e.stopPropagation(); setDeleteTarget(ds); }}
                        style={{ background: "none", border: "1px solid #E4E4E7", borderRadius: 6, padding: "3px 10px", fontSize: 11, fontWeight: 500, color: "#52525B", cursor: "pointer", flexShrink: 0 }}
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
            <div style={{ height: 44, padding: "0 16px", borderBottom: "1px solid #E4E4E7", flexShrink: 0, display: "flex", alignItems: "center", gap: 6 }}>
              {selectedDataset ? (
                <>
                  <span style={{ fontSize: 11, fontWeight: 600, color: "#71717A", letterSpacing: "0.08em", textTransform: "uppercase" }}>{selectedDataset.name}</span>
                  <span style={{ fontSize: 11, color: "#D4D4D8" }}>·</span>
                  <span style={{ fontSize: 11, color: "#A1A1AA" }}>{selectedDocs.length} doc{selectedDocs.length !== 1 ? "s" : ""}</span>
                  <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="hover:bg-cognee-purple-hover cursor-pointer"
                      style={{ background: "#6510F4", color: "#fff", border: "none", borderRadius: 6, padding: "3px 10px", fontSize: 11, fontWeight: 500, cursor: "pointer" }}
                    >
                      Add files
                    </button>
                    <button
                      onClick={() => setShowPasteModal(true)}
                      className="hover:bg-cognee-purple-hover cursor-pointer"
                      style={{ background: "#6510F4", color: "#fff", border: "none", borderRadius: 6, padding: "3px 10px", fontSize: 11, fontWeight: 500, cursor: "pointer" }}
                    >
                      Paste text
                    </button>
                  </div>
                </>
              ) : (
                <span style={{ fontSize: 11, fontWeight: 600, color: "#71717A", letterSpacing: "0.08em", textTransform: "uppercase" }}>Documents</span>
              )}
            </div>

            {/* Doc search */}
            {selectedId && !docsLoading && selectedDocs.length > 0 && (
              <div style={{ display: "flex", alignItems: "center", gap: 8, height: 36, paddingInline: 12, borderBottom: "1px solid #F4F4F5", flexShrink: 0 }}>
                <svg width="13" height="13" viewBox="0 0 16 16" fill="none"><circle cx="7" cy="7" r="4.5" stroke="#A1A1AA" strokeWidth="1.5"/><path d="M10.5 10.5L14 14" stroke="#A1A1AA" strokeWidth="1.5" strokeLinecap="round"/></svg>
                <input
                  type="text"
                  value={docSearch}
                  onChange={(e) => setDocSearch(e.target.value)}
                  placeholder="Filter by name…"
                  style={{ flex: 1, border: "none", outline: "none", fontSize: 12, color: "#18181B", background: "transparent", fontFamily: "inherit" }}
                />
                {docSearch && (
                  <button onClick={() => setDocSearch("")} style={{ background: "none", border: "none", color: "#A1A1AA", fontSize: 13, cursor: "pointer", padding: 0 }}>✕</button>
                )}
              </div>
            )}

            {/* Upload progress */}
            {isUploading && (
              <div style={{ padding: "8px 16px", borderBottom: "1px solid #F4F4F5", background: "#FAFAF9", display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#6510F4" strokeWidth="2" strokeLinecap="round" style={{ animation: "spin 1s linear infinite", flexShrink: 0 }}><path d="M21 12a9 9 0 11-6.219-8.56" /></svg>
                <span style={{ fontSize: 12, color: "#6510F4" }}>Uploading…</span>
              </div>
            )}
            {uploadError && (
              <div style={{ padding: "8px 16px", borderBottom: "1px solid #FEE2E2", background: "#FFF5F5", display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
                <span style={{ fontSize: 12, color: "#EF4444" }}>{uploadError}</span>
                <button onClick={() => setUploadError(null)} style={{ marginLeft: "auto", background: "none", border: "none", color: "#A1A1AA", fontSize: 12, cursor: "pointer" }}>✕</button>
              </div>
            )}

            {/* Content */}
            <div style={{ flex: 1, overflowY: "auto" }}>
              {!selectedId ? (
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", gap: 8 }}>
                  <svg width="32" height="32" viewBox="0 0 32 32" fill="none"><path d="M4 8a2 2 0 012-2h6l2 3h12a2 2 0 012 2v13a2 2 0 01-2 2H6a2 2 0 01-2-2V8z" stroke="#D4D4D8" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
                  <span style={{ fontSize: 13, color: "#A1A1AA" }}>Select a brain</span>
                </div>
              ) : docsLoading ? (
                <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", gap: 8 }}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#A1A1AA" strokeWidth="2" strokeLinecap="round" style={{ animation: "spin 1s linear infinite" }}><path d="M21 12a9 9 0 11-6.219-8.56" /></svg>
                  <span style={{ fontSize: 13, color: "#A1A1AA" }}>Loading…</span>
                </div>
              ) : selectedDocs.length > 0 && (
                <div style={{ display: "flex", alignItems: "center", background: "#FAFAF9", borderBottom: "1px solid #F4F4F5", padding: "6px 16px", flexShrink: 0 }}>
                  <button
                    onClick={() => handleDocSort("name")}
                    className="cursor-pointer"
                    style={{ flex: 1, fontSize: 11, fontWeight: 600, color: "#71717A", background: "none", border: "none", padding: 0, textAlign: "left", fontFamily: "inherit", display: "flex", alignItems: "center", gap: 3 }}
                  >
                    Name
                    <span style={{ fontSize: 9, opacity: docSortKey === "name" ? 1 : 0.3 }}>
                      {docSortKey === "name" ? (docSortDir === "asc" ? "▲" : "▼") : "⇅"}
                    </span>
                  </button>
                  <button
                    onClick={() => handleDocSort("type")}
                    className="cursor-pointer"
                    style={{ width: 52, fontSize: 11, fontWeight: 600, color: "#71717A", background: "none", border: "none", padding: 0, textAlign: "left", fontFamily: "inherit", flexShrink: 0, display: "flex", alignItems: "center", gap: 3 }}
                  >
                    Type
                    <span style={{ fontSize: 9, opacity: docSortKey === "type" ? 1 : 0.3 }}>
                      {docSortKey === "type" ? (docSortDir === "asc" ? "▲" : "▼") : "⇅"}
                    </span>
                  </button>
                  <span style={{ width: 52, fontSize: 11, fontWeight: 600, color: "#71717A", flexShrink: 0 }}>Size</span>
                  <button
                    onClick={() => handleDocSort("added")}
                    className="cursor-pointer"
                    style={{ width: 80, fontSize: 11, fontWeight: 600, color: "#71717A", background: "none", border: "none", padding: 0, textAlign: "left", fontFamily: "inherit", flexShrink: 0, display: "flex", alignItems: "center", gap: 3 }}
                  >
                    Added
                    <span style={{ fontSize: 9, opacity: docSortKey === "added" ? 1 : 0.3 }}>
                      {docSortKey === "added" ? (docSortDir === "asc" ? "▲" : "▼") : "⇅"}
                    </span>
                  </button>
                  <div style={{ width: 60, flexShrink: 0 }} />
                </div>
              )}
              {selectedDocs.length === 0 ? (
                <div
                  style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", gap: 10, cursor: "pointer" }}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <div style={{ width: 44, height: 44, background: "#F0EDFF", borderRadius: 10, display: "flex", alignItems: "center", justifyContent: "center" }}>
                    <EmptyStateIcon />
                  </div>
                  <span style={{ fontSize: 13, color: "#71717A", fontWeight: 500 }}>No documents yet</span>
                  <span style={{ fontSize: 12, color: "#A1A1AA", textAlign: "center", maxWidth: 220, lineHeight: 1.5 }}>
                    Drag &amp; drop files here, or <span style={{ color: "#6510F4", textDecoration: "underline" }}>browse</span>
                  </span>
                </div>
              ) : (
                <>
                  {visibleDocs.length === 0 ? (
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", padding: "32px 16px" }}>
                      <span style={{ fontSize: 13, color: "#A1A1AA" }}>No files match &ldquo;{docSearch}&rdquo;</span>
                    </div>
                  ) : visibleDocs.map((doc, i) => {
                    const displayName = decodeFilename(doc.name);
                    const meta = getExtMeta(displayName, doc.extension);
                    return (
                      <div key={doc.id} style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 16px", borderBottom: i < visibleDocs.length - 1 ? "1px solid #F4F4F5" : "none" }}>
                        <FileIcon {...meta} />
                        <span
                          onClick={() => { setPreviewDoc(doc); setPreviewTab("CHUNKS"); fetchPreview(doc, "CHUNKS"); }}
                          style={{ flex: 1, fontSize: 13, color: "#6510F4", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", cursor: "pointer", textDecoration: "underline" }}
                        >{displayName}</span>
                        <div style={{ display: "flex", alignItems: "center", gap: 12, flexShrink: 0 }}>
                          <span style={{ fontSize: 11, color: "#71717A", fontWeight: 500, minWidth: 32, textAlign: "right" }}>{meta.label}</span>
                          <span style={{ fontSize: 11, color: "#A1A1AA", minWidth: 52, textAlign: "right" }}>{formatSize(doc.size)}</span>
                          <span style={{ fontSize: 11, color: "#A1A1AA", minWidth: 80, textAlign: "right", whiteSpace: "nowrap" }}>{formatDate(doc.createdAt)}</span>
                          <button
                            onClick={() => setDeleteDocTarget(doc)}
                            style={{ background: "none", border: "1px solid #E4E4E7", borderRadius: 6, padding: "3px 10px", fontSize: 11, fontWeight: 500, color: "#52525B", cursor: "pointer", flexShrink: 0 }}
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
          <div style={{ flex: 1, background: "#fff", border: "1px solid #E4E4E7", borderRadius: 12, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 16, padding: 48 }}>
            <div style={{ width: 56, height: 56, background: "#F0EDFF", borderRadius: 12, display: "flex", alignItems: "center", justifyContent: "center" }}>
              <EmptyStateIcon />
            </div>
            <span style={{ fontSize: 16, fontWeight: 600, color: "#18181B" }}>No brains yet</span>
            <p style={{ fontSize: 14, color: "#A1A1AA", margin: 0, maxWidth: 340, textAlign: "center", lineHeight: 1.5 }}>
              Create your first brain to start uploading documents and building knowledge graphs.
            </p>
            <button onClick={() => { trackEvent({ pageName: "Brains", eventName: "dataset_create_modal_opened" }); setShowCreate(true); }}
              className="hover:bg-cognee-purple-hover cursor-pointer"
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
