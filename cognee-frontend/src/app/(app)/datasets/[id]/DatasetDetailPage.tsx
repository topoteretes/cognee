"use client";

import { useState, useEffect, useRef } from "react";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import getDatasets from "@/modules/datasets/getDatasets";
import getDatasetData from "@/modules/datasets/getDatasetData";
import deleteDatasetData from "@/modules/datasets/deleteDatasetData";
import deleteDataset from "@/modules/datasets/deleteDataset";
import addData from "@/modules/ingestion/addData";
import cognifyDataset from "@/modules/datasets/cognifyDataset";

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
  return new Date(dateStr).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

// ── SVG document icon matching Paper reference ──

function FileIcon({ fill, stroke, text, label }: { fill: string; stroke: string; text: string; label: string }) {
  const fontSize = label.length > 3 ? 4.5 : label.length > 2 ? 5 : 5.5;
  return (
    <svg width="16" height="20" viewBox="0 0 16 20" fill="none" style={{ flexShrink: 0 }}>
      <path d="M10 1H3a2 2 0 00-2 2v14a2 2 0 002 2h10a2 2 0 002-2V6l-5-5z" fill={fill} stroke={stroke} strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M10 1v5h5" stroke={stroke} strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
      <text x="8" y="14.5" textAnchor="middle" fontFamily="Inter,system-ui,sans-serif" fontSize={fontSize} fontWeight="700" fill={text}>{label}</text>
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" style={{ flexShrink: 0 }}>
      <path d="M3 4h10M6 4V3h4v1M5 4v8.5a.5.5 0 00.5.5h5a.5.5 0 00.5-.5V4" stroke="#A1A1AA" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// ── Main Page ──

interface Agent {
  id: string;
  email: string;
  agent_type: string;
  agent_short_id: string;
  is_agent: boolean;
  status: string;
}

export default function DatasetDetailPage({ datasetId }: { datasetId: string }) {
  const { cogniInstance, isInitializing } = useCogniInstance();
  const [datasetName, setDatasetName] = useState<string>(datasetId);
  const [files, setFiles] = useState<FileEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [search, setSearch] = useState("");
  const [showShareModal, setShowShareModal] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [sharedWith, setSharedWith] = useState<Set<string>>(new Set());
  const [sharing, setSharing] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!cogniInstance || isInitializing) return;
    // Fetch real dataset name from the datasets list
    getDatasets(cogniInstance).then((datasets) => {
      const ds = Array.isArray(datasets) ? datasets.find((d: { id: string; name: string }) => d.id === datasetId) : null;
      if (ds) setDatasetName(ds.name);
    }).catch(() => {});
    loadFiles();
    // Load agents for sharing (graceful — endpoint may not exist on cloud)
    cogniInstance.fetch("/v1/activity/agents").then((r) => r.ok ? r.json() : []).then((data) => {
      setAgents(Array.isArray(data) ? data : []);
    }).catch(() => setAgents([]));
  }, [cogniInstance, isInitializing]);

  async function loadFiles() {
    if (!cogniInstance) return;
    try {
      const data = await getDatasetData(datasetId, cogniInstance);
      setFiles(Array.isArray(data) ? data.map((d: FileEntry & { rawDataLocation?: string }) => ({
        id: d.id,
        name: d.name || d.rawDataLocation?.split("/").pop() || d.id,
        extension: d.extension,
        mimeType: d.mimeType,
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
    setUploading(true);
    try {
      await addData({ id: datasetId }, Array.from(newFiles), cogniInstance);
      await cognifyDataset({ id: datasetId, name: datasetName, data: [], status: "processing" }, cogniInstance);
      await loadFiles();
    } catch (err) {
      console.error("Upload failed:", err);
    } finally {
      setUploading(false);
    }
  }

  async function handleDelete(fileId: string) {
    if (!cogniInstance) return;
    try {
      await deleteDatasetData(datasetId, fileId, cogniInstance);
      setFiles((prev) => prev.filter((f) => f.id !== fileId));
    } catch (err) {
      console.error("Delete failed:", err);
    }
  }

  async function handleDeleteDataset() {
    if (!cogniInstance) return;
    setDeleting(true);
    try {
      await deleteDataset(datasetId, cogniInstance);
      window.location.href = "/datasets";
    } catch (err) {
      console.error("Delete dataset failed:", err);
      setDeleting(false);
      setShowDeleteConfirm(false);
    }
  }

  async function handleExport() {
    if (!cogniInstance) return;
    try {
      const response = await cogniInstance.fetch(`/v1/activity/export/${datasetId}`);
      if (!response.ok) {
        alert("Export not available. The activity endpoint may not be deployed on this instance.");
        return;
      }
      const markdown = await response.text();
      const blob = new Blob([markdown], { type: "text/markdown" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${datasetName}-memory-export.md`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      alert("Export not available. The activity endpoint may not be deployed on this instance.");
    }
  }

  async function handleShare(principalId: string) {
    if (!cogniInstance) return;
    setSharing(principalId);
    try {
      await cogniInstance.fetch(`/v1/permissions/datasets/${principalId}?permission_name=read`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify([datasetId]),
      });
      setSharedWith((prev) => new Set([...prev, principalId]));
    } catch (err) {
      console.error("Share failed:", err);
    } finally {
      setSharing(null);
    }
  }

  const filtered = search ? files.filter((f) => f.name.toLowerCase().includes(search.toLowerCase())) : files;

  if (loading || isInitializing) {
    return <div style={{ padding: 32, display: "flex", alignItems: "center", justifyContent: "center", height: "100%" }}><span style={{ fontSize: 14, color: "#71717A" }}>Loading files...</span></div>;
  }

  return (
    <div
      style={{ padding: 32, display: "flex", flexDirection: "column", gap: 24, fontFamily: '"Inter", system-ui, sans-serif', height: "100%" }}
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
            <span style={{ fontSize: 22, fontWeight: 600, color: "#18181B" }}>{datasetName}</span>
            {datasetName === "default_dataset" && (
              <span style={{ background: "#F0EDFF", color: "#6C5CE7", fontSize: 11, fontWeight: 500, padding: "2px 8px", borderRadius: 4 }}>Default</span>
            )}
          </div>
          <span style={{ fontSize: 14, color: "#71717A" }}>{files.length} documents</span>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button
            onClick={() => setShowDeleteConfirm(true)}
            className="cursor-pointer hover:bg-red-50"
            style={{ background: "#fff", color: "#EF4444", border: "1px solid #E4E4E7", borderRadius: 6, padding: "8px 16px", fontSize: 13, fontWeight: 500, display: "flex", alignItems: "center", gap: 6 }}
          >
            <TrashIcon />
            Delete
          </button>
          <button
            onClick={() => setShowShareModal(true)}
            className="cursor-pointer hover:bg-cognee-hover"
            style={{ background: "#fff", color: "#3F3F46", border: "1px solid #E4E4E7", borderRadius: 6, padding: "8px 16px", fontSize: 13, fontWeight: 500, display: "flex", alignItems: "center", gap: 6 }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#3F3F46" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="18" cy="5" r="3" /><circle cx="6" cy="12" r="3" /><circle cx="18" cy="19" r="3" /><line x1="8.59" y1="13.51" x2="15.42" y2="17.49" /><line x1="15.41" y1="6.51" x2="8.59" y2="10.49" /></svg>
            Share
          </button>
          <button
            onClick={handleExport}
            className="cursor-pointer hover:bg-cognee-hover"
            style={{ background: "#fff", color: "#3F3F46", border: "1px solid #E4E4E7", borderRadius: 6, padding: "8px 16px", fontSize: 13, fontWeight: 500, display: "flex", alignItems: "center", gap: 6 }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#3F3F46" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" /></svg>
            Export
          </button>
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            className="cursor-pointer hover:bg-cognee-purple-hover"
            style={{ background: "#6510F4", color: "#fff", border: "none", borderRadius: 6, padding: "8px 16px", fontSize: 13, fontWeight: 500, display: "flex", alignItems: "center", gap: 6 }}
          >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" /></svg>
          {uploading ? "Uploading..." : "Upload files"}
          </button>
        </div>
      </div>

      {/* Share modal */}
      {showShareModal && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.3)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={() => setShowShareModal(false)}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "#fff", borderRadius: 12, padding: 24, width: 480, maxHeight: "70vh", overflow: "auto", display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 16px 48px rgba(0,0,0,0.12)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <h2 style={{ fontSize: 18, fontWeight: 600, color: "#18181B", margin: 0 }}>Share dataset</h2>
              <button onClick={() => setShowShareModal(false)} className="cursor-pointer" style={{ background: "none", border: "none", color: "#A1A1AA", fontSize: 18 }}>&#10005;</button>
            </div>
            <p style={{ fontSize: 13, color: "#71717A", margin: 0 }}>Grant read access to <strong>{datasetName}</strong> for agents and users.</p>

            {agents.length === 0 ? (
              <span style={{ fontSize: 13, color: "#A1A1AA", padding: "16px 0" }}>No agents or users found.</span>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                {agents.map((a) => {
                  const isShared = sharedWith.has(a.id);
                  const isSharing = sharing === a.id;
                  const displayName = a.is_agent ? a.agent_type : a.email;
                  const sub = a.is_agent ? a.agent_short_id : (a.email === "default_user@example.com" ? "Owner" : "User");
                  return (
                    <div key={a.id} style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 12px", borderRadius: 8, border: "1px solid #F4F4F5" }}>
                      <div style={{ width: 32, height: 32, borderRadius: 8, background: a.is_agent ? "#6510F4" : "#3B82F6", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                        <span style={{ fontSize: 11, fontWeight: 700, color: "#fff" }}>{displayName.slice(0, 2).toUpperCase()}</span>
                      </div>
                      <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 1, minWidth: 0 }}>
                        <span style={{ fontSize: 14, fontWeight: 500, color: "#18181B" }}>{displayName}</span>
                        <span style={{ fontSize: 12, color: "#A1A1AA" }}>{sub}</span>
                      </div>
                      {isShared ? (
                        <span style={{ fontSize: 12, color: "#22C55E", fontWeight: 500, display: "flex", alignItems: "center", gap: 4 }}>
                          <svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M3.5 8.5L6.5 11.5L12.5 4.5" stroke="#22C55E" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
                          Shared
                        </span>
                      ) : (
                        <button
                          onClick={() => handleShare(a.id)}
                          disabled={isSharing}
                          className="cursor-pointer hover:bg-cognee-purple-hover"
                          style={{ background: "#6510F4", color: "#fff", border: "none", borderRadius: 6, padding: "5px 14px", fontSize: 12, fontWeight: 500 }}
                        >
                          {isSharing ? "Sharing..." : "Share"}
                        </button>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Delete confirmation modal */}
      {showDeleteConfirm && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.3)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={() => setShowDeleteConfirm(false)}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "#fff", borderRadius: 12, padding: 24, width: 420, display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 16px 48px rgba(0,0,0,0.12)" }}>
            <h2 style={{ fontSize: 18, fontWeight: 600, color: "#18181B", margin: 0 }}>Delete dataset</h2>
            <p style={{ fontSize: 13, color: "#71717A", margin: 0 }}>
              Are you sure you want to delete <strong>{datasetName}</strong>? This will permanently remove the dataset and all its files. This action cannot be undone.
            </p>
            <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
              <button onClick={() => setShowDeleteConfirm(false)} className="cursor-pointer" style={{ background: "#fff", border: "1px solid #E4E4E7", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "#3F3F46", fontFamily: "inherit" }}>Cancel</button>
              <button onClick={handleDeleteDataset} disabled={deleting} className="cursor-pointer" style={{ background: "#EF4444", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "#fff", fontFamily: "inherit" }}>
                {deleting ? "Deleting..." : "Delete"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Search */}
      <div style={{ background: "#fff", border: "1px solid #E4E4E7", borderRadius: 8, display: "flex", alignItems: "center", gap: 10, height: 40, paddingInline: 14 }}>
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><circle cx="7" cy="7" r="4.5" stroke="#A1A1AA" strokeWidth="1.5" /><path d="M10.5 10.5L14 14" stroke="#A1A1AA" strokeWidth="1.5" strokeLinecap="round" /></svg>
        <input
          type="text" value={search} onChange={(e) => setSearch(e.target.value)}
          placeholder="Search files..."
          style={{ flex: 1, border: "none", outline: "none", fontSize: 14, color: "#18181B", background: "transparent", fontFamily: "inherit" }}
        />
        {search && <button onClick={() => setSearch("")} className="cursor-pointer" style={{ background: "none", border: "none", color: "#A1A1AA", fontSize: 14 }}>&#10005;</button>}
      </div>

      {/* Drag overlay */}
      {isDragging && (
        <div style={{ position: "fixed", inset: 0, zIndex: 50, background: "rgba(101,16,244,0.04)", border: "2px dashed #6510F4", borderRadius: 12, display: "flex", alignItems: "center", justifyContent: "center", pointerEvents: "none" }}>
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
            <div style={{ width: 52, height: 52, background: "#F0EDFF", borderRadius: 12, display: "flex", alignItems: "center", justifyContent: "center" }}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M12 17V7M12 7L7 12M12 7L17 12" stroke="#6C47FF" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>
            </div>
            <span style={{ fontSize: 15, fontWeight: 600, color: "#6C47FF" }}>Drop files to upload</span>
          </div>
        </div>
      )}

      {/* Files table */}
      {filtered.length > 0 ? (
        <div style={{ background: "#fff", border: "1px solid #E5E7EB", borderRadius: 12, overflow: "hidden" }}>
          <div style={{ display: "flex", alignItems: "center", background: "#FAFAF9", borderBottom: "1px solid #E5E7EB", padding: "12px 20px" }}>
            <span style={{ flex: 1, fontSize: 12, fontWeight: 600, color: "#71717A" }}>Name</span>
            <span style={{ width: 100, fontSize: 12, fontWeight: 600, color: "#71717A", flexShrink: 0 }}>Type</span>
            <span style={{ width: 80, fontSize: 12, fontWeight: 600, color: "#71717A", flexShrink: 0 }}>Size</span>
            <span style={{ width: 140, fontSize: 12, fontWeight: 600, color: "#71717A", flexShrink: 0 }}>Added</span>
            <span style={{ width: 40, flexShrink: 0 }} />
          </div>
          {filtered.map((file, i) => {
            const meta = getExtMeta(file.name, file.extension);
            const typeName = getTypeName(file.name, file.extension);
            return (
              <div
                key={file.id}
                className="hover:bg-cognee-hover"
                style={{ display: "flex", alignItems: "center", padding: "14px 20px", borderBottom: i < filtered.length - 1 ? "1px solid #F4F4F5" : "none", transition: "background 150ms" }}
              >
                <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 10 }}>
                  <FileIcon fill={meta.fill} stroke={meta.stroke} text={meta.text} label={meta.label} />
                  <span style={{ fontSize: 13, fontWeight: 500, color: "#18181B" }}>{decodeURIComponent(file.name)}</span>
                </div>
                <span style={{ width: 100, fontSize: 13, color: "#52525B", flexShrink: 0 }}>{typeName}</span>
                <span style={{ width: 80, fontSize: 13, color: "#52525B", flexShrink: 0 }}>—</span>
                <span style={{ width: 140, fontSize: 13, color: "#A1A1AA", flexShrink: 0 }}>{formatDate(file.createdAt)}</span>
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
        <div style={{ flex: 1, background: "#fff", border: "1px solid #E4E4E7", borderRadius: 12, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 12, padding: 48 }}>
          <span style={{ fontSize: 15, color: "#A1A1AA" }}>{search ? "No files match your search" : "No files yet"}</span>
          <button onClick={() => fileInputRef.current?.click()} className="cursor-pointer hover:bg-cognee-purple-hover" style={{ background: "#6510F4", color: "#fff", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500 }}>Upload files</button>
        </div>
      )}

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
