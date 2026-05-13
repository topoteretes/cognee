"use client";

import { useState, useEffect, useRef } from "react";
import Link from "next/link";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import { useFilter } from "@/ui/layout/FilterContext";
import getDatasets from "@/modules/datasets/getDatasets";
import getDatasetData from "@/modules/datasets/getDatasetData";
import createDataset from "@/modules/datasets/createDataset";
import deleteDataset from "@/modules/datasets/deleteDataset";

interface DatasetRaw {
  id: string;
  name: string;
  createdAt?: string;
  ownerId?: string;
}

interface Dataset extends DatasetRaw {
  documents: number;
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
  const { refreshDatasets: refreshFilterDatasets } = useFilter();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newName, setNewName] = useState("");
  const [creating, setCreating] = useState(false);
  const [search, setSearch] = useState("");
  const [deleteTarget, setDeleteTarget] = useState<Dataset | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!cogniInstance || isInitializing) return;
    loadDatasets();
  }, [cogniInstance, isInitializing]);

  async function loadDatasets() {
    if (!cogniInstance) return;
    try {
      const raw: DatasetRaw[] = await getDatasets(cogniInstance);
      const list = Array.isArray(raw) ? raw : [];

      // Fetch document counts in parallel
      const withCounts = await Promise.all(
        list.map(async (ds) => {
          try {
            const data = await getDatasetData(ds.id, cogniInstance);
            return { ...ds, documents: Array.isArray(data) ? data.length : 0 };
          } catch {
            return { ...ds, documents: 0 };
          }
        }),
      );
      setDatasets(withCounts);
    } catch {
      setDatasets([]);
    } finally {
      setLoading(false);
    }
  }

  const handleDelete = async (ds: Dataset) => {
    if (!cogniInstance) return;
    setDeletingId(ds.id);
    try {
      await deleteDataset(ds.id, cogniInstance);
      setDatasets((prev) => prev.filter((d) => d.id !== ds.id));
      setDeleteTarget(null);
      refreshFilterDatasets();
    } catch (err) {
      console.error("Failed to delete dataset:", err);
    } finally {
      setDeletingId(null);
    }
  };

  const handleCreate = async () => {
    if (!newName.trim() || !cogniInstance) return;
    setCreating(true);
    try {
      const ds = await createDataset({ name: newName.trim().toLowerCase().replace(/\s+/g, "_") }, cogniInstance);
      setDatasets((prev) => [...prev, { ...ds, documents: 0 }]);
      setNewName("");
      setShowCreateModal(false);
      refreshFilterDatasets();
    } catch (err) {
      console.error("Failed to create dataset:", err);
    } finally {
      setCreating(false);
    }
  };

  if (loading || isInitializing) {
    return (
      <div style={{ padding: 32, display: "flex", alignItems: "center", justifyContent: "center", height: "100%" }}>
        <span style={{ fontSize: 14, color: "#71717A" }}>Loading datasets...</span>
      </div>
    );
  }

  const filtered = search
    ? datasets.filter((ds) => ds.name.toLowerCase().includes(search.toLowerCase()))
    : datasets;

  return (
    <div style={{ padding: 32, display: "flex", flexDirection: "column", gap: 24, fontFamily: '"Inter", system-ui, sans-serif', height: "100%" }}>
      {/* Create modal */}
      {showCreateModal && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.3)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={() => setShowCreateModal(false)}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "#fff", borderRadius: 12, padding: 24, width: 420, display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 16px 48px rgba(0,0,0,0.12)" }}>
            <h2 style={{ fontSize: 18, fontWeight: 600, color: "#18181B", margin: 0 }}>Create dataset</h2>
            <p style={{ fontSize: 13, color: "#71717A", margin: 0 }}>Give your dataset a name. You can upload documents after creation.</p>
            <input
              ref={inputRef}
              autoFocus
              type="text"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") handleCreate(); }}
              placeholder="e.g. product-docs, sec-filings..."
              style={{ width: "100%", height: 40, border: "1px solid #E4E4E7", borderRadius: 8, paddingInline: 14, fontSize: 14, color: "#18181B", fontFamily: "inherit", outline: "none", boxSizing: "border-box" }}
              onFocus={(e) => { e.target.style.borderColor = "#6510F4"; e.target.style.boxShadow = "0 0 0 3px rgba(101,16,244,0.1)"; }}
              onBlur={(e) => { e.target.style.borderColor = "#E4E4E7"; e.target.style.boxShadow = "none"; }}
            />
            <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
              <button onClick={() => { setShowCreateModal(false); setNewName(""); }} className="cursor-pointer" style={{ background: "#fff", border: "1px solid #E4E4E7", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "#3F3F46", fontFamily: "inherit" }}>Cancel</button>
              <button onClick={handleCreate} disabled={creating} className="cursor-pointer" style={{ background: newName.trim() ? "#6510F4" : "#E4E4E7", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: newName.trim() ? "#fff" : "#A1A1AA", fontFamily: "inherit" }}>
                {creating ? "Creating..." : "Create"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete confirmation modal */}
      {deleteTarget && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.3)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={() => setDeleteTarget(null)}>
          <div onClick={(e) => e.stopPropagation()} style={{ background: "#fff", borderRadius: 12, padding: 24, width: 420, display: "flex", flexDirection: "column", gap: 16, boxShadow: "0 16px 48px rgba(0,0,0,0.12)" }}>
            <h2 style={{ fontSize: 18, fontWeight: 600, color: "#18181B", margin: 0 }}>Delete dataset</h2>
            <p style={{ fontSize: 13, color: "#71717A", margin: 0 }}>
              Are you sure you want to delete <strong>{deleteTarget.name}</strong>? This will permanently remove the dataset and all its files. This action cannot be undone.
            </p>
            <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
              <button onClick={() => setDeleteTarget(null)} className="cursor-pointer" style={{ background: "#fff", border: "1px solid #E4E4E7", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "#3F3F46", fontFamily: "inherit" }}>Cancel</button>
              <button onClick={() => handleDelete(deleteTarget)} disabled={deletingId === deleteTarget.id} className="cursor-pointer" style={{ background: "#EF4444", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "#fff", fontFamily: "inherit" }}>
                {deletingId === deleteTarget.id ? "Deleting..." : "Delete"}
              </button>
            </div>
          </div>
        </div>
      )}

      {datasets.length > 0 ? (
        <>
          {/* Header */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <h1 style={{ fontSize: 26, fontWeight: 700, letterSpacing: "-0.03em", color: "#18181B", margin: 0 }}>Datasets</h1>
              <p style={{ fontSize: 14, color: "#71717A", margin: 0 }}>Upload documents to build searchable knowledge graphs.</p>
            </div>
            <button onClick={() => setShowCreateModal(true)} className="hover:bg-cognee-purple-hover cursor-pointer" style={{ background: "#6510F4", color: "#fff", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, display: "flex", alignItems: "center", gap: 6 }}>
              <PlusIcon /> New dataset
            </button>
          </div>

          {/* Search */}
          <div style={{ background: "#fff", border: "1px solid #E4E4E7", borderRadius: 8, padding: "10px 16px", display: "flex", alignItems: "center", gap: 10 }}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><circle cx="7" cy="7" r="5.5" stroke="#A1A1AA" strokeWidth="1.3" /><path d="M11 11l3.5 3.5" stroke="#A1A1AA" strokeWidth="1.3" strokeLinecap="round" /></svg>
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search datasets..."
              style={{ flex: 1, border: "none", outline: "none", fontSize: 14, color: "#18181B", background: "transparent", fontFamily: "inherit" }}
            />
            {search && <button onClick={() => setSearch("")} className="cursor-pointer" style={{ background: "none", border: "none", color: "#A1A1AA", fontSize: 14 }}>&#10005;</button>}
          </div>

          {/* Table */}
          <div style={{ border: "1px solid #E4E4E7", borderRadius: 12, overflow: "hidden" }}>
            <div style={{ display: "flex", background: "#FAFAF9", borderBottom: "1px solid #E4E4E7", padding: "10px 16px" }}>
              <span style={{ flex: 1, fontSize: 12, fontWeight: 600, color: "#71717A", minWidth: 0 }}>Name</span>
              <span style={{ width: 100, fontSize: 12, fontWeight: 600, color: "#71717A", flexShrink: 0 }}>Documents</span>
              <span style={{ width: 100, fontSize: 12, fontWeight: 600, color: "#71717A", flexShrink: 0 }}>Status</span>
              <span style={{ width: 140, fontSize: 12, fontWeight: 600, color: "#71717A", flexShrink: 0 }}>Graph Model</span>
              <span style={{ width: 140, fontSize: 12, fontWeight: 600, color: "#71717A", flexShrink: 0 }}>Created</span>
              <span style={{ width: 60, flexShrink: 0 }} />
            </div>
            {filtered.map((ds, i) => (
              <Link
                key={ds.id}
                href={`/datasets/${ds.id}`}
                className="hover:bg-cognee-hover"
                style={{ display: "flex", alignItems: "center", padding: "12px 16px", borderBottom: i < filtered.length - 1 ? "1px solid #F4F4F5" : "none", cursor: "pointer", transition: "background 150ms", textDecoration: "none" }}
              >
                <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
                  <div style={{ width: 8, height: 8, borderRadius: 2, background: "#6510F4", flexShrink: 0 }} />
                  <span style={{ fontSize: 13, fontWeight: 500, color: "#18181B", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{ds.name}</span>
                  {ds.name === "default_dataset" && (
                    <span style={{ background: "#F0EDFF", color: "#6C5CE7", fontSize: 11, fontWeight: 500, padding: "2px 6px", borderRadius: 4 }}>Default</span>
                  )}
                </div>
                <span style={{ width: 100, fontSize: 13, color: "#52525B", flexShrink: 0 }}>{ds.documents}</span>
                <div style={{ width: 100, display: "flex", alignItems: "center", gap: 6, flexShrink: 0 }}>
                  <div style={{ width: 6, height: 6, borderRadius: "50%", background: ds.documents > 0 ? "#22C55E" : "#A1A1AA" }} />
                  <span style={{ fontSize: 13, color: "#52525B" }}>{ds.documents > 0 ? "Ready" : "Empty"}</span>
                </div>
                <span style={{ width: 140, fontSize: 13, color: "#52525B", flexShrink: 0 }}>Automatic</span>
                <span style={{ width: 140, fontSize: 13, color: "#A1A1AA", flexShrink: 0 }}>{formatDate(ds.createdAt)}</span>
                <div style={{ width: 60, display: "flex", justifyContent: "flex-end", flexShrink: 0 }}>
                  <button
                    onClick={(e) => { e.preventDefault(); e.stopPropagation(); setDeleteTarget(ds); }}
                    className="cursor-pointer hover:text-red-500"
                    style={{ background: "none", border: "none", padding: 4, opacity: 0.5, transition: "opacity 150ms" }}
                    onMouseEnter={(e) => { e.currentTarget.style.opacity = "1"; }}
                    onMouseLeave={(e) => { e.currentTarget.style.opacity = "0.5"; }}
                    title="Delete dataset"
                  >
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M3 4h10M6 4V3h4v1M5 4v8.5a.5.5 0 00.5.5h5a.5.5 0 00.5-.5V4" stroke="#A1A1AA" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" /></svg>
                  </button>
                </div>
              </Link>
            ))}
            {filtered.length === 0 && search && (
              <div style={{ padding: "24px 16px", textAlign: "center" }}>
                <span style={{ fontSize: 13, color: "#A1A1AA" }}>No datasets match &quot;{search}&quot;</span>
              </div>
            )}
          </div>
        </>
      ) : (
        <>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <h1 style={{ fontSize: 26, fontWeight: 700, letterSpacing: "-0.03em", color: "#18181B", margin: 0 }}>Datasets</h1>
            <p style={{ fontSize: 14, color: "#71717A", margin: 0 }}>Upload documents to build searchable knowledge graphs.</p>
          </div>
          <div style={{ flex: 1, background: "#fff", border: "1px solid #E4E4E7", borderRadius: 12, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 16, padding: 48 }}>
            <div style={{ width: 56, height: 56, background: "#F0EDFF", borderRadius: 12, display: "flex", alignItems: "center", justifyContent: "center" }}>
              <EmptyStateIcon />
            </div>
            <span style={{ fontSize: 16, fontWeight: 600, color: "#18181B" }}>No datasets yet</span>
            <p style={{ fontSize: 14, color: "#A1A1AA", margin: 0, maxWidth: 340, textAlign: "center", lineHeight: 1.5 }}>
              Create your first dataset to start uploading documents and building knowledge graphs.
            </p>
            <button onClick={() => setShowCreateModal(true)} className="hover:bg-cognee-purple-hover cursor-pointer" style={{ background: "#6510F4", color: "#fff", border: "none", borderRadius: 8, padding: "8px 20px", fontSize: 14, fontWeight: 500, display: "flex", alignItems: "center", gap: 6, marginTop: 12 }}>
              <PlusIcon /> Create dataset
            </button>
          </div>
        </>
      )}
    </div>
  );
}
