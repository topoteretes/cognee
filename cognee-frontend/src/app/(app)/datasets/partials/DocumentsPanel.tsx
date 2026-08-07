"use client";

import { useRef, useState, type ReactElement } from "react";
import { Loader } from "@mantine/core";
import PageLoading from "@/ui/elements/PageLoading";
import SkeletonBar from "@/ui/elements/SkeletonBar";
import EmptyDocIcon from "@/ui/elements/EmptyDocIcon";
import type { BrainUploadStage } from "@/modules/ingestion/useBrainUpload";
import DocumentList, { type DocRow } from "./DocumentList";

// The Documents column of the brains finder: hidden file input, drag-and-drop,
// header with add/paste actions, upload progress/error banners, and the doc
// list (or the appropriate empty/loading state). Owns the file input ref, the
// drag counter, and the drag-over highlight — all purely presentational.
export default function DocumentsPanel<T extends DocRow>({
  selectedId,
  selectedName,
  docsLoading,
  docsError,
  docs,
  isUploading,
  uploadStage,
  uploadError,
  canRetryBuild,
  onUpload,
  onPaste,
  onDeleteDoc,
  onClearUploadError,
  onRetryBuild,
  onRetryDocs,
}: {
  selectedId: string | null;
  selectedName: string | null;
  docsLoading: boolean;
  docsError: boolean;
  docs: T[];
  isUploading: boolean;
  uploadStage: BrainUploadStage;
  uploadError: string | null;
  canRetryBuild: boolean;
  onUpload: (files: File[]) => void;
  onPaste: () => void;
  onDeleteDoc: (doc: T) => void;
  onClearUploadError: () => void;
  onRetryBuild: () => void;
  onRetryDocs: () => void;
}): ReactElement {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dragCounter = useRef(0);
  const [isDragOver, setIsDragOver] = useState(false);

  return (
    <>
      <input
        ref={fileInputRef}
        type="file"
        multiple
        style={{ display: "none" }}
        onChange={(e) => { if (e.target.files?.length) { onUpload(Array.from(e.target.files)); e.target.value = ""; } }}
      />
      <div
        style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", position: "relative" }}
        onDragEnter={(e) => { e.preventDefault(); if (!selectedId) return; dragCounter.current++; setIsDragOver(true); }}
        onDragOver={(e) => { e.preventDefault(); }}
        onDragLeave={(e) => { e.preventDefault(); dragCounter.current--; if (dragCounter.current === 0) setIsDragOver(false); }}
        onDrop={(e) => { e.preventDefault(); dragCounter.current = 0; setIsDragOver(false); if (!selectedId) return; const files = Array.from(e.dataTransfer.files); if (files.length) onUpload(files); }}
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
          {selectedName ? (
            <>
              <span style={{ fontSize: 11, fontWeight: 700, color: "rgba(237,236,234,0.55)", letterSpacing: "0.08em", textTransform: "uppercase" }}>{selectedName}</span>
              <span style={{ fontSize: 11, color: "rgba(255,255,255,0.2)" }}>·</span>
              <span style={{ fontSize: 11, color: "rgba(237,236,234,0.35)", display: "inline-flex", alignItems: "center", gap: 4 }}>
                {docsLoading ? <SkeletonBar width={36} height={8} /> : <>{docs.length} doc{docs.length !== 1 ? "s" : ""}</>}
              </span>
              <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
                <button onClick={() => fileInputRef.current?.click()} className="hover:bg-[#5A0ED6] cursor-pointer" style={{ background: "#6510F4", color: "#fff", border: "none", borderRadius: 6, padding: "3px 10px", fontSize: 11, fontWeight: 500, cursor: "pointer" }}>Add files</button>
                <button onClick={onPaste} className="hover:bg-[#5A0ED6] cursor-pointer" style={{ background: "#6510F4", color: "#fff", border: "none", borderRadius: 6, padding: "3px 10px", fontSize: 11, fontWeight: 500, cursor: "pointer" }}>Paste text</button>
              </div>
            </>
          ) : (
            <span style={{ fontSize: 11, fontWeight: 700, color: "rgba(237,236,234,0.55)", letterSpacing: "0.08em", textTransform: "uppercase" }}>Documents</span>
          )}
        </div>

        {/* Upload progress / error */}
        {isUploading && (
          <div style={{ padding: "8px 16px", borderBottom: "1px solid rgba(255,255,255,0.07)", background: "rgba(255,255,255,0.04)", display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
            <Loader size={12} color="#6510F4" />
            <span style={{ fontSize: 12, color: "#6510F4" }}>
              {uploadStage === "processing" ? "Building knowledge graph…" : "Uploading…"}
            </span>
          </div>
        )}
        {uploadError && (
          <div style={{ padding: "8px 16px", borderBottom: "1px solid rgba(239,68,68,0.3)", background: "rgba(239,68,68,0.1)", display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
            <span style={{ fontSize: 12, color: "#EF4444", flex: 1 }}>{uploadError}</span>
            {canRetryBuild && (
              <button onClick={onRetryBuild} className="cursor-pointer hover:bg-red-500/20" style={{ background: "rgba(239,68,68,0.2)", border: "1px solid rgba(239,68,68,0.35)", borderRadius: 6, padding: "3px 10px", fontSize: 12, fontWeight: 500, color: "#F87171" }}>Retry build</button>
            )}
            <button onClick={onClearUploadError} style={{ background: "none", border: "none", color: "rgba(237,236,234,0.35)", fontSize: 12, cursor: "pointer" }}>✕</button>
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
          ) : docsError && docs.length === 0 ? (
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", gap: 10 }}>
              <span style={{ fontSize: 13, color: "#F87171", fontWeight: 500 }}>Couldn&rsquo;t load documents</span>
              <span style={{ fontSize: 12, color: "rgba(237,236,234,0.35)", textAlign: "center", maxWidth: 220 }}>Your files are safe — try again.</span>
              <button onClick={onRetryDocs} className="cursor-pointer hover:bg-white/10" style={{ background: "rgba(255,255,255,0.06)", color: "#EDECEA", border: "1px solid rgba(255,255,255,0.15)", borderRadius: 8, padding: "6px 14px", fontSize: 12, fontWeight: 500 }}>Retry</button>
            </div>
          ) : docs.length === 0 ? (
            <div
              style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", gap: 10, cursor: "pointer" }}
              onClick={() => fileInputRef.current?.click()}
            >
              <div style={{ width: 44, height: 44, background: "rgba(188,155,255,0.20)", border: "1px solid rgba(188,155,255,0.35)", borderRadius: 10, display: "flex", alignItems: "center", justifyContent: "center" }}>
                <EmptyDocIcon />
              </div>
              <span style={{ fontSize: 13, color: "rgba(237,236,234,0.55)", fontWeight: 500 }}>No documents yet</span>
              <span style={{ fontSize: 12, color: "rgba(237,236,234,0.35)", textAlign: "center", maxWidth: 220 }}>
                Drag &amp; drop files here, or <span style={{ color: "#6510F4", textDecoration: "underline" }}>browse</span>
              </span>
            </div>
          ) : (
            <DocumentList docs={docs} onDelete={onDeleteDoc} />
          )}
        </div>
      </div>
    </>
  );
}
