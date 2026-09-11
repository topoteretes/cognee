"use client";

import type { ReactElement } from "react";
import FileIcon, { getExtMeta } from "@/ui/elements/FileIcon";
import TrashIcon from "@/ui/elements/TrashIcon";
import { formatDate, formatFileSize } from "@/utils/fileFormat";
import isMemoryBlobName from "@/modules/datasets/isMemoryBlobName";

export interface FileRow {
  id: string;
  name: string;
  extension?: string;
  size?: number;
  createdAt?: string;
}

function getTypeName(name: string, ext?: string): string {
  const e = (ext || name.split(".").pop() || "").toLowerCase();
  const names: Record<string, string> = { pdf: "PDF", docx: "DOCX", doc: "DOC", md: "Markdown", txt: "Text", csv: "CSV", json: "JSON" };
  return names[e] || e.toUpperCase();
}

export default function FilesTable({
  files,
  memorySessionIds,
  search,
  loadError,
  onDelete,
  onUploadClick,
  onRetry,
  deletingId,
}: {
  files: FileRow[];
  memorySessionIds: Record<string, string | null>;
  search: string;
  loadError: boolean;
  onDelete: (id: string) => void;
  onUploadClick: () => void;
  onRetry: () => void;
  deletingId?: string | null;
}): ReactElement {
  if (loadError && files.length === 0) {
    return (
      <div style={{ flex: 1, background: "rgba(255,255,255,0.06)", backdropFilter: "blur(12px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 12, padding: 48 }}>
        <span style={{ fontSize: 15, color: "#F87171" }}>Couldn&rsquo;t load files</span>
        <span style={{ fontSize: 13, color: "rgba(237,236,234,0.35)", textAlign: "center", maxWidth: 300 }}>Something went wrong reaching the server. Your files are safe — try again.</span>
        <button onClick={onRetry} className="cursor-pointer hover:bg-white/10" style={{ background: "rgba(255,255,255,0.06)", color: "#EDECEA", border: "1px solid rgba(255,255,255,0.15)", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500 }}>Retry</button>
      </div>
    );
  }

  if (files.length === 0) {
    return (
      <div style={{ flex: 1, background: "rgba(255,255,255,0.06)", backdropFilter: "blur(12px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 12, padding: 48 }}>
        <span style={{ fontSize: 15, color: "rgba(237,236,234,0.35)" }}>{search ? "No files match your search" : "No files yet"}</span>
        <button onClick={onUploadClick} className="cursor-pointer hover:bg-[#5A0ED6]" style={{ background: "#6510F4", color: "#fff", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500 }}>Upload files</button>
      </div>
    );
  }

  return (
    <div style={{ background: "rgba(255,255,255,0.06)", backdropFilter: "blur(12px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, overflow: "hidden" }}>
      <div style={{ display: "flex", alignItems: "center", background: "rgba(255,255,255,0.04)", borderBottom: "1px solid rgba(255,255,255,0.1)", padding: "12px 20px" }}>
        <span style={{ flex: 1, fontSize: 12, fontWeight: 700, color: "rgba(237,236,234,0.55)" }}>Name</span>
        <span style={{ width: 100, fontSize: 12, fontWeight: 700, color: "rgba(237,236,234,0.55)", flexShrink: 0 }}>Type</span>
        <span style={{ width: 80, fontSize: 12, fontWeight: 700, color: "rgba(237,236,234,0.55)", flexShrink: 0 }}>Size</span>
        <span style={{ width: 170, fontSize: 12, fontWeight: 700, color: "rgba(237,236,234,0.55)", flexShrink: 0 }}>Added</span>
        <span style={{ width: 40, flexShrink: 0 }} />
      </div>
      {files.map((file, i) => {
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
            style={{ display: "flex", alignItems: "center", padding: "14px 20px", borderBottom: i < files.length - 1 ? "1px solid rgba(255,255,255,0.07)" : "none", transition: "background 150ms" }}
          >
            <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 10 }}>
              <FileIcon fill={meta.fill} stroke={meta.stroke} text={meta.text} label={meta.label} />
              <span style={{ fontSize: 13, fontWeight: 500, color: "#EDECEA" }}>{displayName}</span>
            </div>
            <span style={{ width: 100, fontSize: 13, color: "rgba(237,236,234,0.55)", flexShrink: 0 }}>{typeName}</span>
            <span style={{ width: 80, fontSize: 13, color: "rgba(237,236,234,0.55)", flexShrink: 0 }}>{formatFileSize(file.size)}</span>
            <span style={{ width: 170, fontSize: 13, color: "rgba(237,236,234,0.35)", flexShrink: 0 }}>{formatDate(file.createdAt, true)}</span>
            <div style={{ width: 40, display: "flex", justifyContent: "flex-end", flexShrink: 0 }}>
              <button
                onClick={() => onDelete(file.id)}
                disabled={deletingId === file.id}
                className="cursor-pointer hover:opacity-100 rounded p-1"
                style={{ background: "none", border: "none", opacity: deletingId === file.id ? 0.3 : 0.5, transition: "opacity 150ms", cursor: deletingId === file.id ? "default" : "pointer" }}
                onMouseEnter={(e) => { if (deletingId !== file.id) e.currentTarget.style.opacity = "1"; }}
                onMouseLeave={(e) => { if (deletingId !== file.id) e.currentTarget.style.opacity = "0.5"; }}
                title="Delete file"
              >
                <TrashIcon />
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
}
