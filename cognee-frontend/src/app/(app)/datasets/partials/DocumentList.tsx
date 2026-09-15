"use client";

import type { ReactElement } from "react";
import FileIcon, { getExtMeta } from "@/ui/elements/FileIcon";
import { decodeFilename, formatDate, formatFileSize } from "@/utils/fileFormat";

export interface DocRow {
  id: string;
  name: string;
  extension?: string;
  size?: number;
  createdAt?: string;
}

export default function DocumentList<T extends DocRow>({
  docs,
  onDelete,
}: {
  docs: T[];
  onDelete: (doc: T) => void;
}): ReactElement {
  return (
    <>
      {docs.map((doc, i) => {
        const displayName = decodeFilename(doc.name);
        const meta = getExtMeta(displayName, doc.extension);
        return (
          <div key={doc.id} style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 16px", borderBottom: i < docs.length - 1 ? "1px solid rgba(255,255,255,0.07)" : "none" }}>
            <FileIcon {...meta} />
            <span style={{ flex: 1, fontSize: 13, color: "#EDECEA", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{displayName}</span>
            <div style={{ display: "flex", alignItems: "center", gap: 12, flexShrink: 0 }}>
              <span style={{ fontSize: 11, color: "rgba(237,236,234,0.55)", fontWeight: 500, minWidth: 32, textAlign: "right" }}>{meta.label}</span>
              <span style={{ fontSize: 11, color: "rgba(237,236,234,0.35)", minWidth: 52, textAlign: "right" }}>{formatFileSize(doc.size)}</span>
              <span style={{ fontSize: 11, color: "rgba(237,236,234,0.35)", minWidth: 80, textAlign: "right", whiteSpace: "nowrap" }}>{formatDate(doc.createdAt)}</span>
              <button
                onClick={() => onDelete(doc)}
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
  );
}
