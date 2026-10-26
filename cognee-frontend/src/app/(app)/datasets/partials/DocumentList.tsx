"use client";

import type { ReactElement } from "react";
import { connectorRepository } from "@/modules/datasets/describeTextBlob";
import FileIcon, { getExtMeta, REPOSITORY_META } from "@/ui/elements/FileIcon";
import { decodeFilename, formatDate, formatFileSize } from "@/utils/fileFormat";
import { fmtCount } from "@/utils/formatCount";
import { capRows, MAX_RENDERED_ROWS } from "@/modules/datasets/maxRenderedRows";
import ProcessingBadge from "./ProcessingBadge";

export interface DocRow {
  /** Stored processing outcome; undefined while it is not known. */
  completed?: boolean;
  id: string;
  name: string;
  label?: string;
  extension?: string;
  /** Set by callers that map API rows to their own shape. */
  size?: number;
  /** What the API sends (DataDTO.data_size). Callers passing raw rows use this. */
  dataSize?: number;
  createdAt?: string;
}

export default function DocumentList<T extends DocRow>({
  docs,
  onDelete,
}: {
  docs: T[];
  onDelete: (doc: T) => void;
}): ReactElement {
  // The paging hook owns the live cap; this protects independent consumers.
  const { visible, hidden } = capRows(docs);

  return (
    <>
      {visible.map((doc, i) => {
        const repository = connectorRepository(doc);
        const displayName = repository ?? decodeFilename(doc.name);
        const meta = repository ? REPOSITORY_META : getExtMeta(displayName, doc.extension);
        return (
          <div key={doc.id} style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 16px", borderBottom: i < visible.length - 1 ? "1px solid rgba(255,255,255,0.07)" : "none" }}>
            <FileIcon {...meta} />
            <span style={{ flex: 1, fontSize: 13, color: "#EDECEA", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{displayName}</span>
            <div style={{ display: "flex", alignItems: "center", gap: 12, flexShrink: 0 }}>
              <ProcessingBadge completed={doc.completed} />
              <span style={{ fontSize: 11, color: "rgba(237,236,234,0.55)", fontWeight: 500, minWidth: 32, textAlign: "right" }}>{meta.label}</span>
              <span style={{ fontSize: 11, color: "rgba(237,236,234,0.35)", minWidth: 52, textAlign: "right" }}>{formatFileSize(doc.size ?? doc.dataSize)}</span>
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
      {hidden > 0 && (
        <div style={{ padding: "8px 16px", fontSize: 11, color: "rgba(237,236,234,0.35)", textAlign: "center" }}>
          Showing {fmtCount(MAX_RENDERED_ROWS)} of {fmtCount(docs.length)} documents
        </div>
      )}
    </>
  );
}
