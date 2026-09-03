import type { ReactElement } from "react";

export interface FileTypeMeta {
  fill: string;
  stroke: string;
  text: string;
  label: string;
}

// Badge colors per file extension. Hardcoded hex is pre-existing debt carried
// over from the two dataset pages; migrating to globals.css custom properties
// is part of the P2 styling sweep (CLO-289).
const EXT_META: Record<string, FileTypeMeta> = {
  pdf: { fill: "#FEE2E2", stroke: "#EF4444", text: "#DC2626", label: "PDF" },
  docx: { fill: "#DBEAFE", stroke: "#3B82F6", text: "#2563EB", label: "DOC" },
  doc: { fill: "#DBEAFE", stroke: "#3B82F6", text: "#2563EB", label: "DOC" },
  md: { fill: "#F3F4F6", stroke: "#6B7280", text: "#374151", label: "MD" },
  txt: { fill: "#F3F4F6", stroke: "#9CA3AF", text: "#6B7280", label: "TXT" },
  csv: { fill: "#DCFCE7", stroke: "#22C55E", text: "#16A34A", label: "CSV" },
  json: { fill: "#FEF3C7", stroke: "#D97706", text: "#B45309", label: "JSON" },
};

export function getExtMeta(name: string, ext?: string): FileTypeMeta {
  const e = (ext || name.split(".").pop() || "").toLowerCase();
  return (
    EXT_META[e] || {
      fill: "#F3F4F6",
      stroke: "#9CA3AF",
      text: "#6B7280",
      label: e.toUpperCase().slice(0, 4) || "FILE",
    }
  );
}

export default function FileIcon({ fill, stroke, text, label }: FileTypeMeta): ReactElement {
  const fontSize = label.length > 3 ? 4.5 : label.length > 2 ? 5 : 5.5;
  return (
    <svg width="16" height="20" viewBox="0 0 16 20" fill="none" style={{ flexShrink: 0 }}>
      <path d="M10 1H3a2 2 0 00-2 2v14a2 2 0 002 2h10a2 2 0 002-2V6l-5-5z" fill={fill} stroke={stroke} strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M10 1v5h5" stroke={stroke} strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
      <text x="8" y="14.5" textAnchor="middle" fontSize={fontSize} fontWeight="700" fill={text}>{label}</text>
    </svg>
  );
}
