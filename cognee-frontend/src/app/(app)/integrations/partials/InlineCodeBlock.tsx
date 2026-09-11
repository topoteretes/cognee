"use client";

import { useCallback, useState, type ReactElement } from "react";

interface InlineCodeBlockProps {
  code: string;
  toCopy?: string;
  loading?: boolean;
}

export default function InlineCodeBlock({ code, toCopy, loading }: InlineCodeBlockProps): ReactElement {
  const [copied, setCopied] = useState(false);

  const doCopy = useCallback(() => {
    if (loading) return;
    navigator.clipboard.writeText(toCopy ?? code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  }, [loading, toCopy, code]);

  return (
    <div onClick={(e) => { e.stopPropagation(); doCopy(); }} style={{ background: "#18181B", borderRadius: 8, padding: "11px 14px", display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, cursor: loading ? "wait" : "pointer" }}>
      <pre style={{ margin: 0, fontSize: 12.5, fontFamily: 'ui-monospace, Menlo, Monaco, "Cascadia Mono", "Segoe UI Mono", "Roboto Mono", monospace', color: loading ? "rgba(237,236,234,0.45)" : "rgba(237,236,234,0.85)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", flex: 1 }}>
        {loading ? "Loading…" : code}
      </pre>
      <button onClick={(e) => { e.stopPropagation(); doCopy(); }} aria-label={copied ? "Copied" : "Copy"} style={{ background: "none", border: "none", cursor: loading ? "wait" : "pointer", flexShrink: 0, padding: 2, borderRadius: 4 }}>
        {copied
          ? <svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M3.5 8.5L6.5 11.5L12.5 4.5" stroke="#22C55E" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
          : <svg width="14" height="14" viewBox="0 0 16 16" fill="none"><rect x="5" y="5" width="8" height="8" rx="1.5" stroke="#6B7280" strokeWidth="1.5" /><path d="M11 3H4.5A1.5 1.5 0 003 4.5V11" stroke="#6B7280" strokeWidth="1.5" strokeLinecap="round" /></svg>}
      </button>
    </div>
  );
}
