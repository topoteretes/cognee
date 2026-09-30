"use client";

import type { ReactElement } from "react";
import { Loader } from "@mantine/core";
import ModalShell from "@/ui/elements/ModalShell";
import type { CreateBrainTemplateKey } from "../createBrainTemplates";

export default function CreateBrainModal({
  value,
  error,
  creating,
  onChange,
  onSubmit,
  onCancel,
}: {
  value: string;
  error: string;
  creating: boolean;
  onChange: (value: string) => void;
  onSubmit: (templateKey: CreateBrainTemplateKey | null) => void;
  onCancel: () => void;
}): ReactElement {
  const trimmed = value.trim();
  // Spaces are masked to hyphens as you type (see onChange) rather than
  // rejected — periods are the only character still blocked outright.
  const hasInvalidChars = trimmed.includes(".");
  const canSubmit = trimmed.length > 0 && !hasInvalidChars;

  return (
    <ModalShell onClose={onCancel}>
      <h2 style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", margin: 0 }}>Create brain</h2>
      <p style={{ fontSize: 13, color: "rgba(237,236,234,0.55)", margin: 0 }}>Give your brain a name. You can upload documents after creation.</p>
      <input
        autoFocus
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value.replace(/ /g, "-"))}
        onKeyDown={(e) => { if (e.key === "Enter" && canSubmit) onSubmit(null); }}
        placeholder="e.g. product-docs, sec-filings..."
        style={{ width: "100%", height: 40, background: "rgba(255,255,255,0.06)", border: `1px solid ${hasInvalidChars ? "#EF4444" : "rgba(255,255,255,0.12)"}`, borderRadius: 8, paddingInline: 14, fontSize: 14, color: "#EDECEA", fontFamily: "inherit", outline: "none", boxSizing: "border-box" }}
        onFocus={(e) => { if (!hasInvalidChars) { e.target.style.borderColor = "#6510F4"; e.target.style.boxShadow = "0 0 0 3px rgba(188,155,255,0.10)"; } }}
        onBlur={(e) => { e.target.style.borderColor = hasInvalidChars ? "#EF4444" : "rgba(255,255,255,0.12)"; e.target.style.boxShadow = "none"; }}
      />
      {hasInvalidChars ? (
        <p style={{ fontSize: 13, color: "#EF4444", margin: 0 }}>Dataset name cannot contain periods.</p>
      ) : (
        <p style={{ fontSize: 12, color: "rgba(237,236,234,0.35)", margin: 0 }}>Spaces become hyphens as you type — periods aren&rsquo;t allowed. Saved as lowercase.</p>
      )}

      {error && <p style={{ fontSize: 13, color: "#EF4444", margin: 0 }}>{error}</p>}
      <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
        <button onClick={onCancel} className="cursor-pointer"
          style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "rgba(237,236,234,0.7)", fontFamily: "inherit" }}>Cancel</button>
        <button onClick={() => onSubmit(null)} disabled={creating || !canSubmit} className="cursor-pointer"
          style={{ display: "flex", alignItems: "center", gap: 6, background: canSubmit ? "#6510F4" : "rgba(255,255,255,0.06)", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: canSubmit ? "#fff" : "rgba(237,236,234,0.35)", fontFamily: "inherit" }}>
          {creating && <Loader size={14} color="#fff" />}
          {creating ? "Creating…" : "Create"}
        </button>
      </div>
    </ModalShell>
  );
}
