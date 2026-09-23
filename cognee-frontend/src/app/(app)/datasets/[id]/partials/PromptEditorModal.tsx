"use client";

import type { ReactElement } from "react";
import { Loader } from "@mantine/core";
import ModalShell from "@/ui/elements/ModalShell";

export default function PromptEditorModal({
  name,
  text,
  saving,
  onNameChange,
  onTextChange,
  onSave,
  onDelete,
  onClose,
}: {
  name: string;
  text: string;
  saving: boolean;
  onNameChange: (value: string) => void;
  onTextChange: (value: string) => void;
  onSave: () => void;
  onDelete: () => void;
  onClose: () => void;
}): ReactElement {
  return (
    <ModalShell width={600} onClose={() => { if (!saving) onClose(); }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h2 style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", margin: 0 }}>Edit Prompt</h2>
        <button onClick={onClose} className="cursor-pointer" style={{ background: "none", border: "none", color: "rgba(237,236,234,0.35)", fontSize: 18 }}>&#10005;</button>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <label style={{ fontSize: 12, fontWeight: 700, color: "rgba(237,236,234,0.55)", textTransform: "uppercase", letterSpacing: 0.3 }}>Name</label>
        <input
          type="text"
          value={name}
          onChange={(e) => onNameChange(e.target.value)}
          placeholder="Prompt name"
          style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 8, padding: "8px 12px", fontSize: 14, fontFamily: "inherit", color: "#EDECEA", outline: "none" }}
        />
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 4, flex: 1, minHeight: 0 }}>
        <label style={{ fontSize: 12, fontWeight: 700, color: "rgba(237,236,234,0.55)", textTransform: "uppercase", letterSpacing: 0.3 }}>Prompt</label>
        <textarea
          value={text}
          onChange={(e) => onTextChange(e.target.value)}
          placeholder="Write your extraction prompt here. This prompt will be used by Cognee when extracting entities and relationships from your data.&#10;&#10;Example: Extract all companies, people, and their relationships from the text. Focus on ownership, employment, and partnership relations."
          style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 8, padding: "10px 12px", fontSize: 13, color: "#EDECEA", outline: "none", resize: "vertical", minHeight: 200, maxHeight: 400, lineHeight: "20px" }}
        />
      </div>

      <div style={{ display: "flex", gap: 8, justifyContent: "space-between" }}>
        <button
          onClick={onDelete}
          className="cursor-pointer hover:opacity-100"
          style={{ background: "none", border: "none", padding: 4, opacity: 0.5, transition: "opacity 150ms", display: "flex", alignItems: "center", justifyContent: "center" }}
          title="Delete prompt"
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M3 4h10M6 4V3h4v1M5 4v8.5a.5.5 0 00.5.5h5a.5.5 0 00.5-.5V4" stroke="#EF4444" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" /></svg>
        </button>
        <div style={{ display: "flex", gap: 8 }}>
          <button
            onClick={onClose}
            className="cursor-pointer"
            style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "rgba(237,236,234,0.7)", fontFamily: "inherit" }}
          >
            Cancel
          </button>
          <button
            onClick={onSave}
            disabled={saving}
            className="cursor-pointer"
            style={{ display: "flex", alignItems: "center", gap: 6, background: "#6510F4", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "#fff", fontFamily: "inherit" }}
          >
            {saving && <Loader size={14} color="#fff" />}
            {saving ? "Saving..." : "Save prompt"}
          </button>
        </div>
      </div>
    </ModalShell>
  );
}
