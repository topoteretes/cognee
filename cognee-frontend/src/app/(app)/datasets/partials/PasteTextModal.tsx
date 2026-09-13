"use client";

import type { ReactElement } from "react";
import ModalShell from "@/ui/elements/ModalShell";

export default function PasteTextModal({
  value,
  pasting,
  onChange,
  onSubmit,
  onCancel,
}: {
  value: string;
  pasting: boolean;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onCancel: () => void;
}): ReactElement {
  return (
    <ModalShell onClose={onCancel}>
      <h2 style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", margin: 0 }}>Paste text</h2>
      <p style={{ fontSize: 13, color: "rgba(237,236,234,0.55)", margin: 0 }}>Paste your text below. It will be added as a document to the selected brain.</p>
      <textarea
        autoFocus
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Paste your text here…"
        rows={8}
        style={{ width: "100%", background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 8, padding: "10px 14px", fontSize: 14, color: "#EDECEA", fontFamily: "inherit", outline: "none", resize: "vertical", boxSizing: "border-box" }}
        onFocus={(e) => { e.target.style.borderColor = "#6510F4"; e.target.style.boxShadow = "0 0 0 3px rgba(188,155,255,0.10)"; }}
        onBlur={(e) => { e.target.style.borderColor = "rgba(255,255,255,0.12)"; e.target.style.boxShadow = "none"; }}
      />
      <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
        <button onClick={onCancel} className="cursor-pointer"
          style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "rgba(237,236,234,0.7)", fontFamily: "inherit" }}>Cancel</button>
        <button onClick={onSubmit} disabled={!value.trim() || pasting} className="cursor-pointer"
          style={{ background: value.trim() ? "#6510F4" : "rgba(255,255,255,0.06)", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: value.trim() ? "#fff" : "rgba(237,236,234,0.35)", fontFamily: "inherit" }}>
          {pasting ? "Adding…" : "Add"}
        </button>
      </div>
    </ModalShell>
  );
}
