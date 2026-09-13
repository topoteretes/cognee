"use client";

import { useState, type ReactElement } from "react";
import { Loader } from "@mantine/core";
import ModalShell from "@/ui/elements/ModalShell";

export default function UploadOntologyModal({
  onSubmit,
  onClose,
}: {
  // Resolves on success (the parent closes this modal); rejects on failure so
  // the form re-enables for another attempt.
  onSubmit: (key: string, file: File, description?: string) => Promise<void>;
  onClose: () => void;
}): ReactElement {
  const [key, setKey] = useState("");
  const [description, setDescription] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent): Promise<void> => {
    e.preventDefault();
    const trimmedKey = key.trim();
    if (!trimmedKey || !file) return;
    setSubmitting(true);
    try {
      await onSubmit(trimmedKey, file, description.trim() || undefined);
    } catch {
      // Parent surfaces the error notification; just re-enable the form.
      setSubmitting(false);
    }
  };

  return (
    <ModalShell width={440} onClose={() => { if (!submitting) onClose(); }}>
      <h2 style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", margin: 0 }}>Upload Ontology</h2>
      <p style={{ fontSize: 13, color: "rgba(237,236,234,0.55)", margin: 0, lineHeight: "20px" }}>
        Upload an OWL ontology file to guide how Cognee structures your knowledge graph.
      </p>
      <form onSubmit={handleSubmit}>
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <label style={{ fontSize: 12, fontWeight: 700, color: "rgba(237,236,234,0.55)", textTransform: "uppercase", letterSpacing: 0.3 }}>Key</label>
            <input value={key} onChange={(e) => setKey(e.target.value)} type="text" required placeholder="e.g. biomedical-ontology" style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 8, padding: "8px 12px", fontSize: 14, fontFamily: "inherit", color: "#EDECEA", outline: "none" }} />
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <label style={{ fontSize: 12, fontWeight: 700, color: "rgba(237,236,234,0.55)", textTransform: "uppercase", letterSpacing: 0.3 }}>OWL File</label>
            <label className="cursor-pointer" style={{ display: "flex", alignItems: "center", gap: 8, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 8, padding: "8px 12px", fontSize: 13, fontFamily: "inherit", color: "rgba(237,236,234,0.55)" }}>
              <svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M8 1v10M4 5l4-4 4 4" stroke="#A1A1AA" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" /><path d="M1 11v2.5A1.5 1.5 0 002.5 15h11a1.5 1.5 0 001.5-1.5V11" stroke="#A1A1AA" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" /></svg>
              <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{file?.name ?? "Choose a .owl file…"}</span>
              <input name="ontologyFile" type="file" required accept=".owl" style={{ display: "none" }} onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
            </label>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <label style={{ fontSize: 12, fontWeight: 700, color: "rgba(237,236,234,0.55)", textTransform: "uppercase", letterSpacing: 0.3 }}>Description <span style={{ fontWeight: 400, textTransform: "none" }}>(optional)</span></label>
            <input value={description} onChange={(e) => setDescription(e.target.value)} type="text" placeholder="What does this ontology define?" style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 8, padding: "8px 12px", fontSize: 14, fontFamily: "inherit", color: "#EDECEA", outline: "none" }} />
          </div>
        </div>
        <div style={{ display: "flex", gap: 8, justifyContent: "flex-end", marginTop: 8 }}>
          <button type="button" onClick={onClose} className="cursor-pointer" style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "rgba(237,236,234,0.7)", fontFamily: "inherit" }}>Cancel</button>
          <button type="submit" disabled={submitting} className="cursor-pointer" style={{ display: "flex", alignItems: "center", gap: 6, background: "#6510F4", border: "none", borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, color: "#fff", fontFamily: "inherit" }}>
            {submitting && <Loader size={14} color="#fff" />}
            {submitting ? "Uploading..." : "Upload"}
          </button>
        </div>
      </form>
    </ModalShell>
  );
}
