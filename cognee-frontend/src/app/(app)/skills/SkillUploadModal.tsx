"use client";

import { useRef, useState } from "react";
import Modal from "@/ui/elements/Modal/Modal";
import type { CogneeInstance } from "@/modules/instances/types";
import rememberSkill, { slugifySkillName } from "@/modules/skills/rememberSkill";

interface DatasetOption {
  id: string;
  name: string;
}

interface SkillUploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  datasets: DatasetOption[];
  instance: CogneeInstance | null;
  /** Called after at least one dataset ingested successfully, to refresh the list. */
  onUploaded: () => void;
}

type PerDatasetResult = { id: string; name: string; ok: boolean; error?: string };

const PRIMARY = "#6510F4";
const ACCENT = "#BC9BFF";
const TEXT = "#EDECEA";

export default function SkillUploadModal({ isOpen, onClose, datasets, instance, onUploaded }: SkillUploadModalProps) {
  const [skillName, setSkillName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [body, setBody] = useState("");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [submitting, setSubmitting] = useState(false);
  const [results, setResults] = useState<PerDatasetResult[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const slug = slugifySkillName(skillName);

  function reset() {
    setSkillName("");
    setFile(null);
    setBody("");
    setSelected(new Set());
    setResults(null);
    setError(null);
    setSubmitting(false);
  }

  function handleClose() {
    if (submitting) return;
    reset();
    onClose();
  }

  function toggleDataset(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  async function handleSubmit() {
    if (!instance) return;
    if (!skillName.trim()) { setError("Give the skill a name."); return; }
    if (!file && !body.trim()) { setError("Upload a SKILL.md file or paste the skill content."); return; }
    if (selected.size === 0) { setError("Select at least one brain to attach the skill to."); return; }

    setSubmitting(true);
    setError(null);
    setResults(null);

    // Content source: the uploaded file's bytes, or the pasted markdown body.
    const content: Blob = file ?? new Blob([body], { type: "text/markdown" });
    const targets = datasets.filter((d) => selected.has(d.id));
    // One remember call per dataset — the backend attaches a skill to a single
    // dataset per call. allSettled so one failure doesn't drop the others.
    const settled = await Promise.allSettled(
      targets.map((d) => rememberSkill(d.id, slug, content, instance)),
    );
    const perDataset: PerDatasetResult[] = targets.map((d, i) => {
      const r = settled[i];
      return r.status === "fulfilled"
        ? { id: d.id, name: d.name, ok: true }
        : { id: d.id, name: d.name, ok: false, error: (r.reason as Error)?.message ?? "Failed" };
    });

    setResults(perDataset);
    setSubmitting(false);

    if (perDataset.some((r) => r.ok)) onUploaded();
    if (perDataset.every((r) => r.ok)) {
      setTimeout(() => handleClose(), 900);
    }
  }

  const canSubmit = !submitting && !!skillName.trim() && (!!file || !!body.trim()) && selected.size > 0 && !!instance;

  return (
    <Modal isOpen={isOpen}>
      <div
        role="dialog"
        aria-modal="true"
        style={{
          width: 520, maxWidth: "92vw", maxHeight: "88vh", display: "flex", flexDirection: "column",
          background: "rgba(15,15,15,0.97)", border: "1px solid rgba(255,255,255,0.14)", borderRadius: 14,
          boxShadow: "0 24px 80px rgba(0,0,0,0.6)", overflow: "hidden",
        }}
      >
        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "18px 20px", borderBottom: "1px solid rgba(255,255,255,0.1)" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
            <span style={{ fontSize: 16, fontWeight: 600, color: TEXT }}>Add a skill</span>
            <span style={{ fontSize: 12, color: "rgba(237,236,234,0.5)" }}>Name it, provide the SKILL.md content, and attach to one or more brains.</span>
          </div>
          <button onClick={handleClose} aria-label="Close" disabled={submitting}
            style={{ background: "none", border: "none", color: "rgba(237,236,234,0.6)", cursor: submitting ? "not-allowed" : "pointer", padding: 4 }}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" /></svg>
          </button>
        </div>

        {/* Body */}
        <div style={{ padding: 20, display: "flex", flexDirection: "column", gap: 18, overflowY: "auto" }}>
          {/* Skill name */}
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <span style={{ fontSize: 11, fontWeight: 700, color: "rgba(237,236,234,0.5)", letterSpacing: "0.06em", textTransform: "uppercase" }}>Skill name</span>
            <input
              value={skillName}
              onChange={(e) => { setSkillName(e.target.value); setError(null); }}
              placeholder="e.g. Weather lookup"
              disabled={submitting}
              style={{ height: 38, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.14)", borderRadius: 8, padding: "0 12px", fontSize: 13, color: TEXT, fontFamily: "inherit", outline: "none" }}
              onFocus={(e) => { e.target.style.borderColor = PRIMARY; }}
              onBlur={(e) => { e.target.style.borderColor = "rgba(255,255,255,0.14)"; }}
            />
            {skillName.trim() && (
              <span style={{ fontSize: 11, color: "rgba(237,236,234,0.4)" }}>
                Ingested as <code style={{ color: ACCENT, fontFamily: "monospace" }}>{slug}/SKILL.md</code>
              </span>
            )}
          </div>

          {/* Content: file upload */}
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <span style={{ fontSize: 11, fontWeight: 700, color: "rgba(237,236,234,0.5)", letterSpacing: "0.06em", textTransform: "uppercase" }}>Skill content</span>
            <input
              ref={fileInputRef}
              type="file"
              accept=".md,text/markdown,text/plain"
              onChange={(e) => { setFile(e.target.files?.[0] ?? null); setError(null); }}
              style={{ display: "none" }}
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={submitting}
              style={{
                display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
                border: `1px dashed ${file ? ACCENT : "rgba(255,255,255,0.25)"}`,
                background: file ? "rgba(188,155,255,0.08)" : "rgba(255,255,255,0.03)",
                borderRadius: 10, padding: "16px 12px", cursor: submitting ? "not-allowed" : "pointer",
                color: file ? ACCENT : "rgba(237,236,234,0.6)", fontSize: 13, fontFamily: "inherit",
              }}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              {file ? file.name : "Upload a SKILL.md file"}
            </button>
            {file && (
              <button onClick={() => { setFile(null); if (fileInputRef.current) fileInputRef.current.value = ""; }} disabled={submitting}
                style={{ alignSelf: "flex-start", background: "none", border: "none", color: "rgba(237,236,234,0.45)", fontSize: 11, cursor: "pointer", padding: 0, textDecoration: "underline" }}>
                Remove file
              </button>
            )}

            {/* …or paste body, only when no file chosen */}
            {!file && (
              <>
                <span style={{ fontSize: 11, color: "rgba(237,236,234,0.4)", textAlign: "center" }}>or paste the skill content</span>
                <textarea
                  value={body}
                  onChange={(e) => { setBody(e.target.value); setError(null); }}
                  placeholder={"---\ndescription: ...\nallowed-tools: [Bash, Read]\n---\n\n# Steps\n1. ..."}
                  disabled={submitting}
                  rows={6}
                  style={{ resize: "vertical", background: "rgba(0,0,0,0.35)", border: "1px solid rgba(255,255,255,0.14)", borderRadius: 8, padding: "10px 12px", fontSize: 12, lineHeight: 1.5, color: TEXT, fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", outline: "none" }}
                  onFocus={(e) => { e.target.style.borderColor = PRIMARY; }}
                  onBlur={(e) => { e.target.style.borderColor = "rgba(255,255,255,0.14)"; }}
                />
              </>
            )}
          </div>

          {/* Dataset multi-select */}
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <span style={{ fontSize: 11, fontWeight: 700, color: "rgba(237,236,234,0.5)", letterSpacing: "0.06em", textTransform: "uppercase" }}>
              Attach to brains{selected.size > 0 ? ` · ${selected.size} selected` : ""}
            </span>
            {datasets.length === 0 ? (
              <span style={{ fontSize: 13, color: "rgba(237,236,234,0.4)" }}>No brains available.</span>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", maxHeight: 200, overflowY: "auto", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 10 }}>
                {datasets.map((d, i) => {
                  const checked = selected.has(d.id);
                  return (
                    <label key={d.id}
                      style={{
                        display: "flex", alignItems: "center", gap: 10, padding: "10px 12px", cursor: "pointer",
                        borderBottom: i < datasets.length - 1 ? "1px solid rgba(255,255,255,0.06)" : "none",
                        background: checked ? "rgba(188,155,255,0.12)" : "transparent",
                      }}
                    >
                      <input type="checkbox" checked={checked} onChange={() => toggleDataset(d.id)} disabled={submitting}
                        style={{ accentColor: PRIMARY, width: 15, height: 15, cursor: "pointer" }} />
                      <span style={{ fontSize: 13, color: TEXT, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{d.name}</span>
                    </label>
                  );
                })}
              </div>
            )}
            <span style={{ fontSize: 11, color: "rgba(237,236,234,0.4)" }}>
              A separate ingestion runs per selected brain — skills are dataset-scoped.
            </span>
          </div>

          {/* Errors / results */}
          {error && (
            <div style={{ background: "rgba(239,68,68,0.16)", border: "1px solid rgba(239,68,68,0.6)", borderRadius: 8, padding: "10px 12px", fontSize: 12, color: "#FCA5A5" }}>{error}</div>
          )}
          {results && (
            <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
              {results.map((r) => (
                <div key={r.id} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: r.ok ? "#86EFAC" : "#FCA5A5" }}>
                  <span>{r.ok ? "✓" : "✕"}</span>
                  <span style={{ color: TEXT }}>{r.name}</span>
                  {!r.ok && r.error && <span style={{ color: "rgba(252,165,165,0.8)" }}>— {r.error}</span>}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div style={{ display: "flex", justifyContent: "flex-end", gap: 10, padding: "14px 20px", borderTop: "1px solid rgba(255,255,255,0.1)" }}>
          <button onClick={handleClose} disabled={submitting}
            style={{ background: "transparent", border: "1px solid rgba(255,255,255,0.2)", borderRadius: 8, padding: "8px 16px", fontSize: 13, color: "rgba(237,236,234,0.8)", cursor: submitting ? "not-allowed" : "pointer", fontFamily: "inherit" }}>
            Cancel
          </button>
          <button onClick={handleSubmit} disabled={!canSubmit}
            style={{
              display: "flex", alignItems: "center", gap: 8, borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, fontFamily: "inherit", border: "none",
              background: canSubmit ? PRIMARY : "rgba(255,255,255,0.08)",
              color: canSubmit ? "#fff" : "rgba(237,236,234,0.4)", cursor: canSubmit ? "pointer" : "not-allowed",
            }}>
            {submitting && <span style={{ width: 12, height: 12, borderRadius: "50%", border: "2px solid rgba(255,255,255,0.35)", borderTopColor: "#fff", animation: "spin 0.7s linear infinite" }} />}
            {submitting ? "Ingesting…" : selected.size > 1 ? `Add to ${selected.size} brains` : "Add skill"}
          </button>
        </div>
      </div>
    </Modal>
  );
}
