"use client";

import { useEffect, useState } from "react";
import Modal from "@/ui/elements/Modal/Modal";
import type { CogneeInstance } from "@/modules/instances/types";
import type { Skill } from "@/modules/skills/types";
import getSkill from "@/modules/skills/getSkill";
import rememberSkill, { skillToMarkdown, slugifySkillName } from "@/modules/skills/rememberSkill";

interface DatasetOption {
  id: string;
  name: string;
}

interface SkillShareModalProps {
  isOpen: boolean;
  onClose: () => void;
  /** The skill being shared (list-level; its procedure is fetched on open). */
  skill: Skill | null;
  /** Dataset the skill is currently shown under — used to fetch its full detail. */
  sourceDatasetId: string | null;
  datasets: DatasetOption[];
  instance: CogneeInstance | null;
  onShared: () => void;
}

type PerDatasetResult = { id: string; name: string; ok: boolean; error?: string };

const PRIMARY = "#6510F4";
const TEXT = "#EDECEA";

export default function SkillShareModal({ isOpen, onClose, skill, sourceDatasetId, datasets, instance, onShared }: SkillShareModalProps) {
  const [full, setFull] = useState<Skill | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [submitting, setSubmitting] = useState(false);
  const [results, setResults] = useState<PerDatasetResult[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Fetch the full skill (incl. procedure) when the modal opens.
  useEffect(() => {
    if (!isOpen || !skill || !instance || !sourceDatasetId) return;
    setFull(null);
    setSelected(new Set());
    setResults(null);
    setError(null);
    setLoadingDetail(true);
    getSkill(instance, sourceDatasetId, skill.id)
      .then(setFull)
      .catch(() => setError("Could not load the skill content to share."))
      .finally(() => setLoadingDetail(false));
  }, [isOpen, skill, instance, sourceDatasetId]);

  if (!skill) return null;

  // Brains the skill is NOT already scoped to (can't re-add where it exists).
  const alreadyIn = new Set(skill.datasetScope);
  if (sourceDatasetId) alreadyIn.add(sourceDatasetId);
  const targets = datasets.filter((d) => !alreadyIn.has(d.id));

  function handleClose() {
    if (submitting) return;
    onClose();
  }

  function toggle(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  async function handleSubmit() {
    if (!instance || !full || !skill) return;
    if (selected.size === 0) { setError("Select at least one brain."); return; }

    setSubmitting(true);
    setError(null);
    setResults(null);

    const slug = slugifySkillName(skill.name);
    const content = new Blob([skillToMarkdown(full)], { type: "text/markdown" });
    const chosen = targets.filter((d) => selected.has(d.id));
    const settled = await Promise.allSettled(
      chosen.map((d) => rememberSkill(d.id, slug, content, instance)),
    );
    const perDataset: PerDatasetResult[] = chosen.map((d, i) => {
      const r = settled[i];
      return r.status === "fulfilled"
        ? { id: d.id, name: d.name, ok: true }
        : { id: d.id, name: d.name, ok: false, error: (r.reason as Error)?.message ?? "Failed" };
    });

    setResults(perDataset);
    setSubmitting(false);
    if (perDataset.some((r) => r.ok)) onShared();
    if (perDataset.every((r) => r.ok)) setTimeout(() => handleClose(), 900);
  }

  const canSubmit = !submitting && !loadingDetail && !!full && selected.size > 0 && !!instance;

  return (
    <Modal isOpen={isOpen}>
      <div role="dialog" aria-modal="true"
        style={{ width: 480, maxWidth: "92vw", maxHeight: "88vh", display: "flex", flexDirection: "column", background: "rgba(15,15,15,0.97)", border: "1px solid rgba(255,255,255,0.14)", borderRadius: 14, boxShadow: "0 24px 80px rgba(0,0,0,0.6)", overflow: "hidden" }}>
        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "18px 20px", borderBottom: "1px solid rgba(255,255,255,0.1)" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
            <span style={{ fontSize: 16, fontWeight: 600, color: TEXT }}>Add “{skill.name}” to more brains</span>
            <span style={{ fontSize: 12, color: "rgba(237,236,234,0.5)" }}>Copies this skill into the selected brains.</span>
          </div>
          <button onClick={handleClose} aria-label="Close" disabled={submitting}
            style={{ background: "none", border: "none", color: "rgba(237,236,234,0.6)", cursor: submitting ? "not-allowed" : "pointer", padding: 4 }}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" /></svg>
          </button>
        </div>

        {/* Body */}
        <div style={{ padding: 20, display: "flex", flexDirection: "column", gap: 14, overflowY: "auto" }}>
          <span style={{ fontSize: 11, fontWeight: 700, color: "rgba(237,236,234,0.5)", letterSpacing: "0.06em", textTransform: "uppercase" }}>
            Brains{selected.size > 0 ? ` · ${selected.size} selected` : ""}
          </span>
          {targets.length === 0 ? (
            <span style={{ fontSize: 13, color: "rgba(237,236,234,0.4)" }}>This skill is already in every available brain.</span>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", maxHeight: 260, overflowY: "auto", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 10 }}>
              {targets.map((d, i) => {
                const checked = selected.has(d.id);
                return (
                  <label key={d.id}
                    style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 12px", cursor: "pointer", borderBottom: i < targets.length - 1 ? "1px solid rgba(255,255,255,0.06)" : "none", background: checked ? "rgba(188,155,255,0.12)" : "transparent" }}>
                    <input type="checkbox" checked={checked} onChange={() => toggle(d.id)} disabled={submitting}
                      style={{ accentColor: PRIMARY, width: 15, height: 15, cursor: "pointer" }} />
                    <span style={{ fontSize: 13, color: TEXT, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{d.name}</span>
                  </label>
                );
              })}
            </div>
          )}

          {loadingDetail && <span style={{ fontSize: 12, color: "rgba(237,236,234,0.4)" }}>Loading skill content…</span>}
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
            style={{ display: "flex", alignItems: "center", gap: 8, borderRadius: 8, padding: "8px 16px", fontSize: 13, fontWeight: 500, fontFamily: "inherit", border: "none", background: canSubmit ? PRIMARY : "rgba(255,255,255,0.08)", color: canSubmit ? "#fff" : "rgba(237,236,234,0.4)", cursor: canSubmit ? "pointer" : "not-allowed" }}>
            {submitting && <span style={{ width: 12, height: 12, borderRadius: "50%", border: "2px solid rgba(255,255,255,0.35)", borderTopColor: "#fff", animation: "spin 0.7s linear infinite" }} />}
            {submitting ? "Adding…" : selected.size > 1 ? `Add to ${selected.size} brains` : "Add to brain"}
          </button>
        </div>
      </div>
    </Modal>
  );
}
