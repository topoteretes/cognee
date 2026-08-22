"use client";

import { useState, useEffect, useCallback, useMemo } from "react";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import { useFilter } from "@/ui/layout/FilterContext";
import { TrackPageView } from "@/modules/analytics";
import getSkills from "@/modules/skills/getSkills";
import getSkill from "@/modules/skills/getSkill";
import type { Skill } from "@/modules/skills/types";
import SkeletonBar from "@/ui/elements/SkeletonBar";
import SkillUploadModal from "./SkillUploadModal";
import SkillShareModal from "./SkillShareModal";

function BuildingIcon() {
  return (
    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#BC9BFF" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
      <rect x="4" y="2" width="16" height="20" rx="1.5" /><path d="M9 22v-4h6v4" /><line x1="8" y1="6" x2="8" y2="6" /><line x1="12" y1="6" x2="12" y2="6" /><line x1="16" y1="6" x2="16" y2="6" /><line x1="8" y1="10" x2="8" y2="10" /><line x1="12" y1="10" x2="12" y2="10" /><line x1="16" y1="10" x2="16" y2="10" /><line x1="8" y1="14" x2="8" y2="14" /><line x1="12" y1="14" x2="12" y2="14" /><line x1="16" y1="14" x2="16" y2="14" />
    </svg>
  );
}

function ToolIcon() {
  return (
    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.55)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
      <path d="M14.7 6.3a4 4 0 0 0-5.4 5.4L3 18v3h3l6.3-6.3a4 4 0 0 0 5.4-5.4l-2.8 2.8-2.1-2.1z" />
    </svg>
  );
}

function ChevronIcon({ open }: { open: boolean }) {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.45)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
      style={{ flexShrink: 0, transform: open ? "rotate(90deg)" : "none", transition: "transform 0.15s" }}>
      <polyline points="9 18 15 12 9 6" />
    </svg>
  );
}

function MaintainerChip({ name, url }: { name: string; url?: string }) {
  const content = (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 5, background: "rgba(188,155,255,0.20)", border: "1px solid rgba(188,155,255,0.35)", borderRadius: 6, padding: "2px 8px", fontSize: 11, fontWeight: 500, color: "#BC9BFF", maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
      <BuildingIcon /> {name}
    </span>
  );
  if (url) {
    return (
      <a href={url} target="_blank" rel="noopener noreferrer" onClick={(e) => e.stopPropagation()} style={{ textDecoration: "none" }} title={`${name} — ${url}`}>
        {content}
      </a>
    );
  }
  return content;
}

function TagPill({ label }: { label: string }) {
  return (
    <span style={{ background: "rgba(255,255,255,0.07)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 5, padding: "1px 7px", fontSize: 10, fontWeight: 500, color: "rgba(237,236,234,0.55)", whiteSpace: "nowrap" }}>
      {label}
    </span>
  );
}

export default function SkillsPage() {
  const { cogniInstance, isInitializing } = useCogniInstance();
  const { datasets, selectedDataset } = useFilter();

  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null);
  // Skills per dataset, populated by scanning every brain. A brain is only
  // shown in the list once it has at least one registered skill.
  const [skillsByDataset, setSkillsByDataset] = useState<Record<string, Skill[]>>({});
  const [scanning, setScanning] = useState(false);
  const [scanned, setScanned] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [uploadOpen, setUploadOpen] = useState(false);
  const [shareSkill, setShareSkill] = useState<Skill | null>(null);
  // Full skill (incl. procedure body), lazily fetched from the detail endpoint on expand.
  const [details, setDetails] = useState<Record<string, Skill>>({});
  const [detailLoadingId, setDetailLoadingId] = useState<string | null>(null);

  // Expand a row and lazily fetch its full detail (procedure body) once.
  const handleToggleExpand = useCallback(
    async (skill: Skill) => {
      const willOpen = expandedId !== skill.id;
      setExpandedId(willOpen ? skill.id : null);
      if (!willOpen || !cogniInstance || !selectedDatasetId || details[skill.id] !== undefined) return;
      setDetailLoadingId(skill.id);
      try {
        const full = await getSkill(cogniInstance, selectedDatasetId, skill.id);
        setDetails((prev) => ({ ...prev, [skill.id]: full }));
      } catch {
        // Leave detail unset — the list-level fields still render.
      } finally {
        setDetailLoadingId((id) => (id === skill.id ? null : id));
      }
    },
    [expandedId, cogniInstance, selectedDatasetId, details],
  );

  // Scan every brain for skills in parallel and keep only the populated ones.
  const scanAll = useCallback(async () => {
    if (!cogniInstance || datasets.length === 0) return;
    setScanning(true);
    setError(null);
    try {
      const entries = await Promise.all(
        datasets.map(async (ds) => {
          try {
            return [ds.id, await getSkills(cogniInstance, ds.id)] as const;
          } catch {
            return [ds.id, [] as Skill[]] as const;
          }
        }),
      );
      setSkillsByDataset(Object.fromEntries(entries));
    } catch {
      setError("Failed to load skills.");
    } finally {
      setScanning(false);
      setScanned(true);
    }
  }, [cogniInstance, datasets]);

  useEffect(() => {
    if (!cogniInstance || isInitializing || datasets.length === 0) return;
    scanAll();
  }, [cogniInstance, isInitializing, datasets, scanAll]);

  // Only brains that actually have skills registered.
  const datasetsWithSkills = useMemo(
    () => datasets.filter((d) => (skillsByDataset[d.id]?.length ?? 0) > 0),
    [datasets, skillsByDataset],
  );

  // Keep the selection valid: default to / fall back to a brain that has skills.
  useEffect(() => {
    if (datasetsWithSkills.length === 0) {
      if (selectedDatasetId) setSelectedDatasetId(null);
      return;
    }
    const stillValid = datasetsWithSkills.some((d) => d.id === selectedDatasetId);
    if (!stillValid) {
      const preferred = selectedDataset && datasetsWithSkills.find((d) => d.id === selectedDataset.id);
      setSelectedDatasetId((preferred ?? datasetsWithSkills[0]).id);
    }
  }, [datasetsWithSkills, selectedDatasetId, selectedDataset]);

  function handleSelectDataset(id: string) {
    if (id === selectedDatasetId) return;
    setExpandedId(null);
    setSelectedDatasetId(id);
  }

  const skills = useMemo(
    () => (selectedDatasetId ? (skillsByDataset[selectedDatasetId] ?? []) : []),
    [selectedDatasetId, skillsByDataset],
  );
  const selectedDatasetName = datasets.find((d) => d.id === selectedDatasetId)?.name ?? "";

  // Client-side search across name, maintainer, description and tags.
  const filteredSkills = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return skills;
    return skills.filter(
      (s) =>
        s.name.toLowerCase().includes(q) ||
        (s.maintainer || "").toLowerCase().includes(q) ||
        (s.description || "").toLowerCase().includes(q) ||
        s.tags.some((t) => t.toLowerCase().includes(q)),
    );
  }, [skills, query]);

  return (
    <div style={{ display: "flex", flexDirection: "column", flex: 1, minHeight: 0, overflow: "hidden" }}>
      <TrackPageView page="Skills" />

      {/* ── Header ── */}
      <div style={{ padding: "24px 32px 16px", display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexShrink: 0 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <h1 style={{ fontSize: 20, fontWeight: 300, color: "#EDECEA", margin: 0, fontFamily: '"TWKLausanne", sans-serif' }}>Skills</h1>
          <p style={{ fontSize: 14, color: "rgba(237,236,234,0.55)", margin: 0 }}>Procedural playbooks loaded by the agent, grouped by their maintainer.</p>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <button onClick={() => setUploadOpen(true)} disabled={datasets.length === 0}
            className="cursor-pointer"
            style={{ background: "#6510F4", color: "#fff", border: "none", borderRadius: 8, padding: "8px 14px", fontSize: 13, fontWeight: 500, display: "flex", alignItems: "center", gap: 6, opacity: datasets.length === 0 ? 0.5 : 1, cursor: datasets.length === 0 ? "not-allowed" : "pointer" }}
            title="Upload a SKILL.md and attach it to brains">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
            </svg>
            Add skill
          </button>
          <button onClick={scanAll} disabled={scanning}
            className="hover:bg-white/10 cursor-pointer"
            style={{ background: "rgba(255,255,255,0.06)", color: "rgba(237,236,234,0.7)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: "8px 12px", fontSize: 13, fontWeight: 500, display: "flex", alignItems: "center", gap: 4 }}
            title="Refresh">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.7)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
              style={scanning ? { animation: "spin 1s linear infinite" } : undefined}>
              <path d="M21 2v6h-6" /><path d="M3 12a9 9 0 0115.36-6.36L21 8" /><path d="M3 22v-6h6" /><path d="M21 12a9 9 0 01-15.36 6.36L3 16" />
            </svg>
          </button>
        </div>
      </div>

      {/* ── Body ── */}
      {scanning && !scanned ? (
        <div style={{ flex: 1, minHeight: 0, display: "flex", overflow: "hidden", marginInline: 32, marginBottom: 32, border: "1px solid rgba(255,255,255,0.12)", borderRadius: 12, background: "rgba(0,0,0,0.82)", backdropFilter: "blur(20px)" }}>
          <div style={{ width: 264, flexShrink: 0, borderRight: "1px solid rgba(255,255,255,0.1)", display: "flex", flexDirection: "column" }}>
            <div style={{ height: 44, padding: "0 14px", borderBottom: "1px solid rgba(255,255,255,0.1)", display: "flex", alignItems: "center" }}>
              <span style={{ fontSize: 11, fontWeight: 700, color: "rgba(237,236,234,0.55)", letterSpacing: "0.08em", textTransform: "uppercase" }}>Brain</span>
            </div>
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} style={{ padding: "11px 14px", borderBottom: "1px solid rgba(255,255,255,0.07)" }}>
                <SkeletonBar width={`${60 + ((i * 13) % 30)}%`} />
              </div>
            ))}
          </div>
          <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
            <div style={{ height: 44, padding: "0 16px", borderBottom: "1px solid rgba(255,255,255,0.1)", display: "flex", alignItems: "center" }}>
              <span style={{ fontSize: 11, fontWeight: 700, color: "rgba(237,236,234,0.45)", letterSpacing: "0.08em", textTransform: "uppercase" }}>Scanning brains for skills…</span>
            </div>
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, padding: "12px 16px", borderBottom: "1px solid rgba(255,255,255,0.07)" }}>
                <SkeletonBar width="40%" />
                <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
                  <SkeletonBar width={48} />
                  <SkeletonBar width={90} />
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : datasetsWithSkills.length > 0 ? (
        <div style={{ flex: 1, minHeight: 0, display: "flex", overflow: "hidden", marginInline: 32, marginBottom: 32, border: "1px solid rgba(255,255,255,0.12)", borderRadius: 12, background: "rgba(0,0,0,0.82)", backdropFilter: "blur(20px)" }}>

          {/* Column 1 — Brains that have skills */}
          <div style={{ width: 264, flexShrink: 0, borderRight: "1px solid rgba(255,255,255,0.1)", display: "flex", flexDirection: "column", overflow: "hidden" }}>
            <div style={{ height: 44, padding: "0 14px", borderBottom: "1px solid rgba(255,255,255,0.1)", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <span style={{ fontSize: 11, fontWeight: 700, color: "rgba(237,236,234,0.55)", letterSpacing: "0.08em", textTransform: "uppercase" }}>Brain</span>
              <span style={{ fontSize: 11, color: "rgba(237,236,234,0.35)" }}>{datasetsWithSkills.length}</span>
            </div>
            <div style={{ flex: 1, overflowY: "auto" }}>
              {datasetsWithSkills.map((ds, i) => {
                const active = ds.id === selectedDatasetId;
                const count = skillsByDataset[ds.id]?.length ?? 0;
                return (
                  <div key={ds.id} onClick={() => handleSelectDataset(ds.id)}
                    style={{
                      display: "flex", alignItems: "center", gap: 8,
                      padding: "8px 14px",
                      borderBottom: i < datasetsWithSkills.length - 1 ? "1px solid rgba(255,255,255,0.07)" : "none",
                      cursor: "pointer",
                      background: active ? "rgba(188,155,255,0.20)" : "transparent",
                      userSelect: "none",
                    }}
                    onMouseEnter={(e) => { if (!active) e.currentTarget.style.background = "rgba(255,255,255,0.06)"; }}
                    onMouseLeave={(e) => { if (!active) e.currentTarget.style.background = "transparent"; }}
                  >
                    <span style={{ flex: 1, fontSize: 13, fontWeight: 500, color: "#EDECEA", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {ds.name}
                    </span>
                    <span style={{ fontSize: 11, color: "rgba(237,236,234,0.35)", flexShrink: 0, minWidth: 16, textAlign: "right" }}>{count}</span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Column 2 — Skills */}
          <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
            <div style={{ height: 44, padding: "0 16px", borderBottom: "1px solid rgba(255,255,255,0.1)", flexShrink: 0, display: "flex", alignItems: "center", gap: 6 }}>
              {selectedDatasetName ? (
                <>
                  <span style={{ fontSize: 11, fontWeight: 700, color: "rgba(237,236,234,0.55)", letterSpacing: "0.08em", textTransform: "uppercase" }}>{selectedDatasetName}</span>
                  <span style={{ fontSize: 11, color: "rgba(255,255,255,0.2)" }}>·</span>
                  <span style={{ fontSize: 11, color: "rgba(237,236,234,0.35)" }}>
                    {query ? `${filteredSkills.length} of ${skills.length}` : skills.length} skill{skills.length !== 1 ? "s" : ""}
                  </span>
                </>
              ) : (
                <span style={{ fontSize: 11, fontWeight: 700, color: "rgba(237,236,234,0.55)", letterSpacing: "0.08em", textTransform: "uppercase" }}>Skills</span>
              )}
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search name, maintainer, tag…"
                style={{ marginLeft: "auto", width: 220, height: 28, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 6, paddingInline: 10, fontSize: 12, color: "#EDECEA", fontFamily: "inherit", outline: "none" }}
                onFocus={(e) => { e.target.style.borderColor = "#6510F4"; }}
                onBlur={(e) => { e.target.style.borderColor = "rgba(255,255,255,0.12)"; }}
              />
            </div>

            <div style={{ flex: 1, overflowY: "auto" }}>
              {error ? (
                <CenterNote text={error} />
              ) : skills.length === 0 ? (
                <SkillsEmptyState />
              ) : filteredSkills.length === 0 ? (
                <CenterNote text={`No skills match "${query}"`} />
              ) : (
                filteredSkills.map((skill, i) => {
                  const open = expandedId === skill.id;
                  return (
                    <div key={skill.id} style={{ borderBottom: i < filteredSkills.length - 1 ? "1px solid rgba(255,255,255,0.07)" : "none" }}>
                      {/* Row */}
                      <div onClick={() => handleToggleExpand(skill)}
                        style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 16px", cursor: "pointer" }}
                        onMouseEnter={(e) => { if (!open) e.currentTarget.style.background = "rgba(255,255,255,0.04)"; }}
                        onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
                      >
                        <ChevronIcon open={open} />
                        <span style={{ width: 7, height: 7, borderRadius: "50%", background: skill.isActive ? "#22C55E" : "#D4D4D8", flexShrink: 0 }} title={skill.isActive ? "Active" : "Inactive"} />
                        <span style={{ fontSize: 13, fontWeight: 500, color: "#EDECEA", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 220 }}>{skill.name}</span>
                        {skill.version && (
                          <span style={{ fontSize: 10, color: "rgba(237,236,234,0.35)", fontFamily: "monospace" }}>v{skill.version}</span>
                        )}
                        <div style={{ display: "flex", alignItems: "center", gap: 6, marginLeft: "auto", flexShrink: 0 }}>
                          <button
                            onClick={(e) => { e.stopPropagation(); setShareSkill(skill); }}
                            title="Add this skill to more brains"
                            aria-label="Add to more brains"
                            className="hover:bg-white/10 cursor-pointer"
                            style={{ display: "flex", alignItems: "center", justifyContent: "center", width: 26, height: 26, borderRadius: 6, background: "transparent", border: "1px solid rgba(255,255,255,0.1)", color: "rgba(237,236,234,0.6)" }}
                          >
                            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                              <circle cx="18" cy="5" r="3" /><circle cx="6" cy="12" r="3" /><circle cx="18" cy="19" r="3" /><line x1="8.59" y1="13.51" x2="15.42" y2="17.49" /><line x1="15.41" y1="6.51" x2="8.59" y2="10.49" />
                            </svg>
                          </button>
                          {skill.tags.slice(0, 2).map((t) => <TagPill key={t} label={t} />)}
                          {skill.declaredTools.length > 0 && (
                            <span style={{ display: "inline-flex", alignItems: "center", gap: 3, fontSize: 11, color: "rgba(237,236,234,0.45)" }}>
                              <ToolIcon /> {skill.declaredTools.length}
                            </span>
                          )}
                          {skill.maintainer
                            ? <MaintainerChip name={skill.maintainer} url={skill.maintainerUrl} />
                            : <span style={{ fontSize: 11, color: "rgba(237,236,234,0.3)", fontStyle: "italic" }}>no maintainer</span>}
                        </div>
                      </div>

                      {/* Expanded detail */}
                      {open && (() => {
                        const detail = details[skill.id];
                        const loadingDetail = detailLoadingId === skill.id;
                        const procedure = detail?.procedure;
                        return (
                        <div style={{ padding: "4px 16px 16px 38px", display: "flex", flexDirection: "column", gap: 12 }}>
                          {skill.description && (
                            <p style={{ fontSize: 13, color: "rgba(237,236,234,0.7)", margin: 0, lineHeight: 1.5, maxWidth: 680 }}>{skill.description}</p>
                          )}
                          <div style={{ display: "flex", flexWrap: "wrap", gap: 24 }}>
                            <DetailField label="Maintainer" value={skill.maintainer || "—"} href={skill.maintainerUrl} />
                            <DetailField label="License" value={skill.license || "—"} />
                            <DetailField label="Repository" value={skill.sourceRepoUrl ? "View source" : "—"} href={skill.sourceRepoUrl} />
                            <DetailField label="Version" value={skill.version ? `v${skill.version}` : "—"} />
                            <DetailField label="Status" value={skill.isActive ? "Active" : "Inactive"} />
                            <DetailField label="Brains" value={skill.datasetScope.length ? String(skill.datasetScope.length) : "—"} />
                            {skill.sourceDir && <DetailField label="Source dir" value={skill.sourceDir} />}
                          </div>
                          {skill.declaredTools.length > 0 && (
                            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                              <span style={{ fontSize: 10, fontWeight: 700, color: "rgba(237,236,234,0.35)", letterSpacing: "0.06em", textTransform: "uppercase" }}>Declared tools</span>
                              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                                {skill.declaredTools.map((t) => (
                                  <span key={t} style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 5, padding: "2px 8px", fontSize: 11, color: "rgba(237,236,234,0.7)", fontFamily: "monospace" }}>{t}</span>
                                ))}
                              </div>
                            </div>
                          )}
                          {/* Skill content (procedure body) — lazily fetched from the detail endpoint */}
                          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                            <span style={{ fontSize: 10, fontWeight: 700, color: "rgba(237,236,234,0.35)", letterSpacing: "0.06em", textTransform: "uppercase" }}>Skill content</span>
                            {loadingDetail ? (
                              <SkeletonBar width="70%" />
                            ) : procedure ? (
                              <pre style={{ margin: 0, maxHeight: 360, overflow: "auto", background: "rgba(0,0,0,0.45)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: "12px 14px", fontSize: 12, lineHeight: 1.55, color: "rgba(237,236,234,0.8)", whiteSpace: "pre-wrap", wordBreak: "break-word", fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" }}>{procedure}</pre>
                            ) : (
                              <span style={{ fontSize: 12, color: "rgba(237,236,234,0.35)", fontStyle: "italic" }}>No content body for this skill.</span>
                            )}
                          </div>
                        </div>
                        );
                      })()}
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>
      ) : (
        <div style={{ flex: 1, display: "flex", flexDirection: "column", paddingInline: 32, paddingBottom: 32 }}>
          <div style={{ flex: 1, background: "rgba(255,255,255,0.06)", backdropFilter: "blur(12px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 16, padding: 48 }}>
            <span style={{ fontSize: 16, fontWeight: 700, color: "#EDECEA" }}>No skills registered yet</span>
            <p style={{ fontSize: 14, color: "rgba(237,236,234,0.35)", margin: 0, maxWidth: 380, textAlign: "center" }}>
              Skills are scoped to a brain. Once a brain has skills ingested via the cognee skills pipeline, it will appear here — brains without skills are hidden.
            </p>
          </div>
        </div>
      )}

      <SkillUploadModal
        isOpen={uploadOpen}
        onClose={() => setUploadOpen(false)}
        datasets={datasets}
        instance={cogniInstance}
        onUploaded={scanAll}
      />

      <SkillShareModal
        isOpen={shareSkill !== null}
        onClose={() => setShareSkill(null)}
        skill={shareSkill}
        sourceDatasetId={selectedDatasetId}
        datasets={datasets}
        instance={cogniInstance}
        onShared={scanAll}
      />

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}

function CenterNote({ text }: { text: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", padding: 24 }}>
      <span style={{ fontSize: 13, color: "rgba(237,236,234,0.45)", textAlign: "center" }}>{text}</span>
    </div>
  );
}

function SkillsEmptyState() {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", gap: 10, padding: 24 }}>
      <div style={{ width: 44, height: 44, background: "rgba(188,155,255,0.20)", border: "1px solid rgba(188,155,255,0.35)", borderRadius: 10, display: "flex", alignItems: "center", justifyContent: "center" }}>
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#BC9BFF" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
          <path d="M14.7 6.3a4 4 0 0 0-5.4 5.4L3 18v3h3l6.3-6.3a4 4 0 0 0 5.4-5.4l-2.8 2.8-2.1-2.1z" />
        </svg>
      </div>
      <span style={{ fontSize: 13, color: "rgba(237,236,234,0.55)", fontWeight: 500 }}>No skills in this brain</span>
      <span style={{ fontSize: 12, color: "rgba(237,236,234,0.35)", textAlign: "center", maxWidth: 280 }}>
        Skills appear here once they are ingested into this brain via the cognee skills pipeline.
      </span>
    </div>
  );
}

function DetailField({ label, value, href }: { label: string; value: string; href?: string }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
      <span style={{ fontSize: 10, fontWeight: 700, color: "rgba(237,236,234,0.35)", letterSpacing: "0.06em", textTransform: "uppercase" }}>{label}</span>
      {href ? (
        <a href={href} target="_blank" rel="noopener noreferrer" style={{ fontSize: 13, color: "#BC9BFF", textDecoration: "none" }}>{value}</a>
      ) : (
        <span style={{ fontSize: 13, color: "rgba(237,236,234,0.7)" }}>{value}</span>
      )}
    </div>
  );
}
