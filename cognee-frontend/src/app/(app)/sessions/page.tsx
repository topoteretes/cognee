"use client";

import { useEffect, useState, useCallback, useMemo } from "react";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import { TrackPageView } from "@/modules/analytics";
import SkeletonBar from "@/ui/elements/SkeletonBar";
import {
  listSessions,
  getSessionStats,
  getSessionDetail,
  type SessionRow,
  type SessionStats,
  type SessionDetail,
  type TraceEntry,
  type TimeRange,
} from "@/modules/sessions/getSessions";

const ACCENT = "#6510F4";
const RANGES: TimeRange[] = ["24h", "7d", "30d", "all"];

function statusColor(status: string): string {
  switch ((status || "").toLowerCase()) {
    case "completed": return "#22C55E";
    case "running":   return "#6510F4";
    case "failed":    return "#EF4444";
    case "abandoned": return "#A1A1AA";
    default:          return "#D4D4D8";
  }
}

function statusLabel(s: SessionRow): string {
  return (s.effective_status || s.status || "unknown").toUpperCase();
}

function shortId(id: string): string {
  return id.length <= 40 ? id : id.slice(0, 40) + "…";
}

// Server returns some timestamps as naive ISO (no timezone designator), but
// they are actually UTC. Appending "Z" forces JS to parse them as UTC instead
// of local — without this, CEST users see everything offset by 2 hours.
function parseServerIso(iso: string): Date {
  const hasTz = /Z$|[+-]\d{2}:?\d{2}$/.test(iso);
  return new Date(hasTz ? iso : iso + "Z");
}

function formatDate(iso: string | null): string {
  if (!iso) return "—";
  return parseServerIso(iso).toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function durationSeconds(s: { started_at: string | null; ended_at: string | null; last_activity_at: string | null }): number {
  if (!s.started_at) return 0;
  const start = new Date(s.started_at).getTime();
  const endIso = s.ended_at || s.last_activity_at;
  if (!endIso) return 0;
  return Math.max(0, (new Date(endIso).getTime() - start) / 1000);
}

function formatDuration(sec: number): string {
  if (sec < 1) return `${sec.toFixed(1)}s`;
  if (sec < 60) return `${Math.round(sec)}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ${Math.round(sec % 60)}s`;
  return `${Math.floor(sec / 3600)}h ${Math.floor((sec % 3600) / 60)}m`;
}

function StatusBadge({ status }: { status: string }) {
  const color = statusColor(status);
  return (
    <span style={{ flexShrink: 0, border: `1px solid ${color}`, color, fontSize: 10, fontWeight: 700, letterSpacing: "0.06em", padding: "2px 8px", borderRadius: 4, textTransform: "uppercase", whiteSpace: "nowrap" }}>
      {status}
    </span>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ flex: "1 1 120px", minWidth: 100, border: "1px solid rgba(255,255,255,0.1)", borderRadius: 10, padding: "14px 16px", display: "flex", flexDirection: "column", gap: 6, background: "rgba(255,255,255,0.06)", backdropFilter: "blur(12px)" }}>
      <span style={{ fontSize: 10, fontWeight: 700, color: "rgba(237,236,234,0.35)", letterSpacing: "0.06em", textTransform: "uppercase", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{label}</span>
      <span style={{ fontSize: 22, fontWeight: 300, color: "#EDECEA", fontFamily: '"TWKLausanne", sans-serif', fontVariantNumeric: "tabular-nums", wordBreak: "break-word" }}>{value}</span>
    </div>
  );
}

// ── Transcript ──────────────────────────────────────────────────────────

interface QAItem {
  qa_id?: string;
  time?: string;
  question?: string;
  answer?: string;
  context?: string;
  feedback_text?: string | null;
  feedback_score?: number | null;
  source?: "recall" | "remember";
}

function asQA(raw: Record<string, unknown>): QAItem {
  // The pod returns qas as opaque dicts. Pluck the known fields safely.
  const s = (v: unknown): string | undefined => (typeof v === "string" ? v : undefined);
  const n = (v: unknown): number | null => (typeof v === "number" ? v : null);
  return {
    qa_id: s(raw.qa_id),
    time: s(raw.time),
    question: s(raw.question),
    answer: s(raw.answer),
    context: s(raw.context),
    feedback_text: s(raw.feedback_text) ?? null,
    feedback_score: n(raw.feedback_score),
  };
}

function formatRelativeTime(iso?: string): string {
  if (!iso) return "—";
  const d = parseServerIso(iso);
  const t = d.getTime();
  if (Number.isNaN(t)) return "—";
  const diffMs = Date.now() - t;
  const diffSec = Math.round(diffMs / 1000);
  if (diffSec < 60) return `${diffSec}s ago`;
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
  if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
  return d.toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

// Visual tokens per source. Recall keeps the existing purple "question" look; remember
// gets a green "saved entry" look so the two are unmistakable side-by-side.
const SOURCE_META = {
  recall: {
    label: "Recall",
    bodyLabel: "Result",
    bg: "rgba(188,155,255,0.20)",
    border: "rgba(188,155,255,0.35)",
    color: "#BC9BFF",
    icon: <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>,
  },
  remember: {
    label: "Remember",
    bodyLabel: "Saved",
    bg: "rgba(134,239,172,0.18)",
    border: "rgba(134,239,172,0.40)",
    color: "#86EFAC",
    icon: <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/></svg>,
  },
} as const;

function MessageCard({ qa, index }: { qa: QAItem; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  const doCopy = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!qa.answer) return;
    try { await navigator.clipboard.writeText(qa.answer); setCopied(true); setTimeout(() => setCopied(false), 1500); } catch {}
  };

  const positiveFeedback = qa.feedback_score != null && qa.feedback_score > 0;
  const negativeFeedback = qa.feedback_score != null && qa.feedback_score < 0;
  const canExpand = !!qa.answer;
  const meta = SOURCE_META[qa.source ?? "recall"];

  return (
    <div style={{
      border: "1px solid rgba(255,255,255,0.08)",
      borderRadius: 10,
      background: "rgba(255,255,255,0.03)",
      overflow: "hidden",
      display: "flex", flexDirection: "column",
    }}>
      {/* Question header — clickable to reveal/hide the answer */}
      <button
        onClick={() => canExpand && setExpanded(v => !v)}
        className={canExpand ? "cursor-pointer" : ""}
        disabled={!canExpand}
        style={{
          background: "none", border: "none", padding: "12px 14px",
          textAlign: "left", display: "flex", flexDirection: "column", gap: 6,
          width: "100%", color: "inherit", font: "inherit",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8, width: "100%" }}>
          <span style={{
            display: "inline-flex", alignItems: "center", gap: 5,
            background: meta.bg,
            border: `1px solid ${meta.border}`,
            color: meta.color,
            borderRadius: 4, padding: "2px 8px",
            fontSize: 10, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase",
          }}>
            {meta.icon}
            {meta.label}
          </span>
          <span style={{ fontSize: 11, color: "rgba(237,236,234,0.45)", fontVariantNumeric: "tabular-nums" }}>{formatRelativeTime(qa.time)}</span>
          {positiveFeedback && (
            <span title={qa.feedback_text || "Positive feedback"} style={{ display: "inline-flex", color: "#22C55E" }}>
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M7 10v12"/><path d="M15 5.88L14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H7"/></svg>
            </span>
          )}
          {negativeFeedback && (
            <span title={qa.feedback_text || "Negative feedback"} style={{ display: "inline-flex", color: "#EF4444" }}>
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M17 14V2"/><path d="M9 18.12L10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H17"/></svg>
            </span>
          )}
          <span style={{ flex: 1 }} />
          <span style={{ fontSize: 11, color: "rgba(237,236,234,0.25)", fontVariantNumeric: "tabular-nums" }}>#{index + 1}</span>
          {canExpand && (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.55)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ transform: expanded ? "rotate(180deg)" : "rotate(0)", transition: "transform 180ms ease" }}>
              <polyline points="6 9 12 15 18 9" />
            </svg>
          )}
        </div>
        <div style={{ fontSize: 13, lineHeight: 1.55, color: "#EDECEA", whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
          {qa.question || <span style={{ color: "rgba(237,236,234,0.35)", fontStyle: "italic" }}>(no question)</span>}
        </div>
      </button>

      {/* Collapsible answer */}
      {canExpand && expanded && (
        <>
          <div style={{ height: 1, background: "rgba(255,255,255,0.07)" }} />
          <div style={{ padding: "10px 14px 14px", display: "flex", flexDirection: "column", gap: 6 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{
                display: "inline-flex", alignItems: "center", gap: 5,
                background: "rgba(255,255,255,0.07)",
                border: "1px solid rgba(255,255,255,0.12)",
                color: "rgba(237,236,234,0.75)",
                borderRadius: 4, padding: "2px 8px",
                fontSize: 10, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase",
              }}>
                <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M20 6L9 17l-5-5" /></svg>
                {meta.bodyLabel}
              </span>
              <span style={{ flex: 1 }} />
              <button
                onClick={doCopy}
                className="cursor-pointer"
                title={copied ? "Copied" : "Copy answer"}
                style={{
                  display: "inline-flex", alignItems: "center", gap: 4,
                  background: "transparent", border: "1px solid rgba(255,255,255,0.12)",
                  color: copied ? "#22C55E" : "rgba(237,236,234,0.55)",
                  borderRadius: 5, padding: "3px 7px",
                  fontSize: 11, fontWeight: 500,
                }}
              >
                {copied
                  ? <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M20 6L9 17l-5-5" /></svg>
                  : <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>}
                {copied ? "Copied" : "Copy"}
              </button>
            </div>
            <div style={{ fontSize: 13, lineHeight: 1.55, color: "rgba(237,236,234,0.85)", whiteSpace: "pre-wrap", wordBreak: "break-word", fontVariantNumeric: "tabular-nums" }}>
              {qa.answer}
            </div>
          </div>
        </>
      )}

      {/* Footer: feedback text if present */}
      {qa.feedback_text && (
        <div style={{ borderTop: "1px solid rgba(255,255,255,0.07)", padding: "8px 14px", fontSize: 11, color: "rgba(237,236,234,0.55)", display: "flex", alignItems: "center", gap: 6 }}>
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
          <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{qa.feedback_text}</span>
        </div>
      )}
    </div>
  );
}

function Transcript({ qas, traces }: { qas: Record<string, unknown>[]; traces: TraceEntry[] }) {
  // /recall always writes a trace with memory_query=question; /remember/entry never does.
  // Build the recall-question set from traces and tag each QA accordingly.
  // ponytail: heuristic only — replace with an explicit `source` field on SessionQAEntry
  // if recalls ever start writing QAs without a matching trace.
  const recallQuestions = useMemo(
    () => new Set(traces.filter(t => t.memory_query).map(t => String(t.memory_query).trim())),
    [traces]
  );
  const items = useMemo(
    () => qas
      .map(asQA)
      .map(q => ({ ...q, source: recallQuestions.has((q.question ?? "").trim()) ? "recall" as const : "remember" as const }))
      .filter(q => q.question || q.answer),
    [qas, recallQuestions]
  );
  const [query, setQuery] = useState("");
  const filtered = useMemo(() => {
    if (!query.trim()) return items;
    const needle = query.toLowerCase();
    return items.filter(q =>
      (q.question?.toLowerCase().includes(needle)) ||
      (q.answer?.toLowerCase().includes(needle))
    );
  }, [items, query]);

  return (
    <div style={{ border: "1px solid rgba(255,255,255,0.1)", borderRadius: 10, padding: 16, display: "flex", flexDirection: "column", gap: 12, background: "rgba(255,255,255,0.06)", backdropFilter: "blur(12px)" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
        <span style={{ fontSize: 11, fontWeight: 700, color: "rgba(237,236,234,0.7)", letterSpacing: "0.06em", textTransform: "uppercase" }}>
          Transcript · {items.length} {items.length === 1 ? "turn" : "turns"}
        </span>
        {items.length > 2 && (
          <div style={{ position: "relative", flex: "0 1 220px" }}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.4)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ position: "absolute", left: 9, top: "50%", transform: "translateY(-50%)", pointerEvents: "none" }}><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search transcript…"
              style={{
                width: "100%", boxSizing: "border-box",
                background: "rgba(0,0,0,0.3)", border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: 6, padding: "5px 8px 5px 26px",
                fontSize: 12, color: "#EDECEA", outline: "none",
              }}
            />
          </div>
        )}
      </div>

      {items.length === 0 ? (
        <div style={{ fontSize: 12, color: "rgba(237,236,234,0.45)", textAlign: "center", padding: "16px 0" }}>
          No transcript entries yet. Send a /remember/entry with this session_id to populate it.
        </div>
      ) : filtered.length === 0 ? (
        <div style={{ fontSize: 12, color: "rgba(237,236,234,0.45)", textAlign: "center", padding: "16px 0" }}>
          No turns match &ldquo;{query}&rdquo;.
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {filtered.map((qa, i) => (
            <MessageCard key={qa.qa_id ?? i} qa={qa} index={items.indexOf(qa)} />
          ))}
        </div>
      )}
    </div>
  );
}

function ToolInvocations({ traces }: { traces: TraceEntry[] }) {
  const counts = new Map<string, number>();
  for (const t of traces) {
    const name = t.origin_function || "unknown";
    counts.set(name, (counts.get(name) ?? 0) + 1);
  }
  const rows = [...counts.entries()].sort((a, b) => b[1] - a[1]);
  if (rows.length === 0) return null;
  const max = Math.max(...rows.map(([, c]) => c));
  return (
    <div style={{ border: "1px solid rgba(255,255,255,0.1)", borderRadius: 10, padding: 16, display: "flex", flexDirection: "column", gap: 10, background: "rgba(255,255,255,0.06)", backdropFilter: "blur(12px)" }}>
      <span style={{ fontSize: 11, fontWeight: 700, color: "rgba(237,236,234,0.7)", letterSpacing: "0.06em", textTransform: "uppercase" }}>Tool invocations</span>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {rows.map(([name, count]) => (
          <div key={name} style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <span style={{ width: 110, flexShrink: 0, fontSize: 12, color: "rgba(237,236,234,0.7)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{name}</span>
            <div style={{ flex: 1, height: 8, background: "rgba(255,255,255,0.1)", borderRadius: 4, overflow: "hidden" }}>
              <div style={{ width: `${(count / max) * 100}%`, height: "100%", background: ACCENT, borderRadius: 4 }} />
            </div>
            <span style={{ width: 24, textAlign: "right", flexShrink: 0, fontSize: 12, color: "rgba(237,236,234,0.55)", fontVariantNumeric: "tabular-nums" }}>{count}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function CopyableId({ value, label }: { value: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      onClick={async (e) => { e.stopPropagation(); try { await navigator.clipboard.writeText(value); setCopied(true); setTimeout(() => setCopied(false), 1500); } catch {} }}
      className="cursor-pointer"
      title={value}
      style={{ background: "none", border: "none", padding: 0, color: copied ? "#22C55E" : "rgba(237,236,234,0.45)", display: "inline-flex", alignItems: "center" }}
      aria-label={label ? `Copy ${label}` : "Copy"}
    >
      {copied
        ? <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M20 6L9 17l-5-5" /></svg>
        : <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>}
    </button>
  );
}

function MetaRow({ label, value, copyable }: { label: string; value: React.ReactNode; copyable?: string }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "100px 1fr", alignItems: "center", gap: 12, minHeight: 22 }}>
      <span style={{ fontSize: 11, fontWeight: 600, color: "rgba(237,236,234,0.45)", letterSpacing: "0.04em", textTransform: "uppercase" }}>{label}</span>
      <div style={{ display: "flex", alignItems: "center", gap: 6, minWidth: 0 }}>
        <span style={{ fontSize: 12, color: "rgba(237,236,234,0.85)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{value}</span>
        {copyable && <CopyableId value={copyable} label={label.toLowerCase()} />}
      </div>
    </div>
  );
}

function SessionDetailPanel({ detail }: { detail: SessionDetail }) {
  const dur = durationSeconds(detail);
  const tokens = (detail.tokens_in ?? 0) + (detail.tokens_out ?? 0);
  const lastActivity = detail.ended_at ?? detail.last_activity_at;
  const startedFull = formatDate(detail.started_at);
  const startedRel = detail.started_at ? formatRelativeTime(detail.started_at) : null;
  const lastFull = formatDate(lastActivity);
  const lastRel = lastActivity ? formatRelativeTime(lastActivity) : null;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 4, minWidth: 0 }}>
          <span style={{ fontSize: 11, fontWeight: 700, color: "rgba(237,236,234,0.35)", letterSpacing: "0.08em", textTransform: "uppercase" }}>Session</span>
          <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
            <span style={{ fontSize: 18, fontWeight: 700, color: "#EDECEA", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {detail.session_id}
            </span>
            <CopyableId value={detail.session_id} label="session id" />
          </div>
        </div>
        <StatusBadge status={statusLabel(detail)} />
      </div>

      {/* Stat cards */}
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
        <StatCard label="Observations" value={String(detail.msg_count ?? 0)} />
        <StatCard label="Tool calls" value={String(detail.tool_calls ?? 0)} />
        <StatCard label="Tokens" value={tokens.toLocaleString()} />
        <StatCard label="Cost" value={`$${(detail.cost_usd ?? 0).toFixed(4)}`} />
        <StatCard label="Duration" value={formatDuration(dur)} />
      </div>

      {/* Metadata — between KPIs and transcript */}
      <div style={{ border: "1px solid rgba(255,255,255,0.1)", borderRadius: 10, padding: 16, display: "flex", flexDirection: "column", gap: 10, background: "rgba(255,255,255,0.06)", backdropFilter: "blur(12px)" }}>
        <span style={{ fontSize: 11, fontWeight: 700, color: "rgba(237,236,234,0.7)", letterSpacing: "0.06em", textTransform: "uppercase" }}>Metadata</span>
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          {detail.dataset_id && (
            <MetaRow label="Dataset" copyable={detail.dataset_id} value={<span style={{ fontFamily: 'ui-monospace, Menlo, Monaco, "Cascadia Mono", "Segoe UI Mono", "Roboto Mono", monospace', fontSize: 11 }}>{detail.dataset_id}</span>} />
          )}
          {detail.last_model && <MetaRow label="Model" value={detail.last_model} />}
          <MetaRow label="Started" value={<span title={startedFull}>{startedFull}{startedRel ? ` · ${startedRel}` : ""}</span>} />
          <MetaRow label={detail.ended_at ? "Ended" : "Last active"} value={<span title={lastFull}>{lastFull}{lastRel ? ` · ${lastRel}` : ""}</span>} />
        </div>
      </div>

      {/* Conversation transcript — what the agent actually asked and stored */}
      <Transcript qas={detail.qas ?? []} traces={detail.traces ?? []} />

      <ToolInvocations traces={detail.traces ?? []} />

      {/* Recent activity (traces) */}
      {(detail.traces ?? []).length > 0 && (
        <div style={{ border: "1px solid rgba(255,255,255,0.1)", borderRadius: 10, padding: 16, display: "flex", flexDirection: "column", gap: 10, background: "rgba(255,255,255,0.06)", backdropFilter: "blur(12px)" }}>
          <span style={{ fontSize: 11, fontWeight: 700, color: "rgba(237,236,234,0.7)", letterSpacing: "0.06em", textTransform: "uppercase" }}>Recent activity</span>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {detail.traces.slice(-12).reverse().map((t, i) => (
              <div key={t.trace_id ?? i} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: "rgba(237,236,234,0.7)" }}>
                <span style={{ width: 6, height: 6, borderRadius: "50%", background: t.status === "error" ? "#EF4444" : "#22C55E", flexShrink: 0 }} />
                <span style={{ fontWeight: 500, color: "#EDECEA" }}>{t.origin_function || "step"}</span>
                {t.session_feedback && <span style={{ color: "rgba(237,236,234,0.55)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>— {t.session_feedback}</span>}
                {t.error_message && <span style={{ color: "#EF4444", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>— {t.error_message}</span>}
              </div>
            ))}
          </div>
        </div>
      )}

    </div>
  );
}

export default function SessionsPage() {
  const { cogniInstance, isInitializing } = useCogniInstance();

  const [range, setRange] = useState<TimeRange>("30d");
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [stats, setStats] = useState<SessionStats | null>(null);
  const [loading, setLoading] = useState(true);

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<SessionDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async () => {
    if (!cogniInstance) { setLoading(false); return; }
    setLoading(true);
    const [page, st] = await Promise.all([
      listSessions(cogniInstance, { range, limit: 100 }),
      getSessionStats(cogniInstance, range),
    ]);
    setSessions(page.sessions);
    setStats(st);
    setLoading(false);
  }, [cogniInstance, range]);

  useEffect(() => { if (!isInitializing) load(); }, [isInitializing, load]);

  // Manual refresh — re-pulls the list + stats + the currently-open session
  // detail so a new agent entry shows up without a full page reload.
  const refresh = useCallback(async () => {
    if (!cogniInstance) return;
    setRefreshing(true);
    const tasks: Promise<unknown>[] = [
      listSessions(cogniInstance, { range, limit: 100 }).then((page) => setSessions(page.sessions)),
      getSessionStats(cogniInstance, range).then(setStats),
    ];
    if (selectedId) {
      tasks.push(getSessionDetail(cogniInstance, selectedId).then(setDetail));
    }
    await Promise.all(tasks);
    setRefreshing(false);
  }, [cogniInstance, range, selectedId]);

  const selectSession = useCallback(async (id: string) => {
    if (!cogniInstance) return;
    setSelectedId(id);
    setDetail(null);
    setDetailLoading(true);
    const d = await getSessionDetail(cogniInstance, id);
    setDetail(d);
    setDetailLoading(false);
  }, [cogniInstance]);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      <TrackPageView page="Sessions" />

      {/* Header + range filter */}
      <div style={{ padding: "24px 32px 16px", display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexShrink: 0 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <h1 style={{ fontSize: 20, fontWeight: 300, color: "#EDECEA", margin: 0, fontFamily: '"TWKLausanne", sans-serif' }}>Sessions</h1>
          <p style={{ fontSize: 14, color: "rgba(237,236,234,0.55)", margin: 0, display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
            <span>Agent runs that wrote to your memory</span>
            {loading && !stats ? (
              <>
                <span>·</span>
                <SkeletonBar width={70} />
                <span>·</span>
                <SkeletonBar width={80} />
                <span>·</span>
                <SkeletonBar width={50} />
              </>
            ) : stats ? (
              <span>{` · ${stats.sessions} sessions · ${Math.round(stats.success_rate * 100)}% success · $${stats.total_spend_usd.toFixed(2)}`}</span>
            ) : null}
          </p>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ display: "flex", gap: 4, background: "rgba(255,255,255,0.06)", backdropFilter: "blur(12px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: 3 }}>
            {RANGES.map((r) => (
              <button key={r} onClick={() => setRange(r)} className="cursor-pointer"
                style={{ background: range === r ? "rgba(188,155,255,0.20)" : "transparent", color: range === r ? "rgba(188,155,255,0.60)" : "rgba(237,236,234,0.55)", border: "none", borderRadius: 6, padding: "5px 12px", fontSize: 12, fontWeight: 500, boxShadow: "none" }}>
                {r}
              </button>
            ))}
          </div>
          <button
            onClick={refresh}
            disabled={refreshing || isInitializing}
            className="hover:bg-white/10 cursor-pointer"
            style={{ background: "rgba(255,255,255,0.06)", color: "rgba(237,236,234,0.7)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, padding: "8px 12px", fontSize: 13, fontWeight: 500, display: "flex", alignItems: "center", gap: 4 }}
            title="Refresh"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.7)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
              style={refreshing ? { animation: "spin 1s linear infinite" } : undefined}>
              <path d="M21 2v6h-6" /><path d="M3 12a9 9 0 0115.36-6.36L21 8" /><path d="M3 22v-6h6" /><path d="M21 12a9 9 0 01-15.36 6.36L3 16" />
            </svg>
          </button>
        </div>
      </div>

      {loading && sessions.length === 0 ? (
        <div style={{ flex: 1, display: "flex", overflow: "hidden", marginInline: 32, marginBottom: 32, border: "1px solid rgba(255,255,255,0.12)", borderRadius: 12, background: "rgba(0,0,0,0.82)", backdropFilter: "blur(20px)" }}>
          <div style={{ width: 360, flexShrink: 0, borderRight: "1px solid rgba(255,255,255,0.1)", display: "flex", flexDirection: "column", overflow: "hidden" }}>
            <div style={{ height: 44, padding: "0 14px", borderBottom: "1px solid rgba(255,255,255,0.1)", flexShrink: 0, display: "flex", alignItems: "center" }}>
              <SkeletonBar width={80} />
            </div>
            <div style={{ flex: 1, overflowY: "auto" }}>
              {Array.from({ length: 8 }).map((_, i) => (
                <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, padding: "12px 14px", borderBottom: i < 7 ? "1px solid rgba(255,255,255,0.07)" : "none", borderLeft: "2px solid transparent" }}>
                  <span style={{ width: 7, height: 7, borderRadius: "50%", background: "rgba(255,255,255,0.15)", flexShrink: 0, marginTop: 2, alignSelf: "flex-start" }} />
                  <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 6 }}>
                    <SkeletonBar width={220} />
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
                      <SkeletonBar width={140} />
                      <SkeletonBar width={60} />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div style={{ flex: 1, overflowY: "auto", padding: "24px 28px 24px 28px" }}>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", gap: 8 }}>
              <span style={{ fontSize: 13, color: "rgba(237,236,234,0.35)" }}>Select a session to inspect its tools, files, and metadata</span>
            </div>
          </div>
        </div>
      ) : sessions.length === 0 ? (
        <div style={{ flex: 1, display: "flex", flexDirection: "column", paddingInline: 32, paddingBottom: 32 }}>
          <div style={{ flex: 1, background: "rgba(255,255,255,0.06)", backdropFilter: "blur(12px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 12, padding: 48 }}>
            <div style={{ width: 56, height: 56, background: "rgba(188,155,255,0.20)", border: "1px solid rgba(188,155,255,0.35)", borderRadius: 12, display: "flex", alignItems: "center", justifyContent: "center" }}>
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M3 5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H8l-5 4z" /><line x1="7" y1="8" x2="15" y2="8" /><line x1="7" y1="12" x2="12" y2="12" /></svg>
            </div>
            <span style={{ fontSize: 16, fontWeight: 700, color: "#EDECEA" }}>No sessions yet</span>
            <p style={{ fontSize: 14, color: "rgba(237,236,234,0.35)", margin: 0, maxWidth: 360, textAlign: "center" }}>
              When an agent connects to Cognee and reads or writes memory, its session will appear here with tools used, observations, and cost.
            </p>
          </div>
        </div>
      ) : (
        <div style={{ flex: 1, display: "flex", overflow: "hidden", marginInline: 32, marginBottom: 32, border: "1px solid rgba(255,255,255,0.12)", borderRadius: 12, background: "rgba(0,0,0,0.82)", backdropFilter: "blur(20px)" }}>
          {/* List */}
          <div style={{ width: 360, flexShrink: 0, borderRight: "1px solid rgba(255,255,255,0.1)", display: "flex", flexDirection: "column", overflow: "hidden" }}>
            <div style={{ height: 44, padding: "0 14px", borderBottom: "1px solid rgba(255,255,255,0.1)", flexShrink: 0, display: "flex", alignItems: "center" }}>
              <span style={{ fontSize: 11, fontWeight: 700, color: "rgba(237,236,234,0.55)", letterSpacing: "0.08em", textTransform: "uppercase" }}>{sessions.length} sessions</span>
            </div>
            <div style={{ flex: 1, overflowY: "auto" }}>
              {sessions.map((s, i) => {
                const active = s.session_id === selectedId;
                return (
                  <div key={s.session_id} onClick={() => selectSession(s.session_id)}
                    style={{ display: "flex", alignItems: "center", gap: 10, padding: "12px 14px", borderBottom: i < sessions.length - 1 ? "1px solid rgba(255,255,255,0.07)" : "none", cursor: "pointer", background: active ? "rgba(188,155,255,0.20)" : "transparent", borderLeft: active ? `2px solid ${ACCENT}` : "2px solid transparent" }}
                    onMouseEnter={(e) => { if (!active) e.currentTarget.style.background = "rgba(255,255,255,0.06)"; }}
                    onMouseLeave={(e) => { if (!active) e.currentTarget.style.background = "transparent"; }}
                  >
                    <span style={{ width: 7, height: 7, borderRadius: "50%", background: statusColor(s.effective_status || s.status), flexShrink: 0, marginTop: 2, alignSelf: "flex-start" }} />
                    <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 3 }}>
                      <span style={{ fontSize: 13, fontWeight: 500, color: "#EDECEA", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{shortId(s.session_id)}</span>
                      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, minWidth: 0 }}>
                        <span style={{ fontSize: 11, color: "rgba(237,236,234,0.35)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1 }}>
                          {formatDate(s.last_activity_at || s.started_at)}{s.last_model ? ` · ${s.last_model}` : ""}
                        </span>
                        <StatusBadge status={statusLabel(s)} />
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Detail */}
          <div style={{ flex: 1, overflowY: "auto", padding: "24px 28px 24px 28px" }}>
            {!selectedId ? (
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", gap: 8 }}>
                <span style={{ fontSize: 13, color: "rgba(237,236,234,0.35)" }}>Select a session to inspect its tools, files, and metadata</span>
              </div>
            ) : detailLoading ? (
              <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%" }}>
                <span style={{ fontSize: 13, color: "rgba(237,236,234,0.35)" }}>Loading session…</span>
              </div>
            ) : detail ? (
              <SessionDetailPanel detail={detail} />
            ) : (
              <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%" }}>
                <span style={{ fontSize: 13, color: "rgba(237,236,234,0.35)" }}>Could not load this session.</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
