"use client";

import { useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import { useFilter } from "@/ui/layout/FilterContext";
import { listSessions, getSessionDetail, type SessionRow, type SessionDetail, type TraceEntry } from "@/modules/sessions/getSessions";

interface PipelineRun {
  id: string;
  pipeline_name: string;
  status: string;
  dataset_id: string | null;
  dataset_name: string | null;
  owner_email: string | null;
  owner_id: string | null;
  created_at: string | null;
  pipeline_run_id: string | null;
}

interface Span {
  name: string;
  span_id: string;
  parent_span_id: string | null;
  duration_ms: number;
  status: string;
  attributes: Record<string, string | number>;
}

interface Trace {
  trace_id: string;
  root_name: string | null;
  duration_ms: number;
  span_count: number;
  spans: Span[];
}

function formatTimestamp(dateStr: string): string {
  const d = new Date(dateStr);
  return d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
}

function formatDate(dateStr: string): string {
  const d = new Date(dateStr);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

function pipelineLabel(name: string): string {
  if (name.includes("cognify")) return "cognee.cognify";
  if (name.includes("add")) return "cognee.add";
  if (name.includes("search")) return "cognee.search";
  return name;
}

function ownerDisplayName(email: string | null): string {
  if (!email) return "System";
  if (email.endsWith("@cognee.agent")) {
    const local = email.split("@")[0];
    const parts = local.split("-");
    const shortId = parts.pop() || "";
    const type = parts.join(" ");
    return `${type} ${shortId}`;
  }
  if (email === "default_user@example.com") return "You";
  return email.split("@")[0];
}

function statusColor(status: string): string {
  if (status.includes("COMPLETED")) return "#22C55E";
  if (status.includes("ERRORED")) return "#EF4444";
  if (status.includes("STARTED") || status.includes("INITIATED")) return "#F59E0B";
  return "#A1A1AA";
}

function statusLabel(status: string): string {
  if (status.includes("COMPLETED")) return "completed";
  if (status.includes("STARTED")) return "started";
  if (status.includes("INITIATED")) return "initiated";
  if (status.includes("ERRORED")) return "error";
  return status.toLowerCase();
}

function spanCategory(name: string): { label: string; color: string } {
  if (name.includes("llm") || name.includes("completion") || name.includes("acreate_structured")) return { label: "LLM", color: "#F59E0B" };
  if (name.includes("vector") || name.includes("embed")) return { label: "Vector", color: "#3B82F6" };
  if (name.includes("graph")) return { label: "Graph", color: "#8B5CF6" };
  if (name.includes("search") || name.includes("retrieval")) return { label: "Search", color: "#6510F4" };
  return { label: "System", color: "#71717A" };
}

function TraceViewer({ trace }: { trace: Trace }) {
  const spans = trace.spans;
  if (!spans.length) return null;

  const totalMs = trace.duration_ms || Math.max(...spans.map((s) => s.duration_ms));

  // Category summary
  const categories: Record<string, { count: number; totalMs: number }> = {};
  for (const s of spans) {
    const cat = spanCategory(s.name);
    if (!categories[cat.label]) categories[cat.label] = { count: 0, totalMs: 0 };
    categories[cat.label].count++;
    categories[cat.label].totalMs += s.duration_ms;
  }

  // Select important spans (>50ms, or named operations)
  const important = spans
    .filter((s) => !s.parent_span_id || s.duration_ms > 50 || s.name.includes("llm") || s.name.includes("search") || s.name.includes("embed") || s.name.includes("completion"))
    .slice(0, 8);

  const searchQuery = spans.find((s) => s.attributes?.["cognee.search.query"])?.attributes?.["cognee.search.query"];
  const resultCount = spans.find((s) => s.attributes?.["cognee.result.count"])?.attributes?.["cognee.result.count"];

  return (
    <div onClick={(e) => e.stopPropagation()} style={{ background: "#fff", border: "1px solid #E4E4E7", borderRadius: 8, padding: 16, marginTop: 4, marginLeft: 28, display: "flex", flexDirection: "column", gap: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontSize: 12, fontWeight: 600, letterSpacing: "0.04em", color: "#71717A", textTransform: "uppercase" }}>
          Trace &middot; {trace.span_count} spans &middot; {totalMs.toFixed(0)}ms
        </span>
        <span style={{ fontSize: 11, color: "#A1A1AA" }}>OpenTelemetry</span>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        {important.map((span, i) => {
          const cat = spanCategory(span.name);
          const widthPct = totalMs > 0 ? Math.max((span.duration_ms / totalMs) * 100, 2) : 2;
          const shortName = span.name.replace("cognee.", "");
          return (
            <div key={span.span_id || i} style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ width: 140, fontSize: 11, color: "#52525B", fontFamily: '"Fira Code", monospace', flexShrink: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{shortName}</span>
              <div style={{ flex: 1, height: 18, background: "#F4F4F5", borderRadius: 3, position: "relative" }}>
                <div style={{ position: "absolute", top: 0, left: 0, width: `${widthPct}%`, height: 18, background: cat.color, opacity: 0.25, borderRadius: 3 }} />
              </div>
              <span style={{ width: 55, fontSize: 11, color: "#A1A1AA", fontFamily: '"Fira Code", monospace', textAlign: "right", flexShrink: 0 }}>{span.duration_ms.toFixed(0)}ms</span>
            </div>
          );
        })}
      </div>

      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginTop: 2 }}>
        {Object.entries(categories).map(([label, data]) => {
          const cat = spanCategory(label.toLowerCase());
          return (
            <span key={label} style={{ fontSize: 11, color: "#71717A" }}>
              <span style={{ display: "inline-block", width: 6, height: 6, borderRadius: "50%", background: cat.color, marginRight: 4 }} />
              {label}: {data.count} ({data.totalMs.toFixed(0)}ms)
            </span>
          );
        })}
      </div>

      {(searchQuery || resultCount !== undefined) && (
        <div style={{ display: "flex", gap: 16 }}>
          {searchQuery && <span style={{ fontSize: 11, color: "#A1A1AA" }}>Query: &quot;{searchQuery}&quot;</span>}
          {resultCount !== undefined && <span style={{ fontSize: 11, color: "#A1A1AA" }}>Results: {resultCount}</span>}
        </div>
      )}
    </div>
  );
}

function TraceStepList({ traces }: { traces: TraceEntry[] }) {
  if (!traces || traces.length === 0) {
    return <span style={{ fontSize: 12, color: "#A1A1AA" }}>No trace steps recorded for this session.</span>;
  }
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {traces.map((t, idx) => {
        const isError = t.status === "error";
        const origin = t.origin_function || "(unknown)";
        const feedback = (t.session_feedback || "").trim();
        const mrv = t.method_return_value;
        let returnBlurb = "";
        if (typeof mrv === "string") returnBlurb = mrv.slice(0, 180);
        else if (mrv != null) {
          try { returnBlurb = JSON.stringify(mrv).slice(0, 180); } catch { returnBlurb = ""; }
        }
        return (
          <div key={t.trace_id ?? idx} style={{ display: "flex", gap: 10, padding: "8px 10px", background: "#FAFAFA", borderRadius: 6, alignItems: "flex-start" }}>
            <span style={{ flexShrink: 0, fontSize: 10, color: "#A1A1AA", width: 24, textAlign: "right", fontVariantNumeric: "tabular-nums", paddingTop: 2 }}>{idx + 1}</span>
            <span style={{ flexShrink: 0, width: 6, height: 6, marginTop: 6, borderRadius: "50%", background: isError ? "#DC2626" : "#16A34A" }} />
            <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 2 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ fontSize: 12, fontWeight: 500, color: "#18181B", fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" }}>{origin}</span>
                <span style={{ fontSize: 11, color: isError ? "#DC2626" : "#16A34A" }}>{t.status || ""}</span>
                {t.time && <span style={{ fontSize: 11, color: "#A1A1AA", marginLeft: "auto" }}>{formatTimestamp(t.time)}</span>}
              </div>
              {feedback && <span style={{ fontSize: 12, color: "#52525B" }}>{feedback}</span>}
              {returnBlurb && !feedback && (
                <span style={{ fontSize: 11, color: "#71717A", fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  → {returnBlurb}{returnBlurb.length >= 180 ? "…" : ""}
                </span>
              )}
              {t.error_message && (
                <span style={{ fontSize: 11, color: "#DC2626" }}>{t.error_message}</span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

const PAGE_SIZE = 10;

export default function ActivityPage() {
  const { cogniInstance, isInitializing } = useCogniInstance();
  const { selectedAgent, selectedDataset } = useFilter();
  const searchParams = useSearchParams();
  const deepLinkSessionId = searchParams.get("session");
  const [runs, setRuns] = useState<PipelineRun[]>([]);
  const [traces, setTraces] = useState<Trace[]>([]);
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [expandedSessionId, setExpandedSessionId] = useState<string | null>(null);
  const [sessionDetails, setSessionDetails] = useState<Record<string, SessionDetail>>({});
  const [sessionDetailLoading, setSessionDetailLoading] = useState<Record<string, boolean>>({});
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(0);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  useEffect(() => {
    if (!cogniInstance || isInitializing) return;

    let cancelled = false;

    Promise.all([
      cogniInstance.fetch("/v1/activity/pipeline-runs")
        .then((r) => (r.ok ? r.json() : []))
        .catch(() => []),
      cogniInstance.fetch("/v1/activity/spans")
        .then((r) => (r.ok ? r.json() : []))
        .catch(() => []),
      listSessions(cogniInstance, { range: "30d", limit: 50 }),
    ])
      .then(([runData, spanData, sessionsPage]) => {
        if (cancelled) return;
        setRuns(Array.isArray(runData) ? runData : []);
        setTraces(Array.isArray(spanData) ? spanData : []);
        setSessions(sessionsPage.sessions);
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [cogniInstance, isInitializing]);

  // Session detail fetcher: lazy on expand.
  const loadSessionDetail = (sid: string) => {
    if (!cogniInstance) return;
    if (sessionDetails[sid] || sessionDetailLoading[sid]) return;
    setSessionDetailLoading((m) => ({ ...m, [sid]: true }));
    getSessionDetail(cogniInstance, sid).then((d) => {
      if (d) setSessionDetails((m) => ({ ...m, [sid]: d }));
      setSessionDetailLoading((m) => ({ ...m, [sid]: false }));
    });
  };

  // Deep link: when ?session=<id> is present, auto-expand and scroll to it.
  useEffect(() => {
    if (!deepLinkSessionId || !cogniInstance) return;
    setExpandedSessionId(deepLinkSessionId);
    loadSessionDetail(deepLinkSessionId);
    // Scroll is attempted after render — run in a microtask so the DOM
    // has the expanded block in place.
    const t = setTimeout(() => {
      const el = document.getElementById(`session-row-${deepLinkSessionId}`);
      if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 200);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deepLinkSessionId, cogniInstance]);

  // Reset page when filter changes
  useEffect(() => { setPage(0); }, [selectedAgent, selectedDataset]);

  if (loading || isInitializing) {
    return <div style={{ padding: 32, display: "flex", alignItems: "center", justifyContent: "center", height: "100%" }}><span style={{ fontSize: 14, color: "#71717A" }}>Loading activity...</span></div>;
  }

  // Filter by selected agent/dataset
  let filtered = runs;
  if (selectedAgent) {
    filtered = filtered.filter((r) => r.owner_id === selectedAgent.id);
  }
  if (selectedDataset) {
    filtered = filtered.filter((r) => r.dataset_id === selectedDataset.id);
  }

  // Group by day
  const grouped: { day: string; runs: PipelineRun[] }[] = [];
  const dayMap = new Map<string, PipelineRun[]>();
  for (const r of filtered) {
    const day = r.created_at
      ? new Date(r.created_at).toLocaleDateString("en-US", { weekday: "long", month: "short", day: "numeric" })
      : "Unknown";
    if (!dayMap.has(day)) { dayMap.set(day, []); grouped.push({ day, runs: dayMap.get(day)! }); }
    dayMap.get(day)!.push(r);
  }

  // Pagination
  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  const pageStart = page * PAGE_SIZE;
  const pageEnd = pageStart + PAGE_SIZE;

  // Flatten for pagination, then re-group
  const pageRuns = filtered.slice(pageStart, pageEnd);
  const pageGrouped: { day: string; runs: PipelineRun[] }[] = [];
  const pageDayMap = new Map<string, PipelineRun[]>();
  for (const r of pageRuns) {
    const day = r.created_at
      ? new Date(r.created_at).toLocaleDateString("en-US", { weekday: "long", month: "short", day: "numeric" })
      : "Unknown";
    if (!pageDayMap.has(day)) { pageDayMap.set(day, []); pageGrouped.push({ day, runs: pageDayMap.get(day)! }); }
    pageDayMap.get(day)!.push(r);
  }

  return (
    <div style={{ padding: 32, display: "flex", flexDirection: "column", gap: 20, fontFamily: '"Inter", system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <h1 style={{ fontSize: 24, fontWeight: 600, color: "#18181B", margin: 0 }}>Activity</h1>
          <span style={{ fontSize: 14, color: "#71717A" }}>
            {selectedAgent
              ? `Showing activity for ${selectedAgent.agent_type}${selectedDataset ? ` / ${selectedDataset.name}` : ""}`
              : selectedDataset
                ? `Showing activity for dataset: ${selectedDataset.name}`
                : "All pipeline runs and system events"}
          </span>
        </div>
        <span style={{ fontSize: 13, color: "#A1A1AA" }}>{filtered.length} events · {sessions.length} sessions</span>
      </div>

      {/* Sessions timeline */}
      {sessions.length > 0 && (
        <div style={{ borderLeft: "2px solid #E4E4E7", marginLeft: 6, paddingLeft: 20 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8, marginLeft: -27 }}>
            <div style={{ width: 10, height: 10, borderRadius: "50%", background: "#16A34A", flexShrink: 0 }} />
            <span style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.06em", color: "#A1A1AA", textTransform: "uppercase" }}>Sessions</span>
          </div>
          {sessions.map((s) => {
            const isExpanded = expandedSessionId === s.session_id;
            const detail = sessionDetails[s.session_id];
            const detailLoading = !!sessionDetailLoading[s.session_id];
            const failed = s.effective_status === "failed" || s.error_count > 0;
            const dot = failed ? "#EF4444" : s.effective_status === "abandoned" ? "#F59E0B" : "#22C55E";
            const label = detail?.label || s.session_id;
            return (
              <div key={s.session_id} id={`session-row-${s.session_id}`} style={{ marginLeft: -27, marginBottom: 4 }}>
                <div
                  onClick={() => {
                    if (isExpanded) {
                      setExpandedSessionId(null);
                    } else {
                      setExpandedSessionId(s.session_id);
                      loadSessionDetail(s.session_id);
                    }
                  }}
                  style={{
                    display: "flex", gap: 14, padding: "10px 12px 10px 28px", position: "relative",
                    borderRadius: 8,
                    background: isExpanded ? "#F0EDFF" : "transparent",
                    border: isExpanded ? "1px solid #DDD6FE" : "1px solid transparent",
                    cursor: "pointer",
                    transition: "background 150ms",
                  }}
                >
                  <div style={{ position: "absolute", left: 3, top: 14, width: 8, height: 8, borderRadius: "50%", background: dot, border: "2px solid #FAFAF9" }} />
                  <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 2 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{ fontSize: 14, fontWeight: 500, color: "#18181B", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 520 }}>{label}</span>
                      <span style={{ fontSize: 12, fontWeight: 500, color: failed ? "#EF4444" : "#6510F4" }}>session</span>
                      <span style={{ fontSize: 12, color: dot }}>{s.effective_status}</span>
                    </div>
                    <span style={{ fontSize: 13, color: "#A1A1AA" }}>
                      {s.last_model ?? "—"} · {s.tokens_in + s.tokens_out} tokens · {s.error_count} error{s.error_count === 1 ? "" : "s"}
                    </span>
                  </div>
                  <div style={{ width: 80, display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 2, flexShrink: 0 }}>
                    <span style={{ fontSize: 11, color: "#A1A1AA", fontVariantNumeric: "tabular-nums" }}>
                      {s.last_activity_at ? formatTimestamp(s.last_activity_at) : ""}
                    </span>
                    <span style={{ fontSize: 11, color: "#D4D4D8" }}>
                      {s.last_activity_at ? formatDate(s.last_activity_at) : ""}
                    </span>
                  </div>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#D4D4D8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0, alignSelf: "center" }}>
                    <polyline points={isExpanded ? "6 9 12 15 18 9" : "9 18 15 12 9 6"} />
                  </svg>
                </div>

                {isExpanded && (
                  <div style={{ margin: "4px 12px 12px 28px", padding: "12px 16px", background: "#FAFAFA", borderRadius: 8, border: "1px solid #F4F4F5" }}>
                    {detailLoading && !detail ? (
                      <span style={{ fontSize: 12, color: "#A1A1AA" }}>Loading trace…</span>
                    ) : detail ? (
                      <TraceStepList traces={detail.traces} />
                    ) : (
                      <span style={{ fontSize: 12, color: "#A1A1AA" }}>No trace data.</span>
                    )}
                  </div>
                )}
              </div>
            );
          })}
          <div style={{ height: 12 }} />
        </div>
      )}

      {/* Timeline */}
      {pageGrouped.length === 0 ? (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8, padding: 48 }}>
          <span style={{ fontSize: 14, color: "#71717A" }}>No activity yet.</span>
          <span style={{ fontSize: 13, color: "#A1A1AA" }}>Upload data, run cognify, or search to see activity here.</span>
        </div>
      ) : (
        <div style={{ borderLeft: "2px solid #E4E4E7", marginLeft: 6, paddingLeft: 20 }}>
          {pageGrouped.map(({ day, runs: dayRuns }) => (
            <div key={day}>
              {/* Day header */}
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8, marginLeft: -27 }}>
                <div style={{ width: 10, height: 10, borderRadius: "50%", background: "#A1A1AA", flexShrink: 0 }} />
                <span style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.06em", color: "#A1A1AA", textTransform: "uppercase" }}>{day}</span>
              </div>

              {/* Events */}
              {dayRuns.map((r) => {
                const sc = statusColor(r.status);
                const agent = ownerDisplayName(r.owner_email);
                const dsName = r.dataset_name || r.dataset_id?.slice(0, 8) || "—";
                const isCompleted = r.status.includes("COMPLETED");
                const isError = r.status.includes("ERRORED");
                const isExpanded = expandedId === r.id;
                const hasTraces = traces.length > 0 && isCompleted;

                return (
                  <div key={r.id} style={{ marginLeft: -27, marginBottom: 4 }}>
                    <div
                      onClick={() => hasTraces && setExpandedId(isExpanded ? null : r.id)}
                      style={{
                        display: "flex", gap: 14, padding: "10px 12px 10px 28px", position: "relative",
                        borderRadius: 8,
                        background: isExpanded ? "#F0EDFF" : "transparent",
                        border: isExpanded ? "1px solid #DDD6FE" : "1px solid transparent",
                        cursor: hasTraces ? "pointer" : "default",
                        transition: "background 150ms",
                      }}
                    >
                      <div style={{ position: "absolute", left: 3, top: 14, width: 8, height: 8, borderRadius: "50%", background: sc, border: "2px solid #FAFAF9" }} />
                      <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 2 }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                          <span style={{ fontSize: 14, fontWeight: 500, color: "#18181B" }}>{agent}</span>
                          <span style={{ fontSize: 12, fontWeight: 500, color: isError ? "#EF4444" : "#6510F4" }}>{pipelineLabel(r.pipeline_name)}</span>
                          <span style={{ fontSize: 12, color: sc }}>{statusLabel(r.status)}</span>
                        </div>
                        <span style={{ fontSize: 13, color: "#A1A1AA" }}>
                          {isCompleted ? "Processed" : isError ? "Failed on" : "Processing"} dataset <span style={{ color: "#52525B" }}>{dsName}</span>
                        </span>
                      </div>
                      <div style={{ width: 80, display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 2, flexShrink: 0 }}>
                        <span style={{ fontSize: 11, color: "#A1A1AA", fontVariantNumeric: "tabular-nums" }}>{r.created_at ? formatTimestamp(r.created_at) : ""}</span>
                        <span style={{ fontSize: 11, color: "#D4D4D8" }}>{r.created_at ? formatDate(r.created_at) : ""}</span>
                      </div>
                      {hasTraces && (
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#D4D4D8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0, alignSelf: "center" }}>
                          <polyline points={isExpanded ? "6 9 12 15 18 9" : "9 18 15 12 9 6"} />
                        </svg>
                      )}
                    </div>

                    {/* Expanded trace viewer */}
                    {isExpanded && traces.length > 0 && (
                      <TraceViewer trace={traces[0]} />
                    )}
                  </div>
                );
              })}

              <div style={{ height: 12 }} />
            </div>
          ))}
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8, paddingTop: 8 }}>
          <button
            onClick={() => setPage(Math.max(0, page - 1))}
            disabled={page === 0}
            className="cursor-pointer"
            style={{ background: "#fff", border: "1px solid #E4E4E7", borderRadius: 6, padding: "6px 14px", fontSize: 13, color: page === 0 ? "#D4D4D8" : "#3F3F46", fontFamily: "inherit" }}
          >
            Previous
          </button>
          <span style={{ fontSize: 13, color: "#71717A" }}>
            Page {page + 1} of {totalPages}
          </span>
          <button
            onClick={() => setPage(Math.min(totalPages - 1, page + 1))}
            disabled={page >= totalPages - 1}
            className="cursor-pointer"
            style={{ background: "#fff", border: "1px solid #E4E4E7", borderRadius: 6, padding: "6px 14px", fontSize: 13, color: page >= totalPages - 1 ? "#D4D4D8" : "#3F3F46", fontFamily: "inherit" }}
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
