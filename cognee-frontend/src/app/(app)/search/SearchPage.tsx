"use client";

import { useState, useRef, useEffect } from "react";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import recallKnowledge from "@/modules/datasets/recallKnowledge";
import getDatasets from "@/modules/datasets/getDatasets";
import { listSessions, type SessionRow } from "@/modules/sessions/getSessions";

type SearchScope = "graph" | "session" | "trace" | "all";

interface SearchResultItem {
  search_result?: string[];
  dataset_id?: string;
  dataset_name?: string;
  // Recall-style fields (when scope includes session/trace)
  question?: string;
  answer?: string;
  origin_function?: string;
  method_return_value?: unknown;
  session_feedback?: string;
  _source?: string;
}

interface HistoryEntry {
  query: string;
  resultCount: number;
  results: SearchResultItem[];
  time: Date;
}

function timeAgo(date: Date): string {
  const diff = Date.now() - date.getTime();
  const secs = Math.floor(diff / 1000);
  if (secs < 10) return "Just now";
  if (secs < 60) return `${secs}s ago`;
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  return `${Math.floor(mins / 60)}h ago`;
}

export default function SearchPage() {
  const { cogniInstance, isInitializing } = useCogniInstance();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResultItem[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [datasets, setDatasets] = useState<{ id: string; name: string }[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null);
  const [showDatasetDropdown, setShowDatasetDropdown] = useState(false);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [expandedHistory, setExpandedHistory] = useState<number | null>(null);
  const [scope, setScope] = useState<SearchScope>("graph");
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!cogniInstance || isInitializing) return;
    getDatasets(cogniInstance)
      .then((data: { id: string; name: string }[]) => setDatasets(Array.isArray(data) ? data : []))
      .catch(() => {});
    listSessions(cogniInstance, { range: "30d", limit: 10 })
      .then((page) => setSessions(page.sessions))
      .catch(() => {});
  }, [cogniInstance, isInitializing]);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setShowDatasetDropdown(false);
      }
    }
    if (showDatasetDropdown) document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [showDatasetDropdown]);

  const selectedDataset = datasets.find((d) => d.id === selectedDatasetId);
  const searchDatasetIds = selectedDatasetId ? [selectedDatasetId] : datasets.map((d) => d.id);

  const handleSearch = async (q: string) => {
    if (!q.trim() || !cogniInstance) return;
    setQuery(q);
    setIsSearching(true);
    setResults([]);
    setError(null);
    setHasSearched(true);

    try {
      const mostRecentSessionId = sessions[0]?.session_id ?? "";
      let sendScope: string | string[];
      let sendSessionId: string | undefined;
      if (scope === "graph") {
        sendScope = "graph";
      } else if (mostRecentSessionId) {
        sendScope = scope === "all" ? ["graph", "session", "trace", "graph_context"] : scope;
        sendSessionId = mostRecentSessionId;
      } else {
        sendScope = "graph";
      }
      const data = await recallKnowledge(cogniInstance, {
        query: q,
        scope: sendScope as never,
        sessionId: sendSessionId,
        datasetIds: searchDatasetIds,
      });
      const resultData = (Array.isArray(data) ? data : []) as SearchResultItem[];
      setResults(resultData);
      // History total count — sum search_result lengths where present,
      // otherwise count rows (recall shape).
      const totalResults = resultData.reduce(
        (sum: number, r: SearchResultItem) =>
          sum + (Array.isArray(r.search_result) ? r.search_result.length : 1),
        0,
      );
      setHistory((prev) => [
        { query: q, resultCount: totalResults, results: resultData, time: new Date() },
        ...prev.filter((h) => h.query !== q),
      ].slice(0, 20));
      setExpandedHistory(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
    } finally {
      setIsSearching(false);
    }
  };

  const suggestions = [
    "What are the main entities?",
    "Summarize the uploaded documents",
    "What relationships exist in the data?",
  ];

  return (
    <div style={{ padding: 32, display: "flex", flexDirection: "column", gap: 24, fontFamily: '"Inter", system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <h1 style={{ fontSize: 22, fontWeight: 600, color: "#18181B", margin: 0 }}>Query</h1>
        <span style={{ fontSize: 14, color: "#71717A" }}>Search your knowledge graph with natural language.</span>
      </div>

      {/* Dataset selector */}
      {datasets.length > 0 && (
        <div ref={dropdownRef} style={{ position: "relative", alignSelf: "flex-start" }}>
          <button
            onClick={() => setShowDatasetDropdown((v) => !v)}
            className="cursor-pointer"
            style={{
              display: "flex", alignItems: "center", gap: 6,
              background: "#fff", border: "1px solid #E4E4E7", borderRadius: 8,
              padding: "7px 14px", fontSize: 13, fontWeight: 500,
              color: selectedDataset ? "#18181B" : "#71717A", fontFamily: "inherit",
            }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={selectedDataset ? "#6510F4" : "#A1A1AA"} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
              <ellipse cx="12" cy="5" rx="9" ry="3" /><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" /><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
            </svg>
            {selectedDataset ? selectedDataset.name : "All datasets"}
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none" style={{ flexShrink: 0 }}><path d="M3 4.5L6 7.5L9 4.5" stroke="#A1A1AA" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" /></svg>
          </button>
          {showDatasetDropdown && (
            <div style={{ position: "absolute", top: 36, left: 0, width: 240, background: "#fff", borderRadius: 10, boxShadow: "0px 8px 30px #0000001F, 0px 0px 0px 1px #0000000F", padding: 6, zIndex: 50 }}>
              <div
                onClick={() => { setSelectedDatasetId(null); setShowDatasetDropdown(false); }}
                className="cursor-pointer"
                style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 12px", borderRadius: 6, background: !selectedDatasetId ? "#F0EDFF" : "transparent" }}
              >
                <span style={{ fontSize: 13, fontWeight: !selectedDatasetId ? 500 : 400, color: !selectedDatasetId ? "#6510F4" : "#3F3F46" }}>All datasets</span>
                {!selectedDatasetId && <svg width="12" height="12" viewBox="0 0 12 12" fill="none" style={{ marginLeft: "auto", flexShrink: 0 }}><path d="M2.5 6L5 8.5L9.5 3.5" stroke="#6510F4" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>}
              </div>
              {datasets.map((d) => (
                <div
                  key={d.id}
                  onClick={() => { setSelectedDatasetId(d.id); setShowDatasetDropdown(false); }}
                  className="cursor-pointer"
                  style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 12px", borderRadius: 6, background: selectedDatasetId === d.id ? "#F0EDFF" : "transparent" }}
                >
                  <span style={{ fontSize: 13, fontWeight: selectedDatasetId === d.id ? 500 : 400, color: selectedDatasetId === d.id ? "#6510F4" : "#3F3F46" }}>{d.name}</span>
                  {selectedDatasetId === d.id && <svg width="12" height="12" viewBox="0 0 12 12" fill="none" style={{ marginLeft: "auto", flexShrink: 0 }}><path d="M2.5 6L5 8.5L9.5 3.5" stroke="#6510F4" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Search bar */}
      {/* Scope pills */}
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        {(["graph", "session", "trace", "all"] as const).map((s) => {
          const needsSession = s !== "graph";
          const disabled = needsSession && sessions.length === 0;
          const active = scope === s && !disabled;
          const label = s === "graph" ? "Graph" : s === "session" ? "Session" : s === "trace" ? "Traces" : "All";
          return (
            <button
              key={s}
              type="button"
              disabled={disabled}
              onClick={() => !disabled && setScope(s)}
              className="cursor-pointer"
              title={disabled ? "No session available to search" : `Search ${label.toLowerCase()}`}
              style={{
                background: active ? "#18181B" : "#FFFFFF",
                color: disabled ? "#D4D4D8" : active ? "#FFFFFF" : "#3F3F46",
                border: active ? "none" : "1px solid #E4E4E7",
                borderRadius: 100,
                paddingBlock: 5,
                paddingInline: 11,
                fontSize: 12,
                lineHeight: "16px",
                fontFamily: "inherit",
                cursor: disabled ? "not-allowed" : "pointer",
              }}
            >
              {label}
            </button>
          );
        })}
      </div>

      <div style={{ background: "#fff", border: `1px solid ${query ? "#6510F4" : "#E5E7EB"}`, borderRadius: 10, padding: "12px 16px", display: "flex", alignItems: "center", gap: 10, boxShadow: "0 1px 3px #0000000A", transition: "border-color 0.2s" }}>
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none" style={{ flexShrink: 0 }}>
          <circle cx="7.5" cy="7.5" r="5.5" stroke="#A1A1AA" strokeWidth="1.5" />
          <path d="M12 12l4.5 4.5" stroke="#A1A1AA" strokeWidth="1.5" strokeLinecap="round" />
        </svg>
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") handleSearch(query); }}
          placeholder="Ask a question about your data..."
          style={{ flex: 1, border: "none", outline: "none", fontSize: 14, color: "#18181B", fontFamily: "inherit", background: "transparent" }}
        />
        {query && (
          <button onClick={() => handleSearch(query)} className="cursor-pointer" style={{ background: "#6510F4", border: "none", borderRadius: 6, padding: "6px 14px", fontSize: 13, fontWeight: 500, color: "#fff" }}>
            Search
          </button>
        )}
      </div>

      {/* Loading */}
      {isSearching && (
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 10, padding: 40 }}>
          <div style={{ width: 16, height: 16, borderRadius: "50%", border: "2px solid #E4E4E7", borderTopColor: "#6510F4", animation: "spin 1s linear infinite" }} />
          <span style={{ fontSize: 13, color: "#71717A" }}>Searching knowledge graph...</span>
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{ background: "#FEF2F2", border: "1px solid #FECACA", borderRadius: 8, padding: "10px 16px", fontSize: 13, color: "#991B1B" }}>
          {error}
        </div>
      )}

      {/* Results */}
      {results.length > 0 && !isSearching && (
        <>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontSize: 12, fontWeight: 500, color: "#71717A" }}>{results.length} result{results.length !== 1 ? "s" : ""}</span>
          </div>

          <div style={{ background: "#fff", border: "1px solid #E5E7EB", borderRadius: 12, overflow: "hidden" }}>
            {results.map((raw, i) => {
              // Normalize recall/search result shapes. The endpoint may
              // return { dataset_name, search_result: [...] } (legacy),
              // a plain string, a recall row ({ question, answer, ... }),
              // or a bare object — coerce all of them into a list of
              // display lines before mapping.
              const r: {
                dataset_name?: string;
                search_result?: unknown;
                question?: string;
                answer?: string;
                _source?: string;
              } = raw as unknown as {
                dataset_name?: string;
                search_result?: unknown;
                question?: string;
                answer?: string;
                _source?: string;
              };
              const label = r.dataset_name || (r._source ? r._source : "Result");
              let lines: string[] = [];
              if (Array.isArray(r.search_result)) {
                lines = (r.search_result as unknown[]).map((x) =>
                  typeof x === "string" ? x : JSON.stringify(x),
                );
              } else if (typeof r.search_result === "string") {
                lines = [r.search_result];
              } else if (typeof raw === "string") {
                lines = [raw as unknown as string];
              } else if (r.question || r.answer) {
                if (r.question) lines.push(`Q: ${r.question}`);
                if (r.answer) lines.push(`A: ${r.answer}`);
              } else {
                try { lines = [JSON.stringify(raw)]; } catch { lines = [String(raw)]; }
              }
              return (
                <div
                  key={i}
                  className="hover:bg-cognee-hover"
                  style={{ padding: "16px 20px", borderBottom: i < results.length - 1 ? "1px solid #F4F4F5" : "none", display: "flex", flexDirection: "column", gap: 8, transition: "background 150ms" }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.04em", color: "#6510F4", textTransform: "uppercase" }}>
                      {label}
                    </span>
                  </div>
                  {lines.map((text, j) => (
                    <p key={j} style={{ fontSize: 13, color: "#18181B", lineHeight: "20px", margin: 0 }}>
                      {text}
                    </p>
                  ))}
                </div>
              );
            })}
          </div>
        </>
      )}

      {/* No results */}
      {hasSearched && !isSearching && results.length === 0 && !error && (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8, padding: 40 }}>
          <span style={{ fontSize: 14, color: "#71717A" }}>No results found.</span>
          <span style={{ fontSize: 13, color: "#A1A1AA" }}>Try a different query, or make sure you&apos;ve cognified your datasets.</span>
        </div>
      )}

      {/* Suggestions (before any search) */}
      {!hasSearched && !isSearching && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
          {suggestions.map((s) => (
            <div key={s} onClick={() => { setQuery(s); handleSearch(s); }} className="cursor-pointer hover:bg-cognee-hover active:bg-cognee-pressed transition-colors" style={{ background: "#fff", border: "1px solid #E4E4E7", borderRadius: 100, padding: "8px 16px", fontSize: 13, color: "#000" }}>
              {s}
            </div>
          ))}
        </div>
      )}

      {/* Search history */}
      {history.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 8, marginTop: 8 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: 12, fontWeight: 600, letterSpacing: "0.06em", color: "#999999", textTransform: "uppercase" }}>Recent searches</span>
          </div>
          {history.map((h, i) => (
            <div key={`${h.query}-${i}`} style={{ display: "flex", flexDirection: "column" }}>
              <div
                onClick={() => {
                  if (expandedHistory === i) { setExpandedHistory(null); }
                  else { setExpandedHistory(i); }
                }}
                className="cursor-pointer hover:bg-cognee-hover"
                style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 12px", borderRadius: 8, transition: "background 150ms" }}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#A1A1AA" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
                  <circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" />
                </svg>
                <span style={{ flex: 1, fontSize: 13, color: "#333333" }}>{h.query}</span>
                <span style={{ fontSize: 11, color: "#A1A1AA", flexShrink: 0 }}>{h.resultCount} result{h.resultCount !== 1 ? "s" : ""}</span>
                <span style={{ fontSize: 11, color: "#D4D4D8", flexShrink: 0 }}>{timeAgo(h.time)}</span>
                <button
                  onClick={(e) => { e.stopPropagation(); setQuery(h.query); handleSearch(h.query); }}
                  className="cursor-pointer hover:bg-cognee-pressed rounded"
                  style={{ background: "none", border: "none", padding: "2px 8px", fontSize: 11, color: "#6510F4", fontFamily: "inherit", flexShrink: 0 }}
                >
                  Re-run
                </button>
              </div>
              {/* Expanded results */}
              {expandedHistory === i && h.results.length > 0 && (
                <div style={{ marginLeft: 36, marginBottom: 8, background: "#fff", border: "1px solid #E5E7EB", borderRadius: 8, overflow: "hidden" }}>
                  {h.results.map((r, ri) => {
                    const label = r.dataset_name || r._source || "Result";
                    let lines: string[] = [];
                    if (Array.isArray(r.search_result)) lines = r.search_result;
                    else if (r.answer) lines = [r.answer];
                    else if (r.question) lines = [`Q: ${r.question}`];
                    else if (r.origin_function) lines = [`${r.origin_function}${r.session_feedback ? ": " + r.session_feedback : ""}`];
                    return (
                      <div key={ri} style={{ padding: "12px 16px", borderBottom: ri < h.results.length - 1 ? "1px solid #F4F4F5" : "none" }}>
                        <span style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.04em", color: "#6510F4", textTransform: "uppercase" }}>{label}</span>
                        {lines.map((text, j) => (
                          <p key={j} style={{ fontSize: 12, color: "#52525B", lineHeight: "18px", margin: "4px 0 0" }}>{text}</p>
                        ))}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}
