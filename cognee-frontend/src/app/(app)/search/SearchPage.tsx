"use client";

import { useState, useRef, useEffect } from "react";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import searchDataset from "@/modules/datasets/searchDataset";
import getDatasets from "@/modules/datasets/getDatasets";

interface SearchResultItem {
  search_result: string[];
  dataset_id: string;
  dataset_name: string;
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
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [expandedHistory, setExpandedHistory] = useState<number | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!cogniInstance || isInitializing) return;
    getDatasets(cogniInstance)
      .then((data: { id: string; name: string }[]) => setDatasets(Array.isArray(data) ? data : []))
      .catch(() => {});
  }, [cogniInstance, isInitializing]);

  const handleSearch = async (q: string) => {
    if (!q.trim() || !cogniInstance) return;
    setQuery(q);
    setIsSearching(true);
    setResults([]);
    setError(null);
    setHasSearched(true);

    try {
      const data = await searchDataset(cogniInstance, {
        query: q,
        searchType: "GRAPH_COMPLETION",
        datasetIds: datasets.map((d) => d.id),
      });
      const resultData = Array.isArray(data) ? data : [];
      setResults(resultData);
      // Add to history (deduplicate: remove previous entry with same query)
      const totalResults = resultData.reduce((sum: number, r: SearchResultItem) => sum + r.search_result.length, 0);
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

      {/* Search bar */}
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
            {results.map((r, i) => (
              <div
                key={i}
                className="hover:bg-cognee-hover"
                style={{ padding: "16px 20px", borderBottom: i < results.length - 1 ? "1px solid #F4F4F5" : "none", display: "flex", flexDirection: "column", gap: 8, transition: "background 150ms" }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.04em", color: "#6510F4", textTransform: "uppercase" }}>
                    {r.dataset_name || "Result"}
                  </span>
                </div>
                {r.search_result.map((text, j) => (
                  <p key={j} style={{ fontSize: 13, color: "#18181B", lineHeight: "20px", margin: 0 }}>
                    {text}
                  </p>
                ))}
              </div>
            ))}
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
                  {h.results.map((r, ri) => (
                    <div key={ri} style={{ padding: "12px 16px", borderBottom: ri < h.results.length - 1 ? "1px solid #F4F4F5" : "none" }}>
                      <span style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.04em", color: "#6510F4", textTransform: "uppercase" }}>{r.dataset_name || "Result"}</span>
                      {r.search_result.map((text, j) => (
                        <p key={j} style={{ fontSize: 12, color: "#52525B", lineHeight: "18px", margin: "4px 0 0" }}>{text}</p>
                      ))}
                    </div>
                  ))}
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
