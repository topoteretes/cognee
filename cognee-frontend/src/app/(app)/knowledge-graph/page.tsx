"use client";

import { useEffect, useState, useRef } from "react";
import Link from "next/link";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import { useFilter } from "@/ui/layout/FilterContext";
import BrainSelector from "@/ui/elements/BrainSelector";
import PageLoading from "@/ui/elements/PageLoading";
import { TrackPageView } from "@/modules/analytics";
import type { DatasetProcessingStatus } from "@/modules/datasets/pollDatasetStatus";

type DisplayStatus = "pending" | "running" | "completed" | "failed" | "empty";

function mapStatus(raw: DatasetProcessingStatus | undefined): DisplayStatus {
  if (!raw) return "empty";
  if (raw === "DATASET_PROCESSING_COMPLETED") return "completed";
  if (raw === "DATASET_PROCESSING_ERRORED") return "failed";
  if (raw === "DATASET_PROCESSING_STARTED") return "running";
  if (raw === "DATASET_PROCESSING_INITIATED") return "pending";
  return "empty";
}

const STATUS_CFG: Record<DisplayStatus, { label: string; color: string; dotBg: string }> = {
  pending:   { label: "Pending",    color: "#F59E0B", dotBg: "rgba(245,158,11,0.15)" },
  running:   { label: "Processing", color: "#F59E0B", dotBg: "rgba(245,158,11,0.15)" },
  completed: { label: "Ready",      color: "#22C55E", dotBg: "rgba(34,197,94,0.15)" },
  failed:    { label: "Failed",     color: "#EF4444", dotBg: "rgba(239,68,68,0.15)" },
  empty:     { label: "Empty",      color: "rgba(237,236,234,0.35)", dotBg: "rgba(255,255,255,0.06)" },
};

export default function KnowledgeGraphPage() {
  const { cogniInstance, isInitializing } = useCogniInstance();
  const { datasets, selectedDataset } = useFilter();

  const [iframeSrc, setIframeSrc] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  // Keep the iframe hidden behind the loading overlay until its content has
  // settled (dark theme applied) — avoids the light-mode boot flicker.
  const [vizReady, setVizReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const blobRef = useRef<string | null>(null);
  const [datasetStatus, setDatasetStatus] = useState<DisplayStatus>("empty");
  const [pollKey, setPollKey] = useState(0);
  const [vizRefreshKey, setVizRefreshKey] = useState(0);
  const prevStatusRef = useRef<DisplayStatus>("empty");

  const activeDataset = selectedDataset ?? datasets[0] ?? null;
  const datasetId = activeDataset?.id ?? null;

  // Reset on dataset change
  useEffect(() => {
    setDatasetStatus("empty");
    prevStatusRef.current = "empty";
  }, [datasetId]);

  // Poll processing status
  useEffect(() => {
    if (!datasetId || !cogniInstance || isInitializing) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    async function check() {
      try {
        const resp = await cogniInstance!.fetch(`/v1/datasets/status?dataset=${datasetId}`);
        if (!resp.ok || cancelled) return;
        const data: Record<string, DatasetProcessingStatus> = await resp.json();
        const status = mapStatus(data[datasetId!]);
        if (!cancelled) setDatasetStatus(status);
        if (!cancelled && (status === "pending" || status === "running")) timer = setTimeout(check, 5000);
      } catch { /* ignore */ }
    }
    check();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [datasetId, cogniInstance, isInitializing, pollKey]);

  // Reload viz when status transitions to completed
  useEffect(() => {
    if (prevStatusRef.current !== datasetStatus) {
      if ((prevStatusRef.current === "running" || prevStatusRef.current === "pending") && datasetStatus === "completed") {
        setVizRefreshKey((k) => k + 1);
      }
      prevStatusRef.current = datasetStatus;
    }
  }, [datasetStatus]);

  // Fetch visualization
  useEffect(() => {
    if (!datasetId || isInitializing) { setLoading(false); return; }
    setLoading(true);
    setIframeSrc(null);
    setError(null);
    setVizReady(false);
    if (blobRef.current) { URL.revokeObjectURL(blobRef.current); blobRef.current = null; }

    const fetchViz = cogniInstance
      ? cogniInstance.fetch(`/v1/visualize?dataset_id=${datasetId}`)
      : global.fetch(`/api/visualize?dataset_id=${datasetId}`, { credentials: "include" });

    fetchViz
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.text(); })
      .then((html) => {
        if (html && html.length > 100 && (html.includes("<!DOCTYPE") || html.includes("<html"))) {
          const closeScript = "<" + "/script>";
          // Dark background CSS beats the first paint (no white flash) and
          // the theme switch runs synchronously instead of post-paint.
          const kgInject =
            '<style>html{background:#0A0A0A!important;color-scheme:dark}#view-tabs{display:none!important}</style>' +
            "<script>(function(){" +
            "document.documentElement.classList.remove('light');" +
            "window._isLightMode=false;" +
            "var t=document.getElementById('theme-toggle');if(t)t.textContent='Light mode';" +
            "})()" + closeScript;
          const blob = new Blob([html.replace("</body>", kgInject + "</body>")], { type: "text/html" });
          const url = URL.createObjectURL(blob);
          blobRef.current = url;
          setIframeSrc(url);
        } else {
          setError("No graph data in this brain yet.");
        }
      })
      .catch((err) => setError(err.message || "Failed to load visualization"))
      .finally(() => setLoading(false));

    return () => { if (blobRef.current) { URL.revokeObjectURL(blobRef.current); blobRef.current = null; } };
  }, [datasetId, isInitializing, vizRefreshKey]);

  if (isInitializing) {
    return <><TrackPageView page="Mindmap" /><div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%" }}><style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#6510F4" strokeWidth="1.5" strokeLinecap="round" style={{ animation: "spin 0.9s linear infinite" }}><path d="M21 12a9 9 0 11-6.219-8.56" /></svg></div></>;
  }

  if (datasets.length === 0) {
    return (
      <><TrackPageView page="Mindmap" />
      <div style={{ padding: 32, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", gap: 12 }}>
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.35)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="2" /><circle cx="12" cy="5" r="1.5" /><circle cx="19" cy="16" r="1.5" /><circle cx="5" cy="16" r="1.5" /><line x1="12" y1="7" x2="12" y2="10" /><line x1="13.7" y1="13.3" x2="17.8" y2="15.2" /><line x1="10.3" y1="13.3" x2="6.2" y2="15.2" /></svg>
        <span style={{ fontSize: 15, fontWeight: 500, color: "#EDECEA" }}>No brains yet</span>
        <span style={{ fontSize: 13, color: "rgba(237,236,234,0.35)", textAlign: "center", maxWidth: 360, lineHeight: "20px" }}>Create a brain and upload documents to visualize your knowledge graph.</span>
        <Link href="/datasets" style={{ background: "#6510F4", color: "#fff", border: "none", borderRadius: 8, padding: "8px 20px", fontSize: 13, fontWeight: 500, textDecoration: "none", marginTop: 4 }}>Go to Datasets</Link>
      </div></>
    );
  }

  const cfg = STATUS_CFG[datasetStatus];

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <TrackPageView page="Mindmap" />

      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "16px 32px", flexShrink: 0, gap: 12, flexWrap: "wrap", borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <h1 style={{ fontSize: 20, fontWeight: 300, color: "#EDECEA", margin: 0, fontFamily: '"TWKLausanne", sans-serif' }}>Mindmap</h1>
          <BrainSelector allowAll={false} />
          {activeDataset && (
            <div style={{ display: "flex", alignItems: "center", gap: 5, background: cfg.dotBg, border: `1px solid ${cfg.color}30`, borderRadius: 6, padding: "3px 8px" }}>
              <div style={{ width: 6, height: 6, borderRadius: "50%", background: cfg.color, ...(datasetStatus === "running" || datasetStatus === "pending" ? { animation: "pulse-dot 1.5s ease-in-out infinite" } : {}) }} />
              <span style={{ fontSize: 12, fontWeight: 500, color: cfg.color }}>{cfg.label}</span>
            </div>
          )}
          {(datasetStatus === "running" || datasetStatus === "pending") && (
            <style>{`@keyframes pulse-dot { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }`}</style>
          )}
        </div>

        <button
          onClick={() => setVizRefreshKey((k) => k + 1)}
          style={{ background: "rgba(0,0,0,0.75)", border: "1px solid rgba(255,255,255,0.18)", color: "#EDECEA", borderRadius: 7, padding: "6px 10px", fontSize: 12, fontWeight: 500, cursor: "pointer", display: "flex", alignItems: "center", gap: 5 }}
          title="Refresh"
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="rgba(237,236,234,0.7)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 2v6h-6" /><path d="M3 12a9 9 0 0115.36-6.36L21 8" /><path d="M3 22v-6h6" /><path d="M21 12a9 9 0 01-15.36 6.36L3 16" /></svg>
          Refresh
        </button>
      </div>

      {/* Graph visualization */}
      <div style={{ flex: 1, position: "relative", overflow: "hidden" }}>
        {(loading || (iframeSrc && !vizReady)) && (
          <div style={{ position: "absolute", inset: 0, zIndex: 1 }}>
            <PageLoading name="Mindmap" />
          </div>
        )}
        {iframeSrc ? (
          <iframe
            key={datasetId}
            src={iframeSrc}
            onLoad={() => setTimeout(() => setVizReady(true), 250)}
            style={{ width: "100%", height: "100%", border: "none", opacity: vizReady ? 1 : 0, transition: "opacity 200ms ease" }}
            title="Mindmap Visualization"
          />
        ) : !loading ? (
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", flexDirection: "column", gap: 12 }}>
            {(datasetStatus === "pending" || datasetStatus === "running") ? (
              <>
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#6510F4" strokeWidth="1.5" strokeLinecap="round" style={{ animation: "spin 2s linear infinite" }}><path d="M21 12a9 9 0 11-6.219-8.56" /></svg>
                <span style={{ fontSize: 15, fontWeight: 500, color: "#EDECEA" }}>Graph is being generated</span>
                <span style={{ fontSize: 13, color: "rgba(237,236,234,0.35)", textAlign: "center", maxWidth: 400, lineHeight: "20px" }}>Your data is being processed. The graph will appear here once ready.</span>
              </>
            ) : (
              <>
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#A1A1AA" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="2" /><circle cx="12" cy="5" r="1.5" /><circle cx="19" cy="16" r="1.5" /><circle cx="5" cy="16" r="1.5" /><line x1="12" y1="7" x2="12" y2="10" /><line x1="13.7" y1="13.3" x2="17.8" y2="15.2" /><line x1="10.3" y1="13.3" x2="6.2" y2="15.2" /></svg>
                <span style={{ fontSize: 15, fontWeight: 500, color: "#EDECEA" }}>No graph data yet</span>
                <span style={{ fontSize: 13, color: "rgba(237,236,234,0.35)", textAlign: "center", maxWidth: 400, lineHeight: "20px" }}>
                  Upload documents to a brain, then configure and re-process from the{" "}
                  <Link href="/schema" style={{ color: "#BC9BFF", textDecoration: "underline" }}>Memory Schema</Link> page.
                </span>
                {error && error !== "No graph data in this brain yet." && (
                  <span style={{ fontSize: 11, color: "#EF4444", textAlign: "center", maxWidth: 400, fontFamily: 'ui-monospace, Menlo, Monaco, "Cascadia Mono", "Segoe UI Mono", "Roboto Mono", monospace', wordBreak: "break-all" }}>{error}</span>
                )}
              </>
            )}
          </div>
        ) : null}
        <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
      </div>
    </div>
  );
}
