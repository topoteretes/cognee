"use client";

import { useEffect, useState, useRef } from "react";
import Link from "next/link";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import { useFilter } from "@/ui/layout/FilterContext";

export default function KnowledgeGraphPage() {
  const { cogniInstance, serviceUrl, isInitializing } = useCogniInstance();
  const { datasets, selectedDataset } = useFilter();
  const [iframeSrc, setIframeSrc] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const blobRef = useRef<string | null>(null);

  const datasetId = selectedDataset?.id || (datasets.length > 0 ? datasets[0].id : null);
  const datasetName = selectedDataset?.name || (datasets.length > 0 ? datasets[0].name : null);

  useEffect(() => {
    if (!datasetId || isInitializing) {
      setLoading(false);
      return;
    }

    setLoading(true);
    setIframeSrc(null);
    setError(null);

    // Revoke previous blob
    if (blobRef.current) {
      URL.revokeObjectURL(blobRef.current);
      blobRef.current = null;
    }

    // Use tenant service URL for cloud, local proxy for local mode
    const fetchVisualization = cogniInstance
      ? cogniInstance.fetch(`/v1/visualize?dataset_id=${datasetId}`)
      : global.fetch(`/api/visualize?dataset_id=${datasetId}`, { credentials: "include" });

    fetchVisualization
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.text();
      })
      .then((html) => {
        if (html && html.length > 100 && (html.includes("<!DOCTYPE") || html.includes("<html"))) {
          const blob = new Blob([html], { type: "text/html" });
          const url = URL.createObjectURL(blob);
          blobRef.current = url;
          setIframeSrc(url);
        } else {
          setError("No graph data in this dataset yet.");
        }
      })
      .catch((err) => {
        setError(err.message || "Failed to load visualization");
      })
      .finally(() => setLoading(false));

    return () => {
      if (blobRef.current) {
        URL.revokeObjectURL(blobRef.current);
        blobRef.current = null;
      }
    };
  }, [datasetId, isInitializing]);

  if (isInitializing) {
    return <div style={{ padding: 32, display: "flex", alignItems: "center", justifyContent: "center", height: "100%" }}><span style={{ fontSize: 14, color: "#71717A" }}>Loading...</span></div>;
  }

  if (datasets.length === 0) {
    return (
      <div style={{ padding: 32, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", gap: 12, fontFamily: '"Inter", system-ui, sans-serif' }}>
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#A1A1AA" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="2" /><circle cx="12" cy="5" r="1.5" /><circle cx="19" cy="16" r="1.5" /><circle cx="5" cy="16" r="1.5" />
          <line x1="12" y1="7" x2="12" y2="10" /><line x1="13.7" y1="13.3" x2="17.8" y2="15.2" /><line x1="10.3" y1="13.3" x2="6.2" y2="15.2" />
        </svg>
        <span style={{ fontSize: 15, fontWeight: 500, color: "#18181B" }}>No datasets yet</span>
        <span style={{ fontSize: 13, color: "#A1A1AA", textAlign: "center", maxWidth: 360, lineHeight: "20px" }}>
          Create a dataset and upload documents to visualize your knowledge graph.
        </span>
        <Link href="/datasets" style={{ background: "#6510F4", color: "#fff", border: "none", borderRadius: 8, padding: "8px 20px", fontSize: 13, fontWeight: 500, textDecoration: "none", marginTop: 4 }}>
          Go to Datasets
        </Link>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", fontFamily: '"Inter", system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "16px 24px", flexShrink: 0 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          <h1 style={{ fontSize: 18, fontWeight: 600, color: "#18181B", margin: 0 }}>Knowledge Graph</h1>
          <span style={{ fontSize: 13, color: "#71717A" }}>
            {datasetName ? `Visualizing: ${datasetName}` : "Select a dataset from the breadcrumb above"}
          </span>
        </div>
      </div>

      {/* Graph */}
      <div style={{ flex: 1, position: "relative", overflow: "hidden", borderTop: "1px solid #E4E4E7" }}>
        {loading && (
          <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", background: "#FAFAF9", zIndex: 1 }}>
            <span style={{ fontSize: 14, color: "#71717A" }}>Loading graph...</span>
          </div>
        )}
        {iframeSrc ? (
          <iframe
            key={datasetId}
            src={iframeSrc}
            style={{ width: "100%", height: "100%", border: "none" }}
            title="Knowledge Graph Visualization"
          />
        ) : !loading ? (
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", flexDirection: "column", gap: 12 }}>
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#A1A1AA" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="2" /><circle cx="12" cy="5" r="1.5" /><circle cx="19" cy="16" r="1.5" /><circle cx="5" cy="16" r="1.5" />
              <line x1="12" y1="7" x2="12" y2="10" /><line x1="13.7" y1="13.3" x2="17.8" y2="15.2" /><line x1="10.3" y1="13.3" x2="6.2" y2="15.2" />
            </svg>
            <span style={{ fontSize: 15, fontWeight: 500, color: "#18181B" }}>{error || "No graph data for this dataset yet"}</span>
            <span style={{ fontSize: 13, color: "#A1A1AA", textAlign: "center", maxWidth: 400, lineHeight: "20px" }}>
              Upload documents to a dataset first, then the knowledge graph will be built automatically. You can upload files from the <Link href="/datasets" style={{ color: "#6510F4", textDecoration: "underline" }}>Datasets</Link> page.
            </span>
          </div>
        ) : null}
      </div>
    </div>
  );
}
