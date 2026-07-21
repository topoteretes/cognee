"use client";

import { useState } from "react";
import SkeletonBar from "@/ui/elements/SkeletonBar";
import type { EnrichmentRun } from "@/modules/sessions/getSessions";
import { formatDate, formatRelativeTime } from "./format";

const INFO_TEXT =
  "As a session accumulates turns or goes idle, Cognee automatically runs improve(): the session's questions, answers and feedback are bridged into the permanent knowledge graph — weighting existing memories by feedback, persisting the conversation, distilling reusable lessons and enriching the graph. Future sessions recall from the enriched graph.";

const STATUS_META = {
  completed: { color: "#22C55E", label: "Completed" },
  running: { color: "#6510F4", label: "Running" },
  failed: { color: "#EF4444", label: "Failed" },
} as const;

function InfoIcon() {
  const [open, setOpen] = useState(false);
  return (
    <span
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      style={{ position: "relative", display: "inline-flex", alignItems: "center", cursor: "help" }}
    >
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="rgba(188,155,255,0.7)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="16" x2="12" y2="12" />
        <line x1="12" y1="8" x2="12.01" y2="8" />
      </svg>
      {open && (
        <span style={{
          position: "absolute", top: "calc(100% + 8px)", left: -10, zIndex: 20,
          width: 300, padding: "10px 12px", borderRadius: 8,
          background: "rgba(20,16,30,0.97)", border: "1px solid rgba(188,155,255,0.35)",
          fontSize: 11, lineHeight: 1.55, color: "rgba(237,236,234,0.85)",
          fontWeight: 400, letterSpacing: "normal", textTransform: "none",
          boxShadow: "0 8px 24px rgba(0,0,0,0.5)",
        }}>
          {INFO_TEXT}
        </span>
      )}
    </span>
  );
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "160px 1fr", alignItems: "center", gap: 12, minHeight: 22 }}>
      <span style={{ fontSize: 11, fontWeight: 600, color: "rgba(237,236,234,0.45)", letterSpacing: "0.04em", textTransform: "uppercase" }}>{label}</span>
      <div style={{ fontSize: 12, color: "rgba(237,236,234,0.85)", minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{value}</div>
    </div>
  );
}

export default function SelfImprovementCard({ datasetId, runs, loading }: { datasetId: string | null; runs: EnrichmentRun[]; loading: boolean }) {
  const info = runs[0] ?? null;
  const status = info ? STATUS_META[info.status] : null;
  const lastFull = info ? formatDate(info.created_at) : null;
  const lastRel = info?.created_at ? formatRelativeTime(info.created_at) : null;

  return (
    <div style={{
      border: "1px solid rgba(188,155,255,0.35)", borderRadius: 10, padding: 16,
      display: "flex", flexDirection: "column", gap: 10,
      background: "linear-gradient(135deg, rgba(101,16,244,0.16), rgba(101,16,244,0.05))",
      backdropFilter: "blur(12px)",
      // backdrop-filter gives every sibling card its own stacking context, so the
      // tooltip's z-index alone can't escape this card — the card itself must sit
      // above the cards that follow it in the DOM.
      position: "relative", zIndex: 5,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#BC9BFF" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 3v3m0 12v3M3 12h3m12 0h3M5.6 5.6l2.2 2.2m8.4 8.4l2.2 2.2M5.6 18.4l2.2-2.2m8.4-8.4l2.2-2.2" />
        </svg>
        <span style={{ fontSize: 11, fontWeight: 700, color: "#BC9BFF", letterSpacing: "0.06em", textTransform: "uppercase" }}>Self-improvement</span>
        <InfoIcon />
      </div>

      {loading ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <SkeletonBar width={220} />
          <SkeletonBar width={160} />
        </div>
      ) : !info ? (
        <span style={{ fontSize: 12, color: "rgba(237,236,234,0.55)", lineHeight: 1.5 }}>
          No graph enrichment yet. Improve runs automatically once this session accumulates turns or goes idle.
        </span>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <Row
            label="Last graph enrichment"
            value={
              <span title={lastFull ?? undefined} style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
                {status && <span style={{ width: 6, height: 6, borderRadius: "50%", background: status.color, flexShrink: 0 }} title={status.label} />}
                {lastFull}{lastRel ? ` · ${lastRel}` : ""}
                {info.status === "running" && <span style={{ fontSize: 11, color: "rgba(188,155,255,0.8)" }}>in progress</span>}
                {info.status === "failed" && <span style={{ fontSize: 11, color: "#EF4444" }}>failed</span>}
              </span>
            }
          />
          <Row
            label="Dataset"
            value={info.dataset_name ?? (
              <span style={{ fontFamily: 'ui-monospace, Menlo, Monaco, "Cascadia Mono", "Segoe UI Mono", "Roboto Mono", monospace', fontSize: 11 }}>{datasetId}</span>
            )}
          />
        </div>
      )}
    </div>
  );
}
