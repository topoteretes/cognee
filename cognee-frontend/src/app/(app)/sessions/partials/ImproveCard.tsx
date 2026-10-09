"use client";

import type { EnrichmentRun } from "@/modules/sessions/getSessions";
import { formatDate, formatRelativeTime } from "./format";

// Matches the SelfImprovementCard purple so all improve surfaces read as one feature.
const PURPLE = { bg: "rgba(188,155,255,0.15)", border: "rgba(188,155,255,0.35)", color: "#BC9BFF" };
const GREEN = "#34D399";
const MS_PER_MINUTE = 60_000;
const MAX_REASON_LENGTH = 160;

function truncateReason(reason: string): string {
  const trimmed = reason.trim();
  if (trimmed.length <= MAX_REASON_LENGTH) return trimmed;
  return `${trimmed.slice(0, MAX_REASON_LENGTH)}…`;
}

function formatBurstDuration(startedAt: string | null, endedAt: string | null): string | null {
  if (!startedAt || !endedAt || startedAt === endedAt) return null;
  const ms = Date.parse(endedAt) - Date.parse(startedAt);
  if (!Number.isFinite(ms) || ms <= 0) return null;
  if (ms < MS_PER_MINUTE) return `${Math.round(ms / 1000)}s`;
  return `${Math.round(ms / MS_PER_MINUTE)}min`;
}

export default function ImproveCard({ run }: { run: EnrichmentRun }) {
  const full = formatDate(run.created_at);
  const duration = formatBurstDuration(run.started_at, run.created_at);
  return (
    <div style={{
      border: "1px solid rgba(255,255,255,0.08)",
      borderRadius: 10,
      background: "rgba(255,255,255,0.03)",
      padding: "12px 14px",
      display: "flex", flexDirection: "column", gap: 6,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, width: "100%" }}>
        <span style={{
          display: "inline-flex", alignItems: "center", gap: 5,
          background: PURPLE.bg,
          border: `1px solid ${PURPLE.border}`,
          color: PURPLE.color,
          borderRadius: 4, padding: "2px 8px",
          fontSize: 10, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase",
        }}>
          <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 3v3m0 12v3M3 12h3m12 0h3M5.6 5.6l2.2 2.2m8.4 8.4l2.2 2.2M5.6 18.4l2.2-2.2m8.4-8.4l2.2-2.2" />
          </svg>
          Improve
        </span>
        <span title={full} style={{ fontSize: 11, color: "rgba(237,236,234,0.45)", fontVariantNumeric: "tabular-nums" }}>{formatRelativeTime(run.created_at)}</span>
        {run.status === "completed" && <span style={{ fontSize: 11, color: GREEN }}>success</span>}
        {run.status === "failed" && <span style={{ fontSize: 11, color: "#EF4444" }}>failed</span>}
        {run.status === "running" && <span style={{ fontSize: 11, color: PURPLE.color }}>in progress</span>}
      </div>
      <div style={{ fontSize: 13, lineHeight: 1.55, color: "rgba(237,236,234,0.7)" }}>
        {run.status === "failed"
          ? "Graph enrichment did not complete — no memory was bridged this run"
          : "Session memory bridged into the knowledge graph"}
      </div>
      {run.status === "failed" && run.failure_reason && (
        <div style={{ fontSize: 11, lineHeight: 1.5, color: "rgba(239,68,68,0.75)" }}>
          {truncateReason(run.failure_reason)}
        </div>
      )}
      <div style={{ display: "flex", gap: 12, fontSize: 11, color: "rgba(237,236,234,0.45)", fontVariantNumeric: "tabular-nums" }}>
        <span>{run.count} {run.count === 1 ? "stage" : "stages"}</span>
        {duration && <span>{duration}</span>}
      </div>
    </div>
  );
}
