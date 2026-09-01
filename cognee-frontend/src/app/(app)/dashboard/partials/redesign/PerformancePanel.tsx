"use client";

import React from "react";
import { AsciiFrame } from "./AsciiFrame";
import { FONT, T } from "./mono";

/** `pct` is null until a recall score exists for the brain. */
export interface TopicScore { name: string; pct: number | null }

interface PerformancePanelProps {
  /** Null until the workspace has a scored recall figure. */
  recallPct: number | null;
  topics: TopicScore[];
  onUpload?: () => void;
  onViewAnalysis?: () => void;
}

const clampPct = (n: number): number => Math.max(0, Math.min(100, n));

/** Rounded track + fill progress bar. */
export function Bar({ pct, color, height = 6 }: { pct: number; color: string; height?: number }): React.ReactElement {
  return (
    <div style={{ flex: 1, height, borderRadius: height / 2, background: "rgba(237,236,234,0.18)", overflow: "hidden" }}>
      <div style={{ width: `${clampPct(pct)}%`, height: "100%", background: color, borderRadius: height / 2, transition: "width 300ms" }} />
    </div>
  );
}

/**
 * PERFORMANCE panel — real-recall (LLM-as-judge) headline plus per-brain
 * coverage bars.
 *
 * MISSING ENDPOINT: nothing scores recall yet, so every score arrives null and
 * the panel shows brain names with an empty bar. Needs a coverage/recall run
 * per dataset — the same payload the Memory Gap Analysis page waits on
 * (see CoverageResult in @/app/(app)/memory-gap-analysis/types). Feed the
 * scores in through `recallPct`/`topics[].pct`; no other change is required.
 */
export function PerformancePanel({
  recallPct,
  topics,
  onUpload,
  onViewAnalysis,
}: PerformancePanelProps): React.ReactElement {
  return (
    <AsciiFrame label="Memory Coverage" meta={recallPct === null ? "LLM-as-judge · not scored yet" : "LLM-as-judge"} minHeight={260}>
      <div style={{ display: "flex", flexDirection: "column", height: "100%", gap: 14 }}>
        {/* Headline recall */}
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ width: 130, flexShrink: 0, display: "inline-flex", alignItems: "baseline", gap: 8 }}>
            <span style={{ ...FONT, fontSize: 12, color: T.muted }}>Real recall</span>
            <span style={{ ...FONT, fontSize: 26, fontWeight: 700, color: recallPct === null ? T.faint : T.text, letterSpacing: "-0.02em", fontVariantNumeric: "tabular-nums" }}>
              {recallPct === null ? "—" : `${recallPct}%`}
            </span>
          </span>
          <Bar pct={recallPct ?? 0} color={T.text} height={8} />
        </div>

        {/* Per-brain coverage */}
        <div style={{ display: "flex", flexDirection: "column", gap: 9, flex: 1, marginTop: 10 }}>
          <span style={{ ...FONT, fontSize: 12, color: T.muted }}>Coverage by brain</span>
          {topics.length === 0 && (
            <span style={{ ...FONT, fontSize: 13, color: T.faint }}>Add data to see coverage by brain</span>
          )}
          {topics.map((t, i) => (
            <div key={`${t.name}-${i}`} style={{ display: "flex", alignItems: "center", gap: 12, ...FONT, fontSize: 13 }}>
              <span style={{ width: 130, color: T.muted, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flexShrink: 0 }}>
                {t.name}
              </span>
              <Bar pct={t.pct ?? 0} color={T.lavender} height={1.5} />
              <span style={{ width: 30, textAlign: "right", color: t.pct === null ? T.faint : T.text, fontVariantNumeric: "tabular-nums", flexShrink: 0 }}>
                {t.pct ?? "—"}
              </span>
            </div>
          ))}
          {topics.length > 0 && recallPct === null && (
            <span style={{ ...FONT, fontSize: 11, color: T.faint, marginTop: 4 }}>Recall scoring isn&apos;t available yet</span>
          )}
        </div>

        {/* Footer row — link sits on the same baseline as the Cost Savings
            card's "View breakdown →"; actions live in the right corner. */}
        <div style={{ marginTop: "auto", display: "flex", alignItems: "flex-end", justifyContent: "space-between", gap: 12 }}>
          <button
            type="button"
            onClick={onViewAnalysis}
            style={{ ...FONT, fontSize: 12, color: T.lavender, background: "none", border: "none", cursor: "pointer", padding: 0, flexShrink: 0, lineHeight: "16px" }}
          >
            View analysis →
          </button>
          <PanelButton label="Upload data" onClick={onUpload} primary />
        </div>
      </div>
    </AsciiFrame>
  );
}

function PanelButton({ label, onClick, primary }: { label: string; onClick?: () => void; primary?: boolean }): React.ReactElement {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        ...FONT,
        fontSize: 13,
        fontWeight: 500,
        cursor: "pointer",
        color: primary ? "#1e1e1c" : T.text,
        background: primary ? T.lavender : "transparent",
        border: `1px solid ${primary ? T.lavender : T.frameStrong}`,
        borderRadius: 0,
        padding: "7px 14px",
        whiteSpace: "nowrap",
        transition: "background 120ms, border-color 120ms",
      }}
    >
      {label}
    </button>
  );
}
