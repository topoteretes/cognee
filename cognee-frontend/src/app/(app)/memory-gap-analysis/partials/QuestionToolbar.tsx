"use client";

import React, { useCallback, useState } from "react";
import { FONT, T } from "@/app/(app)/dashboard/partials/redesign/mono";
import { RADIUS, SIZE, SPACE } from "@/app/(app)/memory-gap-analysis/ui";
import type { SortMode } from "@/app/(app)/memory-gap-analysis/sorting";

export interface ScopeMetrics {
  /** Mean judge score across the current scope. */
  /** Questions in the scope scoring in the gap band. */
  /** Recalls the scope represents — the sum of its dedup cluster sizes. */
  asked: number;
}

interface QuestionToolbarProps {
  query: string;
  onQueryChange: (query: string) => void;
  metrics: ScopeMetrics;
  /** True when a single topic is selected, which re-labels the metrics. */
  topicScoped: boolean;
  sortMode: SortMode;
  onSortModeChange: (mode: SortMode) => void;
}

const SEARCH_WIDTH = 220;

const SORT_OPTIONS: { mode: SortMode; label: string }[] = [
  { mode: "score", label: "Score" },
  { mode: "recalls", label: "Recalls" },
];
const CONTROL_PADDING = `${SPACE.xs + 2}px ${SPACE.md}px`;

function Metric({ label, value, color }: { label: string; value: string; color: string }): React.ReactElement {
  return (
    <span style={{ display: "inline-flex", alignItems: "baseline", gap: SPACE.sm, whiteSpace: "nowrap" }}>
      <span style={{ fontSize: SIZE.meta, color: T.faint }}>{label}</span>
      <span style={{ fontSize: SIZE.meta, color, fontVariantNumeric: "tabular-nums" }}>{value}</span>
    </span>
  );
}

/** Panel-header export control — lavender stroke on hover, per the house hover rules. */
export function ExportButton({ onExport }: { onExport: () => void }): React.ReactElement {
  const [hovered, setHovered] = useState(false);
  return (
    <button
      type="button"
      onClick={onExport}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        ...FONT,
        fontSize: SIZE.meta,
        color: hovered ? T.lavender : T.text,
        background: "transparent",
        border: `1px solid ${hovered ? T.lavender : T.frameStrong}`,
        borderRadius: RADIUS,
        padding: CONTROL_PADDING,
        cursor: "pointer",
        whiteSpace: "nowrap",
        transition: "border-color 120ms, color 120ms",
      }}
    >
      Export
    </button>
  );
}

/**
 * Search, scope metrics and export. The metrics answer "how bad is what I am
 * currently looking at", so they re-scope with the topic and search filters —
 * and they sit in their own inset so they read as a readout, not stray labels.
 */
export function QuestionToolbar({ query, onQueryChange, metrics, topicScoped, sortMode, onSortModeChange }: QuestionToolbarProps): React.ReactElement {
  const handleInput = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => onQueryChange(event.target.value),
    [onQueryChange],
  );

  const prefix = topicScoped ? "Topic " : "";

  return (
    <div style={{ display: "flex", alignItems: "center", gap: SPACE.md, flexWrap: "wrap" }}>
      <input
        type="search"
        value={query}
        onChange={handleInput}
        placeholder="Search questions"
        aria-label="Search questions"
        style={{
          ...FONT,
          width: SEARCH_WIDTH,
          fontSize: SIZE.meta,
          color: T.text,
          background: T.chromeAlt,
          border: `1px solid ${T.frame}`,
          borderRadius: RADIUS,
          padding: CONTROL_PADDING,
          outline: "none",
        }}
      />

      <div style={{ display: "flex", alignItems: "center", gap: SPACE.sm }}>
        <span style={{ ...FONT, fontSize: SIZE.label, color: T.faint }}>Sort</span>
        {SORT_OPTIONS.map((option) => {
          const active = option.mode === sortMode;
          return (
            <button
              key={option.mode}
              type="button"
              onClick={() => onSortModeChange(option.mode)}
              aria-pressed={active}
              className="cursor-pointer"
              style={{
                ...FONT,
                fontSize: SIZE.label,
                fontWeight: active ? 500 : 400,
                color: active ? "var(--color-cognee-lavender)" : "rgba(237,236,234,0.7)",
                background: active ? "var(--color-cognee-lavender-tint-20)" : "rgba(255,255,255,0.06)",
                border: "none",
                borderRadius: 100,
                padding: "4px 12px",
                lineHeight: "16px",
                whiteSpace: "nowrap",
              }}
            >
              {option.label}
            </button>
          );
        })}
      </div>

      <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "flex-end", gap: SPACE.lg, flexWrap: "wrap" }}>
        <div style={{ ...FONT, display: "flex", alignItems: "center", gap: SPACE.lg }}>
          <Metric label={`${prefix}Recall`} value={`x${metrics.asked}`} color="var(--color-cognee-lavender)" />
        </div>
      </div>
    </div>
  );
}
