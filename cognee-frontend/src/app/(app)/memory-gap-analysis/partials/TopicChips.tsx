"use client";

import React, { useCallback, useState } from "react";
import { FONT, T } from "@/app/(app)/dashboard/partials/redesign/mono";
import { SIZE, SPACE } from "@/app/(app)/memory-gap-analysis/ui";
import { SINK_TOPIC_ID } from "@/app/(app)/memory-gap-analysis/types";

export interface TopicChipRow {
  topicId: string;
  label: string;
  questionCount: number;
}

interface TopicChipsProps {
  rows: TopicChipRow[];
  sinkCount: number;
  totalQuestions: number;
  /** null = all topics. */
  selectedTopicId: string | null;
  onSelect: (topicId: string | null) => void;
  onRequestDelete: (topicId: string) => void;
}

interface ChipProps {
  label: string;
  count: number;
  active: boolean;
  topicId: string | null;
  muted: boolean;
  onSelect: (topicId: string | null) => void;
  /** Omitted for chips that cannot be deleted — All topics and Other. */
  onRequestDelete?: (topicId: string) => void;
}

/**
 * One filter chip per topic — the billing filter-chip idiom: pill, tinted when
 * active. Hovering a user-owned topic reveals its delete control inside the chip.
 */
function Chip({ label, count, active, topicId, muted, onSelect, onRequestDelete }: ChipProps): React.ReactElement {
  const [hovered, setHovered] = useState(false);
  const deletable = onRequestDelete !== undefined && topicId !== null;

  const handleDelete = useCallback((event: React.MouseEvent) => {
    event.stopPropagation();
    if (onRequestDelete !== undefined && topicId !== null) onRequestDelete(topicId);
  }, [onRequestDelete, topicId]);

  return (
    <button
      type="button"
      onClick={() => onSelect(topicId)}
      aria-pressed={active}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className="cursor-pointer"
      style={{
        ...FONT,
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        background: active ? "var(--color-cognee-lavender-tint-20)" : "rgba(255,255,255,0.06)",
        color: active ? "var(--color-cognee-lavender)" : muted ? T.faint : "rgba(237,236,234,0.7)",
        border: "none",
        borderRadius: 100,
        padding: "4px 12px",
        fontSize: SIZE.label,
        fontWeight: active ? 500 : 400,
        lineHeight: "16px",
        whiteSpace: "nowrap",
      }}
    >
      {label}
      <span style={{ color: active ? "var(--color-cognee-lavender)" : T.faint, fontVariantNumeric: "tabular-nums" }}>{count}</span>
      {deletable && hovered && (
        <span
          role="button"
          tabIndex={0}
          onClick={handleDelete}
          onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); handleDelete(e as unknown as React.MouseEvent); } }}
          aria-label={`Delete topic ${label}`}
          title={`Delete topic ${label}`}
          style={{ display: "inline-flex", color: "inherit" }}
        >
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" aria-hidden>
            <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </span>
      )}
    </button>
  );
}

/**
 * Topic filter chips for the questions grid. Topics are user-owned, so the row
 * is stable between runs; the sink chip renders muted because it is a holding
 * area, not a topic anyone chose.
 */
export function TopicChips({ rows, sinkCount, totalQuestions, selectedTopicId, onSelect, onRequestDelete }: TopicChipsProps): React.ReactElement {
  return (
    <div role="toolbar" aria-label="Filter by topic" style={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: SPACE.sm }}>
      <Chip label="All topics" count={totalQuestions} active={selectedTopicId === null} topicId={null} muted={false} onSelect={onSelect} />
      {rows.map((row) => (
        <Chip
          key={row.topicId}
          label={row.label}
          count={row.questionCount}
          active={selectedTopicId === row.topicId}
          topicId={row.topicId}
          muted={false}
          onSelect={onSelect}
          onRequestDelete={onRequestDelete}
        />
      ))}
      <Chip label="Other" count={sinkCount} active={selectedTopicId === SINK_TOPIC_ID} topicId={SINK_TOPIC_ID} muted onSelect={onSelect} />
    </div>
  );
}
