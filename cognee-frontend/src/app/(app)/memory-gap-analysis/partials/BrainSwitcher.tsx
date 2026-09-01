"use client";

import React, { useCallback, useState } from "react";
import { FONT, T } from "@/app/(app)/dashboard/partials/redesign/mono";
import { formatScore, scoreColor } from "@/app/(app)/memory-gap-analysis/scoring";
import { RADIUS, SIZE, SPACE } from "@/app/(app)/memory-gap-analysis/ui";

export interface BrainOption {
  id: string;
  name: string;
  /** Null when the brain has never been scored — the chip reads "—". */
  score: number | null;
  /** Questions matching no topic, 0–1 — a taxonomy that needs attention. */
  sinkShare: number;
  needsAttention: boolean;
}

interface BrainSwitcherProps {
  brains: BrainOption[];
  selectedBrainId: string;
  onSelect: (brainId: string) => void;
}

/** Same database glyph the nav uses for Brain, so a dataset reads as a dataset. */
function DatasetIcon({ active, hovered }: { active: boolean; hovered: boolean }): React.ReactElement {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke={active || hovered ? "#EEEEEE" : T.muted} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden style={{ flexShrink: 0 }}>
      <ellipse cx="12" cy="5" rx="9" ry="3" />
      <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
      <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
    </svg>
  );
}

function BrainChip({ brain, active, onSelect }: { brain: BrainOption; active: boolean; onSelect: (brainId: string) => void }): React.ReactElement {
  const handleClick = useCallback(() => onSelect(brain.id), [brain.id, onSelect]);
  const [hovered, setHovered] = useState(false);

  return (
    <button
      type="button"
      onClick={handleClick}
      aria-pressed={active}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      title={
        brain.score === null
          ? "Not scored yet"
          : brain.needsAttention
            ? `${Math.round(brain.sinkShare * 100)}% of questions match no topic`
            : `Coverage ${formatScore(brain.score)} of 5`
      }
      style={{
        ...FONT,
        display: "inline-flex",
        alignItems: "center",
        gap: SPACE.sm,
        fontSize: SIZE.body,
        padding: `${SPACE.sm + 2}px ${SPACE.lg}px`,
        borderRadius: RADIUS,
        cursor: "pointer",
        whiteSpace: "nowrap",
        // Selected dataset inverts: lavender fill, dark ink — like primary buttons.
        color: active || hovered ? "#EEEEEE" : T.muted,
        background: "#000000",
        border: `1px solid ${active ? "#EEEEEE" : T.frame}`,
        transition: "background 120ms, border-color 120ms",
      }}
    >
      <DatasetIcon active={active} hovered={hovered} />
      {brain.name}
      {/* Number-only score chip, same idiom as the question cards. An unscored
          brain keeps the chip shape so the row doesn't reflow once scores land. */}
      <ScoreChip score={brain.score} />
    </button>
  );
}

function ScoreChip({ score }: { score: number | null }): React.ReactElement {
  const color = score === null ? T.faint : scoreColor(score);
  return (
    <span style={{ ...FONT, display: "inline-flex", alignItems: "center", gap: 4, fontSize: 11, fontWeight: 600, letterSpacing: "0.04em", color, background: `color-mix(in srgb, ${color} 12%, transparent)`, borderRadius: 100, padding: "1px 7px", fontVariantNumeric: "tabular-nums", flexShrink: 0 }}>
      <span style={{ width: 4, height: 4, borderRadius: "50%", background: color }} />
      {score === null ? "—" : formatScore(score)}
    </span>
  );
}

/**
 * Brain is the hard scope — one run per brain, never across. Showing every
 * brain's score inline makes the switcher double as the cross-brain roll-up,
 * and the amber dot marks a brain whose sink has outgrown its topics. Brains
 * with no run yet still list, scoreless, so the scope is always the real one.
 */
export function BrainSwitcher({ brains, selectedBrainId, onSelect }: BrainSwitcherProps): React.ReactElement {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: SPACE.sm, flexWrap: "wrap" }}>
      {brains.map((brain) => (
        <BrainChip key={brain.id} brain={brain} active={brain.id === selectedBrainId} onSelect={onSelect} />
      ))}
    </div>
  );
}
