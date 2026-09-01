"use client";

import React from "react";
import { FONT, T } from "@/app/(app)/dashboard/partials/redesign/mono";
import { RADIUS, SIZE } from "@/app/(app)/memory-gap-analysis/ui";
import { formatScore, scoreColor } from "@/app/(app)/memory-gap-analysis/scoring";
import { MAX_SCORE } from "@/app/(app)/memory-gap-analysis/types";

const SEGMENTS = Array.from({ length: MAX_SCORE }, (_, i) => i);
/** Square meter cells, sized to the dataset-switcher button height. */
const SEGMENT = 40;

/**
 * Five-segment meter — the judge scale is 0–5, so a percentage bar would lie
 * about the unit. The fill is lavender like every other progress mark in the
 * dashboard; the verdict chip beside it carries the green/amber/red meaning.
 */
/** Each cell wears its band's legend colour: 0–2 gap red, 2–4 partial amber, 4–5 covered green. */
function segmentColor(index: number): string {
  if (index < 2) return T.red;
  if (index < 4) return T.amber;
  return T.green;
}

function ScoreMeter({ score }: { score: number }): React.ReactElement {
  return (
    <div style={{ position: "relative", flex: 1, minWidth: 0 }}>
      <div style={{ display: "flex", gap: 1 }}>
        {SEGMENTS.map((i) => {
          const fill = Math.max(0, Math.min(1, score - i));
          const color = segmentColor(i);
          return (
            <div key={i} style={{ flex: 1, height: SEGMENT, borderRadius: RADIUS, background: "#000000", border: `1px solid color-mix(in srgb, ${color} 35%, transparent)`, boxSizing: "border-box", overflow: "hidden" }}>
              <div style={{ width: `${fill * 100}%`, height: "100%", background: color, transition: "width 300ms" }} />
            </div>
          );
        })}
      </div>
      {/* The score rides inside the fill, hugging its right edge. */}
      <span
        style={{
          ...FONT,
          position: "absolute",
          top: "50%",
          left: `clamp(0%, ${(Math.max(0, Math.min(score, MAX_SCORE)) / MAX_SCORE) * 100}%, calc(100% - 68px))`,
          transform: "translateY(-50%)",
          paddingLeft: 10,
          fontSize: 28,
          fontWeight: 700,
          color: scoreColor(score),
          letterSpacing: "-0.02em",
          fontVariantNumeric: "tabular-nums",
          lineHeight: 1,
          whiteSpace: "nowrap",
        }}
      >
        {formatScore(score)}
      </span>
    </div>
  );
}

/** Coverage score cluster: label, square meter and the score·verdict chip. */
export function ScoreLine({ overallScore }: { overallScore: number }): React.ReactElement {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, flex: 1, minWidth: 0 }}>
      <ScoreMeter score={overallScore} />
    </div>
  );
}

/** Secondary header action — grey stroke, lavender on hover (Export idiom).
 *  Adding memory is how a low score gets fixed, so it lives beside Run. */
export function AddMemoryButton({ onClick }: { onClick: () => void }): React.ReactElement {
  const [hovered, setHovered] = React.useState(false);
  return (
    <button
      type="button"
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        ...FONT,
        fontSize: SIZE.meta,
        fontWeight: 500,
        color: hovered ? T.lavender : T.text,
        background: "transparent",
        border: `1px solid ${hovered ? T.lavender : T.frameStrong}`,
        borderRadius: RADIUS,
        padding: "7px 14px",
        cursor: "pointer",
        whiteSpace: "nowrap",
        flexShrink: 0,
        transition: "border-color 120ms, color 120ms",
      }}
    >
      Add memory
    </button>
  );
}

/** Primary page action — lavender fill, quiet while a run is in flight. */
export function RunButton({ isRunning, onRun }: { isRunning: boolean; onRun: () => void }): React.ReactElement {
  return (
    <button
      type="button"
      onClick={onRun}
      disabled={isRunning}
      style={{
        ...FONT,
        fontSize: SIZE.meta,
        fontWeight: 500,
        color: isRunning ? T.muted : "#1e1e1c",
        background: isRunning ? "transparent" : T.lavender,
        border: `1px solid ${isRunning ? T.frameStrong : T.lavender}`,
        borderRadius: RADIUS,
        padding: "7px 14px",
        cursor: isRunning ? "default" : "pointer",
        whiteSpace: "nowrap",
        flexShrink: 0,
      }}
    >
      {isRunning ? "Running…" : "Run analysis"}
    </button>
  );
}
