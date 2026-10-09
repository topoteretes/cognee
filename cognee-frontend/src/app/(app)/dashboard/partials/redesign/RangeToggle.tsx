"use client";

import React from "react";
import { FONT, T } from "./mono";

export type DashRange = "24h" | "7d" | "30d";
export const RANGES: DashRange[] = ["24h", "7d", "30d"];

interface RangeToggleProps {
  value: DashRange;
  onChange: (r: DashRange) => void;
}

/** Segmented pill control for the Overview time range. */
export function RangeToggle({ value, onChange }: RangeToggleProps): React.ReactElement {
  return (
    <div
      style={{
        ...FONT,
        display: "inline-flex",
        alignItems: "center",
        gap: 2,
        padding: 2,
        background: "rgba(255,255,255,0.03)",
        border: `1px solid ${T.frame}`,
        borderRadius: 0,
      }}
    >
      {RANGES.map((r) => {
        const active = r === value;
        return (
          <button
            key={r}
            type="button"
            onClick={() => onChange(r)}
            aria-pressed={active}
            style={{
              ...FONT,
              fontSize: 12,
              fontWeight: active ? 600 : 500,
              cursor: "pointer",
              border: "none",
              background: active ? T.purpleSoft : "transparent",
              color: active ? T.lavender : T.muted,
              borderRadius: 0,
              padding: "2px 10px",
              transition: "color 120ms, background 120ms",
            }}
          >
            {r}
          </button>
        );
      })}
    </div>
  );
}
