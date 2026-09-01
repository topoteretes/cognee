"use client";

import React from "react";
import { T } from "@/app/(app)/dashboard/partials/redesign/mono";
import type { QuestionView } from "@/app/(app)/memory-gap-analysis/partials/QuestionGrid";

interface ViewToggleProps {
  view: QuestionView;
  onChange: (view: QuestionView) => void;
}

function GridIcon(): React.ReactElement {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
      <rect x="3" y="3" width="7" height="7" /><rect x="14" y="3" width="7" height="7" />
      <rect x="3" y="14" width="7" height="7" /><rect x="14" y="14" width="7" height="7" />
    </svg>
  );
}

function ListIcon(): React.ReactElement {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden>
      <line x1="4" y1="6" x2="20" y2="6" /><line x1="4" y1="12" x2="20" y2="12" /><line x1="4" y1="18" x2="20" y2="18" />
    </svg>
  );
}

/** Grid/list switcher for the questions panel — square buttons, lavender when active. */
export function ViewToggle({ view, onChange }: ViewToggleProps): React.ReactElement {
  const options: { value: QuestionView; label: string; icon: React.ReactElement }[] = [
    { value: "grid", label: "Grid view", icon: <GridIcon /> },
    { value: "list", label: "List view", icon: <ListIcon /> },
  ];
  return (
    <span style={{ display: "inline-flex" }}>
      {options.map((option) => {
        const active = option.value === view;
        return (
          <button
            key={option.value}
            type="button"
            onClick={() => onChange(option.value)}
            aria-label={option.label}
            aria-pressed={active}
            title={option.label}
            className="cursor-pointer"
            style={{
              display: "inline-flex", alignItems: "center", justifyContent: "center",
              width: 28, height: 28,
              color: active ? "var(--color-cognee-lavender)" : T.muted,
              background: active ? "var(--color-cognee-lavender-tint-10)" : "transparent",
              border: `1px solid ${active ? "rgba(188,155,255,0.55)" : T.frameStrong}`,
              borderRadius: 0,
              marginLeft: -1,
            }}
          >
            {option.icon}
          </button>
        );
      })}
    </span>
  );
}
