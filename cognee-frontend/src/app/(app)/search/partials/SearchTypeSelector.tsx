"use client";

import { useRef } from "react";
import { SEARCH_TYPES, type SearchTypeId } from "@/app/(app)/search/searchTypes";

interface SearchTypeSelectorProps {
  value: SearchTypeId;
  onChange: (id: SearchTypeId) => void;
}

/**
 * Copied from BrainSelector's trigger button (src/ui/elements/BrainSelector.tsx)
 * character for character. The two controls sit against each other, so each
 * option here has to be the same pill the dataset picker is — same glass, same
 * blur, same border, same radius, same padding, same type.
 *
 * It is a copy, not a shared token, because BrainSelector is also rendered by
 * the legacy mindmap page and is not ours to edit. If that button's chrome
 * changes, change it here in the same commit — nothing enforces this, since no
 * test can compare two rendered boxes.
 */
const PICKER_PILL = {
  display: "flex", alignItems: "center", gap: 6,
  background: "rgba(255,255,255,0.06)", backdropFilter: "blur(12px)",
  border: "1px solid rgba(255,255,255,0.1)",
  borderRadius: 8, padding: "6px 10px",
  fontSize: 13, fontWeight: 500, color: "rgba(237,236,234,0.85)",
  fontFamily: "inherit",
} as const;

// The selected option is BrainSelector's own dropdown selection — lavender
// fill and border — with the text raised to full lavender. At the 60% alpha
// those rows use, the label was about 3.4:1 against its fill, below AA for
// 13px text; the rows get away with it because they also carry a checkmark and
// a weight change, and here the colour is the only mark of which one is on.
// #BC9BFF on that fill is about 6.7:1.
const SELECTED_PILL = {
  background: "rgba(188,155,255,0.20)",
  border: "1px solid rgba(188,155,255,0.35)",
  color: "#BC9BFF",
} as const;

// Arrow keys move the selection, as in any radio group; the step each key takes.
const ARROW_STEP: Record<string, number> = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 };

/**
 * Which retrieval lane the next question runs on.
 *
 * Each option is one PICKER_PILL, so radius, height, padding and type cannot
 * drift from the dataset picker beside it — they are the same object. The
 * selected one swaps only its fill and text colour, to the lavender every
 * selected thing in this app uses, including the rows inside the picker's own
 * dropdown. No track, no sliding thumb: a second box meant a second radius
 * nested inside the first, and that is what stopped the two reading as one
 * family.
 *
 * A radio group, not two toggle buttons: exactly one is always on, which is
 * what a screen reader has to announce. Only the selected option is in the
 * tab order, and the arrow keys move between them.
 */
export default function SearchTypeSelector({ value, onChange }: SearchTypeSelectorProps): React.ReactElement {
  const optionRefs = useRef<(HTMLButtonElement | null)[]>([]);

  const handleKeyDown = (event: React.KeyboardEvent<HTMLDivElement>): void => {
    const step = ARROW_STEP[event.key];
    if (step === undefined) return;
    event.preventDefault();
    const current = SEARCH_TYPES.findIndex((type) => type.id === value);
    const next = (current + step + SEARCH_TYPES.length) % SEARCH_TYPES.length;
    onChange(SEARCH_TYPES[next].id);
    optionRefs.current[next]?.focus();
  };

  return (
    <div
      role="radiogroup"
      aria-label="Search type"
      data-testid="search-type-selector"
      onKeyDown={handleKeyDown}
      style={{ display: "flex", alignItems: "center", gap: 6 }}
    >
      {SEARCH_TYPES.map((type, index) => {
        const isSelected = type.id === value;
        return (
          <button
            key={type.id}
            ref={(element) => { optionRefs.current[index] = element; }}
            type="button"
            role="radio"
            aria-checked={isSelected}
            tabIndex={isSelected ? 0 : -1}
            onClick={() => onChange(type.id)}
            title={type.hint}
            className="cursor-pointer"
            style={{
              ...PICKER_PILL,
              ...(isSelected ? SELECTED_PILL : {}),
              transition: "background 150ms, border-color 150ms, color 150ms",
            }}
          >
            {type.label}
          </button>
        );
      })}
    </div>
  );
}
