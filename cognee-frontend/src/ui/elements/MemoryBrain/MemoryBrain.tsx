import { memo } from "react";
import type { ReactElement } from "react";
import classNames from "classnames";
import type { BrainCell } from "./brainCells";
import { brainLayout, litChunks } from "./brainLayout";
import {
  BRAIN_SECTIONS,
  FULL,
  SECTION_FILL,
  SECTION_LABEL,
  type BrainAmounts,
} from "./brainSections";

interface MemoryBrainProps {
  /** How many of each the workspace has: what fills each lobe. Every zero draws the empty brain. */
  amounts: BrainAmounts;
  /** What it holds is still being read in: the filled squares blink until it settles. */
  filling?: boolean;
}

// A plain function and not a component: there are over ten thousand squares,
// and each one as a component would be a thousand more things to reconcile.
function square(cell: BrainCell, isFilled: boolean, filling: boolean): ReactElement {
  return (
    <rect
      key={`${cell.x}-${cell.y}`}
      x={cell.x + 0.1}
      y={cell.y + 0.1}
      width="0.8"
      height="0.8"
      className={classNames(
        isFilled ? SECTION_FILL[cell.section] : "fill-cognee-fg/14",
        "motion-safe:transition-colors motion-safe:duration-700",
        { "motion-safe:animate-pulse": filling && isFilled },
      )}
    />
  );
}

/**
 * The memory as a brain of squares, one lobe to each thing a workspace
 * connects or builds. A lobe fills with its own shade of lavender as its
 * count grows toward full, a chunk of squares at a time. It knows nothing of
 * onboarding or the Overview: it draws the amounts it is given, and with no
 * motion preference it breathes, which `prefers-reduced-motion` turns off.
 * Its width is its container's. It is memoised, so a parent that renders for
 * other reasons does not redraw its ten thousand squares.
 */
export const MemoryBrain = memo(function MemoryBrain({
  amounts,
  filling = false,
}: MemoryBrainProps): ReactElement {
  const { cols, rows, cells, beats } = brainLayout();
  const lit = litChunks(amounts);
  const filled = new Set(
    cells.flatMap((cell, index) => (cell.chunk < lit[cell.section] ? [index] : [])),
  );
  const total = cells.length.toLocaleString("en-US");

  return (
    <svg
      role="img"
      aria-label={`Your memory: ${filled.size.toLocaleString("en-US")} of ${total} squares filled`}
      viewBox={`0 0 ${cols} ${rows}`}
      className="h-auto w-full"
    >
      {BRAIN_SECTIONS.map((section) => (
        <g key={section}>
          <title>{`${SECTION_LABEL[section]}: ${amountText(amounts[section])} of ${FULL[section]}`}</title>
          {[...(beats.get(section) ?? [])].map(([beat, indices]) => (
            <g key={beat} className={`brain-cell-pulse ${beat}`}>
              {indices.map((index) => square(cells[index], filled.has(index), filling))}
            </g>
          ))}
        </g>
      ))}
    </svg>
  );
});

// A lobe can hold a part of an item, and reads with at most one decimal: 1.5, and 2 and not 2.0.
function amountText(amount: number): string {
  return String(Math.round(amount * 10) / 10);
}
