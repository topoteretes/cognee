import { brainGrid, GRID_SCALE } from "./brainGrid";
import type { BrainSection } from "./brainSections";

/** Where each lobe sits on the map; a square belongs to the nearest one. */
const CENTRES: readonly (readonly [BrainSection, number, number])[] = [
  ["teams", 33, 8],
  ["dataSources", 10, 27],
  ["integrations", 19, 51],
  ["agents", 57, 9],
  ["agents", 61, 24],
  ["sessions", 44, 37],
  ["connections", 31, 57],
];

/** The brain faces right and rows count a little more than columns, which flattens the borders the way the sulci run. */
const ROW_WEIGHT = 1.3;

/** The stem is a column of its own, below this row and between these columns. */
const STEM_FROM_ROW = 50;
const STEM_COLUMNS = [26, 34] as const;

/** How far round a square its lobe's border is smoothed, in drawn squares. */
const LOBE_SMOOTH = 2;
/** How round each lobe's corners are, in drawn squares. */
const LOBE_CORNER = 6;
/** How much of itself a lobe keeps when its corners are rounded. */
const KEEP_AT_LEAST = 0.7;

type Labels = (BrainSection | null)[][];

/** The lobe of a map character. */
function lobeAt(x: number, y: number): BrainSection {
  if (y >= STEM_FROM_ROW && x >= STEM_COLUMNS[0] && x <= STEM_COLUMNS[1]) return "connections";
  let best = CENTRES[0];
  let bestDistance = Infinity;
  for (const centre of CENTRES) {
    const distance = (x - centre[1]) ** 2 + ROW_WEIGHT * (y - centre[2]) ** 2;
    if (distance < bestDistance) {
      best = centre;
      bestDistance = distance;
    }
  }
  return best[0];
}

/** Where a disc of `radius` fits wholly inside `section`, and everything it reaches from there. */
function opening(labels: Labels, section: BrainSection, radius: number): boolean[][] {
  const disc: [number, number][] = [];
  for (let dy = -radius; dy <= radius; dy++) {
    for (let dx = -radius; dx <= radius; dx++) {
      if (dx * dx + dy * dy <= radius * radius) disc.push([dx, dy]);
    }
  }
  const fits = labels.map((row, y) =>
    row.map((own, x) => own === section && disc.every(([dx, dy]) => labels[y + dy]?.[x + dx] === section)),
  );
  return labels.map((row, y) =>
    row.map((own, x) => own === section && disc.some(([dx, dy]) => fits[y + dy]?.[x + dx])),
  );
}

/** Each square takes the lobe most of those around it have, so the borders run smooth instead of in steps. */
function smoothBorders(labels: Labels): Labels {
  let current = labels;
  for (let pass = 0; pass < 2; pass++) {
    const from = current;
    current = from.map((row, y) =>
      row.map((own, x) => {
        if (!own) return own;
        const votes = new Map<BrainSection, number>();
        for (let dy = -LOBE_SMOOTH; dy <= LOBE_SMOOTH; dy++) {
          for (let dx = -LOBE_SMOOTH; dx <= LOBE_SMOOTH; dx++) {
            const other = from[y + dy]?.[x + dx];
            if (other) votes.set(other, (votes.get(other) ?? 0) + 1);
          }
        }
        let best = own;
        for (const [section, count] of votes) if (count > (votes.get(best) ?? 0)) best = section;
        return best;
      }),
    );
  }
  return current;
}

/**
 * Each lobe smoothed and its corners rounded: a lobe keeps only what a disc of
 * `LOBE_CORNER` fits in and sweeps, so its corners come out round. What a lobe
 * gives up is left empty, part of the gap between it and the next. A narrow
 * lobe, the stem, takes a smaller disc, down to none, rather than vanish.
 */
function roundedLobes(labels: Labels): Labels {
  const smooth = smoothBorders(labels);
  const result: Labels = smooth.map((row) => row.map(() => null));
  const sections = new Set(smooth.flat().filter((own): own is BrainSection => own !== null));
  for (const section of sections) {
    const size = smooth.flat().filter((own) => own === section).length;
    for (let radius = LOBE_CORNER; radius >= 0; radius--) {
      const opened = opening(smooth, section, radius);
      const kept = opened.flat().filter(Boolean).length;
      if (kept >= size * KEEP_AT_LEAST || radius === 0) {
        opened.forEach((row, y) =>
          row.forEach((inside, x) => {
            if (inside) result[y][x] = section;
          }),
        );
        break;
      }
    }
  }
  return result;
}

/** The lobe of every square of the drawn grid, null outside the brain. */
export function brainLobes(): Labels {
  return roundedLobes(
    brainGrid().map((row, y) =>
      [...row].map((cell, x) => (cell !== "." ? lobeAt(Math.floor(x / GRID_SCALE), Math.floor(y / GRID_SCALE)) : null)),
    ),
  );
}
