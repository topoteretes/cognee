import { brainCells, type BrainCell } from "./brainCells";
import { brainGrid } from "./brainGrid";
import { BRAIN_SECTIONS, FULL, type BrainAmounts, type BrainSection } from "./brainSections";

/** Seed of the beat each square breathes on, so the server and the client draw the same one. */
const PULSE_SEED = 733;
const PULSE_SPEEDS = 3;
const PULSE_DELAYS = 10;

type Beats = Map<string, number[]>;

export interface BrainLayout {
  cols: number;
  rows: number;
  cells: BrainCell[];
  /** How many chunks of squares each lobe fills in. */
  chunks: Record<BrainSection, number>;
  /**
   * Squares sharing a beat are animated as one group: 30 beats a lobe instead
   * of thousands of separate animations, which keeps the page smooth.
   */
  beats: Map<BrainSection, Beats>;
}

function seeded(seed: number): () => number {
  let state = seed;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function beatsOf(cells: BrainCell[]): Map<BrainSection, Beats> {
  const random = seeded(PULSE_SEED);
  const lobes = new Map<BrainSection, Beats>();
  cells.forEach((cell, index) => {
    const beat = `pulse-t${Math.floor(random() * PULSE_SPEEDS)} pulse-d${Math.floor(random() * PULSE_DELAYS)}`;
    const lobe = lobes.get(cell.section) ?? new Map<string, number[]>();
    lobe.set(beat, [...(lobe.get(beat) ?? []), index]);
    lobes.set(cell.section, lobe);
  });
  return lobes;
}

let layout: BrainLayout | null = null;

/** Everything the brain's shape gives, worked out once: it does not change. */
export function brainLayout(): BrainLayout {
  if (layout) return layout;
  const cells = brainCells();
  const grid = brainGrid();
  const chunks = cells.reduce(
    (count, cell) => ({ ...count, [cell.section]: Math.max(count[cell.section] ?? 0, cell.chunk + 1) }),
    {} as Record<BrainSection, number>,
  );
  layout = { cols: grid[0].length, rows: grid.length, cells, chunks, beats: beatsOf(cells) };
  return layout;
}

/**
 * How many of each lobe's chunks are lit for the amounts: its share of full,
 * in whole chunks, so what is added lands as chunks of squares, more for what
 * weighs more.
 */
export function litChunks(amounts: BrainAmounts): Record<BrainSection, number> {
  const { chunks } = brainLayout();
  return Object.fromEntries(
    BRAIN_SECTIONS.map((section) => [section, Math.round(Math.min(amounts[section] / FULL[section], 1) * chunks[section])]),
  ) as Record<BrainSection, number>;
}
