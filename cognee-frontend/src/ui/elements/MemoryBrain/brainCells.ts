import { brainGrid } from "./brainGrid";
import { brainLobes } from "./brainLobes";
import { SECTION_SIDE, type BrainSection } from "./brainSections";

export type BrainCell = {
  x: number;
  y: number;
  section: BrainSection;
  /** Its place in its lobe's fill order: 0 fills first. */
  order: number;
  /**
   * Its chunk's place in its lobe's fill order. A lobe fills a chunk of
   * squares at a time, a 4x4 block of the grid or what of one the lobe holds,
   * so whatever is added lands as chunks, not one square at a time.
   */
  chunk: number;
};

type PlainCell = Pick<BrainCell, "x" | "y" | "section">;
type Chunk = { cx: number; cy: number; members: PlainCell[] };
type Chunks = Map<string, Chunk>;

/** The side of a chunk, in cells. */
const CHUNK = 4;
/** Seed of the order chunks are picked in, so the server and the client draw the same brain. */
const SEED = 1173;

const SIDES: readonly (readonly [number, number])[] = [
  [1, 0],
  [-1, 0],
  [0, 1],
  [0, -1],
];

function seeded(seed: number): () => number {
  let state = seed;
  return () => {
    state = (state * 1103515245 + 12345) % 2147483648;
    return state / 2147483648;
  };
}

/**
 * Every cell of memory with its lobe. A cell on the border with another lobe
 * is left out, a one-cell gap like a fold, so the lobes read as separate parts
 * even before they fill.
 */
function lobeCells(): PlainCell[] {
  const labels = brainLobes();
  const at = (x: number, y: number) => labels[y]?.[x] ?? null;
  const cells: PlainCell[] = [];
  brainGrid().forEach((row, y) => {
    [...row].forEach((cell, x) => {
      const section = labels[y][x];
      if (cell !== "#" || !section) return;
      const right = at(x + 1, y);
      const below = at(x, y + 1);
      if ((right && right !== section) || (below && below !== section)) return;
      cells.push({ x, y, section });
    });
  });
  return cells;
}

/**
 * Where a lobe's spokes land, in cells: its outermost cell on the given side,
 * at the middle of the rows that reach that far.
 */
function lobePort(cells: PlainCell[], section: BrainSection, side: "left" | "right") {
  const own = cells.filter((cell) => cell.section === section);
  const edge = side === "left" ? Math.min(...own.map((cell) => cell.x)) : Math.max(...own.map((cell) => cell.x));
  const rows = own.filter((cell) => cell.x === edge).map((cell) => cell.y);
  return {
    x: side === "left" ? edge : edge + 1,
    y: (Math.min(...rows) + Math.max(...rows) + 1) / 2,
  };
}

function chunksBySection(cells: PlainCell[]): Map<BrainSection, Chunks> {
  const bySection = new Map<BrainSection, Chunks>();
  for (const cell of cells) {
    const cx = Math.floor(cell.x / CHUNK);
    const cy = Math.floor(cell.y / CHUNK);
    const own = bySection.get(cell.section) ?? new Map<string, Chunk>();
    bySection.set(cell.section, own);
    const chunk = own.get(`${cx}:${cy}`) ?? { cx, cy, members: [] };
    chunk.members.push(cell);
    own.set(`${cx}:${cy}`, chunk);
  }
  return bySection;
}

function nearestTo(pool: Chunk[], start: { x: number; y: number }): Chunk {
  const distance = (chunk: Chunk) => Math.hypot(chunk.cx * CHUNK - start.x, chunk.cy * CHUNK - start.y);
  return pool.reduce((best, chunk) => (distance(chunk) < distance(best) ? chunk : best));
}

/** How many of a chunk's eight neighbours have filled. */
function filledAround(chunk: Chunk, own: Chunks, left: Set<Chunk>): number {
  let count = 0;
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      const next = own.get(`${chunk.cx + dx}:${chunk.cy + dy}`);
      if ((dx || dy) && next && !left.has(next)) count++;
    }
  }
  return count;
}

/**
 * The order a lobe fills its chunks in: it starts with the chunk where its
 * spokes land, and every chunk after is the one touching the most of what has
 * filled (a seeded random one of those), so the chunks pack against each other
 * with no gaps behind.
 */
function fillOrder(own: Chunks, start: { x: number; y: number }, random: () => number): Chunk[] {
  const left = new Set(own.values());
  const frontier = new Set<Chunk>();
  const dealt: Chunk[] = [];
  const take = (chunk: Chunk) => {
    left.delete(chunk);
    frontier.delete(chunk);
    dealt.push(chunk);
    for (const [dx, dy] of SIDES) {
      const next = own.get(`${chunk.cx + dx}:${chunk.cy + dy}`);
      if (next && left.has(next)) frontier.add(next);
    }
  };
  take(nearestTo([...left], start));
  while (left.size > 0) {
    // A part of the lobe cut off by a fold carries on from its nearest chunk.
    if (frontier.size === 0) {
      take(nearestTo([...left], start));
      continue;
    }
    const candidates = [...frontier];
    const most = Math.max(...candidates.map((chunk) => filledAround(chunk, own, left)));
    const pool = candidates.filter((chunk) => filledAround(chunk, own, left) === most);
    take(pool[Math.floor(random() * pool.length)]);
  }
  return dealt;
}

/** Numbers the dealt chunks, and the cells in them, within their lobe. */
function ranked(dealt: Chunk[]): BrainCell[] {
  const order = new Map<BrainSection, number>();
  const chunkRank = new Map<BrainSection, number>();
  return dealt.flatMap(({ members }) => {
    const section = members[0].section;
    const chunk = chunkRank.get(section) ?? 0;
    chunkRank.set(section, chunk + 1);
    return members.map((cell) => {
      const index = order.get(section) ?? 0;
      order.set(section, index + 1);
      return { ...cell, order: index, chunk };
    });
  });
}

function startOf(section: BrainSection, own: Chunks, cells: PlainCell[]): { x: number; y: number } {
  if (section === "connections") {
    return { x: 0, y: Math.min(...[...own.values()].map((chunk) => chunk.cy)) * CHUNK };
  }
  return lobePort(cells, section, SECTION_SIDE[section]);
}

let cached: BrainCell[] | null = null;

/** Every cell of the brain, with its lobe and its place in that lobe's fill. Worked out once. */
export function brainCells(): BrainCell[] {
  if (cached) return cached;
  const cells = lobeCells();
  const random = seeded(SEED);
  const dealt: Chunk[] = [];
  for (const [section, own] of chunksBySection(cells)) {
    dealt.push(...fillOrder(own, startOf(section, own, cells), random));
  }
  cached = ranked(dealt);
  return cached;
}
