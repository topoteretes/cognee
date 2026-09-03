// Canvas text has no DOM layout engine to consult, so "avoid overlapping
// labels" here means deciding, every frame, which candidate labels get
// drawn at all — the same greedy point-label declutter technique dense-point
// tools (Mapbox, kepler.gl) use: sort by priority, accept a label only if
// its box doesn't overlap one already accepted. A true force-directed label
// layout (nudging overlapping labels apart) would need its own simulation
// running every frame for a few hundred candidates — likely-jittery
// overkill for what's really a "which labels lose" decision.
//
// Boxes are computed in world units (the same space as node x/y), which is
// valid because callers already draw at a font size compensated for zoom
// (divided by sqrt(k)) — ctx.measureText returns widths in that same
// pre-transform space, so a world-space AABB test is equivalent to a
// screen-space one regardless of the current zoom level.
export interface LabelCandidate {
  key: string;
  x: number;
  y: number;
  width: number;
  height: number;
  priority: number;
}

export interface Box {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

function boxesOverlap(a: Box, b: Box): boolean {
  return a.x1 < b.x2 && a.x2 > b.x1 && a.y1 < b.y2 && a.y2 > b.y1;
}

// World-space bucket width for the collision index below. Roughly a couple
// of label widths: small enough that a box lands in a handful of buckets,
// large enough that the per-box bucket walk stays short even when the zoom
// floor inflates node circles at low k.
const GRID_CELL_WORLD = 64;

// A uniform-grid collision index. Testing every candidate against one flat
// array — seeded with an obstacle per visible entity — made the per-frame
// pass O(labels x entities): on the dense graphs this whole layout exists
// for, six figures of overlap tests inside a 60fps RAF loop. Bucketing turns
// each test into a walk of the few cells a box actually touches.
//
// Nested column→row maps rather than one map on a packed numeric key: world
// coordinates go negative in every direction here, and any single-number
// packing of a signed (col, row) pair has aliasing cases that would silently
// merge two distant cells into one bucket.
type Grid = Map<number, Map<number, Box[]>>;

function forEachCell(box: Box, visit: (col: number, row: number) => void): void {
  const colMin = Math.floor(box.x1 / GRID_CELL_WORLD);
  const colMax = Math.floor(box.x2 / GRID_CELL_WORLD);
  const rowMin = Math.floor(box.y1 / GRID_CELL_WORLD);
  const rowMax = Math.floor(box.y2 / GRID_CELL_WORLD);
  for (let col = colMin; col <= colMax; col += 1) {
    for (let row = rowMin; row <= rowMax; row += 1) visit(col, row);
  }
}

function insert(grid: Grid, box: Box): void {
  forEachCell(box, (col, row) => {
    let column = grid.get(col);
    if (!column) {
      column = new Map();
      grid.set(col, column);
    }
    const bucket = column.get(row);
    if (bucket) bucket.push(box);
    else column.set(row, [box]);
  });
}

function collides(grid: Grid, box: Box): boolean {
  let hit = false;
  forEachCell(box, (col, row) => {
    if (hit) return;
    // A box spanning several cells sits in each of their buckets, so the
    // same pair can be compared more than once — harmless for a boolean
    // answer, and cheaper than deduplicating.
    const bucket = grid.get(col)?.get(row);
    if (bucket && bucket.some((b) => boxesOverlap(box, b))) hit = true;
  });
  return hit;
}

// y is treated as the text baseline (alphabetic textBaseline, this module's
// only caller convention) — the box extends one line-height above it and a
// small descender allowance below.
//
// `obstacles` seeds the collision index with boxes that block a label but can
// never themselves be evicted or "picked" — the caller uses this for node
// circles (a label overlapping some OTHER entity's dot in a dense cluster
// read as belonging to that dot, not its own, which label-vs-label collision
// alone never caught since a label naturally clears its own circle by
// drawing below it).
//
// A candidate with `priority: Infinity` is placed unconditionally. Sorting
// alone could not deliver the "a label the user selected or hovered never
// loses" guarantee the caller writes that priority for: sorting only settles
// label-vs-label ties, and the obstacles are already in the index before the
// first candidate is considered, so a node circle silently outranked it.
export function pickNonOverlappingLabels(candidates: LabelCandidate[], obstacles: Box[] = []): Set<string> {
  const sorted = [...candidates].sort((a, b) => b.priority - a.priority);
  const grid: Grid = new Map();
  obstacles.forEach((b) => insert(grid, b));
  const picked = new Set<string>();
  sorted.forEach((c) => {
    const box: Box = {
      x1: c.x - c.width / 2,
      y1: c.y - c.height,
      x2: c.x + c.width / 2,
      y2: c.y + c.height * 0.25,
    };
    if (c.priority !== Infinity && collides(grid, box)) return;
    insert(grid, box);
    picked.add(c.key);
  });
  return picked;
}
