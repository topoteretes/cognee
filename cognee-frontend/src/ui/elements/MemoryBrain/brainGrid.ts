import { BRAIN_MAP } from "./brainMap";

/**
 * How many squares across each character of the map is drawn as. The map is
 * coarse; each character becomes a block of smaller squares, so the brain
 * keeps its outline with more, finer squares in it. The lobes are still worked
 * out on the map's own grid.
 */
export const GRID_SCALE = 2;

/** How many times the shape is smoothed: twice rounds a character's steps off without losing the folds. */
const SMOOTH_PASSES = 2;
/** How many of its nine a fold square needs to stay a fold; a fresh one needs one more. */
const FOLD_KEEP = 9;
/** How far round a square the outline is smoothed: the wider, the simpler the silhouette. */
const OUTLINE_RADIUS = 3;
/** A fold smaller than this many squares is a pinhole, not a fold: it fills in. */
const PINHOLE = 40;

const NEIGHBOURS: readonly (readonly [number, number])[] = [
  [1, 0],
  [-1, 0],
  [0, 1],
  [0, -1],
];

/**
 * The silhouette with its small bumps and notches taken out: a square is
 * inside when most of those within `OUTLINE_RADIUS` are, twice over. What it
 * adds inside the brain is memory; folds keep their place.
 */
function simplerOutline(grid: string[]): string[] {
  let current = grid;
  const area = (OUTLINE_RADIUS * 2 + 1) ** 2;
  for (let pass = 0; pass < SMOOTH_PASSES; pass++) {
    const from = current;
    const at = (x: number, y: number) => from[y]?.[x] ?? ".";
    current = from.map((row, y) =>
      [...row]
        .map((cell, x) => {
          let inside = 0;
          for (let dy = -OUTLINE_RADIUS; dy <= OUTLINE_RADIUS; dy++) {
            for (let dx = -OUTLINE_RADIUS; dx <= OUTLINE_RADIUS; dx++) {
              if (at(x + dx, y + dy) !== ".") inside++;
            }
          }
          if (inside * 2 <= area) return ".";
          return cell === "." ? "#" : cell;
        })
        .join(""),
    );
  }
  return current;
}

/** Fills in the folds too small to read as folds. */
function withoutPinholes(grid: string[]): string[] {
  const cells = grid.map((row) => [...row]);
  const seen = new Set<string>();
  cells.forEach((row, y) =>
    row.forEach((cell, x) => {
      if (cell !== ":" || seen.has(`${x}:${y}`)) return;
      const piece: [number, number][] = [];
      const queue: [number, number][] = [[x, y]];
      seen.add(`${x}:${y}`);
      for (let next = queue.pop(); next; next = queue.pop()) {
        const [cx, cy] = next;
        piece.push([cx, cy]);
        for (const [dx, dy] of NEIGHBOURS) {
          const nx = cx + dx;
          const ny = cy + dy;
          if (cells[ny]?.[nx] === ":" && !seen.has(`${nx}:${ny}`)) {
            seen.add(`${nx}:${ny}`);
            queue.push([nx, ny]);
          }
        }
      }
      if (piece.length < PINHOLE) for (const [px, py] of piece) cells[py][px] = "#";
    }),
  );
  return cells.map((row) => row.join(""));
}

/**
 * The traced shape, refined at the size it is drawn: every square takes what
 * most of the nine around it are, inside or out, fold or not. The steps a
 * character wide round off into curves, the outline and the folds flow as a
 * brain's do, and a stray square or pinhole goes.
 */
function smoothed(grid: string[]): string[] {
  const width = grid[0]?.length ?? 0;
  let current = simplerOutline(grid);
  for (let pass = 0; pass < SMOOTH_PASSES; pass++) {
    const from = current;
    const at = (x: number, y: number) => from[y]?.[x] ?? ".";
    current = from.map((row, y) =>
      Array.from({ length: width }, (_, x) => {
        let inside = 0;
        let folds = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const cell = at(x + dx, y + dy);
            if (cell !== ".") inside++;
            if (cell === ":") folds++;
          }
        }
        if (inside < 5) return ".";
        // Folds take more than a majority to stay, so they thin to grooves
        // and the brain is mostly memory, not holes.
        return folds >= FOLD_KEEP + 1 || (row[x] === ":" && folds >= FOLD_KEEP) ? ":" : "#";
      }).join(""),
    );
  }
  return withoutPinholes(current);
}

let grid: string[] | null = null;

/** The brain's map at the size it is drawn: every character a block of squares, smoothed. */
export function brainGrid(): string[] {
  if (grid) return grid;
  const blocks = BRAIN_MAP.flatMap((row) => {
    const wide = [...row].map((cell) => cell.repeat(GRID_SCALE)).join("");
    return Array.from({ length: GRID_SCALE }, () => wide);
  });
  grid = smoothed(blocks);
  return grid;
}
