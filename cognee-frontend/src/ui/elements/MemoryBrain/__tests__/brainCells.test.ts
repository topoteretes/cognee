import { brainCells } from "../brainCells";
import { brainLayout } from "../brainLayout";

// The numbers below were taken from the design's own code
// (cognee-design, src/components/ui/graph/brainSections.ts, `brainCells("brain")`),
// so a drift in the port shows as a different brain.
const DESIGN_CELLS: Record<string, { cells: number; chunks: number }> = {
  teams: { cells: 2337, chunks: 169 },
  agents: { cells: 2206, chunks: 157 },
  dataSources: { cells: 2629, chunks: 186 },
  sessions: { cells: 1848, chunks: 134 },
  integrations: { cells: 1433, chunks: 108 },
  connections: { cells: 232, chunks: 26 },
};
const DESIGN_TOTAL = 10685;
const DESIGN_HASH = 334946785;

function fingerprint(cells: ReturnType<typeof brainCells>): number {
  let hash = 0;
  for (const cell of cells) {
    hash = (hash * 31 + cell.x * 7919 + cell.y * 104729 + cell.chunk * 13 + cell.section.length) | 0;
  }
  return hash;
}

describe("the brain's shape", () => {
  it("has the cells the design draws", () => {
    expect(brainCells()).toHaveLength(DESIGN_TOTAL);
  });

  it("gives each lobe the cells and chunks the design gives it", () => {
    const { cells, chunks } = brainLayout();

    for (const [section, expected] of Object.entries(DESIGN_CELLS)) {
      expect(cells.filter((cell) => cell.section === section)).toHaveLength(expected.cells);
      expect(chunks[section as keyof typeof chunks]).toBe(expected.chunks);
    }
  });

  it("puts every cell where the design puts it, in the order the design fills it", () => {
    expect(fingerprint(brainCells())).toBe(DESIGN_HASH);
  });

  it("is drawn on a grid of 142 by 128 squares", () => {
    const { cols, rows } = brainLayout();

    expect([cols, rows]).toEqual([142, 128]);
  });
});
