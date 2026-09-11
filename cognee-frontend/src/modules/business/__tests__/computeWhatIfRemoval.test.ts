import type { SemanticLink } from "../sceneTypes";
import { computeWhatIfRemoval } from "../computeWhatIfRemoval";

// ── helpers ──────────────────────────────────────────────────────────────────

function link(sid: string, tid: string): SemanticLink {
  return { source: sid, target: tid, _sid: sid, _tid: tid };
}

// ── tests ─────────────────────────────────────────────────────────────────────

describe("computeWhatIfRemoval", () => {
  describe("guard clauses", () => {
    it("returns nothing orphaned when the entity isn't in the graph at all", () => {
      const result = computeWhatIfRemoval("GHOST", [link("A", "B")]);
      expect(result).toEqual({ orphanedIds: new Set(), islandCount: 0 });
    });

    it("returns nothing orphaned for an empty link list", () => {
      const result = computeWhatIfRemoval("A", []);
      expect(result.orphanedIds.size).toBe(0);
    });

    it("returns nothing orphaned when the entity's component is just itself plus one neighbor", () => {
      // A-B only: removing A leaves B alone, but there's nothing else for B
      // to be cut off FROM — a lone pair isn't a "single point of failure".
      const result = computeWhatIfRemoval("A", [link("A", "B")]);
      expect(result.orphanedIds.size).toBe(0);
    });
  });

  describe("bridging hub", () => {
    it("finds the entities stranded when a bridge node is removed", () => {
      // Two triangles joined only through hub H: A-B-A_extra forms one side,
      // C-D-C_extra the other, H is the sole connector.
      const links = [
        link("A", "A_extra"), link("A", "H"),
        link("H", "C"), link("C", "C_extra"),
      ];
      const result = computeWhatIfRemoval("H", links);
      expect(result.islandCount).toBe(2);
      // The smaller/equal fragment becomes "orphaned" — both sides are the
      // same size here (2 nodes each), so exactly one of them is reported.
      expect(result.orphanedIds.size).toBe(2);
    });

    it("does not include the removed entity itself in the orphaned set", () => {
      const links = [link("A", "H"), link("H", "B"), link("B", "B_extra")];
      const result = computeWhatIfRemoval("H", links);
      expect(result.orphanedIds.has("H")).toBe(false);
    });

    it("treats the largest surviving fragment as intact, not orphaned", () => {
      // H bridges a lone node A to a larger cluster B-C-D — removing H
      // strands only A, not the whole B-C-D side.
      const links = [link("A", "H"), link("H", "B"), link("B", "C"), link("C", "D")];
      const result = computeWhatIfRemoval("H", links);
      expect(result.orphanedIds).toEqual(new Set(["A"]));
    });
  });

  describe("non-bridging entity", () => {
    it("finds nothing orphaned when removing a node from a cycle (redundant path exists)", () => {
      // Ring A-B-C-D-A: removing any one node leaves the rest still connected
      // via the opposite arc.
      const links = [link("A", "B"), link("B", "C"), link("C", "D"), link("D", "A")];
      const result = computeWhatIfRemoval("A", links);
      expect(result.orphanedIds.size).toBe(0);
    });

    it("finds nothing orphaned for a leaf node (removing it strands no one else)", () => {
      const links = [link("A", "B"), link("B", "C"), link("B", "D")];
      // C is a leaf hanging off B — removing C affects only C, not the rest.
      const result = computeWhatIfRemoval("C", links);
      expect(result.orphanedIds.size).toBe(0);
    });
  });

  describe("multiple islands", () => {
    it("collects orphaned ids from every smaller fragment, not just one", () => {
      // H connects to three separate leaves — removing H stands up 3
      // singleton islands, all reported.
      const links = [link("H", "A"), link("H", "B"), link("H", "C")];
      const result = computeWhatIfRemoval("H", links);
      expect(result.islandCount).toBe(3);
      expect(result.orphanedIds).toEqual(new Set(["A", "B", "C"]));
    });
  });

  describe("unrelated components", () => {
    it("ignores entities and edges outside the removed node's own component", () => {
      const links = [link("H", "A"), link("H", "B"), link("X", "Y"), link("Y", "Z")];
      const result = computeWhatIfRemoval("H", links);
      expect(result.orphanedIds.has("X")).toBe(false);
      expect(result.orphanedIds.has("Y")).toBe(false);
      expect(result.orphanedIds.has("Z")).toBe(false);
    });
  });
});
