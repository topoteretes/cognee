import type { BusinessEntity, SemanticLink } from "../sceneTypes";
import { buildReasoningTrace } from "../reasoningTrace";

// ── helpers ──────────────────────────────────────────────────────────────────

function link(sid: string, tid: string, relation?: string): SemanticLink {
  return { source: sid, target: tid, _sid: sid, _tid: tid, relation };
}

function entity(id: string, overrides: Partial<BusinessEntity> = {}): BusinessEntity {
  return { id, ...overrides };
}

function entityById(...entities: BusinessEntity[]): Record<string, BusinessEntity> {
  return Object.fromEntries(entities.map((e) => [e.id, e]));
}

// ── tests ─────────────────────────────────────────────────────────────────────

describe("buildReasoningTrace", () => {
  describe("guard clauses", () => {
    it("returns an empty trace for an empty id set", () => {
      const steps = buildReasoningTrace(new Set(), [], {});
      expect(steps).toEqual([]);
    });
  });

  describe("single id", () => {
    it("returns one step narrating the sole id by name", () => {
      const steps = buildReasoningTrace(new Set(["A"]), [], entityById(entity("A", { name: "Acme Robotics" })));
      expect(steps).toEqual([{ id: "A", narration: "following the trail: Acme Robotics" }]);
    });

    it("falls back to type when the entity has no name", () => {
      const steps = buildReasoningTrace(new Set(["A"]), [], entityById(entity("A", { type: "Organization" })));
      expect(steps[0].narration).toBe("following the trail: Organization");
    });

    it("falls back to a generic label when the entity has neither name nor type", () => {
      const steps = buildReasoningTrace(new Set(["A"]), [], {});
      expect(steps[0].narration).toBe("following the trail: an unnamed record");
    });
  });

  describe("root selection", () => {
    it("starts the walk at the most-connected id within the induced subgraph", () => {
      // B is the hub (2 edges within the id set); A and C each have 1.
      const links = [link("A", "B"), link("B", "C")];
      const ids = entityById(entity("A", { name: "A" }), entity("B", { name: "B" }), entity("C", { name: "C" }));
      const steps = buildReasoningTrace(new Set(["A", "B", "C"]), links, ids);
      expect(steps[0].id).toBe("B");
    });
  });

  describe("graph-walk order", () => {
    it("hops to graph neighbors in BFS order, not the id set's iteration order", () => {
      // Chain A-B-C; iterating the Set in reverse (C, B, A) must still walk A -> B -> C
      // because A is the only degree-1 endpoint tied for root with C, and ties
      // resolve to whichever the reduce sees first — assert on reachability/order instead.
      const links = [link("A", "B"), link("B", "C")];
      const steps = buildReasoningTrace(new Set(["C", "B", "A"]), links, {});
      const order = steps.map((s) => s.id);
      // B must come immediately before or after each of its neighbors in a valid walk;
      // since B is the unique hub (degree 2) it must be visited before both leaves.
      expect(order.indexOf("B")).toBeLessThan(order.indexOf("A"));
      expect(order.indexOf("B")).toBeLessThan(order.indexOf("C"));
    });

    it("includes a relation label on hops that cross a labeled edge", () => {
      const links = [link("A", "B", "works_at")];
      const steps = buildReasoningTrace(
        new Set(["A", "B"]),
        links,
        entityById(entity("A", { name: "Elena Cruz" }), entity("B", { name: "Meridian Dynamics" })),
      );
      expect(steps[1].narration).toBe("→ works_at → Meridian Dynamics");
    });

    it("omits the relation arrow label when the connecting edge has none", () => {
      const links = [link("A", "B")];
      const steps = buildReasoningTrace(new Set(["A", "B"]), links, entityById(entity("B", { name: "B" })));
      expect(steps[1].narration).toBe("→ B");
    });
  });

  describe("disconnected ids", () => {
    it("still appends an id with no edge into the walked component, as a trailing step", () => {
      const links = [link("A", "B")];
      const steps = buildReasoningTrace(new Set(["A", "B", "ISOLATED"]), links, {});
      expect(steps.map((s) => s.id)).toContain("ISOLATED");
    });

    it("ignores edges where the other endpoint is outside the id set", () => {
      const links = [link("A", "B"), link("B", "OUTSIDE")];
      const steps = buildReasoningTrace(new Set(["A", "B"]), links, {});
      expect(steps.map((s) => s.id)).toEqual(expect.arrayContaining(["A", "B"]));
      expect(steps.map((s) => s.id)).not.toContain("OUTSIDE");
    });
  });

  describe("step cap", () => {
    it("caps the walk at 8 steps even when more ids contributed to the answer", () => {
      const ids = Array.from({ length: 12 }, (_, i) => `n${i}`);
      const links = ids.slice(0, -1).map((id, i) => link(id, ids[i + 1]));
      const steps = buildReasoningTrace(new Set(ids), links, {});
      expect(steps.length).toBe(8);
    });
  });
});
