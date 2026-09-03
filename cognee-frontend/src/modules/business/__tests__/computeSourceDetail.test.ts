import type { BrainState, SemanticLink, TypeNode } from "../sceneTypes";
import type { BusinessGraphNode } from "../types";
import { computeSourceDetail } from "../computeSourceDetail";

// ── helpers ──────────────────────────────────────────────────────────────────

function link(sid: string, tid: string, bridge = false): SemanticLink {
  return { source: sid, target: tid, _sid: sid, _tid: tid, _bridge: bridge };
}

function typeNode(name: string, sets: Record<string, number>): TypeNode {
  return { name, members: [], sets };
}

function node(id: string, overrides: Partial<BusinessGraphNode> = {}): BusinessGraphNode {
  return { id, ...overrides };
}

function baseBrainState(overrides: Partial<BrainState> = {}): BrainState {
  return {
    byId: {},
    entities: [],
    entityById: {},
    sourceNames: [],
    setColor: {},
    isSessionSet: () => false,
    setEntityCount: {},
    setDocCount: {},
    setMemberCount: {},
    semanticLinks: [],
    docLinks: [],
    anchors: {},
    typeNodes: [],
    typeLinks: [],
    importanceMax: 1,
    plumbingNodes: [],
    plumbingEntityId: {},
    importanceCut: 0,
    connectedIds: new Set(),
    hub: null,
    ...overrides,
  };
}

// ── tests ─────────────────────────────────────────────────────────────────────

describe("computeSourceDetail", () => {
  it("carries through the counts, color, and display name the Sources rail already shows", () => {
    const brainState = baseBrainState({
      setColor: { crm: "#43D9E8" },
      setEntityCount: { crm: 12 },
      setDocCount: { crm: 3 },
    });
    const detail = computeSourceDetail("crm", brainState);
    expect(detail).toMatchObject({ name: "crm", displayName: "crm", color: "#43D9E8", entityCount: 12, docCount: 3 });
  });

  it("falls back to the neutral gray when a source has no assigned color", () => {
    const detail = computeSourceDetail("crm", baseBrainState());
    expect(detail.color).toBe("#7E8CA6");
  });

  describe("type breakdown", () => {
    it("includes only types with a nonzero count for this source, sorted descending", () => {
      const brainState = baseBrainState({
        typeNodes: [typeNode("Person", { crm: 3 }), typeNode("Company", { crm: 7 }), typeNode("Deal", { other: 5 })],
      });
      const detail = computeSourceDetail("crm", brainState);
      expect(detail.typeBreakdown).toEqual([
        { type: "Company", count: 7 },
        { type: "Person", count: 3 },
      ]);
    });

    it("caps the breakdown at 5 rows", () => {
      const typeNodes = Array.from({ length: 8 }, (_, i) => typeNode(`Type${i}`, { crm: i + 1 }));
      const detail = computeSourceDetail("crm", baseBrainState({ typeNodes }));
      expect(detail.typeBreakdown.length).toBe(5);
    });
  });

  describe("document names", () => {
    it("lists document/record names from the plumbing layer that belong to this source", () => {
      const plumbingNodes = [
        node("d1", { name: "invoice_2024.pdf", belongs_to_set: ["crm"] }),
        node("d2", { name: "unrelated.pdf", belongs_to_set: ["other"] }),
      ];
      const detail = computeSourceDetail("crm", baseBrainState({ plumbingNodes }));
      expect(detail.documentNames).toEqual(["invoice_2024.pdf"]);
      expect(detail.documentTotal).toBe(1);
    });

    it("caps the shown names at 6 but reports the true total", () => {
      const plumbingNodes = Array.from({ length: 10 }, (_, i) =>
        node(`d${i}`, { name: `doc${i}.pdf`, belongs_to_set: ["crm"] }),
      );
      const detail = computeSourceDetail("crm", baseBrainState({ plumbingNodes }));
      expect(detail.documentNames.length).toBe(6);
      expect(detail.documentTotal).toBe(10);
    });

    it("excludes plumbing nodes with no name even if they belong to the source", () => {
      const plumbingNodes = [node("d1", { belongs_to_set: ["crm"] })];
      const detail = computeSourceDetail("crm", baseBrainState({ plumbingNodes }));
      expect(detail.documentNames).toEqual([]);
    });
  });

  describe("bridge count", () => {
    it("counts distinct other sources this source's entities bridge into", () => {
      const byId = {
        a: node("a", { belongs_to_set: ["crm"] }),
        b: node("b", { belongs_to_set: ["marketing"] }),
        c: node("c", { belongs_to_set: ["support"] }),
      };
      const semanticLinks = [link("a", "b", true), link("a", "c", true)];
      const detail = computeSourceDetail("crm", baseBrainState({ byId, semanticLinks }));
      expect(detail.bridgeCount).toBe(2);
    });

    it("ignores non-bridge links entirely", () => {
      const byId = {
        a: node("a", { belongs_to_set: ["crm"] }),
        b: node("b", { belongs_to_set: ["marketing"] }),
      };
      const semanticLinks = [link("a", "b", false)];
      const detail = computeSourceDetail("crm", baseBrainState({ byId, semanticLinks }));
      expect(detail.bridgeCount).toBe(0);
    });

    it("ignores bridge links that don't touch this source at all", () => {
      const byId = {
        a: node("a", { belongs_to_set: ["marketing"] }),
        b: node("b", { belongs_to_set: ["support"] }),
      };
      const semanticLinks = [link("a", "b", true)];
      const detail = computeSourceDetail("crm", baseBrainState({ byId, semanticLinks }));
      expect(detail.bridgeCount).toBe(0);
    });

    it("does not count this source itself even if a link's other side also belongs to it", () => {
      const byId = {
        a: node("a", { belongs_to_set: ["crm"] }),
        b: node("b", { belongs_to_set: ["crm", "marketing"] }),
      };
      const semanticLinks = [link("a", "b", true)];
      const detail = computeSourceDetail("crm", baseBrainState({ byId, semanticLinks }));
      expect(detail.bridgeCount).toBe(1);
    });
  });
});
