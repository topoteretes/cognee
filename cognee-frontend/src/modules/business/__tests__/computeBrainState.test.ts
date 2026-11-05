import type { BusinessGraphNode, BusinessGraphLink } from "../types";
import { computeBrainState } from "../computeBrainState";
import { CODE_SOURCE } from "../codeGraph";
import { ENTITY_FALLBACK_COLOR, SOURCE_FALLBACK_COLOR, TYPE_PALETTE } from "../colorEncoding";

// cognee 1.6.1 stamps entity_type on every node, code or not — falling back
// to the node's own class ("Entity") when no is_a edge resolved a real one —
// so a fixture omitting it doesn't match a real payload and can't catch a
// regression that trusts entity_type for a non-code node.
function node(id: string, overrides: Partial<BusinessGraphNode> = {}): BusinessGraphNode {
  return { id, stage: "other", entity_type: overrides.type ?? "Entity", ...overrides };
}

function link(source: string, target: string, overrides: Partial<BusinessGraphLink> = {}): BusinessGraphLink {
  return { source, target, ...overrides };
}

describe("computeBrainState with code-graph nodes", () => {
  it("promotes a code node into the entity layer", () => {
    const state = computeBrainState([node("a", { type: "CodeSymbol" })], []);
    expect(state.entities.map((e) => e.id)).toEqual(["a"]);
  });

  it("does not promote a non-code stage-other node into the entity layer", () => {
    const state = computeBrainState([node("a", { type: "TextSummary" })], []);
    expect(state.entities).toEqual([]);
  });

  it("types a code entity by entity_type when no is_a edge resolved one", () => {
    const state = computeBrainState(
      [node("a", { type: "CodeSymbol", entity_type: "CodeSymbol" })],
      [],
    );
    expect(state.entityType.a).toBe("CodeSymbol");
  });

  it("prefers an is_a edge's type over entity_type when both exist", () => {
    const nodes = [
      node("a", { type: "CodeSymbol", entity_type: "CodeSymbol", stage: "entity" }),
      node("t", { stage: "type", name: "Function" }),
    ];
    const links = [link("a", "t", { relation: "is_a" })];
    const state = computeBrainState(nodes, links);
    expect(state.entityType.a).toBe("Function");
  });

  it("buckets a plain entity with no is_a edge as other, unaffected by its own entity_type", () => {
    // entity_type here defaults to "Entity" (node()'s fallback, matching a
    // real unresolved node) — this must NOT become a Types-rail bucket.
    const state = computeBrainState([node("a", { stage: "entity" })], []);
    expect(state.entityType.a).toBe("other");
  });

  it("regression: does not surface a resolved semantic entity_type for a non-code entity", () => {
    // cognee 1.6.1 also stamps a REAL semantic type ("Person") on a resolved
    // non-code entity when its is_a edge exists — but that edge may not be
    // in a bounded read, and the fallback here is deliberately code-only, so
    // this still buckets as "other" rather than surfacing "Person" from
    // entity_type alone.
    const state = computeBrainState([node("a", { stage: "entity", entity_type: "Person" })], []);
    expect(state.entityType.a).toBe("other");
  });

  it("builds a type link between two code types connected by a semantic edge", () => {
    // Before the shared typeOf() resolver, typeLinks used typeName (is_a
    // edges only) while typeNodes used entity_type — every code pair
    // collapsed to other→other and was dropped by the a===b guard.
    const nodes = [node("a", { type: "CodeSymbol" }), node("b", { type: "CodeModule" })];
    const links = [link("a", "b", { relation: "calls", edge_class: "semantic" })];
    const state = computeBrainState(nodes, links);
    expect(state.typeNodes.map((t) => t.name).sort()).toEqual(["CodeModule", "CodeSymbol"]);
    expect(state.typeLinks).toEqual([{ a: "CodeSymbol", b: "CodeModule", relation: "calls", count: 1 }]);
  });

  it("gives a sourceless code node the CODE_SOURCE tag", () => {
    const state = computeBrainState([node("a", { type: "CodeModule" })], []);
    expect(state.encodings.source.sourceColor(CODE_SOURCE)).not.toBe(SOURCE_FALLBACK_COLOR);
    expect(state.sourceNames).toContain(CODE_SOURCE);
  });

  it("paints a sourceless code node with the CODE_SOURCE color, not the fallback", () => {
    const state = computeBrainState([node("a", { type: "CodeModule" })], []);
    expect(state.encodings.source.entityFill([CODE_SOURCE])).toBe(state.encodings.source.sourceColor(CODE_SOURCE));
    expect(state.encodings.source.entityFill([CODE_SOURCE])).not.toBe(ENTITY_FALLBACK_COLOR);
  });

  it("keeps a code node's real belongs_to_set instead of CODE_SOURCE", () => {
    const state = computeBrainState(
      [node("a", { type: "CodeModule", belongs_to_set: ["github"] })],
      [],
    );
    expect(state.sourceNames).toEqual(["github"]);
    expect(state.sourceNames).not.toContain(CODE_SOURCE);
  });

  it("places a semantic code-to-code link in semanticLinks", () => {
    const nodes = [node("a", { type: "CodeSymbol" }), node("b", { type: "CodeSymbol" })];
    const links = [link("a", "b", { relation: "calls", edge_class: "semantic" })];
    const state = computeBrainState(nodes, links);
    expect(state.semanticLinks).toHaveLength(1);
    expect(state.semanticLinks[0].relation).toBe("calls");
  });

  it("excludes a semantic link touching a non-promoted stage-other node", () => {
    const nodes = [node("a", { type: "CodeSymbol" }), node("b", { type: "TextSummary" })];
    const links = [link("a", "b", { relation: "calls", edge_class: "semantic" })];
    const state = computeBrainState(nodes, links);
    expect(state.semanticLinks).toHaveLength(0);
  });
});

describe("computeBrainState color encodings", () => {
  function typedEntity(id: string, typeName: string): BusinessGraphNode[] {
    return [node(id, { stage: "entity" }), node(`type-${typeName}`, { stage: "type", name: typeName })];
  }

  it("colors the biggest types of the brain in the type encoding, in rank order", () => {
    const nodes = [...typedEntity("a", "person"), ...typedEntity("b", "person"), ...typedEntity("c", "country")];
    const links = [
      link("a", "type-person", { relation: "is_a" }),
      link("b", "type-person", { relation: "is_a" }),
      link("c", "type-country", { relation: "is_a" }),
    ];
    const { type } = computeBrainState(nodes, links).encodings;
    expect(type.entityFill([], "person")).toBe(TYPE_PALETTE[0]);
    expect(type.entityFill([], "country")).toBe(TYPE_PALETTE[1]);
  });

  it("gives a session set an ordinary source color, never amber", () => {
    const nodes = [
      node("ns", { type: "NodeSet", name: "session_learnings", stage: "other" }),
      node("e", { stage: "entity", belongs_to_set: ["session_learnings"] }),
    ];
    const { source } = computeBrainState(nodes, []).encodings;
    expect(source.sourceColor("session_learnings")).not.toBe("#F5A83C");
    expect(source.sourceColor("session_learnings")).not.toBe(SOURCE_FALLBACK_COLOR);
  });

  it("colors a session set that no entity is tagged with yet", () => {
    const nodes = [node("ns", { type: "NodeSet", name: "agent_trace_feedbacks", stage: "other" })];
    const state = computeBrainState(nodes, []);
    expect(state.sessionSetNames).toEqual(["agent_trace_feedbacks"]);
    expect(state.encodings.source.sourceColor("agent_trace_feedbacks")).not.toBe(SOURCE_FALLBACK_COLOR);
  });
});
