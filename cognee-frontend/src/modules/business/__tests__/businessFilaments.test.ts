import type { BusinessEntity } from "../sceneTypes";
import { sourcesWithEntities } from "../canvas/businessFilaments";
import { UNCATEGORIZED_SOURCE } from "../computeBrainState";

function entity(id: string, overrides: Partial<BusinessEntity> = {}): BusinessEntity {
  return { id, ...overrides };
}

describe("sourcesWithEntities", () => {
  it("collects every set an entity belongs to", () => {
    const names = sourcesWithEntities([
      entity("a", { belongs_to_set: ["slack", "docs"] }),
      entity("b", { belongs_to_set: ["docs"] }),
    ]);
    expect(names).toEqual(new Set(["slack", "docs"]));
  });

  it("omits a source registered as a NodeSet but referenced by no entity", () => {
    // The CLO-597 filament bug: deriveSourceNames unions in dataset sources
    // with zero extracted entities (e.g. a "skills" NodeSet), and a filament
    // was drawn to their fallback anchor. Such a source never appears here,
    // so the caller's filter drops its filament.
    const names = sourcesWithEntities([entity("a", { belongs_to_set: ["docs"] })]);
    expect(names.has("skills")).toBe(false);
  });

  it("falls back to source_node_set, then uncategorized, like setsOf", () => {
    const names = sourcesWithEntities([
      entity("a", { source_node_set: "slack, docs" }),
      entity("b"),
    ]);
    expect(names).toEqual(new Set(["slack", "docs", UNCATEGORIZED_SOURCE]));
  });

  it("returns an empty set for no entities", () => {
    expect(sourcesWithEntities([]).size).toBe(0);
  });
});
