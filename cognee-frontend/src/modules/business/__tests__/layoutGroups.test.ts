import { CODE_NODE_TYPES } from "../codeGraph";
import { CODE_GROUP_ORDER, resolveLayoutGroups } from "../layoutGroups";
import type { BusinessEntity } from "../sceneTypes";

function entity(id: string): BusinessEntity {
  return { id };
}

describe("resolveLayoutGroups", () => {
  it("only gives sectors to types that draw as code", () => {
    CODE_GROUP_ORDER.forEach((type) => expect(CODE_NODE_TYPES.has(type)).toBe(true));
  });

  it("groups code entities by their type", () => {
    const { groupOf } = resolveLayoutGroups([entity("a"), entity("b"), entity("c")], {
      a: "ExternalDependency", b: "CodeModule", c: "CodeSymbol",
    });
    expect(groupOf).toEqual({ a: "ExternalDependency", b: "CodeModule", c: "CodeSymbol" });
  });

  it("always returns the full fixed order, whichever types are present", () => {
    const first = resolveLayoutGroups([entity("a")], { a: "CodeModule" });
    const second = resolveLayoutGroups([entity("a"), entity("b")], { a: "CodeModule", b: "ApiEndpoint" });
    expect(first.groupOrder).toEqual(CODE_GROUP_ORDER);
    expect(second.groupOrder).toEqual(CODE_GROUP_ORDER);
  });

  it("leaves an entity with any other type out of every group", () => {
    const { groupOf } = resolveLayoutGroups([entity("a"), entity("b")], { a: "country", b: "other" });
    expect(groupOf).toEqual({ a: null, b: null });
  });

  it("leaves a CodeRepository node at its territory's center", () => {
    const { groupOf } = resolveLayoutGroups([entity("r"), entity("m")], { r: "CodeRepository", m: "CodeModule" });
    expect(groupOf).toEqual({ r: null, m: "CodeModule" });
  });
});
