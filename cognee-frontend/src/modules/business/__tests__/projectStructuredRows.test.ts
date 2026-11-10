import { computeBrainState } from "@/modules/business/computeBrainState";
import { projectStructuredRows } from "@/modules/business/projectStructuredRows";

function graph() {
  return computeBrainState([
    { id: "a", type: "DltRow", name: "Ana" },
    { id: "b", type: "DltRow", name: "Ben" },
    { id: "c", type: "DltRow", name: "Cleo" },
    { id: "table", type: "SchemaTable", name: "people" },
    { id: "cell", type: "ColumnValue", name: "Atlas Labs" },
  ], [
    { source: "a", target: "table", relation: "is_row_of" },
    { source: "b", target: "table", relation: "is_row_of" },
    { source: "a", target: "cell", relation: "has_company" },
    { source: "c", target: "cell", relation: "has_company" },
    { source: "a", target: "b", relation: "knows" },
  ]);
}

describe("projectStructuredRows", () => {
  it("keeps matching rows and context without pulling in other rows through a shared table or cell", () => {
    const brain = graph();
    const projected = projectStructuredRows(brain, new Set(["a"]));
    expect(projected.entities.map((node) => node.id)).toEqual(["a", "table", "cell"]);
    expect(projected.graphLinks.map((link) => [link._sid, link._tid])).toEqual([["a", "table"], ["a", "cell"]]);
    expect(projected.entities[0]).toBe(brain.entityById.a);
    expect(brain.entities).toHaveLength(5);
    expect(brain.graphLinks).toHaveLength(5);
    expect(projected.typeNodes).toEqual([]);
    expect(projected.importanceCut).toBe(0);
  });

  it("retains real relationships between matching rows and restores the full scene on clear", () => {
    const brain = graph();
    expect(projectStructuredRows(brain, new Set(["a", "b"])).graphLinks.some((link) => link.relation === "knows")).toBe(true);
    expect(projectStructuredRows(brain, null)).toBe(brain);
  });

  it("shows an empty result for no matches, including type and plumbing layers", () => {
    const result = projectStructuredRows(graph(), new Set(["missing"]));
    expect(result.entities).toEqual([]);
    expect(result.graphLinks).toEqual([]);
    expect(result.typeNodes).toEqual([]);
    expect(result.plumbingNodes).toEqual([]);
  });
});
