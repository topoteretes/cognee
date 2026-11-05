import { computeBrainState } from "@/modules/business/computeBrainState";
import { computeNeighborIds } from "@/modules/business/canvas/businessGraphFocus";
import { normalizeStructuredNodes } from "@/modules/business/structuredGraph";
import { computeSourceDetail } from "@/modules/business/computeSourceDetail";
import type { BusinessGraphLink, BusinessGraphNode } from "@/modules/business/types";

// Shapes returned by SDK preprocess(): DltRow uses chunk, while tables and
// optional shared cell values use schema. No LLM entity is present.
const csvNodes: BusinessGraphNode[] = [
  { id: "table", name: "people", type: "SchemaTable", stage: "schema" },
  { id: "ana", name: "person: Ana, company: Atlas", type: "DltRow", stage: "chunk" },
  { id: "luka", name: "person: Luka, company: Atlas", type: "DltRow", stage: "chunk" },
];
const csvLinks: BusinessGraphLink[] = [
  { source: "ana", target: "table", relation: "is_row_of", edge_class: "semantic" },
  { source: "luka", target: "table", relation: "is_row_of", edge_class: "semantic" },
];

describe("structured graphs in the business scene", () => {
  it("displays a CSV-only graph with its row links, source and type filters", () => {
    const state = computeBrainState(csvNodes, csvLinks);

    expect(state.entities.map((node) => node.id)).toEqual(["table", "ana", "luka"]);
    expect(state.graphLinks.map((link) => [link._sid, link._tid, link.relation])).toEqual([
      ["ana", "table", "is_row_of"], ["luka", "table", "is_row_of"],
    ]);
    expect(state.sourceNames).toEqual(["Table: people"]);
    expect(state.entityType).toEqual({ table: "Table", ana: "Row", luka: "Row" });
    expect(state.typeLinks).toEqual([{ a: "Row", b: "Table", relation: "is_row_of", count: 2 }]);
    expect(computeNeighborIds(state.graphLinks, "table")).toEqual(new Set(["table", "ana", "luka"]));
    expect(state.plumbingNodes).toEqual([]);
    expect(state.semanticLinks).toEqual([]);
    expect(state.hub).toBeNull();
  });

  it.each([
    ["DltRow", "Row"], ["TableRow", "Row"], ["SchemaTable", "Table"],
    ["SchemaRelationship", "Table relationship"], ["DatabaseSchema", "Database schema"],
    ["TableType", "Table type"], ["ColumnValue", "Column value"], ["DltColumn", "Column value"],
  ])("types every documented structured node: %s", (type, bucket) => {
    const state = computeBrainState([{ id: "n", type, stage: "schema", name: "x" }], []);
    expect(state.entityType.n).toBe(bucket);
    expect(state.plumbingNodes).toEqual([]);
  });

  it("counts promoted structured nodes as entities and members of their source", () => {
    const state = computeBrainState(csvNodes.map((node) => ({ ...node, belongs_to_set: ["csv"] })), csvLinks);
    expect(state.setEntityCount).toEqual({ csv: 3 });
    expect(state.setMemberCount).toEqual({ csv: 3 });
    expect(state.plumbingEntityId).toEqual({});
  });

  it.each(["DltColumn", "ColumnValue"])("connects rows through an optional shared %s", (type) => {
    const state = computeBrainState([
      ...csvNodes,
      { id: "atlas", type, stage: "schema", name: "people:company:Atlas" },
    ], [
      ...csvLinks,
      { source: "ana", target: "atlas", relation: "company", edge_class: "semantic" },
      { source: "luka", target: "atlas", relation: "company", edge_class: "semantic" },
    ]);

    expect(state.entityType.atlas).toBe("Column value");
    expect(computeNeighborIds(state.graphLinks, "atlas")).toEqual(new Set(["atlas", "ana", "luka"]));
    expect(state.connectedIds.has("atlas")).toBe(true);
  });

  it("keeps structural and foreign-key links between structured nodes", () => {
    const state = computeBrainState([
      ...csvNodes,
      { id: "fk", type: "SchemaRelationship", stage: "schema", name: "manager" },
    ], [
      ...csvLinks,
      { source: "table", target: "fk", relation: "has_relationship", edge_class: "structural" },
      { source: "ana", target: "luka", relation: "manager", edge_class: "semantic" },
    ]);

    expect(state.graphLinks.map((link) => link.relation)).toEqual([
      "is_row_of", "is_row_of", "has_relationship", "manager",
    ]);
    expect(state.graphLinks[2].edge_class).toBe("structural");
  });

  it("preserves document and code behavior in a mixed graph without promoting ordinary chunks", () => {
    const state = computeBrainState([
      ...csvNodes,
      { id: "doc", type: "TextDocument", stage: "document" },
      { id: "chunk", type: "DocumentChunk", stage: "chunk" },
      { id: "person", type: "Entity", stage: "entity", name: "Ana" },
      { id: "code", type: "CodeSymbol", stage: "other", entity_type: "CodeSymbol" },
    ], [
      ...csvLinks,
      { source: "chunk", target: "person", relation: "contains", edge_class: "structural" },
      { source: "code", target: "person", relation: "mentions", edge_class: "semantic" },
      { source: "person", target: "ana", relation: "is_part_of", edge_class: "structural" },
    ]);

    expect(state.entities.map((node) => node.id)).toEqual(["table", "ana", "luka", "person", "code"]);
    // The structural edge between a row and an ordinary entity is drawn too,
    // so the entity is not orphaned; only semanticLinks stays business-only.
    expect(state.graphLinks.map((link) => link.relation)).toEqual(["is_row_of", "is_row_of", "mentions", "is_part_of"]);
    expect(state.semanticLinks.map((link) => link.relation)).toEqual(["mentions"]);
    expect(state.connectedIds.has("person")).toBe(true);
    expect(state.docLinks.map((link) => link.relation)).toEqual(["contains"]);
    expect(state.plumbingNodes.map((node) => node.id)).toEqual(["doc", "chunk"]);
    expect(state.entityType.person).toBe("other");
    expect(state.entityType.code).toBe("CodeSymbol");
  });

  it("keeps an isolated row visible and ignores dangling edges in a bounded graph read", () => {
    const state = computeBrainState([csvNodes[1]], csvLinks);
    expect(state.entities.map((node) => node.id)).toEqual(["ana"]);
    expect(state.graphLinks).toEqual([]);
  });

  it("keeps a genuinely empty graph empty", () => {
    const state = computeBrainState([], []);
    expect(state.entities).toEqual([]);
    expect(state.graphLinks).toEqual([]);
  });

  it("preserves cached payloads, properties and explicit sources", () => {
    const original = Object.freeze({
      id: "row", type: "TableRow", stage: "schema", name: "Ana", properties: "person Ana",
      belongs_to_set: ["crm"],
    });
    const state = computeBrainState([original], []);
    expect(original.stage).toBe("schema");
    expect(state.byId.row).toEqual({ ...original, stage: "entity", originalName: "Ana" });
    expect(state.sourceNames).toEqual(["crm"]);
    expect(state.entityType.row).toBe("Row");
  });

  it("does not promote unrelated or malformed types", () => {
    const nodes: BusinessGraphNode[] = [
      { id: "actor", type: "User", stage: "other" },
      { id: "summary", type: "TextSummary", stage: "summary" },
      { id: "missing", stage: "schema" },
      { id: "unknown", type: "toString", stage: "other" },
    ];
    expect(normalizeStructuredNodes(nodes)).toEqual(nodes);
    expect(computeBrainState(nodes, []).entities).toEqual([]);
  });

  it("excludes tabular connectivity from hubs and source bridges while retaining real relationships", () => {
    const state = computeBrainState([
      ...csvNodes.map((node) => ({ ...node, belongs_to_set: [node.id === "table" ? "schema" : "csv"] })),
      { id: "a", type: "Entity", stage: "entity", name: "Alice", belongs_to_set: ["crm"] },
      { id: "b", type: "Entity", stage: "entity", name: "Bob", belongs_to_set: ["sales"] },
      { id: "c", type: "Entity", stage: "entity", name: "Carol", belongs_to_set: ["sales"] },
    ], [
      ...csvLinks,
      { source: "a", target: "b", edge_class: "semantic", relation: "knows" },
      { source: "a", target: "c", edge_class: "semantic", relation: "knows" },
    ]);
    expect(state.graphLinks).toHaveLength(4);
    expect(state.semanticLinks).toHaveLength(2);
    expect(state.hub).toEqual({ entityId: "a", name: "Alice", degree: 2, sourceCount: 2 });
    expect(computeSourceDetail("csv", state, state.encodings.source).bridgeCount).toBe(0);
    expect(computeSourceDetail("crm", state, state.encodings.source).bridgeCount).toBe(1);
  });
});
