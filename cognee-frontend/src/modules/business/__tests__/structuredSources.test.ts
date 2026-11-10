import { computeBrainState } from "@/modules/business/computeBrainState";
import { groupStructuredSources } from "@/modules/business/structuredSources";
import type { BusinessGraphLink, BusinessGraphNode } from "@/modules/business/types";

const nodes: BusinessGraphNode[] = [
  { id: "people", type: "SchemaTable", name: "people" },
  { id: "projects", type: "SchemaTable", name: "projects" },
  { id: "ana", type: "DltRow", document_name: "people.csv", document_id: "doc", name: "Ana" },
  { id: "task", type: "DltRow", document_name: "projects.csv", name: "Graph UI" },
  { id: "atlas", type: "DltColumn", name: "people:company:Atlas" },
  { id: "doc", type: "dlt_source", name: "people" },
];
const links: BusinessGraphLink[] = [
  { source: "ana", target: "people", relation: "is_row_of", edge_class: "semantic" },
  { source: "task", target: "projects", relation: "is_row_of", edge_class: "semantic" },
  { source: "ana", target: "atlas", relation: "company", edge_class: "semantic" },
  { source: "task", target: "ana", relation: "owner", edge_class: "semantic" },
];

describe("structured source groups", () => {
  it("accepts a belongs_to_set sent as a single string", () => {
    const grouped = groupStructuredSources([
      { id: "r", type: "DltRow", belongs_to_set: "crm" as unknown as string[] },
      { id: "t", type: "SchemaTable", name: "people" },
    ], [{ source: "r", target: "t", relation: "is_row_of", edge_class: "semantic" }]);
    expect(grouped.find((node) => node.id === "t")?.belongs_to_set).toEqual(["crm"]);
  });

  it("keeps a string belongs_to_set as the node's source through the whole scene", () => {
    const state = computeBrainState([
      { id: "r", type: "DltRow", belongs_to_set: "crm" as unknown as string[] },
      { id: "t", type: "SchemaTable", name: "people" },
    ], [{ source: "r", target: "t", relation: "is_row_of", edge_class: "semantic" }]);
    expect(state.sourceNames).toEqual(["crm"]);
    expect(state.byId.r.belongs_to_set).toEqual(["crm"]);
  });

  it("drops non-string entries from a belongs_to_set array so they never reach the source names", () => {
    const state = computeBrainState([
      { id: "r", type: "DltRow", belongs_to_set: ["crm", 5, ""] as unknown as string[] },
    ], []);
    expect(state.sourceNames).toEqual(["crm"]);
  });

  it("reads sources from source_node_set when belongs_to_set is absent", () => {
    const grouped = groupStructuredSources([
      { id: "r", type: "DltRow", source_node_set: "crm, sales" },
      { id: "t", type: "SchemaTable", name: "people" },
    ], [{ source: "r", target: "t", relation: "is_row_of", edge_class: "semantic" }]);
    expect(grouped.find((node) => node.id === "t")?.belongs_to_set).toEqual(["crm", "sales"]);
  });

  it("lets only a row hand its sources to a cell value", () => {
    const grouped = groupStructuredSources([
      { id: "t", type: "SchemaTable", name: "people", belongs_to_set: ["schema"] },
      { id: "cell", type: "ColumnValue", name: "people:company:Atlas" },
    ], [{ source: "t", target: "cell", relation: "company", edge_class: "semantic" }]);
    expect(grouped.find((node) => node.id === "cell")?.belongs_to_set).toBeUndefined();
  });

  it("resolves link endpoints that are node objects", () => {
    const grouped = groupStructuredSources(nodes, [
      { source: { id: "ana" }, target: { id: "people" }, relation: "is_row_of", edge_class: "semantic" },
    ] as unknown as BusinessGraphLink[]);
    expect(grouped.find((node) => node.id === "people")?.belongs_to_set).toEqual(["File: people.csv"]);
  });

  it("does not name a source after a schema header fragment", () => {
    const grouped = groupStructuredSources([
      { id: "r", type: "DltRow", name: "Table: people Columns:\n  - person (TEXT)" },
    ], []);
    expect(grouped[0].belongs_to_set).toBeUndefined();
  });

  it("gives each file its own source filter and layout territory without joining files across foreign keys", () => {
    const state = computeBrainState(nodes, links);
    expect(state.sourceNames).toEqual(["File: people.csv", "File: projects.csv"]);
    expect(state.setEntityCount).toEqual({ "File: people.csv": 3, "File: projects.csv": 2 });
    expect(state.entityAnchors.ana).not.toEqual(state.entityAnchors.task);
    expect(state.entityAnchors.ana).toEqual(state.entityAnchors.atlas);
    expect(state.byId.doc.belongs_to_set).toEqual(["File: people.csv"]);
    expect(state.graphLinks).toHaveLength(4);
    expect(state.semanticLinks).toEqual([]);
    expect(nodes[2].belongs_to_set).toBeUndefined();
  });

  it("uses table names for manifest-only and bounded payloads without fabricating a filename", () => {
    const grouped = groupStructuredSources([
      { id: "row", type: "DltRow", document_name: "dlt_manifest_123.txt", text: "Table: people\n\nRow Data:\n  person: Ana" },
    ], []);
    expect(grouped[0].belongs_to_set).toEqual(["Table: people"]);
    expect(groupStructuredSources([{ id: "row", type: "DltRow", name: "Table: people Columns: - name" }], [])[0].belongs_to_set).toBeUndefined();
  });

  it("does not overwrite explicit source tags or spread groups through unrelated entities", () => {
    const tagged = { ...nodes[2], belongs_to_set: ["crm"] };
    const external = { id: "external", type: "Entity", stage: "entity", name: "Atlas" };
    const grouped = groupStructuredSources([tagged, external], [{ source: "ana", target: "external", relation: "works_at" }]);
    expect(grouped[0]).toBe(tagged);
    expect(grouped[1]).toBe(external);
  });

  it("groups relational rows and schema relationships by their owning table", () => {
    const grouped = groupStructuredSources([
      { id: "table", type: "TableType", name: "people" },
      { id: "row", type: "TableRow", name: "people:1" },
      { id: "fk", type: "SchemaRelationship", name: "manager" },
    ], [
      { source: "row", target: "table", relation: "is_part_of" },
      { source: "table", target: "fk", relation: "has_relationship" },
    ]);
    expect(grouped.map((node) => node.belongs_to_set)).toEqual([
      ["Table: people"], ["Table: people"], ["Table: people"],
    ]);
  });

  it("retains both owners of a shared value without merging the rows' groups", () => {
    const grouped = groupStructuredSources(nodes, [...links, { source: "task", target: "atlas", relation: "company" }]);
    expect(grouped.find((node) => node.id === "atlas")?.belongs_to_set).toEqual(["File: people.csv", "File: projects.csv"]);
    expect(grouped.find((node) => node.id === "task")?.belongs_to_set).toEqual(["File: projects.csv"]);
  });

  it("does not mistake a CSV filename for an agent session source", () => {
    const state = computeBrainState([{ id: "r", type: "DltRow", document_name: "session_learnings.csv" }], []);
    expect(state.sourceNames).toEqual(["File: session_learnings.csv"]);
    expect(state.sessionSetNames).toEqual([]);
  });
});
