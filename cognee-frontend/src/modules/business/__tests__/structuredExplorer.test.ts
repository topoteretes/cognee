import { buildStructuredExplorer } from "@/modules/business/structuredExplorer";
import { computeBrainState } from "@/modules/business/computeBrainState";
import type { BusinessGraphNode } from "@/modules/business/types";

function row(id: string, file: string, value = "Atlas Labs"): BusinessGraphNode {
  return { id, type: "DltRow", document_name: file, text: `Table: people\nRow Data:\n  company: ${value}\n  empty: ` };
}

describe("buildStructuredExplorer", () => {
  it("separates equal values across files from recorded links without fabricating edges", () => {
    const brain = computeBrainState([row("a", "people.csv"), row("b", "accounts.csv")], []);
    const index = buildStructuredExplorer(brain);
    expect(index.recordedRowLinks).toBe(0);
    expect(index.sharedValueNodes).toBe(0);
    expect(index.sourceCount).toBe(2);
    expect(index.possibleMatches).toEqual([{
      column: "company", value: "Atlas Labs", rowIds: new Set(["a", "b"]),
      sources: new Set(["File: people.csv", "File: accounts.csv"]),
    }]);
    expect(brain.graphLinks).toEqual([]);
    expect(index.columns.get("empty")?.get("")).toEqual(new Set(["a", "b"]));
  });

  it("counts actual row links separately from shared cell paths and table membership", () => {
    const brain = computeBrainState([
      row("a", "people.csv"), row("b", "accounts.csv"),
      { id: "value", type: "ColumnValue", name: "Atlas Labs" },
      { id: "table", type: "SchemaTable", name: "people" },
    ], [
      { source: "a", target: "b", relation: "works_for" },
      { source: "a", target: "value", relation: "has_company" },
      { source: "value", target: "b", relation: "has_company" },
      { source: "a", target: "table", relation: "is_row_of" },
      { source: "b", target: "table", relation: "is_row_of" },
    ]);
    const index = buildStructuredExplorer(brain);
    expect(index.recordedRowLinks).toBe(1);
    expect(index.sharedValueNodes).toBe(1);
  });

  it("does not treat same-file duplicates, different columns or case changes as cross-source exact matches", () => {
    const index = buildStructuredExplorer(computeBrainState([
      row("a", "people.csv"), row("b", "people.csv"), row("c", "other.csv", "atlas labs"),
      { id: "d", type: "DltRow", document_name: "other.csv", text: "Row Data:\n  customer: Atlas Labs" },
    ], []));
    expect(index.possibleMatches).toEqual([]);
    expect(index.columns.get("company")?.get("Atlas Labs")).toEqual(new Set(["a", "b"]));
  });

  it("reports unavailable values and does not manufacture sources for unknown rows", () => {
    const index = buildStructuredExplorer(computeBrainState([
      { id: "a", type: "DltRow", name: "Truncated header" },
      { id: "b", type: "TableRow", properties: "{'company': 'Atlas Labs'}" },
    ], []));
    expect(index.unavailableRows).toBe(2);
    expect(index.columns.size).toBe(0);
    expect(index.sourceCount).toBe(0);
    expect(buildStructuredExplorer(null).rowIds.size).toBe(0);
  });

  it("indexes relational JSON rows and prototype-like column names safely", () => {
    const index = buildStructuredExplorer(computeBrainState([
      { id: "a", type: "TableRow", properties: '{"__proto__":"value","amount":0,"enabled":false}' },
    ], []));
    expect(index.columns.get("__proto__")?.get("value")).toEqual(new Set(["a"]));
    expect(index.columns.get("amount")?.get("0")).toEqual(new Set(["a"]));
    expect(index.columns.get("enabled")?.get("false")).toEqual(new Set(["a"]));
  });
});

it("scans rows sharing one value in linear time", () => {
  const rows: BusinessGraphNode[] = Array.from({ length: 20000 }, (_, i) => ({
    id: `r${i}`, type: "DltRow", document_name: "people.csv", text: "Row Data:\n  country: Bosnia",
  }));
  const brain = computeBrainState(rows, []);
  const started = performance.now();
  const explorer = buildStructuredExplorer(brain);
  // The all-pairs scan this replaced took over two seconds here.
  expect(performance.now() - started).toBeLessThan(1000);
  expect(explorer.possibleMatches).toEqual([]);
});
