import { parseRowFields, structuredRowFields } from "@/modules/business/structuredRowFields";

it("trims the column of a nested-indentation cell", () => {
  expect(parseRowFields("Row Data:\n  a: 1\n    b: 2").map((field) => field.column)).toEqual(["a", "b"]);
});

it("treats an indented line with a colon as a new column, not a continuation", () => {
  expect(parseRowFields("Row Data:\n  note: first\n  at: 10:30").map((field) => field.column)).toEqual(["note", "at"]);
});

describe("structured row fields", () => {
  it("treats text before the first field as unparseable rather than inventing columns", () => {
    expect(parseRowFields("Row Data:\nnot a field\n  a: 1")).toEqual([]);
  });

  it("ignores JSON that is not an object for a TableRow", () => {
    const node = { id: "r", type: "TableRow" };
    expect(["null", "[1]", "3"].map((content) => structuredRowFields(node, content))).toEqual([[], [], []]);
  });

  it("reads values only after Row Data and keeps their order, punctuation and empty cells", () => {
    expect(parseRowFields("Table: people\r\nColumns:\r\n  - person (TEXT)\r\n\r\nRow Data:\r\n  person: Ana Horvat\r\n  note: hello, world: again\r\n  manager: \r\n  count: 0")).toEqual([
      { column: "person", value: "Ana Horvat" },
      { column: "note", value: "hello, world: again" },
      { column: "manager", value: "" },
      { column: "count", value: "0" },
    ]);
  });

  it("preserves multiline values and duplicate column names without losing data in a map", () => {
    expect(parseRowFields("Row Data:\n  note: First line\nSecond line\n\nThird line\n  note: Another value")).toEqual([
      { column: "note", value: "First line\nSecond line\n\nThird line" },
      { column: "note", value: "Another value" },
    ]);
  });

  it("falls back to original content when there is no row section or its format is unsupported", () => {
    expect(parseRowFields("Table: people\nColumns:\n  name: TEXT")).toEqual([]);
    expect(parseRowFields("Row Data:\nunknown format")).toEqual([]);
  });

  it("handles a relational JSON object without interpreting Python expressions", () => {
    const row = { id: "r", type: "TableRow" };
    expect(structuredRowFields(row, '{"name":"Ana","count":0,"active":false,"manager":null}')).toEqual([
      { column: "name", value: "Ana" }, { column: "count", value: "0" },
      { column: "active", value: "false" }, { column: "manager", value: "null" },
    ]);
    expect(structuredRowFields(row, "{'name': 'Ana'}")).toEqual([]);
  });
});
