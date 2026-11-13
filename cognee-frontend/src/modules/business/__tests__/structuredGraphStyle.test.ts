import { isTabularStructureLink, structuredNodeKind, traceStructuredShape } from "@/modules/business/structuredGraphStyle";

function fakeContext(): { ctx: CanvasRenderingContext2D; calls: unknown[][] } {
  const calls: unknown[][] = [];
  const record = (name: string) => (...args: number[]): void => { calls.push([name, ...args]); };
  const path: Pick<CanvasRenderingContext2D, "rect" | "moveTo" | "lineTo" | "closePath"> = {
    rect: record("rect"), moveTo: record("moveTo"), lineTo: record("lineTo"), closePath: record("closePath"),
  };
  return { ctx: path as CanvasRenderingContext2D, calls };
}

describe("structuredGraphStyle", () => {
  it("distinguishes tables, rows and cells across both tabular ingestion paths", () => {
    expect(["SchemaTable", "TableType", "DltRow", "TableRow", "DltColumn", "ColumnValue", "Entity"].map((type) => structuredNodeKind({ id: type, type })))
      .toEqual(["table", "table", "row", "row", "value", "value", null]);
  });
  it("deemphasizes membership and cell links in either direction, never row-to-row relationships", () => {
    const row = { id: "row", type: "DltRow" }, table = { id: "table", type: "SchemaTable" }, cell = { id: "cell", type: "DltColumn" };
    expect(isTabularStructureLink(row, table)).toBe(true);
    expect(isTabularStructureLink(cell, row)).toBe(true);
    expect(isTabularStructureLink(row, { id: "other", type: "TableRow" })).toBe(false);
    expect(isTabularStructureLink(row, { id: "person", type: "Entity" })).toBe(false);
  });
  it("leaves row and non-tabular node tracing to the existing circle and code renderers", () => {
    const { ctx, calls } = fakeContext();
    expect(traceStructuredShape(ctx, { id: "row", type: "DltRow" }, 0, 0, 10)).toBe(false);
    expect(traceStructuredShape(ctx, { id: "code", type: "CodeModule" }, 0, 0, 10)).toBe(false);
    expect(calls).toEqual([]);
  });
  it("traces a table as a rectangle inside the circular envelope", () => {
    const { ctx, calls } = fakeContext();
    expect(traceStructuredShape(ctx, { id: "t", type: "SchemaTable" }, 100, 50, 10)).toBe(true);
    expect(calls).toEqual([["rect", 92, 44, 16, 12]]);
  });
  it("traces a cell value as a diamond", () => {
    const { ctx, calls } = fakeContext();
    expect(traceStructuredShape(ctx, { id: "v", type: "ColumnValue" }, 100, 50, 10)).toBe(true);
    expect(calls).toEqual([["moveTo", 100, 40], ["lineTo", 110, 50], ["lineTo", 100, 60], ["lineTo", 90, 50], ["closePath"]]);
  });
});
