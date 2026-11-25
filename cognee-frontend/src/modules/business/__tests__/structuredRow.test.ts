import { structuredRowContent, structuredRowLabel, structuredRowValues } from "@/modules/business/structuredRow";
import { normalizeStructuredNodes } from "@/modules/business/structuredGraph";
import type { BusinessGraphNode } from "@/modules/business/types";

const row: BusinessGraphNode = {
  id: "12345678-abcd", type: "DltRow", stage: "chunk", chunk_index: 2,
  name: "Table: people Columns: - person (TEXT) - company (TEXT)",
  text: "Table: people\nColumns:\n  - person (TEXT)\n  - company (TEXT)\n\nRow Data:\n  person: Ana Horvat\n  company: Atlas Labs\n  description: Ana works at Atlas Labs.",
};

describe("structured row labels", () => {
  it("uses a person value rather than the schema header and preserves the complete row for details", () => {
    const [normalized] = normalizeStructuredNodes([row]);
    expect(normalized.name).toBe("Ana Horvat");
    expect(normalized.originalName).toBe(row.name);
    expect(structuredRowContent(normalized)).toBe(row.text);
    expect(row.name).toMatch(/^Table:/);
  });

  it.each([
    ["product", "Orion Laptop"], ["task", "Build CSV upload"], ["Full Name", "Ana Horvat"],
  ])("uses the %s column as a label", (column, value) => {
    expect(structuredRowLabel({ ...row, text: `Row Data:\r\n  ${column}: ${value}\r\n  id: 42` })).toBe(value);
  });

  it("does not interpret schema column declarations as row values", () => {
    expect(structuredRowLabel({ ...row, text: "Table: people\nColumns:\n  name: TEXT" })).toBe("Row 3");
  });

  it("labels a support ticket by its issue rather than the product shared by every ticket", () => {
    expect(structuredRowLabel({
      ...row, text: "Row Data:\n  product: Knowledge Hub\n  issue: Empty CSV graph\n  customer: Blue River",
    })).toBe("Empty CSV graph");
  });

  it("ignores empty, null-like and oversized labels", () => {
    expect(structuredRowLabel({
      ...row, text: `Row Data:\n  name: None\n  person: \n  title: ${"x".repeat(100)}\n  id: 42`,
    })).toBe("Row 3");
  });

  it("uses a stable identifier when the compact payload omits the row position and values", () => {
    const compact = { id: row.id, type: "DltRow", name: row.name };
    expect(structuredRowLabel(compact)).toBe("Row 12345678");
    expect(structuredRowLabel({ ...compact, chunk_index: -1 })).toBe("Row 12345678");
  });

  it("keeps relational row identities and their unparsed properties intact", () => {
    const relational = { id: "row", type: "TableRow", name: "people:42", properties: "{'name': 'Ana'}" };
    expect(structuredRowLabel(relational)).toBe("people:42");
    expect(structuredRowContent(relational)).toBe(relational.properties);
  });

  it("preserves the original text even when it arrived only in name", () => {
    const [normalized] = normalizeStructuredNodes([{ id: row.id, type: "DltRow", name: row.text as string }]);
    expect(normalized.name).toBe("Ana Horvat");
    expect(structuredRowContent(normalized)).toBe(row.text);
    expect(normalizeStructuredNodes([normalized])).toEqual([normalized]);
  });
});

describe("row values and labels", () => {
  it("returns no values for a compact row or one marked as asked, while content falls back to the name", () => {
    const compact = { id: "r", type: "DltRow", name: "Table: people Columns:\n  - person (TEXT)" };
    expect(structuredRowValues(compact)).toBe("");
    expect(structuredRowValues({ ...compact, text: "" })).toBe("");
    expect(structuredRowContent(compact)).toBe(compact.name);
  });

  it("keeps the first value of a column, so a spoofed continuation line cannot rename the row", () => {
    const row = { id: "r", type: "DltRow", text: "Row Data:\n  name: Ana\n  name: Evil" };
    expect(structuredRowLabel(row)).toBe("Ana");
  });

  it("keeps the label of a person called Nan but skips the SDK's null markers", () => {
    expect(structuredRowLabel({ id: "a", type: "DltRow", text: "Row Data:\n  name: Nan" })).toBe("Nan");
    expect(structuredRowLabel({ id: "b", type: "DltRow", text: "Row Data:\n  name: None\n  company: Atlas" })).toBe("Atlas");
  });
});
