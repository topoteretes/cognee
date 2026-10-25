import type { BusinessGraphNode } from "./types";
import { parseRowFields } from "./structuredRowFields";

const LABEL_COLUMNS = ["name", "full_name", "person", "title", "task", "issue", "product", "customer", "company"];
const MAX_LABEL_LENGTH = 80;
// What the SDK writes for a missing cell (str(None), float nan). Matched
// case-sensitively so a person called "Nan" keeps their label.
const NULL_TOKENS = /^(None|nan|NaN|null)$/;

export function isStructuredRow(node: BusinessGraphNode): boolean {
  return node.type === "DltRow" || node.type === "TableRow";
}

// A compact streamed row carries only a truncated schema header; the values
// live in `text` / `properties`, which only the full JSON read sends.
export function hasRowDetail(node: BusinessGraphNode): boolean {
  return typeof node.text === "string" || typeof node.properties === "string";
}

function firstText(values: unknown[]): string {
  return values.find((value): value is string => typeof value === "string" && value.trim() !== "") ?? "";
}

// The row's real values, with no fallback: a compact row has none, and its
// name is only a truncated schema header that must not be shown as values.
export function structuredRowValues(node: BusinessGraphNode): string {
  return isStructuredRow(node) ? firstText([node.text, node.properties]) : "";
}

// Values when present, otherwise the name, which labels and source grouping
// can still read a table name from.
export function structuredRowContent(node: BusinessGraphNode): string {
  if (!isStructuredRow(node)) return "";
  return structuredRowValues(node) || firstText([node.originalName, node.name]);
}

// Only the Row Data section contains values. The preceding Columns section
// describes the schema and must never supply a person's or product's name.
function labelFromRowText(text: string): string | undefined {
  const values = new Map<string, string>();
  for (const field of parseRowFields(text)) {
    const column = field.column.trim().toLowerCase().replace(/\s+/g, "_");
    const value = field.value.trim();
    if (value && !value.includes("\n") && value.length <= MAX_LABEL_LENGTH &&
        !NULL_TOKENS.test(value) && !values.has(column)) values.set(column, value);
  }
  return LABEL_COLUMNS.map((column) => values.get(column)).find(Boolean);
}

export function structuredRowLabel(node: BusinessGraphNode): string {
  const label = labelFromRowText(structuredRowContent(node));
  if (label) return label;
  const name = node.originalName ?? node.name;
  // Relational TableRow nodes already have a useful table:primary-key name;
  // a compact DltRow name is usually a truncated schema header instead.
  if (typeof name === "string" && name.trim() && name.length <= MAX_LABEL_LENGTH &&
      !name.includes("\n") && !/^(Table:|Unnamed\s)/.test(name)) return name;
  const index = node.chunk_index;
  if (typeof index === "number" && Number.isSafeInteger(index) && index >= 0) return `Row ${index + 1}`;
  return `Row ${node.id.slice(0, 8)}`;
}
