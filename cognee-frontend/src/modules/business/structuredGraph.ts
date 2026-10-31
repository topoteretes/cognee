import type { BusinessGraphNode } from "./types";
import { isStructuredRow, structuredRowLabel } from "./structuredRow";

const STRUCTURED_TYPES: Readonly<Record<string, string>> = {
  DltRow: "Row",
  TableRow: "Row",
  SchemaTable: "Table",
  SchemaRelationship: "Table relationship",
  DatabaseSchema: "Database schema",
  TableType: "Table type",
  ColumnValue: "Column value",
  DltColumn: "Column value",
};

export function structuredTypeName(node: BusinessGraphNode): string | undefined {
  return node.type && Object.hasOwn(STRUCTURED_TYPES, node.type)
    ? STRUCTURED_TYPES[node.type]
    : undefined;
}

// Unlike codeGraph.ts, which derives its short name at render time, a row's
// label is written onto `name` here: its raw name is a truncated schema header
// ("Table: people Columns: ..."), identical for every row of a table, so
// search, tooltips and path narration would otherwise show N indistinguishable
// rows. The wire name is kept in `originalName` and nothing matches on it.
//
// Structured ingestion already creates a graph without LLM entities. Use the
// canvas's entity layer for those records too, retaining their original types
// and properties for the inspector and leaving the cached API payload intact.
export function normalizeStructuredNodes(nodes: BusinessGraphNode[]): BusinessGraphNode[] {
  return nodes.map((node) => {
    if (!structuredTypeName(node)) return node;
    return {
      ...node,
      stage: "entity",
      ...(isStructuredRow(node) ? {
        name: structuredRowLabel(node),
        originalName: node.originalName ?? node.name,
      } : {}),
    };
  });
}
