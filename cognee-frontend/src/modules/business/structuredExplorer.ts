import { setsOf, UNCATEGORIZED_SOURCE } from "./computeBrainState";
import { isStructuredRow, structuredRowContent } from "./structuredRow";
import { structuredRowFields } from "./structuredRowFields";
import type { BrainState } from "./sceneTypes";

export interface SharedColumnValue {
  column: string;
  value: string;
  rowIds: Set<string>;
  sources: Set<string>;
}

export interface StructuredExplorer {
  rowIds: Set<string>;
  unavailableRows: number;
  columns: Map<string, Map<string, Set<string>>>;
  sourceCount: number;
  recordedRowLinks: number;
  sharedValueNodes: number;
  possibleMatches: SharedColumnValue[];
}

function distinctSources(a: Set<string>, b: Set<string>): boolean {
  // Overlapping source attribution is ambiguous, not proof of two files.
  return a.size > 0 && b.size > 0 && ![...a].some((source) => b.has(source));
}

// Equal source sets cannot be disjoint, so only distinct signatures need
// comparing. This keeps the check linear in the rows sharing a value instead
// of quadratic, which froze the render on a CSV with one repeated cell value.
function hasDistinctSourcePair(sourceSets: Iterable<Set<string>>): boolean {
  const bySignature = new Map<string, Set<string>>();
  for (const sources of sourceSets) {
    if (sources.size > 0) bySignature.set([...sources].sort().join("\u0000"), sources);
  }
  const unique = [...bySignature.values()];
  return unique.some((a, i) => unique.slice(i + 1).some((b) => distinctSources(a, b)));
}

function sourcesForRow(node: BrainState["entities"][number]): Set<string> {
  const file = typeof node.document_name === "string" ? node.document_name.split(/[\\/]/).pop() : undefined;
  return new Set(file && /\.csv$/i.test(file) ? [`File: ${file}`] : setsOf(node).filter((source) => source !== UNCATEGORIZED_SOURCE));
}

export function buildStructuredExplorer(brain: BrainState | null): StructuredExplorer {
  const result: StructuredExplorer = {
    rowIds: new Set(), unavailableRows: 0, columns: new Map(), sourceCount: 0,
    recordedRowLinks: 0, sharedValueNodes: 0, possibleMatches: [],
  };
  if (!brain) return result;
  const sourcesByRow = new Map<string, Set<string>>();
  for (const node of brain.entities) {
    if (!isStructuredRow(node)) continue;
    result.rowIds.add(node.id);
    sourcesByRow.set(node.id, sourcesForRow(node));
    const fields = structuredRowFields(node, structuredRowContent(node));
    if (!fields.length) result.unavailableRows += 1;
    for (const { column, value } of fields) {
      const values = result.columns.get(column) ?? new Map<string, Set<string>>();
      const ids = values.get(value) ?? new Set<string>();
      ids.add(node.id);
      values.set(value, ids);
      result.columns.set(column, values);
    }
  }
  result.sourceCount = new Set([...sourcesByRow.values()].flatMap((sources) => [...sources])).size;
  result.possibleMatches = findSharedValues(result.columns, sourcesByRow);
  countRecordedConnections(brain, sourcesByRow, result);
  return result;
}

function findSharedValues(
  columns: StructuredExplorer["columns"], sourcesByRow: Map<string, Set<string>>,
): SharedColumnValue[] {
  const matches: SharedColumnValue[] = [];
  for (const [column, values] of columns) {
    for (const [value, rowIds] of values) {
      // Blank/null placeholders provide no useful candidate identity.
      if (!value.trim() || /^(null|none|nan)$/i.test(value.trim())) continue;
      const sources = [...rowIds].map((id) => sourcesByRow.get(id) ?? new Set<string>());
      if (hasDistinctSourcePair(sources)) {
        matches.push({ column, value, rowIds, sources: new Set(sources.flatMap((s) => [...s])) });
      }
    }
  }
  return matches.sort((a, b) => b.rowIds.size - a.rowIds.size || a.column.localeCompare(b.column) || a.value.localeCompare(b.value));
}

function countRecordedConnections(
  brain: BrainState, sourcesByRow: Map<string, Set<string>>, result: StructuredExplorer,
): void {
  const valueOwners = new Map<string, Set<string>>();
  for (const link of brain.graphLinks) {
    const source = sourcesByRow.get(link._sid), target = sourcesByRow.get(link._tid);
    if (source && target && distinctSources(source, target)) result.recordedRowLinks += 1;
    for (const [rowId, cellId] of [[link._sid, link._tid], [link._tid, link._sid]]) {
      const type = brain.entityById[cellId]?.type;
      if (!sourcesByRow.has(rowId) || (type !== "DltColumn" && type !== "ColumnValue")) continue;
      const owners = valueOwners.get(cellId) ?? new Set<string>();
      owners.add(rowId);
      valueOwners.set(cellId, owners);
    }
  }
  for (const owners of valueOwners.values()) {
    if (hasDistinctSourcePair([...owners].map((id) => sourcesByRow.get(id) ?? new Set<string>()))) result.sharedValueNodes += 1;
  }
}
