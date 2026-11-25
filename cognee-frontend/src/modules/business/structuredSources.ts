import { endId } from "./linkEndpoint";
import { isStructuredRow, structuredRowContent } from "./structuredRow";
import type { BusinessGraphLink, BusinessGraphNode } from "./types";

function explicitSources(node: BusinessGraphNode): string[] {
  // The SDK's preprocessor also accepts a single name as a plain string.
  const sets: unknown = node.belongs_to_set;
  if (Array.isArray(sets)) {
    const names = sets.filter((name): name is string => typeof name === "string" && name !== "");
    if (names.length) return names;
  }
  if (typeof sets === "string" && sets) return [sets];
  return node.source_node_set?.split(",").map((name) => name.trim()).filter(Boolean) ?? [];
}

function isTable(node: BusinessGraphNode | undefined): boolean {
  return node?.type === "SchemaTable" || node?.type === "TableType";
}

function isSourceDocument(node: BusinessGraphNode | undefined): boolean {
  return node?.type === "dlt_source" || node?.type === "DltSourceDocument";
}

function fileName(node: BusinessGraphNode): string | undefined {
  if (typeof node.document_name !== "string") return undefined;
  const basename = node.document_name.split(/[\\/]/).pop()?.trim();
  // The SDK can send a generated manifest name or a filename stem. Neither
  // proves the original filename, so use the table instead in those cases.
  // Only .csv is accepted: other extensions (a generated manifest can carry
  // one) are not verified to be the uploaded file, so they fall back to the
  // table rather than risk naming a source after a pipeline artifact.
  return basename && /\.csv$/i.test(basename) ? basename : undefined;
}

function addSources(groups: Map<string, Set<string>>, id: string, sources: string[]): void {
  const group = groups.get(id) ?? new Set<string>();
  sources.forEach((source) => group.add(source));
  groups.set(id, group);
}

function rowSources(row: BusinessGraphNode, table: BusinessGraphNode | undefined): string[] {
  const explicit = explicitSources(row);
  if (explicit.length) return explicit;
  const file = fileName(row);
  if (file) return [`File: ${file}`];
  const tableSources = table && explicitSources(table);
  if (tableSources?.length) return tableSources;
  const content = structuredRowContent(row);
  const tableName = table?.name || (content.includes("\n") ? /^Table: ([^:\r\n]+?)[ \t]*\r?$/m.exec(content)?.[1] : undefined);
  return tableName ? [`Table: ${tableName.trim()}`] : [];
}

// Ownership only: foreign-key and cell-value matches must not merge two
// files' territories. Derive row groups first, then share them with their
// table, source document and cell nodes, without traversing arbitrary edges.
export function groupStructuredSources(
  nodes: BusinessGraphNode[], links: BusinessGraphLink[],
): BusinessGraphNode[] {
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const tables = new Map<string, BusinessGraphNode>();
  for (const link of links) {
    const row = byId.get(endId(link.source)), table = byId.get(endId(link.target));
    if (row && isStructuredRow(row) && table && isTable(table) &&
        (link.relation === "is_row_of" || link.relation === "is_part_of" || link.relation === "is_a")) {
      tables.set(row.id, table);
    }
  }
  const groups = new Map<string, Set<string>>();
  for (const node of nodes) {
    if (!isStructuredRow(node)) continue;
    const sources = rowSources(node, tables.get(node.id));
    addSources(groups, node.id, sources);
    const table = tables.get(node.id);
    if (table) addSources(groups, table.id, sources);
  }
  for (const node of nodes) {
    if (isTable(node) && !groups.get(node.id)?.size) {
      const sources = explicitSources(node);
      addSources(groups, node.id, sources.length ? sources : node.name ? [`Table: ${node.name}`] : []);
    }
  }
  inheritOwnedSources(nodes, links, groups, byId);
  return nodes.map((node) => {
    const explicit = explicitSources(node);
    // setsOf reads the array as sent, so a plain string or an array holding
    // other values is normalised here or the node would fall back to
    // uncategorized, or carry a non-name into the source list.
    const rawSets: unknown = node.belongs_to_set;
    const needsNormalising = typeof rawSets === "string" || (Array.isArray(rawSets) && rawSets.length !== explicit.length);
    if (explicit.length) return needsNormalising ? { ...node, belongs_to_set: explicit } : node;
    const sources = [...(groups.get(node.id) ?? [])].sort();
    return sources.length ? { ...node, belongs_to_set: sources } : node;
  });
}

function inheritOwnedSources(
  nodes: BusinessGraphNode[], links: BusinessGraphLink[], groups: Map<string, Set<string>>,
  byId: Map<string, BusinessGraphNode>,
): void {
  // A bounded read may omit the row→document edge but still carry document_id.
  for (const row of nodes) {
    if (isStructuredRow(row) && typeof row.document_id === "string" && isSourceDocument(byId.get(row.document_id))) {
      addSources(groups, row.document_id, [...(groups.get(row.id) ?? [])]);
    }
  }
  for (const link of links) {
    const source = byId.get(endId(link.source)), target = byId.get(endId(link.target));
    if (!source || !target) continue;
    const ownsCell = isStructuredRow(source) && (target.type === "DltColumn" || target.type === "ColumnValue");
    const ownsDocument = isStructuredRow(source) && link.relation === "is_part_of" &&
      isSourceDocument(target);
    const ownsRelationship = isTable(source) && target.type === "SchemaRelationship" &&
      (link.relation === "has_foreign_key" || link.relation === "has_relationship");
    if (ownsCell || ownsDocument || ownsRelationship) {
      const sources = groups.get(source.id) ?? new Set(explicitSources(source));
      addSources(groups, target.id, [...sources]);
    }
  }
}
