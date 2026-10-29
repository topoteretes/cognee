import type { BusinessGraphNode } from "./types";

export interface RowField {
  column: string;
  value: string;
}

// DLT renders cells after this marker; the Columns section above it is
// schema metadata, not values. Keep continuations, blank cells and order.
export function parseRowFields(text: string): RowField[] {
  const marker = /^Row Data:[ \t]*\r?$/m.exec(text);
  if (!marker) return [];
  const fields: RowField[] = [];
  const lines = text.slice(marker.index + marker[0].length).replace(/^\r?\n/, "").split(/\r?\n/);
  for (const line of lines) {
    const match = /^  ([^:\r\n]+):(?: |$)(.*)$/.exec(line);
    if (match) fields.push({ column: match[1].trim(), value: match[2] });
    else if (fields.length) fields[fields.length - 1].value += `\n${line}`;
    else if (line.trim()) return [];
  }
  return fields;
}

export function structuredRowFields(node: BusinessGraphNode, content: string): RowField[] {
  if (node.type === "DltRow") return parseRowFields(content);
  if (node.type !== "TableRow") return [];
  // Some providers send JSON; the legacy Python dict representation cannot
  // be parsed losslessly here and stays available as raw content instead.
  try {
    const value: unknown = JSON.parse(content);
    if (!value || typeof value !== "object" || Array.isArray(value)) return [];
    return Object.entries(value).map(([column, cell]) => ({
      column, value: typeof cell === "string" ? cell : JSON.stringify(cell),
    }));
  } catch {
    return [];
  }
}
