import { isStructuredRow } from "./structuredRow";
import type { BusinessGraphNode } from "./types";

export function structuredNodeKind(node: BusinessGraphNode): "table" | "row" | "value" | null {
  if (isStructuredRow(node)) return "row";
  if (node.type === "SchemaTable" || node.type === "TableType") return "table";
  if (node.type === "DltColumn" || node.type === "ColumnValue") return "value";
  return null;
}

export function isTabularStructureLink(source: BusinessGraphNode, target: BusinessGraphNode): boolean {
  const a = structuredNodeKind(source), b = structuredNodeKind(target);
  return (a === "row" && (b === "table" || b === "value")) ||
    (b === "row" && (a === "table" || a === "value"));
}

// All shapes stay within the circular hover/ring envelope used by the canvas.
export function traceStructuredShape(
  ctx: CanvasRenderingContext2D, node: BusinessGraphNode, x: number, y: number, r: number,
): boolean {
  const kind = structuredNodeKind(node);
  if (kind === "table") {
    ctx.rect(x - r * 0.8, y - r * 0.6, r * 1.6, r * 1.2);
    return true;
  }
  if (kind === "value") {
    ctx.moveTo(x, y - r);
    ctx.lineTo(x + r, y);
    ctx.lineTo(x, y + r);
    ctx.lineTo(x - r, y);
    ctx.closePath();
    return true;
  }
  return false;
}
