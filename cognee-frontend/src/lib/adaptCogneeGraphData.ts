import type { CogneeGraphResponse } from "@/types/CogneeAPI";
import type { Node, Edge } from "@/ui/rendering/graph/types";

export function validateCogneeGraphResponse(data: unknown): data is CogneeGraphResponse {
  if (data == null || typeof data !== "object") {
    return false;
  }

  const obj = data as Record<string, unknown>;

  if (!Array.isArray(obj.nodes) || !Array.isArray(obj.edges)) {
    return false;
  }

  for (const raw of obj.nodes) {
    const node = raw as Record<string, unknown>;
    if (typeof node.id !== "string" || typeof node.label !== "string" || typeof node.type !== "string") {
      return false;
    }
  }

  for (const raw of obj.edges) {
    const edge = raw as Record<string, unknown>;
    if (typeof edge.source !== "string" || typeof edge.target !== "string" || typeof edge.label !== "string") {
      return false;
    }
  }

  return true;
}

export function adaptCogneeGraphData(response: CogneeGraphResponse): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = response.nodes.map((dp) => ({
    id: dp.id,
    label: dp.label,
    type: dp.type,
  }));

  const nodeIdSet = new Set(nodes.map((n) => n.id));

  const edges: Edge[] = response.edges
    .filter((e) => nodeIdSet.has(e.source) && nodeIdSet.has(e.target))
    .map((e, index) => ({
      id: `edge_${e.source}_${e.target}_${index}`,
      label: e.label,
      source: e.source,
      target: e.target,
    }));

  return { nodes, edges };
}
