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

  if (obj.nodes.length > 0) {
    const first = obj.nodes[0] as Record<string, unknown>;
    if (typeof first.id !== "string" || typeof first.label !== "string" || typeof first.type !== "string") {
      return false;
    }
  }

  if (obj.edges.length > 0) {
    const first = obj.edges[0] as Record<string, unknown>;
    if (typeof first.source !== "string" || typeof first.target !== "string" || typeof first.label !== "string") {
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
