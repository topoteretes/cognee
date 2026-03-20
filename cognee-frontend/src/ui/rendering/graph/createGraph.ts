import createNgraph, { Graph } from "ngraph.graph";
import { Edge, Node } from "./types";

export default function createGraph(
  nodes: Node[],
  edges: Edge[],
  forNode?: (node: Node) => void,
  forEdge?: (node: Edge) => void
): Graph {
  const graph = createNgraph();

  for (const node of nodes) {
    graph.addNode(node.id, {
      id: node.id,
      label: node.label,
    });
    forNode?.(node);
  }
  for (const edge of edges) {
    graph.addLink(edge.source, edge.target, {
      id: edge.id,
      label: edge.label,
    });
    forEdge?.(edge);
  }

  return graph;
}
