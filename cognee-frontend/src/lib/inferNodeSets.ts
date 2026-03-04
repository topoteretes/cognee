import type { Node, Edge } from "@/ui/rendering/graph/types";
import type { NodeSet, RetrievalResult } from "@/types/NodeSet";

interface InferNodeSetsOptions {
  minSetSize?: number;
  maxSets?: number;
}

interface InferNodeSetsResult {
  typeAttributes: Map<string, number>;
  inferredSets: NodeSet[];
}

export function inferNodeSets(
  nodes: Node[],
  edges: Edge[],
  options: InferNodeSetsOptions = {}
): InferNodeSetsResult {
  const minSetSize = Math.max(1, options.minSetSize ?? 5);
  const maxSets = Math.max(1, options.maxSets ?? 15);

  // Group nodes by type
  const typeGroups = new Map<string, string[]>();
  for (const node of nodes) {
    const existing = typeGroups.get(node.type);
    if (existing) {
      existing.push(node.id);
    } else {
      typeGroups.set(node.type, [node.id]);
    }
  }

  // typeAttributes: type -> count
  const typeAttributes = new Map<string, number>();
  for (const [type, ids] of typeGroups) {
    typeAttributes.set(type, ids.length);
  }

  // Build adjacency for cohesion calculation
  const adjacency = new Map<string, Set<string>>();
  for (const edge of edges) {
    let sourceSet = adjacency.get(edge.source);
    if (!sourceSet) {
      sourceSet = new Set();
      adjacency.set(edge.source, sourceSet);
    }
    sourceSet.add(edge.target);

    let targetSet = adjacency.get(edge.target);
    if (!targetSet) {
      targetSet = new Set();
      adjacency.set(edge.target, targetSet);
    }
    targetSet.add(edge.source);
  }

  // Create NodeSets for types meeting minSetSize
  const inferredSets: NodeSet[] = [];
  for (const [type, nodeIds] of typeGroups) {
    if (nodeIds.length < minSetSize) {
      continue;
    }

    // Compute cohesion: internal edges / max possible internal edges
    const memberSet = new Set(nodeIds);
    let internalEdges = 0;
    for (const nodeId of nodeIds) {
      const neighbors = adjacency.get(nodeId);
      if (neighbors) {
        for (const neighbor of neighbors) {
          if (memberSet.has(neighbor)) {
            internalEdges++;
          }
        }
      }
    }
    // Each edge counted twice (once from each endpoint)
    internalEdges = Math.floor(internalEdges / 2);

    const n = nodeIds.length;
    const maxPossible = (n * (n - 1)) / 2;
    const cohesion = maxPossible > 0 ? internalEdges / maxPossible : 0;

    inferredSets.push({
      id: `inferred_${type}`,
      name: type,
      description: `Inferred set of ${n} ${type} nodes`,
      nodeIds,
      size: n,
      definition: `Nodes with type="${type}"`,
      stability: "stable",
      source: "model-inferred",
      lastUpdated: new Date(),
      cohesion: Math.min(cohesion, 1),
      confidence: 0.9,
    });
  }

  // Sort by size descending, then truncate
  inferredSets.sort((a, b) => b.size - a.size);
  inferredSets.splice(maxSets);

  return { typeAttributes, inferredSets };
}

export function mockRetrievalSearch(
  query: string,
  nodes: Node[],
  inferredSets: NodeSet[]
): RetrievalResult[] {
  const lowerQuery = query.toLowerCase();
  const results: RetrievalResult[] = [];

  // Search individual nodes
  for (const node of nodes) {
    const label = node.label.toLowerCase();
    if (label.includes(lowerQuery)) {
      const score = Math.min(lowerQuery.length / label.length, 1.0);
      results.push({
        type: "node",
        nodeId: node.id,
        nodeLabel: node.label,
        why: `Label contains "${query}"`,
        similarityScore: score,
        signals: [
          { name: "label-match", weight: score, value: node.label },
        ],
        confidence: score,
      });
    }
  }

  // Search node sets
  for (const nodeSet of inferredSets) {
    const name = nodeSet.name.toLowerCase();
    if (name.includes(lowerQuery)) {
      const score = Math.min(lowerQuery.length / name.length, 1.0);
      results.push({
        type: "nodeSet",
        nodeSet,
        why: `Set name contains "${query}"`,
        similarityScore: score,
        signals: [
          { name: "set-name-match", weight: score, value: nodeSet.name },
        ],
        confidence: score,
      });
    }
  }

  // Sort by score descending, take top 20
  results.sort((a, b) => b.similarityScore - a.similarityScore);
  return results.slice(0, 20);
}
