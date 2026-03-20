/**
 * Node Sets: The primary abstraction for grouping nodes
 * Replaces fixed types with dynamic, inferred, overlapping groups
 */

export type NodeSetSource =
  | "model-inferred"    // Created by AI/ML model
  | "user-defined"      // Manually created by user
  | "query-result"      // Result of a search query
  | "imported"          // From external source
  | "set-algebra";      // Created by combining other sets

export type NodeSetStability =
  | "stable"      // Won't change often
  | "evolving"    // Changes gradually
  | "ephemeral";  // Temporary, will be removed

export interface NodeSet {
  id: string;
  name: string;
  description?: string;

  // Required properties
  nodeIds: string[];           // Member node IDs
  size: number;                // Number of nodes
  definition: string;          // How it was created (e.g., "semantic cluster around 'AI'")
  stability: NodeSetStability;
  source: NodeSetSource;
  lastUpdated: Date;

  // Confidence metrics
  confidence?: number;         // 0-1, how confident we are in this grouping
  cohesion?: number;          // 0-1, how tightly connected members are

  // Set algebra metadata
  parentSets?: string[];      // If created from other sets
  operation?: "union" | "intersect" | "diff";

  // Retrieval metadata
  retrievalScore?: number;    // If this set was retrieved
  retrievalSignals?: string[]; // Why it was retrieved

  // Visual properties (for rendering)
  color?: string;
  visible?: boolean;
}

/**
 * Set operations
 */
export function unionSets(sets: NodeSet[], name: string): NodeSet {
  const allNodeIds = new Set<string>();
  sets.forEach(set => set.nodeIds.forEach(id => allNodeIds.add(id)));

  return {
    id: `union_${Date.now()}`,
    name,
    nodeIds: Array.from(allNodeIds),
    size: allNodeIds.size,
    definition: `Union of: ${sets.map(s => s.name).join(", ")}`,
    stability: "ephemeral",
    source: "set-algebra",
    lastUpdated: new Date(),
    parentSets: sets.map(s => s.id),
    operation: "union",
  };
}

export function intersectSets(sets: NodeSet[], name: string): NodeSet {
  if (sets.length === 0) {
    return {
      id: `intersect_${Date.now()}`,
      name,
      nodeIds: [],
      size: 0,
      definition: "Empty intersection",
      stability: "ephemeral",
      source: "set-algebra",
      lastUpdated: new Date(),
    };
  }

  const intersection = new Set(sets[0].nodeIds);
  sets.slice(1).forEach(set => {
    const setIds = new Set(set.nodeIds);
    intersection.forEach(id => {
      if (!setIds.has(id)) intersection.delete(id);
    });
  });

  return {
    id: `intersect_${Date.now()}`,
    name,
    nodeIds: Array.from(intersection),
    size: intersection.size,
    definition: `Intersection of: ${sets.map(s => s.name).join(", ")}`,
    stability: "ephemeral",
    source: "set-algebra",
    lastUpdated: new Date(),
    parentSets: sets.map(s => s.id),
    operation: "intersect",
  };
}

export function diffSets(setA: NodeSet, setB: NodeSet, name: string): NodeSet {
  const diff = new Set(setA.nodeIds);
  setB.nodeIds.forEach(id => diff.delete(id));

  return {
    id: `diff_${Date.now()}`,
    name,
    nodeIds: Array.from(diff),
    size: diff.size,
    definition: `${setA.name} minus ${setB.name}`,
    stability: "ephemeral",
    source: "set-algebra",
    lastUpdated: new Date(),
    parentSets: [setA.id, setB.id],
    operation: "diff",
  };
}

/**
 * Retrieval result with explanation
 */
export interface RetrievalResult {
  type: "node" | "nodeSet" | "suggestedSet";

  // For nodes
  nodeId?: string;
  nodeLabel?: string;

  // For node sets
  nodeSet?: NodeSet;

  // For suggested sets
  suggestedSetDefinition?: string;
  suggestedNodeIds?: string[];

  // Explanation (critical for trust)
  why: string;                    // Human-readable explanation
  similarityScore: number;        // 0-1
  signals: {
    name: string;                 // e.g., "semantic", "recency", "provenance"
    weight: number;               // Contribution to final score
    value: string | number;       // The actual value
  }[];

  // Confidence
  confidence: number;             // 0-1, how confident we are in this retrieval
}

/**
 * Recall simulation: "If the agent were asked X, what would be retrieved?"
 */
export interface RecallSimulation {
  query: string;
  rankedMemories: RetrievalResult[];
  activatedSets: NodeSet[];
  conflicts?: {
    nodeId: string;
    reason: string;
    conflictingSets: string[];
  }[];
}
