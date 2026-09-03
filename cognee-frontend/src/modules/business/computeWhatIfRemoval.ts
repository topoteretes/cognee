import type { SemanticLink } from "./sceneTypes";

export interface WhatIfRemoval {
  orphanedIds: Set<string>;
  islandCount: number;
}

const EMPTY: WhatIfRemoval = { orphanedIds: new Set(), islandCount: 0 };

// A component's own single-entity or single-neighbor case can never
// fragment — there's nothing left on the other side to strand.
const MIN_COMPONENT_SIZE_TO_FRAGMENT = 3;

function buildAdjacency(semanticLinks: SemanticLink[], excludeId?: string): Map<string, string[]> {
  const adjacency = new Map<string, string[]>();
  const addEdge = (a: string, b: string): void => {
    const list = adjacency.get(a);
    if (list) list.push(b);
    else adjacency.set(a, [b]);
  };
  semanticLinks.forEach((l) => {
    if (excludeId && (l._sid === excludeId || l._tid === excludeId)) return;
    addEdge(l._sid, l._tid);
    addEdge(l._tid, l._sid);
  });
  return adjacency;
}

function connectedComponent(start: string, adjacency: Map<string, string[]>, visited: Set<string>): Set<string> {
  const component = new Set<string>([start]);
  visited.add(start);
  const queue: string[] = [start];
  while (queue.length) {
    const current = queue.shift();
    if (current === undefined) break;
    for (const next of adjacency.get(current) ?? []) {
      if (visited.has(next)) continue;
      visited.add(next);
      component.add(next);
      queue.push(next);
    }
  }
  return component;
}

// The graph-dashboard companion to the hub metric (computeHubInsight):
// "if this entity disappeared, what else would lose its way back to the
// rest of the model?" Scoped to entityId's own connected component first
// (BFS on the FULL graph) since nothing outside it can possibly be
// affected — cheap even on graphs with many small, unrelated components.
// Removing entityId's edges from that component and re-running BFS splits
// it into fragments; the largest fragment is treated as "the graph, still
// intact" (it kept its own path to most of what mattered), and every
// smaller fragment is an orphaned island.
export function computeWhatIfRemoval(entityId: string, semanticLinks: SemanticLink[]): WhatIfRemoval {
  const fullAdjacency = buildAdjacency(semanticLinks);
  if (!fullAdjacency.has(entityId)) return EMPTY;

  const ownComponent = connectedComponent(entityId, fullAdjacency, new Set());
  if (ownComponent.size < MIN_COMPONENT_SIZE_TO_FRAGMENT) return EMPTY;

  const survivorsAdjacency = buildAdjacency(semanticLinks, entityId);
  const visited = new Set<string>([entityId]);
  const fragments: Set<string>[] = [];
  ownComponent.forEach((id) => {
    if (visited.has(id)) return;
    fragments.push(connectedComponent(id, survivorsAdjacency, visited));
  });
  if (fragments.length <= 1) return EMPTY;

  fragments.sort((a, b) => b.size - a.size);
  // The largest fragment is treated as "the graph, still standing" UNLESS
  // it's a singleton too — a hub whose every neighbor was a leaf (a pure
  // star) leaves nothing behind big enough to call "the rest of the graph";
  // every one of those leaves is equally orphaned, not just all-but-one.
  const orphaned = fragments[0].size > 1 ? fragments.slice(1) : fragments;
  const orphanedIds = new Set<string>();
  orphaned.forEach((fragment) => fragment.forEach((id) => orphanedIds.add(id)));
  return { orphanedIds, islandCount: fragments.length };
}
