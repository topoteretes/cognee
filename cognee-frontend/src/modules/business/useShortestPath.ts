"use client";

import { useMemo } from "react";
import type { SemanticLink } from "./sceneTypes";
import { edgeKey } from "./canvas/businessGraphFocus";

export interface ShortestPath {
  pathIds: Set<string>;
  pathEdgeKeys: Set<string>;
}

const EMPTY_PATH: ShortestPath = { pathIds: new Set(), pathEdgeKeys: new Set() };

// Shift-click a second record while one is already selected (see
// BusinessCanvas's handleClick) to trace how the two are connected — new
// in this port, no source equivalent. A plain BFS over the undirected
// semantic-link graph; these graphs are small enough (hundreds to a few
// thousand entities) that BFS is instant, so there's no need for anything
// smarter than "shortest by hop count".
export function useShortestPath(
  semanticLinks: SemanticLink[] | undefined,
  fromId: string | null,
  toId: string | null,
): ShortestPath {
  return useMemo(() => {
    if (!semanticLinks || !fromId || !toId || fromId === toId) return EMPTY_PATH;
    const adjacency = new Map<string, string[]>();
    const addEdge = (a: string, b: string): void => {
      const list = adjacency.get(a);
      if (list) list.push(b);
      else adjacency.set(a, [b]);
    };
    semanticLinks.forEach((l) => { addEdge(l._sid, l._tid); addEdge(l._tid, l._sid); });

    const prev = new Map<string, string>();
    const visited = new Set<string>([fromId]);
    const queue: string[] = [fromId];
    let reached = false;
    while (queue.length && !reached) {
      const current = queue.shift();
      if (current === undefined) break;
      for (const next of adjacency.get(current) ?? []) {
        if (visited.has(next)) continue;
        visited.add(next);
        prev.set(next, current);
        if (next === toId) { reached = true; break; }
        queue.push(next);
      }
    }
    if (!reached) return EMPTY_PATH;

    const pathIds = new Set<string>([toId]);
    const pathEdgeKeys = new Set<string>();
    let cur = toId;
    while (cur !== fromId) {
      const parent = prev.get(cur);
      if (!parent) break;
      pathEdgeKeys.add(edgeKey(parent, cur));
      pathIds.add(parent);
      cur = parent;
    }
    return { pathIds, pathEdgeKeys };
  }, [semanticLinks, fromId, toId]);
}
