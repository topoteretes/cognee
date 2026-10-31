import type { SemanticLink } from "../sceneTypes";

// Shared by the "click a record" focus-mode (dim everyone but its direct
// neighbors), the shortest-path highlight (useShortestPath), and the
// orphan-record indicator — all three read the same semanticLinks array,
// just asking a different question of it.

export function computeNeighborIds(semanticLinks: SemanticLink[], centerId: string): Set<string> {
  const ids = new Set<string>([centerId]);
  semanticLinks.forEach((l) => {
    if (l._sid === centerId) ids.add(l._tid);
    else if (l._tid === centerId) ids.add(l._sid);
  });
  return ids;
}

export function computeConnectedIds(semanticLinks: SemanticLink[]): Set<string> {
  const ids = new Set<string>();
  semanticLinks.forEach((l) => { ids.add(l._sid); ids.add(l._tid); });
  return ids;
}

// Undirected edge identity — a path's edges and a rendered link's endpoints
// don't necessarily agree on which side is "source", so lookups on either
// need to land on the same key.
export function edgeKey(a: string, b: string): string {
  return a < b ? `${a}|${b}` : `${b}|${a}`;
}
