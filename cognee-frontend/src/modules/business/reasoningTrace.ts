import type { BusinessEntity, SemanticLink } from "./sceneTypes";

export interface ReasoningStep {
  id: string;
  narration: string;
}

// Long answers would turn the walk into a slog before landing on the actual
// answer card — cap it so the "wow" stays snappy regardless of how many
// facts a search touched.
export const MAX_TRACE_STEPS = 8;

function pickRoot(idList: string[], degree: Map<string, number>): string {
  return idList.reduce(
    (best, id) => ((degree.get(id) ?? 0) > (degree.get(best) ?? 0) ? id : best),
    idList[0],
  );
}

function nameOf(id: string, entityById: Record<string, BusinessEntity>): string {
  return entityById[id]?.name || entityById[id]?.type || "an unnamed record";
}

interface AdjacentNode {
  id: string;
  relation?: string;
}

function buildInducedAdjacency(
  ids: Set<string>,
  semanticLinks: SemanticLink[],
): { adjacency: Map<string, AdjacentNode[]>; degree: Map<string, number> } {
  const adjacency = new Map<string, AdjacentNode[]>();
  const degree = new Map<string, number>();
  const addHop = (from: string, to: string, relation?: string): void => {
    adjacency.set(from, [...(adjacency.get(from) ?? []), { id: to, relation }]);
    degree.set(from, (degree.get(from) ?? 0) + 1);
  };
  semanticLinks.forEach((l) => {
    if (!ids.has(l._sid) || !ids.has(l._tid)) return;
    addHop(l._sid, l._tid, l.relation);
    addHop(l._tid, l._sid, l.relation);
  });
  return { adjacency, degree };
}

function narrationFor(step: AdjacentNode, index: number, entityById: Record<string, BusinessEntity>): string {
  const name = nameOf(step.id, entityById);
  if (index === 0) return `following the trail: ${name}`;
  return step.relation ? `→ ${step.relation} → ${name}` : `→ ${name}`;
}

// Orders a search answer's contributing node ids into a walkable sequence,
// hopping along real edges of the induced subgraph (edges where both ends
// are among `ids`) from its most-connected member, instead of the id set's
// arbitrary iteration order. Ids with no edge into the walked component
// still get a trailing step, so nothing the answer actually relied on is
// silently dropped from the walk.
export function buildReasoningTrace(
  ids: Set<string>,
  semanticLinks: SemanticLink[],
  entityById: Record<string, BusinessEntity>,
): ReasoningStep[] {
  const idList = Array.from(ids);
  if (!idList.length) return [];

  const { adjacency, degree } = buildInducedAdjacency(ids, semanticLinks);
  const root = pickRoot(idList, degree);
  const visited = new Set<string>([root]);
  const queue: string[] = [root];
  const order: AdjacentNode[] = [{ id: root }];
  while (queue.length) {
    const current = queue.shift();
    if (current === undefined) break;
    for (const next of adjacency.get(current) ?? []) {
      if (visited.has(next.id)) continue;
      visited.add(next.id);
      queue.push(next.id);
      order.push(next);
    }
  }
  idList.forEach((id) => {
    if (!visited.has(id)) order.push({ id });
  });

  return order
    .slice(0, MAX_TRACE_STEPS)
    .map((step, index) => ({ id: step.id, narration: narrationFor(step, index, entityById) }));
}
