import type { BrainState } from "./sceneTypes";
import { isStructuredRow } from "./structuredRow";
import { setsOf } from "./computeBrainState";

// A rendering projection, sharing the simulation's positioned node objects.
// Filtering must not restart physics, announce growth, or invent edges.
export function projectStructuredRows(brain: BrainState, rowIds: Set<string> | null): BrainState {
  if (rowIds === null) return brain;
  const visible = new Set([...rowIds].filter((id) => brain.entityById[id] && isStructuredRow(brain.entityById[id])));
  for (const link of brain.graphLinks) {
    for (const [row, neighbor] of [[link._sid, link._tid], [link._tid, link._sid]]) {
      const node = brain.entityById[neighbor];
      if (rowIds.has(row) && node && !isStructuredRow(node)) visible.add(neighbor);
    }
  }
  const entities = brain.entities.filter((node) => visible.has(node.id));
  const graphLinks = brain.graphLinks.filter((link) => visible.has(link._sid) && visible.has(link._tid));
  return {
    ...brain, entities, graphLinks,
    entityById: Object.fromEntries(entities.map((node) => [node.id, node])),
    sourceNames: [...new Set(entities.flatMap(setsOf))],
    // Exact row results stay visible at every zoom, including low-importance
    // rows that the unfiltered overview normally suppresses.
    typeNodes: [], typeLinks: [], importanceCut: 0,
    semanticLinks: brain.semanticLinks.filter((link) => visible.has(link._sid) && visible.has(link._tid)),
    // A document link has one non-entity end that is never in `visible`, so it
    // is kept when its entity end is, unlike the entity-to-entity links above.
    docLinks: brain.docLinks.filter((link) => visible.has(link._sid) || visible.has(link._tid)),
    plumbingNodes: brain.plumbingNodes.filter((node) => visible.has(brain.plumbingEntityId[node.id])),
    connectedIds: new Set(graphLinks.flatMap((link) => [link._sid, link._tid])),
  };
}
