import type { BrainState } from "./sceneTypes";
import { setsOf, sourceLabel } from "./computeBrainState";

export interface SourceTypeBreakdownEntry {
  type: string;
  count: number;
}

export interface SourceDetail {
  name: string;
  displayName: string;
  color: string;
  entityCount: number;
  docCount: number;
  typeBreakdown: SourceTypeBreakdownEntry[];
  documentNames: string[];
  documentTotal: number;
  bridgeCount: number;
}

const MAX_TYPE_ROWS = 5;
const MAX_DOCUMENT_ROWS = 6;

function buildTypeBreakdown(name: string, brainState: BrainState): SourceTypeBreakdownEntry[] {
  return brainState.typeNodes
    .map((t) => ({ type: t.name, count: t.sets[name] || 0 }))
    .filter((row) => row.count > 0)
    .sort((a, b) => b.count - a.count)
    .slice(0, MAX_TYPE_ROWS);
}

function buildDocumentNames(name: string, brainState: BrainState): { names: string[]; total: number } {
  const matches = brainState.plumbingNodes.filter((n) => setsOf(n).includes(name) && n.name);
  return { names: matches.slice(0, MAX_DOCUMENT_ROWS).map((n) => String(n.name)), total: matches.length };
}

// How many OTHER sources this one's entities are linked into — reuses the
// same _bridge flag computeHubInsight relies on for the graph-wide hub
// metric, just aggregated per source instead of per entity.
function countBridges(name: string, brainState: BrainState): number {
  const otherSources = new Set<string>();
  const addOthers = (sets: string[]): void => sets.forEach((s) => { if (s !== name) otherSources.add(s); });
  brainState.semanticLinks.forEach((l) => {
    if (!l._bridge) return;
    const sSets = setsOf(brainState.byId[l._sid]);
    const tSets = setsOf(brainState.byId[l._tid]);
    if (sSets.includes(name)) addOthers(tSets);
    else if (tSets.includes(name)) addOthers(sSets);
  });
  return otherSources.size;
}

// Everything a "which source is this" click could reasonably want beyond
// what the Sources rail tile already shows (entity/doc counts): a type
// breakdown (the type layer already tallies members per source, just never
// read in this direction), an actual document/record name sample (from the
// plumbing layer), and how many other sources this one bridges into (the
// hub metric's own _bridge flag, aggregated per source). All derived from
// BrainState the canvas already computed — no extra fetch.
export function computeSourceDetail(name: string, brainState: BrainState): SourceDetail {
  const { names, total } = buildDocumentNames(name, brainState);
  return {
    name,
    displayName: sourceLabel(name),
    color: brainState.setColor[name] || "#7E8CA6",
    entityCount: brainState.setEntityCount[name] || 0,
    docCount: brainState.setDocCount[name] || 0,
    typeBreakdown: buildTypeBreakdown(name, brainState),
    documentNames: names,
    documentTotal: total,
    bridgeCount: countBridges(name, brainState),
  };
}
