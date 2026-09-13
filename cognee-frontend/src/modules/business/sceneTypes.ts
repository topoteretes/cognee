import type { BusinessGraphNode, BusinessGraphLink } from "./types";

// An entity node with the simulation fields d3-force reads/writes and the
// canvas draw loop maintains. Cloned fresh per computeBrainState call (never
// the same object the wire payload used) so switching brains can't leak one
// brain's pinned position into another's.
export interface BusinessEntity extends BusinessGraphNode {
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
  vx?: number;
  vy?: number;
  _r?: number;
  // The radius businessEntityLayer last DREW this entity at, including its
  // breathe pulse and newborn scale — written every frame by the draw pass
  // and read by hitTestEntity, so the clickable disc is exactly the visible
  // one. `_r` above stays the simulation's collision radius; since CLO-604
  // the two are deliberately different curves.
  _rVisual?: number;
}

export interface SemanticLink extends BusinessGraphLink {
  _sid: string;
  _tid: string;
  _bridge?: boolean;
}

// The business-model (L0) layer: one node per entity type, floating at the
// centroid of its members.
export interface TypeNode {
  name: string;
  members: BusinessEntity[];
  sets: Record<string, number>;
  x?: number;
  y?: number;
  _r?: number;
  _visible?: BusinessEntity[];
}

export interface TypeLink {
  a: string;
  b: string;
  relation: string;
  count: number;
}

export interface Anchor {
  x: number;
  y: number;
}

// Everything the canvas needs about ONE brain (one governance+content merge),
// recomputed by computeBrainState whenever the enabled layers or the focused
// dataset change.
export interface BrainState {
  byId: Record<string, BusinessEntity | BusinessGraphNode>;
  entities: BusinessEntity[];
  entityById: Record<string, BusinessEntity>;
  sourceNames: string[];
  setColor: Record<string, string>;
  isSessionSet: (name: string) => boolean;
  setEntityCount: Record<string, number>;
  setDocCount: Record<string, number>;
  // Every real node tagged with a set, any stage — not just entity/document.
  // Some sources (e.g. Skill nodes, stage "other") never populate either of
  // the counts above, so a source can be fully loaded yet show 0 there; this
  // is what tells "genuinely still processing" apart from "has content, just
  // not entity/document-staged" (COG-6233).
  setMemberCount: Record<string, number>;
  semanticLinks: SemanticLink[];
  docLinks: SemanticLink[];
  anchors: Record<string, Anchor>;
  typeNodes: TypeNode[];
  typeLinks: TypeLink[];
  importanceMax: number;
  // The Records (plumbing) layer: every pipeline node the entity/type
  // layers never draw — chunks, documents, summaries, context — plus, for
  // each, the entity it actually links to, so the layer can draw it WHERE
  // that relationship is instead of on a meaningless hash-scatter ring.
  plumbingNodes: BusinessGraphNode[];
  plumbingEntityId: Record<string, string>;
  // Precomputed here (not per draw frame) — see DrawState in businessDraw.ts
  // for why: neither depends on x/y, only on the graph's own data.
  importanceCut: number;
  connectedIds: Set<string>;
  // The highest-degree entity — a "single point of failure" callout, same
  // idea as a graph-dashboard hub metric: whichever record the model relies
  // on most, and how many distinct sources it bridges. null on a graph too
  // small/sparse for the number to mean anything (degree < 2).
  hub: { entityId: string; name: string; degree: number; sourceCount: number } | null;
}
