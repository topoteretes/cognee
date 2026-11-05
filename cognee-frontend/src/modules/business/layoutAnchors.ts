import type { Anchor } from "./sceneTypes";

// The layout's view of one entity: which sets it sits in and which sub-group
// of its territory it belongs to (null = the territory's own center). Kept to
// this shape so the module needs nothing from computeBrainState.
export interface LayoutEntity {
  id: string;
  sets: string[];
  group: string | null;
}

export interface LayoutAnchors {
  // One gravity point per source / session set, also the fallback target of
  // a source's filament before any of its entities has settled.
  anchors: Record<string, Anchor>;
  // Where the simulation pulls each entity: its sets' mean anchor, shifted
  // toward its group's sector of the territory.
  entityAnchors: Record<string, Anchor>;
}

const MIN_SOURCE_RING_RADIUS = 300;
const SOURCE_RING_X_SCALE = 1.25;
const SOURCE_RING_Y_SCALE = 0.6;
const SESSION_ARC_X = 480;
const SESSION_ARC_Y = 430;
const SESSION_ARC_STEP = 0.55;
const SESSION_ARC_DROP = 60;
// A territory's radius grows with the square root of its entity count (area
// is what scales with count), and its groups sit on a ring inside it.
const TERRITORY_UNIT = 12;
const GROUP_RING_FRACTION = 0.55;
const ORIGIN: Anchor = { x: 0, y: 0 };

function territoryRadius(entityCount: number): number {
  return TERRITORY_UNIT * Math.sqrt(entityCount);
}

// Which set an entity's group offset is measured in: the first real source,
// else its first set (a session-only entity).
function primarySet(sets: string[], isSessionSet: (name: string) => boolean): string | undefined {
  return sets.find((s) => !isSessionSet(s)) ?? sets[0];
}

// The source ring has to grow with its territories, or two big sources end up
// closer together than their own radii and their territories interleave.
// Neighbours must clear the sum of their radii along the ring's flattest
// direction (the y squash), hence the division by that scale. Session sets
// are indexed on the ring but then placed on their own arc (see
// buildSetAnchors), so a pair involving one does not constrain the ring.
function sourceRingRadius(
  sourceNames: string[],
  countOf: (name: string) => number,
  isSessionSet: (name: string) => boolean,
): number {
  if (sourceNames.length < 2) return 0;
  const chord = 2 * Math.sin(Math.PI / sourceNames.length) * SOURCE_RING_Y_SCALE;
  const needed = sourceNames.map((name, i) => {
    const next = sourceNames[(i + 1) % sourceNames.length];
    if (isSessionSet(name) || isSessionSet(next)) return 0;
    return (territoryRadius(countOf(name)) + territoryRadius(countOf(next))) / chord;
  });
  return Math.max(MIN_SOURCE_RING_RADIUS, ...needed);
}

// Circular layout for sources; session-memory sets get their own arc BELOW
// the sources, since without an anchor those entities all pull to (0,0) and
// bury the center.
function buildSetAnchors(
  sourceNames: string[],
  sessionNames: string[],
  ringRadius: number,
): Record<string, Anchor> {
  const anchors: Record<string, Anchor> = {};
  sourceNames.forEach((name, i) => {
    const angle = (i / Math.max(sourceNames.length, 1)) * Math.PI * 2 - Math.PI / 2;
    anchors[name] = {
      x: Math.cos(angle) * ringRadius * SOURCE_RING_X_SCALE,
      y: Math.sin(angle) * ringRadius * SOURCE_RING_Y_SCALE,
    };
  });
  sessionNames.forEach((name, i) => {
    const spread = (i - (sessionNames.length - 1) / 2) * SESSION_ARC_STEP;
    anchors[name] = { x: Math.sin(spread) * SESSION_ARC_X, y: SESSION_ARC_Y + Math.abs(spread) * SESSION_ARC_DROP };
  });
  return anchors;
}

function meanAnchor(sets: string[], anchors: Record<string, Anchor>): Anchor {
  if (!sets.length) return ORIGIN;
  const sum = sets.reduce((acc, s) => {
    const a = anchors[s] ?? ORIGIN;
    return { x: acc.x + a.x, y: acc.y + a.y };
  }, ORIGIN);
  return { x: sum.x / sets.length, y: sum.y / sets.length };
}

// `groupOrder` is the full, fixed list of group names: an entity's sector is
// its index in it, so the sector of an existing entity never depends on which
// other groups currently exist.
export function buildLayoutAnchors(
  entities: LayoutEntity[],
  groupOrder: string[],
  sourceNames: string[],
  sessionNames: string[],
  isSessionSet: (name: string) => boolean,
): LayoutAnchors {
  const territoryCount: Record<string, number> = {};
  entities.forEach((e) => {
    const territory = primarySet(e.sets, isSessionSet);
    if (territory) territoryCount[territory] = (territoryCount[territory] || 0) + 1;
  });
  const ringRadius = sourceRingRadius(sourceNames, (name) => territoryCount[name] || 0, isSessionSet);
  const anchors = buildSetAnchors(sourceNames, sessionNames, ringRadius);

  const entityAnchors: Record<string, Anchor> = {};
  entities.forEach((e) => {
    const base = meanAnchor(e.sets, anchors);
    const territory = primarySet(e.sets, isSessionSet);
    const index = e.group ? groupOrder.indexOf(e.group) : -1;
    if (!territory || index < 0) {
      entityAnchors[e.id] = base;
      return;
    }
    const angle = (index / groupOrder.length) * Math.PI * 2 - Math.PI / 2;
    const radius = GROUP_RING_FRACTION * territoryRadius(territoryCount[territory]);
    entityAnchors[e.id] = { x: base.x + Math.cos(angle) * radius, y: base.y + Math.sin(angle) * radius };
  });
  return { anchors, entityAnchors };
}
