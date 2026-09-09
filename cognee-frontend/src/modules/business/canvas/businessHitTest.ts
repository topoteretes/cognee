import type { BusinessEntity, TypeNode } from "../sceneTypes";

export interface Transform {
  x: number;
  y: number;
  k: number;
}

export function worldPoint(offsetX: number, offsetY: number, transform: Transform): [number, number] {
  return [(offsetX - transform.x) / transform.k, (offsetY - transform.y) / transform.k];
}

// Below this zoom, hovering/clicking hits the business-model (type) layer
// first; at or above it, entities are the direct target.
export const TYPE_NODE_ZOOM_THRESHOLD = 1.4;
const TYPE_NODE_HIT_PADDING = 8;
const TYPE_NODE_DEFAULT_RADIUS = 20;
// Screen-space 18px hit radius, converted to world units by dividing by k —
// so the target feels the same size on screen at any zoom level. It is a pad
// AROUND the entity's drawn disc, never a substitute for it: see entityRadius.
const ENTITY_HIT_RADIUS_PX = 18;

// The radius the entity layer actually DREW this frame (businessEntityLayer
// stores it per entity), falling back to the physics radius for an entity
// that hasn't been drawn yet. Since CLO-604 decoupled render size from
// physics size the two differ by up to ~5 world units, and hit-testing the
// physics radius left the visible rim of every mid-importance node dead to
// hover and click once ENTITY_HIT_RADIUS_PX/k shrank below that gap.
function entityRadius(n: BusinessEntity): number {
  return n._rVisual ?? n._r ?? 0;
}

export interface TypeNodeHit {
  kind: "type";
  node: TypeNode;
}
export interface EntityHit {
  kind: "entity";
  node: BusinessEntity;
}
export type SceneHit = TypeNodeHit | EntityHit | null;

export function hitTestTypeNode(typeNodes: TypeNode[], mx: number, my: number): TypeNode | null {
  let best: TypeNode | null = null;
  let bestDist = Infinity;
  typeNodes.forEach((tn) => {
    if (tn.x == null || tn.y == null) return;
    const d = Math.hypot(tn.x - mx, tn.y - my);
    if (d < (tn._r || TYPE_NODE_DEFAULT_RADIUS) + TYPE_NODE_HIT_PADDING && d < bestDist) {
      best = tn;
      bestDist = d;
    }
  });
  return best;
}

// bestD tightens as closer candidates are found — a later, farther entity
// can't unseat an already-closer one even if it would pass the initial
// screen-space threshold on its own.
//
// stickyId short-circuits the search to whatever's already hovered, as long
// as the cursor is still within ITS radius. Without this, re-running this
// same test every animation frame (BusinessCanvas's continuous re-hover,
// needed so a hover doesn't go stale while the simulation is still settling)
// lets two entities close enough to be near-tied in distance flip the
// winner from frame to frame — and since the hovered entity's label always
// wins pickNonOverlappingLabels' priority tie-break, that flip visibly
// swaps which neighboring label is showing every frame, reading as the
// whole hover trembling rather than holding steady.
export function hitTestEntity(
  entities: BusinessEntity[],
  mx: number,
  my: number,
  k: number,
  stickyId?: string | null,
): BusinessEntity | null {
  const threshold = ENTITY_HIT_RADIUS_PX / k;
  if (stickyId) {
    const current = entities.find((n) => n.id === stickyId);
    if (current && current.x != null && current.y != null) {
      const d = Math.hypot(current.x - mx, current.y - my);
      if (d < entityRadius(current) + threshold) return current;
    }
  }
  let best: BusinessEntity | null = null;
  let bestDist = threshold;
  entities.forEach((n) => {
    if (n.x == null || n.y == null) return;
    const d = Math.hypot(n.x - mx, n.y - my);
    if (d < entityRadius(n) + bestDist && d < (best ? bestDist : Infinity)) {
      best = n;
      bestDist = d;
    }
  });
  return best;
}

// CLO-606's agent marker (businessAgentPresence.ts) — a comfortable minimum
// screen-space target, converted to world units the same way
// ENTITY_HIT_RADIUS_PX is. It is a FLOOR, not the target itself: the marker
// pulses and grows an emphasis ring, and past k≈4.6 its drawn radius alone
// overtakes this, so the caller passes what it actually drew.
const AGENT_MARKER_MIN_HIT_RADIUS_PX = 14;

export function hitTestAgentMarker(
  markerWorld: { x: number; y: number },
  offsetX: number,
  offsetY: number,
  transform: Transform,
  drawnRadiusWorld: number,
): boolean {
  const [mx, my] = worldPoint(offsetX, offsetY, transform);
  const threshold = Math.max(AGENT_MARKER_MIN_HIT_RADIUS_PX / transform.k, drawnRadiusWorld);
  return Math.hypot(markerWorld.x - mx, markerWorld.y - my) < threshold;
}

// Shared by hover and click (customer_tutorial.html:6567 mousemove, :6269
// click) — the click handler in the export does no picking of its own, so a
// real implementation reuses this rather than adding new distance math.
export function hitTestScene(
  offsetX: number,
  offsetY: number,
  transform: Transform,
  typeNodes: TypeNode[],
  entities: BusinessEntity[],
  spotlightActive: boolean,
  stickyEntityId?: string | null,
  // The schema crossfade threshold for this graph's size (see
  // computeTypeFadeKMax) — the type layer must be hit-testable exactly when
  // it's drawn, and where it's drawn is fit-scale-relative, not a fixed k.
  typeThresholdK: number = TYPE_NODE_ZOOM_THRESHOLD,
): SceneHit {
  const [mx, my] = worldPoint(offsetX, offsetY, transform);
  if (transform.k < typeThresholdK && !spotlightActive) {
    const typeHit = hitTestTypeNode(typeNodes, mx, my);
    if (typeHit) return { kind: "type", node: typeHit };
  }
  const entityHit = hitTestEntity(entities, mx, my, transform.k, stickyEntityId);
  return entityHit ? { kind: "entity", node: entityHit } : null;
}
