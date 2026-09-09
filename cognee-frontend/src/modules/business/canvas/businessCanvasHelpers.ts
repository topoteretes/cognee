import type { BrainState } from "../sceneTypes";
import type { Spotlight } from "./businessDraw";

// Ports the hover handler's `!spotlight` check (customer_tutorial.html
// ~7280) — same expiry bug as draw()'s crossfade: `spotlight !== null`
// checks the state OBJECT's presence, not whether its 9s window has
// actually elapsed, so type-node hover stayed disabled for the rest of the
// session after the first search event fired.
export function isSpotlightActive(spotlight: Spotlight | null): boolean {
  return spotlight !== null && performance.now() < spotlight.until;
}

// The rail a source/operator/dataset card sits in (ScrollFadeContainer) can
// overflow-scroll — a card below the fold is still mounted with a real
// layout position, just clipped from view by its ancestor's overflow, not
// removed. getBoundingClientRect() reports that real (invisible) position
// regardless, so a filament to a scrolled-past card used to draw from
// wherever it happens to sit — often well past the rail's own visible
// bottom edge — reading as a line crossing the canvas from empty space with
// no visible card at its start. Cached per element (not recomputed every
// frame) since the scrollable ancestor relationship never changes for a
// given DOM node — only getBoundingClientRect (already called every frame
// regardless) needs to run per frame.
const scrollAncestorCache = new WeakMap<HTMLElement, HTMLElement | null>();

function findScrollAncestor(el: HTMLElement): HTMLElement | null {
  const cached = scrollAncestorCache.get(el);
  if (cached !== undefined) return cached;
  let node = el.parentElement;
  while (node) {
    const overflowY = getComputedStyle(node).overflowY;
    if (overflowY === "auto" || overflowY === "scroll") {
      scrollAncestorCache.set(el, node);
      return node;
    }
    node = node.parentElement;
  }
  scrollAncestorCache.set(el, null);
  return null;
}

export function cardScreenPosition(
  cardMap: Record<string, HTMLElement | null> | undefined,
  key: string,
  canvasRect: DOMRect,
): { x: number; y: number } | null {
  const el = cardMap?.[key];
  if (!el) return null;
  const r = el.getBoundingClientRect();
  const scrollAncestor = findScrollAncestor(el);
  if (scrollAncestor) {
    const containerRect = scrollAncestor.getBoundingClientRect();
    const centerY = r.top + r.height / 2;
    if (centerY < containerRect.top || centerY > containerRect.bottom) return null;
  }
  return { x: r.right - canvasRect.left, y: r.top - canvasRect.top + r.height / 2 };
}

// Point at t along the SAME quadratic curve `ctx.quadraticCurveTo` draws —
// shared by the type-model layer's link labels and the semantic-link
// traveling particles, both of which need a point ON the exact curve
// they're drawing, not an approximation.
export function quadraticPoint(
  ax: number, ay: number, mx: number, my: number, bx: number, by: number, t: number,
): [number, number] {
  const u = 1 - t;
  return [u * u * ax + 2 * u * t * mx + t * t * bx, u * u * ay + 2 * u * t * my + t * t * by];
}

// Stable per-key phase in [0, 1) so animated elements sharing a cycle
// (filament particles, link particles) don't all move in lockstep.
export function hashSeed(key: string): number {
  let h = 0;
  for (let i = 0; i < key.length; i++) h = (h * 31 + key.charCodeAt(i)) | 0;
  return Math.abs(h % 1000) / 1000;
}

export const EMPTY_ANSWERED_IDS: Set<string> = new Set();
export const EMPTY_PATH_IDS: Set<string> = new Set();
export const EMPTY_PATH_EDGE_KEYS: Set<string> = new Set();

export const EMPTY_BRAIN_STATE: Pick<
  BrainState,
  | "entities" | "typeNodes" | "typeLinks" | "semanticLinks" | "docLinks" | "byId" | "setColor" | "sourceNames"
  | "anchors" | "importanceMax" | "isSessionSet" | "plumbingNodes" | "plumbingEntityId"
  | "importanceCut" | "connectedIds"
> = {
  entities: [],
  typeNodes: [],
  typeLinks: [],
  semanticLinks: [],
  docLinks: [],
  byId: {},
  setColor: {},
  sourceNames: [],
  anchors: {},
  importanceMax: 1,
  isSessionSet: () => false,
  plumbingNodes: [],
  plumbingEntityId: {},
  importanceCut: 0,
  connectedIds: new Set(),
};
