import { mean } from "d3-array";
import type { BusinessEntity, TypeNode, TypeLink, SemanticLink } from "../sceneTypes";
import type { BusinessGraphNode } from "../types";
import type { Transform } from "./businessHitTest";
import { drawTypeModelLayer } from "./businessTypeLayer";
import { inLens, drawEntities, computeVisibleEntityIds } from "./businessEntityLayer";
import { drawDocumentBadges, drawPlumbingLayer } from "./businessAuxLayers";
import { drawSourceHulls } from "./businessSourceHulls";
import { quadraticPoint, hashSeed } from "./businessCanvasHelpers";
import { edgeKey } from "./businessGraphFocus";

// Who opened this spotlight. Four producers set `spotlight` with four
// different window widths and only one of them has an agent behind it, so
// anything that annotates a spotlight (CLO-606's agent marker) has to be
// able to tell them apart — "some spotlight is active" is not the same
// question as "the asking agent's answer is on screen".
export type SpotlightSource = "answer" | "trace" | "insight" | "whatIf";

export interface Spotlight {
  ids: Set<string>;
  // Both bounds, not just `until`: the three producers use three different
  // window widths (9s answer, 1.8s trace step, 12s what-if), so a consumer
  // that needs "how far into this window are we" can only get it by
  // assuming one — which is exactly how the marker's fade envelope ended up
  // hardcoding 9000ms and reading a negative elapsed on the 12s window.
  startedAt: number;
  until: number;
  source: SpotlightSource;
  question?: string;
}

export interface DrawState {
  transform: Transform;
  level: number;
  entities: BusinessEntity[];
  typeNodes: TypeNode[];
  typeLinks: TypeLink[];
  semanticLinks: SemanticLink[];
  docLinks: SemanticLink[];
  byId: Record<string, BusinessEntity | BusinessGraphNode>;
  setColor: Record<string, string>;
  sourceNames: string[];
  newbornAt: Record<string, number>;
  hoveredId: string | null;
  selectedId: string | null;
  spotlight: Spotlight | null;
  focusSets: Set<string> | null;
  reducedMotion: boolean;
  importanceMax: number;
  isSessionSet: (name: string) => boolean;
  // Every entity id that has EVER answered a live search this session — a
  // lasting "this has been useful" mark, distinct from spotlight's current
  // 9s window (see businessEntityLayer's drawEntities).
  answeredIds: Set<string>;
  // The shortest path between two shift-clicked entities (useShortestPath) —
  // empty sets when no path is active. edgeKeys use businessGraphFocus's
  // undirected edgeKey so a rendered link's (source,target) order never
  // matters for the lookup.
  pathIds: Set<string>;
  pathEdgeKeys: Set<string>;
  // Records-level inputs (see BrainState) — each pipeline leftover node and
  // the entity it links to, so drawPlumbingLayer places it meaningfully.
  plumbingNodes: BusinessGraphNode[];
  plumbingEntityId: Record<string, string>;
  // Where the schema crossfade sits for THIS graph's size — computed by
  // BusinessCanvas from the current fit scale (computeTypeFadeKMax), since
  // only it knows the viewport dimensions.
  typeFadeKMax: number;
  // Precomputed once per brainState (computeBrainState) instead of being
  // resorted/rebuilt every draw frame — importanceCut only depends on
  // entities' importance values, connectedIds only on semanticLinks, and
  // neither changes between frames of the same graph.
  importanceCut: number;
  connectedIds: Set<string>;
}

// Crossfade window between the model (type) layer and the instance layer.
// The threshold is relative to the graph's own fit scale, not absolute zoom:
// a fixed k=1.55 assumed a small demo graph that fits the viewport around
// that zoom, but a large ingested graph only fits at a much lower k — so
// zooming out to see all of it crossed the fixed threshold long before the
// whole graph was visible, and the schema "arrived too early" (reviewer 2b),
// making it impossible to ever see the entire entity graph at once.
const TYPE_FADE_K_MAX_CAP = 1.55;
const TYPE_FADE_K_MIN = 0.4;
const TYPE_FADE_WINDOW_RATIO = 0.16;

// The k below which the schema starts crossfading in, for a graph whose
// fit-to-viewport scale is kFit (see useBusinessCamera.computeFitScale):
// just under the fit scale, so the full entity graph is reachable first.
export function computeTypeFadeKMax(kFit: number): number {
  return Math.min(TYPE_FADE_K_MAX_CAP, Math.max(TYPE_FADE_K_MIN, kFit * 0.9));
}

// How present the instance (entity) layer is at this zoom: 1 fully drawn,
// 0 fully replaced by the schema layer, fractional mid-crossfade. Exported
// because the source filaments are drawn OUTSIDE draw() (screen space,
// before the camera transform — see BusinessCanvas) yet must fade with the
// entities they point at, the same way drawSourceHulls already does. Left
// ungated, they kept threading at full strength to entity clusters the
// schema view no longer renders, reading as lines crossing the canvas
// toward nothing.
export function computeInstanceAlpha(
  transformK: number,
  typeFadeKMax: number,
  hasTypeNodes: boolean,
  spotlightActive: boolean,
): number {
  if (spotlightActive || !hasTypeNodes) return 1;
  const fadeWindow = Math.max(0.08, typeFadeKMax * TYPE_FADE_WINDOW_RATIO);
  const typeAlpha = Math.max(0, Math.min(1, (typeFadeKMax - transformK) / fadeWindow));
  return 1 - typeAlpha;
}

const OVERLAP_RELAX_PASSES = 12;
// Labels need ~40px of clearance beyond the two circles just touching.
const OVERLAP_MIN_GAP = 46;

// Recomputes each type node's centroid from its (possibly lens-filtered)
// members every frame — cheap relative to the simulation tick, and keeps the
// model layer honest about which entities are actually in view. The
// relaxation pass afterward (customer_tutorial.html ~7160) pushes apart any
// pair of type circles still overlapping after re-centroiding — with many
// distinct types (a code-analysis dataset can have a dozen-plus) their
// member centroids often land close together, and without this the model
// layer itself becomes the "too many overlapping circles" clutter, not just
// the links between them.
function layoutTypeNodes(typeNodes: TypeNode[], focusSets: Set<string> | null): void {
  typeNodes.forEach((tn) => {
    const visible = focusSets ? tn.members.filter((m) => inLens(m, focusSets)) : tn.members;
    tn._visible = visible;
    const pool = visible.length ? visible : tn.members;
    tn.x = mean(pool, (m) => m.x ?? 0) || 0;
    tn.y = mean(pool, (m) => m.y ?? 0) || 0;
    tn._r = 14 + 7 * Math.log2(1 + (visible.length || tn.members.length));
  });
  for (let pass = 0; pass < OVERLAP_RELAX_PASSES; pass++) {
    for (let i = 0; i < typeNodes.length; i++) {
      for (let j = i + 1; j < typeNodes.length; j++) {
        const a = typeNodes[i], b = typeNodes[j];
        const dx = (b.x ?? 0) - (a.x ?? 0), dy = (b.y ?? 0) - (a.y ?? 0);
        const dist = Math.hypot(dx, dy) || 1;
        const min = (a._r ?? 0) + (b._r ?? 0) + OVERLAP_MIN_GAP;
        if (dist < min) {
          const push = (min - dist) / 2;
          const ux = dx / dist, uy = dy / dist;
          a.x = (a.x ?? 0) - ux * push;
          a.y = (a.y ?? 0) - uy * push;
          b.x = (b.x ?? 0) + ux * push;
          b.y = (b.y ?? 0) + uy * push;
        }
      }
    }
  }
}

// The source's own demo graphs never exceed a few dozen semantic links, so
// drawing every one at a fixed opacity never had to reckon with volume. A
// real ingested dataset (a code repo, years of CRM activity) can have
// thousands — at a fixed opacity those don't read as "connected", they read
// as a solid hairball. Fading in proportion to volume keeps a small graph's
// links exactly as visible as before (no fade below the threshold) while a
// dense one recedes into a texture instead of dominating the canvas.
const DENSE_LINK_THRESHOLD = 400;
// Lowered from 0.15 and squared below (UX audit, 2026-08-21): a several-
// hundred-entity ingested dataset routinely clears several thousand semantic
// links, and the original linear threshold/count falloff still floored at
// 15% opacity — visible enough that the RESTING (nothing hovered/selected)
// canvas still read as a hairball on staging screenshots. Squaring the ratio
// falls off much faster past the threshold while leaving anything AT or
// below it (fade=1, untouched) exactly as before; the lower floor only
// matters once the squared term already undershoots it. Highlighted links
// (hover/selection/spotlight/path) bypass this fade entirely — see
// `highlighted ? 1 : ...` below — so this only mutes the idle state, never
// interactivity. Starting values, not measured against a specific dataset —
// tune against a real dense screenshot before treating these as final.
const MIN_DENSITY_FADE = 0.06;
const DENSITY_FADE_EXPONENT = 2;
const PARTICLE_TRAVEL_MS = 2200;

// Exported so it can be unit-tested without reaching into the draw loop
// (drawSemanticLinks itself isn't exported — see computeTypeFadeKMax above
// for the same "pull the pure formula out" pattern in this file).
export function computeLinkDensityFade(linkCount: number): number {
  if (linkCount <= DENSE_LINK_THRESHOLD) return 1;
  return Math.max(MIN_DENSITY_FADE, Math.pow(DENSE_LINK_THRESHOLD / linkCount, DENSITY_FADE_EXPONENT));
}

// Ports the semantic-link draw block (customer_tutorial.html ~7180): a
// slight quadratic-curve bow (perpendicular offset = 0.12 * length), not a
// straight line — and three distinct treatments, not two. Amber is reserved
// for a link between two entities BOTH in the current spotlight (the same
// "amber = live signal" convention colorForSet already guards); a bridge
// (cross-source) link gets a bone/white tone, never amber — this was
// drawing bridges in amber before, which quietly broke that convention.
//
// A fourth, source-less treatment: any link touching the CLICKED entity
// (selectedId) draws in the same bone tone as its selection ring, at full
// opacity regardless of density fade or spotlight dimming — click-to-select
// itself has no source equivalent (new in this port), and without this a
// selected node's ring had nothing showing what it actually connects to.
// A fifth, green treatment marks the shortest-path highlight (also new,
// see useShortestPath) and wins over all the others — it's the one thing
// the user explicitly asked to trace, so it should never lose to "this
// link also happens to be in the live spotlight."
function drawLinkRelationLabel(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  transformK: number,
): void {
  const fontSize = 10 / Math.sqrt(transformK);
  ctx.textAlign = "center";
  ctx.font = `500 ${fontSize}px ui-monospace, monospace`;
  const tw = ctx.measureText(text).width;
  ctx.fillStyle = "rgba(14,21,38,0.88)";
  ctx.beginPath();
  ctx.roundRect(x - tw / 2 - 5, y - fontSize, tw + 10, fontSize + 4, 7);
  ctx.fill();
  ctx.fillStyle = "rgba(233,238,246,0.9)";
  ctx.fillText(text, x, y);
}

function drawSemanticLinks(
  ctx: CanvasRenderingContext2D,
  state: DrawState,
  entityById: Record<string, BusinessEntity>,
  alpha: number,
  dimmed: number,
  now: number,
  visibleEntityIds: Set<string>,
): void {
  const spotIds = state.spotlight && now < state.spotlight.until ? state.spotlight.ids : null;
  const densityFade = computeLinkDensityFade(state.semanticLinks.length);
  state.semanticLinks.forEach((l) => {
    const s = entityById[l._sid], t = entityById[l._tid];
    if (!s || !t || s.x == null || t.x == null || s.y == null || t.y == null) return;
    // Same gate drawEntities uses to decide whether an endpoint is actually
    // drawn (focus lens, importance cut, newborn fade) — a link to a hidden
    // endpoint used to still draw at full length, reading as a loose,
    // unconnected line to nothing.
    if (!visibleEntityIds.has(s.id) || !visibleEntityIds.has(t.id)) return;
    const inSpot = spotIds ? spotIds.has(s.id) && spotIds.has(t.id) : false;
    const touchesSelected = state.selectedId !== null && (s.id === state.selectedId || t.id === state.selectedId);
    const inPath = state.pathEdgeKeys.has(edgeKey(l._sid, l._tid));
    const highlighted = inPath || inSpot || touchesSelected;
    // Focus-mode: once something is selected, links that neither touch it
    // nor sit on an active path recede further — the same "everyone but
    // the neighborhood fades" the entity layer applies below.
    const focusDim = state.selectedId !== null && !highlighted ? 0.25 : 1;
    const dx = t.x - s.x, dy = t.y - s.y;
    const mx = (s.x + t.x) / 2 - dy * 0.12, my = (s.y + t.y) / 2 + dx * 0.12;
    ctx.beginPath();
    ctx.moveTo(s.x, s.y);
    ctx.quadraticCurveTo(mx, my, t.x, t.y);
    ctx.globalAlpha = highlighted ? 1 : alpha * densityFade * focusDim;
    if (inPath) {
      ctx.strokeStyle = "#56DB7D";
      ctx.lineWidth = 2 / state.transform.k;
    } else if (inSpot) {
      ctx.strokeStyle = "#F5A83C";
      ctx.lineWidth = 1.8 / state.transform.k;
    } else if (touchesSelected) {
      ctx.strokeStyle = "#E9EEF6";
      ctx.lineWidth = 1.8 / state.transform.k;
    } else if (l._bridge) {
      ctx.strokeStyle = `rgba(233,238,246,${0.42 * dimmed})`;
      ctx.lineWidth = 1.4 / state.transform.k;
    } else {
      ctx.strokeStyle = `rgba(126,140,166,${0.4 * dimmed})`;
      ctx.lineWidth = 1.1 / state.transform.k;
    }
    ctx.stroke();
    // The relation name — invisible at the instance layer otherwise — only
    // for links a click or a path actually calls out, so it answers "how
    // are these two connected" instead of adding more label soup.
    if ((touchesSelected || inPath) && l.relation) {
      const [lx, ly] = quadraticPoint(s.x, s.y, mx, my, t.x, t.y, 0.5);
      drawLinkRelationLabel(ctx, String(l.relation).replace(/_/g, " "), lx, ly, state.transform.k);
    }
    // A single traveling glint per link — new in this port (no source
    // equivalent), the same "data is flowing" language as the source-card
    // filaments, just spread across the whole model instead of only the
    // rail. Reuses the stroke's own just-set globalAlpha (density fade,
    // selection/spotlight override, all already baked in) and only further
    // dims it with a smooth sine pulse, so it never reads brighter than the
    // line it rides on. Skipped above the same density threshold that
    // fades the lines themselves — a hairball doesn't need more motion.
    if (!state.reducedMotion && state.semanticLinks.length <= DENSE_LINK_THRESHOLD) {
      const travelT = ((now / PARTICLE_TRAVEL_MS + hashSeed(l._sid + l._tid)) % 1 + 1) % 1;
      const [px, py] = quadraticPoint(s.x, s.y, mx, my, t.x, t.y, travelT);
      ctx.globalAlpha = ctx.globalAlpha * Math.sin(travelT * Math.PI);
      ctx.beginPath();
      ctx.arc(px, py, 1.3 / state.transform.k, 0, Math.PI * 2);
      ctx.fillStyle = inPath ? "#56DB7D" : inSpot ? "#F5A83C" : touchesSelected ? "#E9EEF6" : "#43D9E8";
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  });
}

export function draw(
  ctx: CanvasRenderingContext2D,
  state: DrawState,
  now: number,
  // See businessEntityLayer.drawEntities — cleared unconditionally here (not
  // inside drawEntities) because at low zoom entities aren't drawn at all
  // (the type-model layer replaces them, instAlpha <= 0.01 below), and a
  // stale id from the last frame they WERE drawn would still hit-test.
  visibleIdsOut?: Set<string>,
): void {
  visibleIdsOut?.clear();
  // Ports `const spot = spotlight && now < spotlight.until ? spotlight :
  // (spotlight = null)` (customer_tutorial.html ~7130) — source recomputes
  // this expiry check every frame; checking raw `state.spotlight` truthiness
  // instead meant that once a single search event had EVER fired (including
  // the replay-on-load every session gets), the type layer stayed
  // permanently suppressed and entities stayed permanently 16%-dimmed for
  // the rest of the session, long after the spotlight's actual 9s window.
  const spotlightActive = state.spotlight !== null && now < state.spotlight.until;
  // A graph with no type layer (no is_a/instance_of links produced any type
  // nodes) has nothing to crossfade INTO — fading entities out at far zoom
  // would leave an empty canvas, so they stay fully drawn at every zoom.
  const instAlpha = computeInstanceAlpha(
    state.transform.k, state.typeFadeKMax, state.typeNodes.length > 0, spotlightActive,
  );
  const typeAlpha = 1 - instAlpha;
  const dimmed = spotlightActive ? 0.16 : 1;

  layoutTypeNodes(state.typeNodes, state.focusSets);
  // Hulls fade out with the entity layer they shade: territory polygons
  // span the ENTITY extents, so left at full strength under the schema view
  // they dwarfed the compact type cluster (reviewer 2b's screenshot) —
  // giant colored areas around a tiny model.
  if (state.level <= 1 && instAlpha > 0.01) {
    drawSourceHulls(
      ctx, state.entities, state.sourceNames, state.setColor, state.focusSets,
      dimmed * instAlpha, state.transform.k,
    );
  }
  if (typeAlpha > 0.01) {
    drawTypeModelLayer(
      ctx, state.typeNodes, state.typeLinks, state.setColor, state.isSessionSet,
      typeAlpha, dimmed, state.focusSets, state.transform.k,
    );
  }
  if (instAlpha > 0.01) {
    const entityById: Record<string, BusinessEntity> = {};
    state.entities.forEach((n) => { entityById[n.id] = n; });
    // Computed only in this branch: below it, the type-model layer replaces
    // entities entirely, so nothing here is actually on screen to hit-test
    // against — visibleIdsOut stays at the empty clear() above instead of
    // reporting entities as visible that this frame never drew.
    const visibleEntityIds = computeVisibleEntityIds(state, now);
    visibleEntityIds.forEach((id) => visibleIdsOut?.add(id));
    drawSemanticLinks(ctx, state, entityById, instAlpha, dimmed, now, visibleEntityIds);
    // level 3 is plumbing (BusinessCanvas folds the toggle into level
    // before calling draw), so level >= 2 already covers source's
    // "level >= 2 || plumbing".
    if (state.level >= 2) drawDocumentBadges(ctx, state, entityById);
    drawEntities(ctx, state, instAlpha, dimmed, now, visibleEntityIds);
  }
  // Plumbing is a toggle, not a zoom level — level 3 means it's on
  // (BusinessCanvas folds plumbingRef into level before calling draw), so
  // this only fires on that exact toggle, not merely "zoomed to level 2+".
  if (state.level === 3) {
    const entityById: Record<string, BusinessEntity> = {};
    state.entities.forEach((n) => { entityById[n.id] = n; });
    drawPlumbingLayer(ctx, state.plumbingNodes, state.plumbingEntityId, entityById, state.transform.k, state.focusSets);
  }
}
