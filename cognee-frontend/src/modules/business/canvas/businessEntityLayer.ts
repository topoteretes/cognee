import type { BusinessEntity } from "../sceneTypes";
import { setsOf } from "../computeBrainState";
import { pickNonOverlappingLabels, type LabelCandidate, type Box } from "./businessLabelLayout";
import { computeNeighborIds } from "./businessGraphFocus";
import type { DrawState } from "./businessDraw";

export function inLens(n: BusinessEntity, focusSets: Set<string> | null): boolean {
  return !focusSets || setsOf(n).some((s) => focusSets.has(s));
}

// A handful of entities carry a real but sentence-length name (session
// summaries, ticket descriptions) rather than a short label — is_unnamed
// alone doesn't catch those. Truncating defends the canvas the same way
// businessTypeLayer's MAX_LABELS_PER_PAIR does for relation-label soup.
const MAX_LABEL_CHARS = 44;

function truncateLabel(name: string): string {
  return name.length > MAX_LABEL_CHARS ? `${name.slice(0, MAX_LABEL_CHARS - 1)}…` : name;
}

// Screen-space legibility floor (UX audit, 2026-08-20): on a several-hundred-
// entity dataset, computeFitScale's default landing k lands around 0.4-0.8,
// which draws _r (5-16 world px) as 2-6 screen px and the label font as
// ~6-10 screen px — sub-legible on first paint, before the user has zoomed
// at all. Bumping _r itself would be self-defeating: it feeds
// forceCollide's radius (useBusinessSimulation.ts), so a larger _r spreads
// the layout and lowers kFit by roughly the same amount, mostly cancelling
// out. A screen-space floor only kicks in at the low-k end where the
// problem actually is, and leaves normal zoomed-in sizing (k >= ~1) alone.
export const MIN_NODE_SCREEN_PX = 4.5;
const MIN_LABEL_SCREEN_PX = 10;

// Render-only importance-to-radius curve (UX audit, 2026-08-21): `_r`
// (useBusinessSimulation.ts, 5-16 world px) also feeds forceCollide and the
// camera's fit scale, so widening it there to make importance visible
// spreads the whole layout and lowers kFit by roughly the same amount — the
// same self-cancelling trade that file's own comment already rules out for
// MIN_NODE_SCREEN_PX above. A dense cluster still needs SOME node to look
// bigger than another, so this recomputes a second, physics-independent
// radius purely for drawing: a wider range (5-20 vs 5-16) and a sub-linear
// exponent (0.6) that lifts mid-importance entities further off the
// MIN_NODE_SCREEN_PX floor, instead of the near-linear curve collapsing most
// of a power-law-distributed dataset into the same floored size. Starting
// values, not measured against a specific dataset — tune against a real
// dense screenshot before treating these as final.
const RENDER_MIN_R = 5;
export const RENDER_MAX_R = 20;
const RENDER_IMPORTANCE_EXPONENT = 0.6;

function visualRadius(importance: number, importanceMax: number): number {
  const ratio = importanceMax > 0 ? Math.max(0, Math.min(1, importance / importanceMax)) : 0;
  return RENDER_MIN_R + (RENDER_MAX_R - RENDER_MIN_R) * Math.pow(ratio, RENDER_IMPORTANCE_EXPONENT);
}

// Halo behind each label — a plain fill alone read illegibly against
// whatever happened to be behind it (a same-toned node fill, another node's
// color, the dark background at low contrast). Stroked BEFORE the fill in
// the same font/position so it reads as an outline, not a background box —
// the relation-label callout (drawLinkRelationLabel above) already solves
// this with a filled rounded rect, but that doesn't fit a label sitting
// directly below a small circle without covering neighboring content.
const LABEL_HALO_RGB = "14,21,38";
const LABEL_HALO_ALPHA = 0.85;
const LABEL_HALO_WIDTH_RATIO = 0.28;

// Ports topImportanceCut (customer_tutorial.html ~7040): below L1, a
// curated dataset (≤150 entities) still shows every one, but a large
// ingested one only shows its top ~100 by importance — otherwise every dot
// AND every label draws at once below the type-layer crossfade, which is
// exactly the "too many points" clutter the label-collision pass alone
// doesn't fix (that only thins labels, not the dots themselves).
const DENSE_ENTITY_THRESHOLD = 150;
const TOP_IMPORTANCE_RANK = 99;

export function topImportanceCut(entities: BusinessEntity[]): number {
  if (entities.length <= DENSE_ENTITY_THRESHOLD) return 0;
  const sorted = entities.map((n) => n.importance || 0).sort((a, b) => b - a);
  return sorted[TOP_IMPORTANCE_RANK] || 0;
}

// The single source of truth for "does this entity actually render this
// frame" — the same three gates drawEntities applies below (focus lens,
// below-L1 importance cut, not-yet-faded-in newborn), extracted so
// drawSemanticLinks can skip a link the moment either endpoint is one of
// these. Without this, a link still drew at full length to whichever
// endpoint the gates hid, reading as a loose, unconnected line — reported
// as "entities load at different times, leaving edges looking unconnected."
export function computeVisibleEntityIds(state: DrawState, now: number): Set<string> {
  const spotIds = state.spotlight && now < state.spotlight.until ? state.spotlight.ids : null;
  const showAll = state.level >= 1;
  const importanceCut = state.importanceCut;
  const visible = new Set<string>();
  state.entities.forEach((n) => {
    if (n.x == null || n.y == null) return;
    const inSpot = spotIds?.has(n.id) ?? false;
    if (!inLens(n, state.focusSets) && !inSpot) return;
    if (!showAll && (n.importance || 0) < importanceCut && !inSpot) return;
    const { alpha: bornAlpha } = newbornAlphaAndScale(now, state.newbornAt[n.id], state.reducedMotion);
    if (bornAlpha <= 0) return;
    visible.add(n.id);
  });
  return visible;
}

// Ports hash() (customer_tutorial.html ~7100) — a stable per-id phase in
// [0, 2π), used here for the breathing animation's per-entity offset and
// (via businessDraw's drawDocumentBadges) for scattering document badges.
export function hashPhase(id: string): number {
  let h = 0;
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) | 0;
  return (Math.abs(h) / 2147483647) * 6.28;
}

const NEWBORN_FADE_MS = 700;

function newbornAlphaAndScale(
  now: number,
  bornAt: number | undefined,
  reducedMotion: boolean,
): { alpha: number; scale: number } {
  if (reducedMotion || bornAt === undefined || now >= bornAt + NEWBORN_FADE_MS) return { alpha: 1, scale: 1 };
  if (now < bornAt) return { alpha: 0, scale: 0 };
  const t = (now - bornAt) / NEWBORN_FADE_MS;
  return { alpha: t, scale: Math.min(1, t) };
}

interface EntityLabelMeta {
  text: string;
  x: number;
  y: number;
  fontSize: number;
  fade: number;
}

export function drawEntities(
  ctx: CanvasRenderingContext2D,
  state: DrawState,
  alpha: number,
  dimmed: number,
  now: number,
  // The frame's visible set (see computeVisibleEntityIds) — BusinessCanvas
  // hit-tests against the same set the caller passes in here, so hovering a
  // record the lens/importance-cut/newborn-fade hid never shows its tooltip.
  visibleEntityIds: Set<string>,
): void {
  const spotIds = state.spotlight && now < state.spotlight.until ? state.spotlight.ids : null;
  // Only for the label-priority cutoff below (L0 labels well-above-average
  // entities even when not spotlighted) — the draw/hide decision itself now
  // comes entirely from visibleEntityIds. Precomputed once per brainState
  // (see computeBrainState) rather than resorting every frame.
  const importanceCut = state.importanceCut;
  const candidates: LabelCandidate[] = [];
  const labelMeta: Record<string, EntityLabelMeta> = {};
  // Fed to pickNonOverlappingLabels as obstacles below — a label overlapping
  // some OTHER entity's circle in a dense cluster read as belonging to that
  // circle, not its own, which label-vs-label collision alone never caught.
  const nodeBoxes: Box[] = [];
  // Focus-mode: click-to-select has no source equivalent, and neither does
  // dimming everyone but the clicked record's direct neighbors — but once
  // a record is selected, "what does this actually touch" is the question,
  // and a full-brightness graph around it buries the answer.
  const neighborIds = state.selectedId ? computeNeighborIds(state.semanticLinks, state.selectedId) : null;
  const connectedIds = state.connectedIds;

  state.entities.forEach((n) => {
    // The x/y check is redundant with visibleEntityIds (computeVisibleEntityIds
    // already required both) but keeps them narrowed from number|undefined
    // for TypeScript through the rest of this closure.
    if (!visibleEntityIds.has(n.id) || n.x == null || n.y == null) return;
    const { alpha: bornAlpha, scale } = newbornAlphaAndScale(now, state.newbornAt[n.id], state.reducedMotion);
    const inSpot = spotIds?.has(n.id) ?? false;
    const isSpot = !spotIds || spotIds.has(n.id);
    const inFocus = !state.selectedId || n.id === state.selectedId
      || (neighborIds?.has(n.id) ?? false) || state.pathIds.has(n.id);
    const fade = (spotIds ? (isSpot ? 1 : dimmed) : 1) * alpha * bornAlpha * (inFocus ? 1 : 0.15);
    // The "alive" signature — a subtle ±4.5% radius pulse, phase-offset per
    // id so entities don't all breathe in lockstep.
    const breathe = state.reducedMotion ? 1 : 1 + 0.045 * Math.sin(now / 1400 + hashPhase(n.id));
    const r = Math.max(visualRadius(n.importance || 0, state.importanceMax), MIN_NODE_SCREEN_PX / state.transform.k) * breathe * scale;
    // Published for hitTestEntity: the hit disc has to be the disc the user
    // can actually see, and only this pass knows the zoom floor, the breathe
    // phase and the newborn scale that produced it.
    n._rVisual = r;
    const bornAt = state.newbornAt[n.id];
    nodeBoxes.push({ x1: n.x - r, y1: n.y - r, x2: n.x + r, y2: n.y + r });

    if (inSpot) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, r + 6 / state.transform.k, 0, Math.PI * 2);
      ctx.strokeStyle = "#F5A83C";
      ctx.lineWidth = 2 / state.transform.k;
      ctx.stroke();
    }
    // A newborn keeps rippling for 1.2s — well past its ~0.7s fade-in — so
    // "this just arrived" reads as a distinct event, not just a fade.
    if (bornAt !== undefined && now - bornAt < 1200 && now - bornAt >= 0) {
      const elapsed = now - bornAt;
      ctx.beginPath();
      ctx.arc(n.x, n.y, r + 4 + elapsed / 90, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(67,217,232,${1 - elapsed / 1200})`;
      ctx.lineWidth = 1.2 / state.transform.k;
      ctx.stroke();
    }

    const sets = setsOf(n);
    const sourceSets = sets.filter((s) => !state.isSessionSet(s));
    const sessionMember = sets.length > sourceSets.length;
    const fillSet = sourceSets[0] || sets[0];
    const fill = (fillSet && state.setColor[fillSet]) || "#8A7BD8";
    ctx.beginPath();
    ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
    ctx.globalAlpha = fade;
    ctx.fillStyle = fill;
    ctx.fill();
    // A second ring marks an entity that bridges sources, or carries agent
    // session memory — the same "amber = agent-made" convention as the
    // session-set anchors and colors elsewhere.
    if (sourceSets.length > 1 || sessionMember) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, r + 2.2 / state.transform.k, 0, Math.PI * 2);
      ctx.strokeStyle = sessionMember && sourceSets.length <= 1
        ? "#F5A83C"
        : state.setColor[sourceSets[1]] || (sessionMember ? "#F5A83C" : "#E9EEF6");
      ctx.lineWidth = 1.6 / state.transform.k;
      ctx.stroke();
    }
    // A dashed cyan ring marks an entity that has EVER answered a live
    // search, pulled from the whole session's accumulated event log
    // (useBusinessLiveUpdates' answeredIds) — a lasting "this has been
    // useful" fact, unlike the amber spotlight ring above which only shows
    // for the current 9s window. Gated to L1+ so the model layer's
    // crossfade (L0) doesn't pick up yet another ring on top of type nodes.
    if (state.level >= 1 && state.answeredIds.has(n.id)) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, r + 1.4 / state.transform.k, 0, Math.PI * 2);
      ctx.setLineDash([2.4 / state.transform.k, 2 / state.transform.k]);
      ctx.strokeStyle = "rgba(67,217,232,0.85)";
      ctx.lineWidth = 1.4 / state.transform.k;
      ctx.stroke();
      ctx.setLineDash([]);
    }
    // A tight dashed ring flags a record with no semantic links at all —
    // new in this port, a data-quality signal source has no equivalent
    // for. Gated to L1+ like the answered-ring, at a distinct radius so
    // the two dashed rings never coincide on the same never-answered,
    // never-connected record.
    if (state.level >= 1 && !connectedIds.has(n.id)) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, r + 0.7 / state.transform.k, 0, Math.PI * 2);
      ctx.setLineDash([1.5 / state.transform.k, 2.5 / state.transform.k]);
      ctx.strokeStyle = "rgba(126,140,166,0.6)";
      ctx.lineWidth = 1 / state.transform.k;
      ctx.stroke();
      ctx.setLineDash([]);
    }
    // The shortest-path highlight (useShortestPath, also new in this
    // port) — every record on the traced path except the origin, which
    // already reads clearly enough from its own selection ring below.
    if (state.pathIds.has(n.id) && n.id !== state.selectedId) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, r + 3.6 / state.transform.k, 0, Math.PI * 2);
      ctx.strokeStyle = "#56DB7D";
      ctx.lineWidth = 1.8 / state.transform.k;
      ctx.stroke();
    }
    // Hover's r+3/k ring ports source exactly (customer_tutorial.html
    // ~6999-7001: solid C.bone, 1/k width) — this had drifted to a 0.7-alpha
    // stroke, a fainter ring than source ever draws. Selected has no source
    // equivalent (the ticket calls click-to-select out as new, not
    // ported) — reusing hover's ring position but heavier keeps one
    // consistent "ring means pointed-at" convention rather than inventing
    // a second one.
    // Pushed out to r+6/k (was r+3/k): at that radius, the selected ring's
    // own 2.5/k width overlapped the dual-source ring at r+2.2/k, painting
    // the white select stroke straight over it — selecting or hovering a
    // bridge entity erased the only visual cue that it had a second source
    // (COG-6233).
    if (n.id === state.selectedId || n.id === state.hoveredId) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, r + 6 / state.transform.k, 0, Math.PI * 2);
      ctx.lineWidth = (n.id === state.selectedId ? 2.5 : 1) / state.transform.k;
      ctx.strokeStyle = "#E9EEF6";
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    // Ports the labeled condition (customer_tutorial.html ~7020): L0 shows
    // label_priority entities plus anything well above the importance cut,
    // L1+ shows every named one — is_unnamed entities never get a label at
    // any level, since their "name" is a synthesized "Unnamed Entity (id)"
    // placeholder, not real content.
    const labeled = state.level >= 1
      ? !n.is_unnamed
      : Boolean(n.label_priority) || (n.importance || 0) >= importanceCut * 1.4;
    if (!labeled || n.is_unnamed || bornAlpha < 1 || !n.name) return;
    // Text drawn in the same world space as the nodes scales with
    // ctx.scale(k) same as everything else — dividing by sqrt(k) keeps it
    // readable instead of ballooning to dominate the screen once zoomed
    // into the "Records" level. That same /sqrt(k) shrinks the ON-SCREEN
    // size at low k (screen px = worldFontSize * k = base * sqrt(k)) — the
    // MIN_LABEL_SCREEN_PX floor below only engages once that would drop
    // under legibility, at the low end of the fit-scale range.
    const baseFontSize = Math.max(10, Math.min(16, 10 + 6 * ((n.importance || 0) / state.importanceMax)));
    const fontSize = Math.max(baseFontSize / Math.sqrt(state.transform.k), MIN_LABEL_SCREEN_PX / state.transform.k);
    ctx.font = `600 ${fontSize}px 'TWKLausanne', sans-serif`;
    const text = truncateLabel(String(n.name));
    const width = ctx.measureText(text).width;
    const y = n.y + r + fontSize + 2;
    // Selected/hovered always wins a collision — a label the user just
    // asked for by clicking or hovering should never lose to a merely
    // more-important neighbor. Otherwise importance decides which label a
    // crowded cluster keeps.
    const priority = n.id === state.selectedId || n.id === state.hoveredId || state.pathIds.has(n.id)
      ? Infinity
      : n.importance || 0;
    candidates.push({ key: n.id, x: n.x, y, width, height: fontSize, priority });
    labelMeta[n.id] = { text, x: n.x, y, fontSize, fade };
  });

  const accepted = pickNonOverlappingLabels(candidates, nodeBoxes);
  ctx.textAlign = "center";
  accepted.forEach((key) => {
    const m = labelMeta[key];
    ctx.font = `600 ${m.fontSize}px 'TWKLausanne', sans-serif`;
    // Halo drawn first so the fill on top reads as crisp text, not a
    // washed-out outline — see LABEL_HALO_COLOR above.
    ctx.lineWidth = m.fontSize * LABEL_HALO_WIDTH_RATIO;
    ctx.strokeStyle = `rgba(${LABEL_HALO_RGB},${LABEL_HALO_ALPHA * m.fade})`;
    ctx.strokeText(m.text, m.x, m.y);
    ctx.fillStyle = `rgba(233,238,246,${0.92 * m.fade})`;
    ctx.fillText(m.text, m.x, m.y);
  });
}
