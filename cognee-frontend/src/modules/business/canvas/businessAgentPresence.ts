import type { BusinessEntity } from "../sceneTypes";
import type { Spotlight } from "./businessDraw";
import { MIN_NODE_SCREEN_PX, RENDER_MAX_R } from "./businessEntityLayer";

// CLO-606: draws the asking agent itself on the canvas while its answer is
// spotlighted — a marker near the retrieved subgraph, a temporary filament
// to it, nothing at all once the spotlight expires (the CLO-604 resting-
// state rule: no permanent lines). Every function here is pure world-space
// math except drawAgentPresence, kept separate so the gate/position/alpha
// logic is unit-testable without a canvas.

export interface AgentPresence {
  markerWorld: { x: number; y: number };
  centroidWorld: { x: number; y: number };
  alpha: number;
}

// The agent a live search event's answer is currently attributed to, plus
// the window that attribution is good for (useBusinessQaSurface's
// AskingWindow, resolved to a display name in BusinessView) — null whenever
// no agent is asking, which is this feature's other half of the gate.
//
// The window is carried here rather than derived from the spotlight because
// the two do not line up: a reasoning-trace walk hands a series of ~1.8s
// step spotlights over to the answer's own 9s one, so "how far into the
// agent's turn are we" is a question only the agent's own window can answer.
export interface AskingAgent {
  id: string;
  name: string;
  startedAt: number;
  until: number;
}

// Which spotlights this marker is allowed to annotate. The auto-insight and
// what-if-removal spotlights have no agent behind them, so painting an amber
// marker and an agent's NAME next to those clusters attributes a claim to
// someone who never made it — the gate has to check whose spotlight it is,
// not merely that one is active.
const AGENT_SPOTLIGHT_SOURCES: ReadonlySet<Spotlight["source"]> = new Set(["answer", "trace"]);

// Clearance between the spotlighted bounding box and the marker, expressed
// the way the entity layer expresses its own sizes. A flat world constant
// broke at low zoom: the entity layer floors its render radius at
// MIN_NODE_SCREEN_PX/k, so below k≈0.22 a node's world radius alone already
// exceeds a fixed 40, putting the marker inside the cluster it points at.
const MARKER_GAP_SCREEN_PX = 10;

// At k=1 the three terms sum to exactly 40, the flat world constant this
// replaces — so the marker sits where it always did at rest, and only its
// off-nominal-zoom behaviour changes.
export function markerPaddingWorld(transformK: number): number {
  const largestNodeWorld = Math.max(RENDER_MAX_R, MIN_NODE_SCREEN_PX / transformK);
  return largestNodeWorld + markerRadiusWorld(transformK) + MARKER_GAP_SCREEN_PX / transformK;
}

// Bounding box of the spotlighted entities' current positions, marker placed
// outside its top-right corner. Entities without a settled x/y (not yet
// seeded by the simulation) are skipped — null means nothing to anchor to.
export function computeMarkerPosition(
  entities: BusinessEntity[],
  ids: Set<string>,
  transformK: number,
): { x: number; y: number } | null {
  let minY = Infinity;
  let maxX = -Infinity;
  let found = false;
  entities.forEach((e) => {
    if (!ids.has(e.id) || e.x == null || e.y == null) return;
    found = true;
    maxX = Math.max(maxX, e.x);
    minY = Math.min(minY, e.y);
  });
  if (!found) return null;
  const padding = markerPaddingWorld(transformK);
  return { x: maxX + padding, y: minY - padding };
}

// The filament's other end — plain mean position, not buildAnchors' gravity
// point (same "point at where the cluster actually IS" reasoning as
// businessFilaments.filamentTargets).
export function computeCentroid(
  entities: BusinessEntity[],
  ids: Set<string>,
): { x: number; y: number } | null {
  let sx = 0;
  let sy = 0;
  let n = 0;
  entities.forEach((e) => {
    if (!ids.has(e.id) || e.x == null || e.y == null) return;
    sx += e.x;
    sy += e.y;
    n += 1;
  });
  if (!n) return null;
  return { x: sx / n, y: sy / n };
}

const FADE_IN_MS = 300;
const FADE_OUT_MS = 1500;

// Fades in over the window's first FADE_IN_MS, fades out over its last
// FADE_OUT_MS, full opacity in between. 0 once the window has expired. Both
// bounds come from the caller, so a window of any width (a 9s answer, a
// walk-plus-answer stretch of 16s) plays the same envelope.
export function computePresenceAlpha(now: number, startedAt: number, until: number): number {
  const remaining = until - now;
  if (remaining <= 0) return 0;
  const elapsed = now - startedAt;
  if (elapsed < 0) return 0;
  const fadeIn = elapsed < FADE_IN_MS ? elapsed / FADE_IN_MS : 1;
  const fadeOut = remaining < FADE_OUT_MS ? remaining / FADE_OUT_MS : 1;
  return Math.max(0, Math.min(1, Math.min(fadeIn, fadeOut)));
}

// The gate this whole feature hangs off: the agent's own window is open, the
// spotlight on screen is one the agent produced, and there is a positioned
// cluster to anchor to. Anything else draws nothing at all — the CLO-604
// resting-state rule.
export function computeAgentPresence(
  entities: BusinessEntity[],
  spotlight: Spotlight,
  agent: AskingAgent,
  now: number,
  transformK: number,
): AgentPresence | null {
  if (!AGENT_SPOTLIGHT_SOURCES.has(spotlight.source)) return null;
  const alpha = computePresenceAlpha(now, agent.startedAt, agent.until);
  if (alpha <= 0) return null;
  const markerWorld = computeMarkerPosition(entities, spotlight.ids, transformK);
  const centroidWorld = computeCentroid(entities, spotlight.ids);
  if (!markerWorld || !centroidWorld) return null;
  return { markerWorld, centroidWorld, alpha };
}

// Screen-size-floored world size — same compensate-then-floor formula as
// businessTypeLayer's worldSize, kept local since each canvas layer file
// owns its own copy rather than sharing one export.
const MIN_LABEL_SCREEN_PX = 10;
function worldSize(base: number, k: number): number {
  return Math.max(base / Math.sqrt(k), MIN_LABEL_SCREEN_PX / k);
}

const AMBER = "#F5A83C";
const FILAMENT_ALPHA = 0.35;
// Same perpendicular-bow ratio drawSemanticLinks uses for its quadratic
// curves — a subtle bow, not a straight line, consistent with every other
// curved connector this canvas draws.
const FILAMENT_BOW_RATIO = 0.12;

function drawFilamentToMarker(ctx: CanvasRenderingContext2D, presence: AgentPresence, transformK: number): void {
  const { markerWorld: m, centroidWorld: c, alpha } = presence;
  const dx = m.x - c.x;
  const dy = m.y - c.y;
  const mx = (c.x + m.x) / 2 - dy * FILAMENT_BOW_RATIO;
  const my = (c.y + m.y) / 2 + dx * FILAMENT_BOW_RATIO;
  ctx.beginPath();
  ctx.moveTo(c.x, c.y);
  ctx.quadraticCurveTo(mx, my, m.x, m.y);
  ctx.strokeStyle = `rgba(245,168,60,${FILAMENT_ALPHA * alpha})`;
  ctx.lineWidth = 1.4 / transformK;
  ctx.stroke();
}

const MARKER_BASE_RADIUS = 5;
const PULSE_PERIOD_MS = 1100;
const PULSE_AMPLITUDE = 0.3;
const EMPHASIS_RING_OFFSET_WORLD = 5;

function markerPulse(reducedMotion: boolean, now: number): number {
  return reducedMotion ? 1 : 1 + PULSE_AMPLITUDE * (0.5 + 0.5 * Math.sin(now / PULSE_PERIOD_MS));
}

// The one radius both the renderer and the hit-test read. `now`/`reducedMotion`
// are optional so a caller that only needs the marker's resting size (the
// padding math above, which must not pulse or the marker would drift) can
// ask for it without inventing a timestamp; the hit-test passes them so the
// clickable target breathes with the dot instead of lagging behind it at the
// top of every pulse.
export function markerRadiusWorld(transformK: number, reducedMotion = true, now = 0): number {
  return worldSize(MARKER_BASE_RADIUS, transformK) * markerPulse(reducedMotion, now);
}

// The emphasis ring sits outside the dot, so an emphasized marker's visible
// extent — the thing a user aims at — is this, not the dot alone.
export function markerHitRadiusWorld(
  transformK: number,
  reducedMotion: boolean,
  now: number,
  emphasized: boolean,
): number {
  const r = markerRadiusWorld(transformK, reducedMotion, now);
  return emphasized ? r + EMPHASIS_RING_OFFSET_WORLD / transformK : r;
}

function drawMarkerDot(
  ctx: CanvasRenderingContext2D,
  presence: AgentPresence,
  transformK: number,
  reducedMotion: boolean,
  emphasized: boolean,
  now: number,
): void {
  const { markerWorld: m, alpha } = presence;
  const r = markerRadiusWorld(transformK, reducedMotion, now);
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  ctx.arc(m.x, m.y, r, 0, Math.PI * 2);
  ctx.fillStyle = AMBER;
  ctx.fill();
  // Rail→canvas hover sync: brighter, larger halo when OperatorsRail's own
  // hover (or this marker's own hover, reported back up) targets this agent.
  if (emphasized) {
    ctx.beginPath();
    ctx.arc(m.x, m.y, r + EMPHASIS_RING_OFFSET_WORLD / transformK, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(245,168,60,0.9)";
    ctx.lineWidth = 2.2 / transformK;
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

const LABEL_GAP_WORLD = 8;
const LABEL_HALO_WIDTH_RATIO = 0.28;

function drawMarkerLabel(ctx: CanvasRenderingContext2D, presence: AgentPresence, name: string, transformK: number): void {
  const { markerWorld: m, alpha } = presence;
  const r = markerRadiusWorld(transformK);
  const fontSize = worldSize(12, transformK);
  const x = m.x + r + LABEL_GAP_WORLD / transformK;
  ctx.font = `600 ${fontSize}px 'TWKLausanne', sans-serif`;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  // Halo stroked before the fill, same as businessEntityLayer's node labels —
  // reads as an outline against whatever's behind it, not a background box.
  ctx.lineWidth = fontSize * LABEL_HALO_WIDTH_RATIO;
  ctx.strokeStyle = `rgba(14,21,38,${0.85 * alpha})`;
  ctx.strokeText(name, x, m.y);
  ctx.fillStyle = `rgba(245,168,60,${alpha})`;
  ctx.fillText(name, x, m.y);
}

export interface AgentPresenceDrawParams {
  presence: AgentPresence;
  name: string;
  transformK: number;
  reducedMotion: boolean;
  emphasized: boolean;
  now: number;
}

// Called from inside the same camera-transformed ctx the entity/type layers
// draw into (BusinessCanvas), so the marker/filament/label all pan and zoom
// with the cluster they're pointing at.
export function drawAgentPresence(ctx: CanvasRenderingContext2D, params: AgentPresenceDrawParams): void {
  const { presence, name, transformK, reducedMotion, emphasized, now } = params;
  drawFilamentToMarker(ctx, presence, transformK);
  drawMarkerDot(ctx, presence, transformK, reducedMotion, emphasized, now);
  drawMarkerLabel(ctx, presence, name, transformK);
}
