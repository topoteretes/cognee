import type { Anchor, BusinessEntity } from "../sceneTypes";
import type { Transform } from "./businessHitTest";
import { hashSeed } from "./businessCanvasHelpers";
import { setsOf } from "../computeBrainState";

// Sources with at least one entity actually in the graph — `sourceNames`
// also includes dataset sources registered with zero extracted entities
// (COG-6233 union in deriveSourceNames), which otherwise thread a stray
// filament to a fallback anchor with nothing real on the other end. The
// caller filters its filament list through this BEFORE drawFilaments runs.
export function sourcesWithEntities(entities: BusinessEntity[]): Set<string> {
  const names = new Set<string>();
  entities.forEach((e) => setsOf(e).forEach((s) => names.add(s)));
  return names;
}

// The filament's actual destination — where the settled cluster visually
// IS, not buildAnchors' theoretical gravity point. That anchor is only a
// weak (0.07-strength) pull among much stronger link/charge forces, so a
// cluster routinely settles somewhere else entirely; pointing the filament
// at the anchor instead of the real cluster made it exit toward empty
// space once the camera (correctly) framed the actual settled positions
// instead of the anchor (COG-6233). A source with no positioned entities
// keeps the raw anchor value here, but never reaches drawFilaments — the
// caller drops it via sourcesWithEntities above.
export function filamentTargets(
  entities: BusinessEntity[],
  anchors: Record<string, Anchor>,
): Record<string, Anchor> {
  const sums: Record<string, { sx: number; sy: number; n: number }> = {};
  entities.forEach((e) => {
    if (e.x == null || e.y == null) return;
    setsOf(e).forEach((s) => {
      const bucket = (sums[s] = sums[s] || { sx: 0, sy: 0, n: 0 });
      bucket.sx += e.x as number;
      bucket.sy += e.y as number;
      bucket.n += 1;
    });
  });
  const targets: Record<string, Anchor> = { ...anchors };
  Object.entries(sums).forEach(([name, { sx, sy, n }]) => {
    targets[name] = { x: sx / n, y: sy / n };
  });
  return targets;
}

// Cubic bezier point at t, matching the curve ctx.bezierCurveTo actually
// draws — so a drifting particle can be placed exactly on the line it
// travels, not just approximately near it.
function bez(p0: number, p1: number, p2: number, p3: number, t: number): number {
  const mt = 1 - t;
  return mt * mt * mt * p0 + 3 * mt * mt * t * p1 + 3 * mt * t * t * p2 + t * t * t * p3;
}

// The source's fixed control-point offsets (110-180px) assume the card and
// its target are always comfortably far apart, which held for its fixed
// rail-card layout but not for this port's simulation-driven positions —
// close together, a fixed offset makes the two control points cross and the
// curve loops back on itself. Scaling the offset down with distance keeps
// the same curve shape when far apart and just fades to a straight line
// instead of looping when close.
export function clampedOffset(distance: number, maxOffset: number): number {
  return Math.min(maxOffset, Math.abs(distance) / 2.2);
}

// Ports the source-card filaments (customer_tutorial.html:6640-6663): a
// screen-space curve from each source card's right edge to its territory's
// world-space anchor (projected through the camera transform), with two
// particles drifting along it — the visual thread connecting the rails to
// the canvas. Drawn in screen space, before the caller applies the camera
// transform for the rest of the scene.
export function drawFilaments(
  ctx: CanvasRenderingContext2D,
  cardRects: Record<string, { x: number; y: number }>,
  anchors: Record<string, Anchor>,
  transform: Transform,
  now: number,
  // How present the entity layer this thread points into currently is (see
  // businessDraw's computeInstanceAlpha) — a filament aimed at an entity
  // cluster the schema view has faded out must fade with it, the same way
  // drawSourceHulls already fades the territory it shades.
  layerAlpha: number,
): void {
  Object.entries(cardRects).forEach(([name, card]) => {
    const anchor = anchors[name];
    if (!anchor) return;
    const x0 = card.x, y0 = card.y;
    const cx = transform.x + anchor.x * transform.k, cy = transform.y + anchor.y * transform.k;
    const offset = clampedOffset(cx - x0, 105);
    const c1x = x0 + offset, c1y = y0, c2x = cx - offset, c2y = cy;

    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.bezierCurveTo(c1x, c1y, c2x, c2y, cx, cy);
    ctx.strokeStyle = `rgba(126,140,166,${0.18 * layerAlpha})`;
    ctx.lineWidth = 1;
    ctx.stroke();

    const seed = hashSeed(name);
    for (let p = 0; p < 2; p++) {
      const t = ((now / 2600 + p * 0.5 + seed) % 1 + 1) % 1;
      const px = bez(x0, c1x, c2x, cx, t), py = bez(y0, c1y, c2y, cy, t);
      ctx.beginPath();
      ctx.arc(px, py, 1.6, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(67,217,232,${0.5 * (1 - t) * layerAlpha})`;
      ctx.fill();
    }
  });
}
