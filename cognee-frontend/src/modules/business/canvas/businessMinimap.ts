import type { BusinessEntity } from "../sceneTypes";
import type { Transform } from "./businessHitTest";
import { setsOf } from "../computeBrainState";

// A fixed overview panel in the canvas's bottom-right corner — new in this
// port (no source equivalent; the Mindmap view has one, and it's the
// standard "where am I" affordance for a pannable canvas). Drawn in screen
// space by the main draw loop, after the world-space scene, so it always
// sits on top.
const MINIMAP_W = 150;
const MINIMAP_H = 104;
const MINIMAP_MARGIN = 12;
// BusinessDock is an absolute-positioned overlay pinned to the canvas's own
// bottom edge (narration line + altimeter buttons + live/tour controls), not
// accounted for by the plain MINIMAP_MARGIN below it — without this, the
// dock clipped the bottom third of the minimap instead of sitting beside it.
const MINIMAP_BOTTOM_MARGIN = 84;
const BOUNDS_PAD = 40;

interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface MinimapProjection {
  rect: Rect;
  x0: number;
  y0: number;
  scale: number;
  ox: number;
  oy: number;
}

export function minimapRect(width: number, height: number): Rect {
  return {
    x: width - MINIMAP_W - MINIMAP_MARGIN,
    y: height - MINIMAP_H - MINIMAP_BOTTOM_MARGIN,
    w: MINIMAP_W,
    h: MINIMAP_H,
  };
}

export function minimapContains(mx: number, my: number, width: number, height: number): boolean {
  const r = minimapRect(width, height);
  return mx >= r.x && mx <= r.x + r.w && my >= r.y && my <= r.y + r.h;
}

function projection(entities: BusinessEntity[], width: number, height: number): MinimapProjection | null {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  entities.forEach((n) => {
    if (n.x == null || n.y == null) return;
    if (n.x < x0) x0 = n.x;
    if (n.x > x1) x1 = n.x;
    if (n.y < y0) y0 = n.y;
    if (n.y > y1) y1 = n.y;
  });
  if (x0 === Infinity) return null;
  x0 -= BOUNDS_PAD; y0 -= BOUNDS_PAD; x1 += BOUNDS_PAD; y1 += BOUNDS_PAD;
  const rect = minimapRect(width, height);
  const scale = Math.min(rect.w / (x1 - x0 || 1), rect.h / (y1 - y0 || 1));
  return {
    rect,
    x0,
    y0,
    scale,
    ox: rect.x + (rect.w - (x1 - x0) * scale) / 2,
    oy: rect.y + (rect.h - (y1 - y0) * scale) / 2,
  };
}

// Maps a canvas-space click back to WORLD coordinates when it lands inside
// the minimap — the caller pans the camera there; null means "not a minimap
// click, hit-test the scene as usual".
export function minimapWorldPoint(
  mx: number,
  my: number,
  entities: BusinessEntity[],
  width: number,
  height: number,
): { x: number; y: number } | null {
  if (!minimapContains(mx, my, width, height)) return null;
  const p = projection(entities, width, height);
  if (!p) return null;
  return { x: p.x0 + (mx - p.ox) / p.scale, y: p.y0 + (my - p.oy) / p.scale };
}

export function drawMinimap(
  ctx: CanvasRenderingContext2D,
  entities: BusinessEntity[],
  setColor: Record<string, string>,
  transform: Transform,
  width: number,
  height: number,
): void {
  if (entities.length < 2) return;
  const p = projection(entities, width, height);
  if (!p) return;
  const r = p.rect;

  ctx.fillStyle = "rgba(20,29,51,0.92)";
  ctx.strokeStyle = "#2A3652";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(r.x, r.y, r.w, r.h, 8);
  ctx.fill();
  ctx.stroke();

  entities.forEach((n) => {
    if (n.x == null || n.y == null) return;
    const px = p.ox + (n.x - p.x0) * p.scale, py = p.oy + (n.y - p.y0) * p.scale;
    const set = setsOf(n)[0];
    ctx.fillStyle = (set && setColor[set]) || "#8A7BD8";
    ctx.globalAlpha = 0.8;
    ctx.fillRect(px - 1, py - 1, 2, 2);
  });
  ctx.globalAlpha = 1;

  // The visible world region (screen corners pulled back through the camera
  // transform), clamped so a deep zoom-out doesn't spill past the panel.
  const vx = p.ox + ((0 - transform.x) / transform.k - p.x0) * p.scale;
  const vy = p.oy + ((0 - transform.y) / transform.k - p.y0) * p.scale;
  const vw = (width / transform.k) * p.scale, vh = (height / transform.k) * p.scale;
  const cx = Math.max(r.x + 1, vx), cy = Math.max(r.y + 1, vy);
  const cw = Math.min(r.x + r.w - 1, vx + vw) - cx, ch = Math.min(r.y + r.h - 1, vy + vh) - cy;
  if (cw > 0 && ch > 0) {
    ctx.strokeStyle = "rgba(67,217,232,0.8)";
    ctx.lineWidth = 1;
    ctx.strokeRect(cx, cy, cw, ch);
  }
}
