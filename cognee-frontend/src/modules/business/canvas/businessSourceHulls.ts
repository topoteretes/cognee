import { mean } from "d3-array";
import { polygonHull } from "d3-polygon";
import { color as d3color } from "d3-color";
import type { BusinessEntity } from "../sceneTypes";
import { setsOf, sourceLabel } from "../computeBrainState";
import { truncate } from "../textUtils";

// A node_set name is user/pipeline-supplied and unbounded in length (a
// 1000+ char name is possible) — without a cap it draws full-width every
// frame and costs a full-string measureText on top (COG-6233).
const HULL_LABEL_MAX_CHARS = 60;

// Source territories: soft convex hulls shaded in the source's color, with
// the caption riding the hull's crown (customer_tutorial.html ~6754-6793).
export function drawSourceHulls(
  ctx: CanvasRenderingContext2D,
  entities: BusinessEntity[],
  sourceNames: string[],
  setColor: Record<string, string>,
  focusSets: Set<string> | null,
  dimmed: number,
  transformK: number,
): void {
  ctx.textAlign = "center";
  const names = focusSets ? [...focusSets] : sourceNames;
  names.forEach((name) => {
    const members = entities.filter((n) => setsOf(n).includes(name) && n.x != null && n.y != null);
    if (members.length < 2) return;
    const pad = 34;
    const pts: [number, number][] = [];
    members.forEach((n) => {
      for (let a = 0; a < 8; a++) {
        pts.push([
          (n.x ?? 0) + Math.cos((a * Math.PI) / 4) * ((n._r ?? 5) + pad),
          (n.y ?? 0) + Math.sin((a * Math.PI) / 4) * ((n._r ?? 5) + pad),
        ]);
      }
    });
    const hull = polygonHull(pts);
    if (!hull) return;
    const col = d3color(setColor[name] || "#888");
    if (!col) return;
    const rgb = col.rgb();
    // Smooths the hull into an organic blob instead of a jagged polygon:
    // each edge's midpoint is a curve endpoint, with the shared vertex as
    // the curve's control point (customer_tutorial.html ~7175).
    ctx.beginPath();
    for (let i = 0; i < hull.length; i++) {
      const p = hull[i], q = hull[(i + 1) % hull.length];
      const mx = (p[0] + q[0]) / 2, my = (p[1] + q[1]) / 2;
      if (i === 0) ctx.moveTo(mx, my);
      else ctx.quadraticCurveTo(p[0], p[1], mx, my);
    }
    const p0 = hull[0], p1 = hull[1 % hull.length];
    ctx.quadraticCurveTo(p0[0], p0[1], (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2);
    ctx.closePath();
    ctx.fillStyle = `rgba(${rgb.r},${rgb.g},${rgb.b},${0.06 * dimmed})`;
    ctx.fill();
    ctx.strokeStyle = `rgba(${rgb.r},${rgb.g},${rgb.b},${0.22 * dimmed})`;
    ctx.lineWidth = 1 / transformK;
    ctx.stroke();
    const topY = Math.min(...hull.map((p) => p[1])) - 12;
    ctx.font = "600 13px 'TWKLausanne', sans-serif";
    ctx.fillStyle = `rgba(${rgb.r},${rgb.g},${rgb.b},${0.85 * dimmed})`;
    ctx.fillText(truncate(sourceLabel(name), HULL_LABEL_MAX_CHARS), mean(members, (n) => n.x ?? 0) || 0, topY);
  });
}
