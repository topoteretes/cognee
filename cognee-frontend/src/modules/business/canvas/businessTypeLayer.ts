import { color as d3color } from "d3-color";
import type { TypeNode, TypeLink } from "../sceneTypes";
import { quadraticPoint } from "./businessCanvasHelpers";
import { truncate } from "../textUtils";

// A type name derives from an is_a/instance_of link target, itself
// pipeline/LLM-produced text with no length guarantee — without a cap it
// draws (and gets measureText'd for label-claim reservation) full-width
// every frame (COG-6233).
const TYPE_LABEL_MAX_CHARS = 44;

// UX audit (2026-08-20): unlike businessEntityLayer's labels, these were
// drawn at a FIXED world-space size with no /sqrt(k) zoom compensation —
// screen size = worldSize * k, so on a dataset where the schema view's own
// fade window (computeTypeFadeKMax, businessDraw.ts) sits at a low k, the
// L0 type labels (what the "Business" dock button lands on) rendered at
// ~5px. worldSize() below applies the same compensate-then-floor formula
// businessEntityLayer uses, so a screen size never falls under
// MIN_LABEL_SCREEN_PX regardless of how low k gets.
const MIN_LABEL_SCREEN_PX = 10;
function worldSize(base: number, k: number): number {
  return Math.max(base / Math.sqrt(k), MIN_LABEL_SCREEN_PX / k);
}

interface Rect {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

// Ports the claim()-based label reservation (customer_tutorial.html
// ~7150-7200): node labels claim their space FIRST and always win; link
// labels then try a handful of candidate positions along their curve and
// take the first that doesn't collide — in WORLD coordinates, since labels
// scale with the camera and world-space collision is the honest test.
// Silently going unlabeled rather than stacking text is the same "soup, but
// hover still reveals everything" trade-off source itself accepts for a
// schema with more types/relations than its own demo ever produced.
function makeLabelClaimer(): {
  claim: (x: number, y: number, w: number, h: number) => boolean;
  block: (x0: number, y0: number, y1: number, x1: number) => void;
} {
  const occupied: Rect[] = [];
  const overlaps = (rect: Rect): boolean =>
    occupied.some((r) => rect.x0 < r.x1 && rect.x1 > r.x0 && rect.y0 < r.y1 && rect.y1 > r.y0);
  return {
    claim(x, y, w, h) {
      const rect: Rect = { x0: x - w / 2 - 3, x1: x + w / 2 + 3, y0: y - h + 2, y1: y + 4 };
      if (overlaps(rect)) return false;
      occupied.push(rect);
      return true;
    },
    block(x0, y0, x1, y1) {
      occupied.push({ x0, y0, x1, y1 });
    },
  };
}

// A code-analysis-style graph can have dozens of distinct type PAIRS
// (variable↔function, class↔entity, …) where source's own type schemas
// (organization/campaign/opportunity/…) rarely exceed a handful — fading
// the connecting lines in proportion to pair count keeps a small schema
// exactly as visible as before while a dense one recedes into a texture.
// Label legibility doesn't need this: the claimer above already thins text
// on its own, the same way it does in source.
const DENSE_PAIR_THRESHOLD = 10;
const MIN_PAIR_DENSITY_FADE = 0.2;

// The claimer only stops two labels from occupying the exact same
// rectangle — it does nothing to cap how many distinct relations exist
// between one type PAIR. A code-analysis-style schema can carry dozens of
// relation kinds between the same two types (concept↔other), each on its
// own barely-offset bow (see the fan formula below); every one still finds
// SOME unclaimed sliver of space, so the claimer's "no pixel overlap"
// guarantee holds while the result is a solid column of stacked text —
// confirmed on staging with a 733-record dataset zoomed to the schema view.
// Labeling only the pair's most frequent relations (already sorted by
// count just below) keeps the signal and drops the rest; their connecting
// lines still draw, just unlabeled.
const MAX_LABELED_RELATIONS_PER_PAIR = 3;

// Ports the type-model layer's whole draw block (customer_tutorial.html
// ~7130-7205) as one function: node labels claim space, links draw (with
// their own claim-checked labels), then node circles paint on top — one
// pass, not two independently-ordered ones, since node and link labels
// share the same claimer.
export function drawTypeModelLayer(
  ctx: CanvasRenderingContext2D,
  typeNodes: TypeNode[],
  typeLinks: TypeLink[],
  setColor: Record<string, string>,
  isSessionSet: (name: string) => boolean,
  typeAlpha: number,
  dimmed: number,
  focusSets: Set<string> | null,
  transformK: number,
): void {
  ctx.textAlign = "center";
  const byName: Record<string, TypeNode> = {};
  typeNodes.forEach((tn) => { byName[tn.name] = tn; });
  const { claim, block } = makeLabelClaimer();
  // claim()'s return value used to be discarded here — every type node's
  // label got drawn in the third loop below regardless of whether its
  // reserved space actually collided with an earlier (denser-packed, so
  // higher-priority by draw order) node's label. That's what let near-
  // identical labels (e.g. several per-session "session_learning — <date>
  // (session …)" type nodes sitting close together) stack directly on top
  // of each other — the space-reservation system existed but only ever
  // gated the link labels drawn after it, never the node labels it was
  // meant to protect in the first place.
  const labeledTypeNames = new Set<string>();

  typeNodes.forEach((tn) => {
    if (tn.x == null || tn.y == null) return;
    const r = tn._r || 20;
    // The final draw below (third loop) renders TWO lines under the node —
    // the name, then a "N records" line beneath it — but this claim used to
    // reserve a single fixed-height (30px) box sized for the name alone.
    // The records line fell outside that box, so a link label drawn right
    // after (thinking the space below the name was free) could land right
    // on top of it — confirmed on staging where a "connected to" link label
    // overlapped an "154 records" count. worldSize() also grows at low zoom
    // (screen-size-floored, so world-size scales up as k shrinks), so a
    // fixed 30px was never right at any zoom, just less visibly wrong at
    // k≈1. Reserving the actual two-line extent fixes both.
    const nameSize = worldSize(15, transformK);
    const recordSize = worldSize(11, transformK);
    ctx.font = `700 ${nameSize}px 'TWKLausanne', sans-serif`;
    const nameWidth = ctx.measureText(truncate(tn.name, TYPE_LABEL_MAX_CHARS)).width;
    ctx.font = `500 ${recordSize}px 'TWKLausanne', sans-serif`;
    const shown = tn._visible?.length || tn.members.length;
    const recordWidth = ctx.measureText(`${shown} ${shown === 1 ? "record" : "records"}`).width;
    const w = Math.max(nameWidth, recordWidth, 70);
    const h = nameSize + recordSize + 14;
    if (claim(tn.x, tn.y + r + h - 8, w, h)) labeledTypeNames.add(tn.name);
    block(tn.x - r, tn.y - r, tn.x + r, tn.y + r);
  });

  const pairs: Record<string, TypeLink[]> = {};
  typeLinks.forEach((tl) => {
    const key = [tl.a, tl.b].sort().join("␟");
    (pairs[key] = pairs[key] || []).push(tl);
  });
  const pairCount = Object.keys(pairs).length;
  const densityFade = pairCount > DENSE_PAIR_THRESHOLD
    ? Math.max(MIN_PAIR_DENSITY_FADE, DENSE_PAIR_THRESHOLD / pairCount)
    : 1;

  Object.values(pairs).forEach((fullGroup) => {
    const group = [...fullGroup].sort((x, y) => y.count - x.count);
    group.forEach((tl, idx) => {
      const a = byName[tl.a], b = byName[tl.b];
      if (!a || a.x == null || a.y == null || !b || b.x == null || b.y == null) return;
      if (focusSets && (!a._visible?.length || !b._visible?.length)) return;
      const dx = b.x - a.x, dy = b.y - a.y;
      // Fan: first relation bows one way, second the other, growing wider.
      const bow = (idx % 2 === 0 ? 1 : -1) * (0.1 + Math.floor(idx / 2) * 0.14);
      const mx = (a.x + b.x) / 2 - dy * bow, my = (a.y + b.y) / 2 + dx * bow;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.quadraticCurveTo(mx, my, b.x, b.y);
      ctx.strokeStyle = `rgba(126,140,166,${0.5 * typeAlpha * dimmed * densityFade})`;
      ctx.lineWidth = Math.min(3, 1 + tl.count * 0.4) / transformK;
      ctx.stroke();

      const br = b._r || 20;
      const ang = Math.atan2(b.y - my, b.x - mx);
      const ax2 = b.x - Math.cos(ang) * (br + 4), ay2 = b.y - Math.sin(ang) * (br + 4);
      ctx.beginPath();
      ctx.moveTo(ax2, ay2);
      ctx.lineTo(ax2 - Math.cos(ang - 0.45) * 7, ay2 - Math.sin(ang - 0.45) * 7);
      ctx.lineTo(ax2 - Math.cos(ang + 0.45) * 7, ay2 - Math.sin(ang + 0.45) * 7);
      ctx.closePath();
      ctx.fillStyle = `rgba(126,140,166,${0.6 * typeAlpha * dimmed * densityFade})`;
      ctx.fill();

      if (idx >= MAX_LABELED_RELATIONS_PER_PAIR) return;
      const text = String(tl.relation).replace(/_/g, " ");
      ctx.font = `500 ${worldSize(10.5, transformK)}px ui-monospace, monospace`;
      const tw = ctx.measureText(text).width;
      for (const t of [0.5, 0.34, 0.66, 0.22, 0.78]) {
        const [lx, ly] = quadraticPoint(a.x, a.y, mx, my, b.x, b.y, t);
        if (!claim(lx, ly, tw, 14)) continue;
        ctx.globalAlpha = typeAlpha * dimmed * densityFade;
        ctx.fillStyle = "rgba(14,21,38,0.88)";
        ctx.beginPath();
        ctx.roundRect(lx - tw / 2 - 5, ly - 11, tw + 10, 15, 7);
        ctx.fill();
        ctx.fillStyle = "rgba(233,238,246,0.85)";
        ctx.fillText(text, lx, ly);
        ctx.globalAlpha = 1;
        break;
      }
    });
  });

  typeNodes.forEach((tn) => {
    if (tn.x == null || tn.y == null) return;
    if (focusSets && !tn._visible?.length) return;
    // The dominant NON-session source decides the fill — a type is "mostly
    // crm with a sliver of google_drive," one color, not a proportional pie
    // (source doesn't draw one; a single dominant-source fill is what it
    // actually does).
    const ranked = Object.keys(tn.sets).sort((x, y) => tn.sets[y] - tn.sets[x]);
    const dominant = focusSets ? [...focusSets][0] : ranked.find((x) => !isSessionSet(x)) || ranked[0];
    const fill = d3color((dominant && setColor[dominant]) || "#8A7BD8");
    const rgb = fill ? fill.rgb() : { r: 138, g: 123, b: 216 };
    const r = tn._r || 20;
    ctx.globalAlpha = typeAlpha * dimmed;
    ctx.beginPath();
    ctx.arc(tn.x, tn.y, r, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(${rgb.r},${rgb.g},${rgb.b},0.28)`;
    ctx.fill();
    ctx.strokeStyle = `rgba(${rgb.r},${rgb.g},${rgb.b},0.9)`;
    ctx.lineWidth = 1.6 / transformK;
    ctx.stroke();
    if (labeledTypeNames.has(tn.name)) {
      // Offsets grow with the (now zoom-floored) label size instead of a
      // fixed world +17/+31 — otherwise text enlarged by worldSize() at low
      // k would sit closer to (or overlap) the circle than at normal zoom.
      const nameSize = worldSize(15, transformK);
      const recordSize = worldSize(11, transformK);
      ctx.font = `700 ${nameSize}px 'TWKLausanne', sans-serif`;
      ctx.fillStyle = `rgba(233,238,246,${typeAlpha})`;
      ctx.fillText(truncate(tn.name, TYPE_LABEL_MAX_CHARS), tn.x, tn.y + r + nameSize + 2);
      ctx.font = `500 ${recordSize}px 'TWKLausanne', sans-serif`;
      ctx.fillStyle = `rgba(126,140,166,${typeAlpha})`;
      const shown = tn._visible?.length || tn.members.length;
      ctx.fillText(`${shown} ${shown === 1 ? "record" : "records"}`, tn.x, tn.y + r + nameSize + recordSize + 6);
    }
    ctx.globalAlpha = 1;
  });
}
