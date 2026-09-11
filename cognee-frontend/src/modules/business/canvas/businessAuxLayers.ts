import type { BusinessEntity } from "../sceneTypes";
import type { BusinessGraphNode } from "../types";
import { setsOf } from "../computeBrainState";
import { hashPhase, inLens } from "./businessEntityLayer";
import type { DrawState } from "./businessDraw";

// Ports the L2 "documents unfold near their entities" block
// (customer_tutorial.html ~7195): a dashed tether plus a small rounded
// swatch, colored by the document's own source — content made visible as a
// physical object near the entity it's about, not just a number in a
// tooltip. Source's scatter offset reuses hash()'s [0, 2π) phase through a
// %20/%16 that leaves it barely varying (an inconsequential quirk, not a
// deliberate range) — mapping the same phase across the full 20/16px range
// gives the intended-looking scatter without copying that quirk.
export function drawDocumentBadges(
  ctx: CanvasRenderingContext2D,
  state: DrawState,
  entityById: Record<string, BusinessEntity>,
): void {
  state.docLinks.forEach((l) => {
    const sIsEntity = state.byId[l._sid]?.stage === "entity";
    const doc = sIsEntity ? state.byId[l._tid] : state.byId[l._sid];
    const e = sIsEntity ? entityById[l._sid] : entityById[l._tid];
    if (!doc || !e || e.x == null || e.y == null) return;
    // A source focus lens dims everything outside it but never removed
    // OTHER sources' entities from the simulation (they keep real x/y
    // scattered across the whole graph) — without this check, zooming to
    // L2+ while a lens was active drew every OTHER source's document
    // badges too, reading as unrelated data suddenly appearing on zoom
    // (COG-6233).
    if (!inLens(e, state.focusSets)) return;
    const phase = hashPhase(doc.id);
    const dx = e.x + 26 + (phase / 6.28) * 20;
    const dy = e.y + 18 + (((phase * 3) % 6.28) / 6.28) * 16;
    ctx.setLineDash([3 / state.transform.k, 3 / state.transform.k]);
    ctx.beginPath();
    ctx.moveTo(e.x, e.y);
    ctx.lineTo(dx, dy);
    ctx.strokeStyle = "rgba(126,140,166,0.3)";
    ctx.lineWidth = 0.8 / state.transform.k;
    ctx.stroke();
    ctx.setLineDash([]);
    // Ports setsOf(d) — the document's OWN source tag, not the entity it's
    // tethered to. They usually agree, but a document belongs to whatever
    // ingested it; the entity extracted from it is what's being described.
    const sets = setsOf(doc);
    ctx.fillStyle = (sets.length && state.setColor[sets[0]]) || "#7E8CA6";
    ctx.globalAlpha = 0.75;
    ctx.beginPath();
    ctx.roundRect(dx - 5, dy - 6, 10, 12, 2);
    ctx.fill();
    ctx.globalAlpha = 1;
  });
}

// Radius band for a record's deterministic orbit around its linked entity —
// close enough to read as "belongs to this", far enough not to sit on it.
const PLUMBING_ORBIT_MIN = 22;
const PLUMBING_ORBIT_SPREAD = 26;
// How far past the settled graph's extent the unlinked leftovers ring sits.
const UNLINKED_RING_MARGIN = 120;

// Ports "L3: plumbing" (customer_tutorial.html ~7220) — Summary, Context,
// chunk and document nodes the entity/type layers never draw — reworked
// from the source's hash-scatter: those dots landed on a Lissajous curve
// around the WORLD ORIGIN, unrelated to both the data and wherever the
// graph actually settled, which read as the Records level "showing data
// strangely". Each record now orbits the entity it genuinely links to
// (plumbingEntityId, resolved from the real link list in computeBrainState),
// with a short tether making the relationship explicit; records with no
// entity link at all fall back to a ring around the settled graph itself,
// so even the leftovers follow the data instead of an arbitrary fixed spot.
export function drawPlumbingLayer(
  ctx: CanvasRenderingContext2D,
  plumbingNodes: BusinessGraphNode[],
  plumbingEntityId: Record<string, string>,
  entityById: Record<string, BusinessEntity>,
  transformK: number,
  focusSets: Set<string> | null,
): void {
  let cx = 0, cy = 0, count = 0, maxR = 0;
  Object.values(entityById).forEach((e) => {
    if (e.x == null || e.y == null) return;
    cx += e.x; cy += e.y; count += 1;
  });
  if (count) { cx /= count; cy /= count; }
  Object.values(entityById).forEach((e) => {
    if (e.x == null || e.y == null) return;
    maxR = Math.max(maxR, Math.hypot(e.x - cx, e.y - cy));
  });
  const unlinkedRingR = maxR + UNLINKED_RING_MARGIN;

  ctx.globalAlpha = 0.55;
  plumbingNodes.forEach((n) => {
    // Same reasoning as drawDocumentBadges above — an unfiltered plumbing
    // layer surfaced every OTHER source's leftover chunks/records while a
    // focus lens was active (COG-6233).
    if (!inLens(n, focusSets)) return;
    const phase = hashPhase(n.id);
    const anchor = entityById[plumbingEntityId[n.id]];
    let px: number, py: number;
    if (anchor && anchor.x != null && anchor.y != null) {
      const orbit = PLUMBING_ORBIT_MIN + (anchor._r ?? 5) + ((phase * 5) % 1) * PLUMBING_ORBIT_SPREAD;
      px = anchor.x + Math.cos(phase) * orbit;
      py = anchor.y + Math.sin(phase) * orbit;
      ctx.setLineDash([2 / transformK, 3 / transformK]);
      ctx.beginPath();
      ctx.moveTo(anchor.x, anchor.y);
      ctx.lineTo(px, py);
      ctx.strokeStyle = "rgba(91,104,128,0.4)";
      ctx.lineWidth = 0.7 / transformK;
      ctx.stroke();
      ctx.setLineDash([]);
    } else {
      px = cx + Math.sin(phase) * unlinkedRingR;
      py = cy + Math.cos(phase * 7) * unlinkedRingR * 0.75;
    }
    ctx.beginPath();
    ctx.arc(px, py, 3 / Math.sqrt(transformK), 0, Math.PI * 2);
    ctx.fillStyle = "#5B6880";
    ctx.fill();
  });
  ctx.globalAlpha = 1;
}
