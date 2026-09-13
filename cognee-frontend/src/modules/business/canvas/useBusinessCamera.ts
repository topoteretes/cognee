"use client";

import { useCallback, useEffect, useRef, type RefObject } from "react";
import { select, zoom as d3zoom, zoomIdentity, mean, type ZoomTransform } from "d3";
import type { BusinessEntity } from "../sceneTypes";
import { inLens } from "./businessEntityLayer";

// Hysteresis thresholds for the four semantic zoom levels
// (customer_tutorial.html:6410-6416) — L3 ("plumbing") is a toggle, not
// zoom-driven, so it isn't in this table.
const LEVEL_UP = [1.7, 3.2];
const LEVEL_DOWN = [1.4, 2.7];

// A single pass min/max — `Math.max(...array)` spreads the whole array onto
// the call stack (RangeError past ~100k args) and allocates a fresh spread
// every call. computeFitScale runs every draw frame, so this is on the hot
// path (COG-6233).
function minMax(values: number[]): { min: number; max: number } {
  let min = Infinity, max = -Infinity;
  for (const v of values) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return { min, max };
}

// The zoom at which the whole settled graph fits the viewport (same margins
// fitToEntities uses: 420px of side rails, 210px of top/bottom chrome, 85%
// breathing room, capped at 1.5 so a tiny graph doesn't blow up). Exported
// because the schema crossfade is defined relative to THIS, not to absolute
// zoom — see computeTypeFadeKMax in businessDraw.
export function computeFitScale(entities: BusinessEntity[], width: number, height: number): number {
  if (!entities.length || width <= 0 || height <= 0) return 1.5;
  const xs = minMax(entities.map((n) => n.x ?? 0)), ys = minMax(entities.map((n) => n.y ?? 0));
  const spanX = xs.max - xs.min + 120;
  const spanY = ys.max - ys.min + 120;
  return Math.max(
    0.1,
    Math.min(1.5, 0.85 * Math.min((width - 420) / (spanX || 1), (height - 210) / (spanY || 1))),
  );
}

function nextLevel(prev: number, k: number): number {
  let next = prev;
  if (prev === 0 && k > LEVEL_UP[0]) next = 1;
  else if (prev === 1 && k < LEVEL_DOWN[0]) next = 0;
  if (next <= 1 && k > LEVEL_UP[1]) next = 2;
  else if (next === 2 && k < LEVEL_DOWN[1]) next = 1;
  return next;
}

export interface BusinessCamera {
  transformRef: RefObject<ZoomTransform>;
  // level/plumbing are draw state, not UI state — read every frame by the
  // canvas's own RAF loop, never by a React render. Keeping them in refs
  // (not useState) is what the ticket calls the single biggest risk in this
  // port: a state setter here would recreate this hook's return object on
  // every zoom tick, and BusinessCanvas's draw effect depending on that
  // object would tear down and restart its RAF loop on every tick instead
  // of running as one persistent loop.
  levelRef: RefObject<number>;
  plumbingRef: RefObject<boolean>;
  lastInteractionRef: RefObject<number>;
  fitToEntities: (entities: BusinessEntity[], width: number, height: number, animate: boolean) => void;
  goToAltimeterLevel: (
    targetLevel: number, width: number, height: number, entities: BusinessEntity[], focusSets: Set<string> | null,
  ) => void;
  // Pan to a world point keeping the current zoom — the minimap's
  // click-to-navigate, which repositions without re-leveling.
  centerOnWorld: (wx: number, wy: number, width: number, height: number) => void;
  // Snapshot/restore for interruptions (the tour): put a captured transform
  // back instantly, cancelling any in-flight camera transition.
  applyTransform: (t: ZoomTransform, animate: boolean, durationMs?: number) => void;
}

// Ports the camera/zoom half of customer_tutorial.html's Simulation section
// (6396-6453): d3.zoom for pan/pinch/wheel, plus the two imperative moves
// (fit-to-content, jump-to-altimeter-level) the rest of the view drives.
//
// onLevelChange is for UI that needs to react to the level (e.g. highlighting
// the active altimeter button) — it fires only on an actual transition, not
// every zoom tick, so a consumer can safely put it in React state without
// reintroducing the render-loop-restart bug levelRef/plumbingRef exist to avoid.
export function useBusinessCamera(
  canvasRef: RefObject<HTMLCanvasElement | null>,
  onLevelChange?: (level: number, plumbing: boolean) => void,
): BusinessCamera {
  const transformRef = useRef<ZoomTransform>(zoomIdentity);
  const zoomBehaviorRef = useRef<ReturnType<typeof d3zoom<HTMLCanvasElement, unknown>> | null>(null);
  const lastInteractionRef = useRef(0);
  const levelRef = useRef(0);
  const plumbingRef = useRef(false);
  // Kept fresh every render so the zoom handler below (bound once) never
  // closes over a stale callback, without needing the effect that binds it
  // to re-run — rebinding d3-zoom on every render of a changing prop would
  // be its own version of the render-loop-restart bug.
  const onLevelChangeRef = useRef(onLevelChange);
  onLevelChangeRef.current = onLevelChange;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    // Google-Maps-style wheel handling (d3-zoom's own default): scroll and
    // pinch both zoom, panning is click-drag. A Figma-style split (scroll =
    // pan, ctrl/pinch = zoom) was tried here and reverted on feedback —
    // zoom-on-scroll is what this graph's users reach for first.
    const behavior = d3zoom<HTMLCanvasElement, unknown>()
      .scaleExtent([0.25, 8])
      .on("zoom", (event) => {
        transformRef.current = event.transform;
        lastInteractionRef.current = performance.now();
        const next = nextLevel(levelRef.current, event.transform.k);
        if (next !== levelRef.current) {
          levelRef.current = next;
          onLevelChangeRef.current?.(next, plumbingRef.current);
        }
      });
    zoomBehaviorRef.current = behavior;
    select(canvas).call(behavior);
    return () => {
      select(canvas).on(".zoom", null);
    };
  }, [canvasRef]);

  // fit() and the altimeter buttons animate at different speeds in source
  // (700ms vs 600ms) — durationMs defaults to fit's, goToAltimeterLevel
  // overrides it below.
  const applyTransform = useCallback((t: ZoomTransform, animate: boolean, durationMs = 700): void => {
    const canvas = canvasRef.current;
    const behavior = zoomBehaviorRef.current;
    if (!canvas || !behavior) return;
    const selection = select(canvas);
    if (animate) selection.transition().duration(durationMs).call(behavior.transform, t);
    else selection.call(behavior.transform, t);
  }, [canvasRef]);

  const fitToEntities = useCallback(
    (entities: BusinessEntity[], width: number, height: number, animate: boolean): void => {
      if (!entities.length) {
        // No real entities to fit to (e.g. a dataset with only plumbing/Skill
        // nodes) — still recenter on world origin at the default scale,
        // rather than leaving whatever transform a PREVIOUS dataset left
        // behind. The unlinked-plumbing ring (businessAuxLayers.ts) is drawn
        // around world (0,0) precisely so this keeps it in frame instead of
        // stranded wherever the last dataset's camera happened to be
        // (COG-6233).
        applyTransform(zoomIdentity.translate(width / 2, height / 2 - 34).scale(1.5), animate);
        return;
      }
      const xs = minMax(entities.map((n) => n.x ?? 0)), ys = minMax(entities.map((n) => n.y ?? 0));
      const minX = xs.min - 60, maxX = xs.max + 60;
      const minY = ys.min - 60, maxY = ys.max + 60;
      const k = computeFitScale(entities, width, height);
      const t = zoomIdentity
        .translate(width / 2, height / 2 - 34)
        .scale(k)
        .translate(-(minX + maxX) / 2, -(minY + maxY) / 2);
      applyTransform(t, animate);
    },
    [applyTransform],
  );

  const goToAltimeterLevel = useCallback(
    (
      targetLevel: number, width: number, height: number, entities: BusinessEntity[],
      focusSets: Set<string> | null,
    ): void => {
      lastInteractionRef.current = performance.now();
      if (targetLevel === 3) {
        plumbingRef.current = !plumbingRef.current;
        onLevelChangeRef.current?.(levelRef.current, plumbingRef.current);
        return;
      }
      plumbingRef.current = false;
      // An explicit altimeter click is a direct level choice, not a passive
      // zoom — set it immediately rather than waiting for the animated
      // transition to happen to cross the hysteresis threshold on its own.
      levelRef.current = targetLevel;
      onLevelChangeRef.current?.(targetLevel, false);
      // A source focus lens dims OTHER sources' entities but never moves
      // them out of the simulation — they keep real x/y elsewhere in the
      // whole graph. Framing on the FULL entity set here recentered the
      // camera on the whole graph's centroid at a much tighter zoom,
      // stranding the small focused cluster outside the viewport — reading
      // as "everything disappeared" after clicking an altimeter level while
      // a lens was active (COG-6233). Fall back to the full set only if the
      // lens somehow matches nothing, so this never zooms to an empty mean.
      const inFocus = focusSets ? entities.filter((n) => inLens(n, focusSets)) : entities;
      const framed = inFocus.length ? inFocus : entities;
      // Business (level 0) is the schema view, and where the schema
      // crossfade sits now depends on the graph's own fit scale (see
      // computeTypeFadeKMax) — a fixed k=1 landed a large graph on the
      // entity layer instead, with the button seemingly doing nothing.
      const k = targetLevel === 0
        ? Math.min(1, 0.8 * computeFitScale(framed, width, height))
        : targetLevel === 1 ? 2.1 : 3.8;
      const cx = mean(framed, (n) => n.x ?? 0) || 0;
      const cy = mean(framed, (n) => n.y ?? 0) || 0;
      const t = zoomIdentity.translate(width / 2, height / 2).scale(k).translate(-cx, -cy);
      applyTransform(t, true, 600);
    },
    [applyTransform],
  );

  const centerOnWorld = useCallback(
    (wx: number, wy: number, width: number, height: number): void => {
      lastInteractionRef.current = performance.now();
      const k = transformRef.current.k;
      const t = zoomIdentity.translate(width / 2, height / 2).scale(k).translate(-wx, -wy);
      applyTransform(t, true, 400);
    },
    [applyTransform],
  );

  return { transformRef, levelRef, plumbingRef, lastInteractionRef, fitToEntities, goToAltimeterLevel, centerOnWorld, applyTransform };
}
