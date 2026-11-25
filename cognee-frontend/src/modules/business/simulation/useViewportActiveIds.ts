"use client";

import { useEffect, useRef, useState, type RefObject } from "react";
import type { ZoomTransform } from "d3";
import type { BusinessEntity } from "../sceneTypes";

// Below this count d3-force already keeps up (existing datasets run in the
// hundreds) — capping would only add restart churn for no measurable win, so
// it stays off and every caller gets today's behavior (null = simulate
// everyone).
//
// Deliberately kept ABOVE getBrainGraph's MAX_NODES (1000), so one dataset
// stays uncapped however dense it is, and this covers only what the fetch cap
// cannot: a scene merging several datasets' graphs (CLO-578).
//
// It briefly sat at 500, below that cap, and the results argued against
// capping rather than for it. Measured at 792 entities: d3-force handled them
// without trouble, but the capping machinery itself never stopped moving the
// graph — every membership change rebuilds the simulation, and only the
// visible subset is laid out, so the offscreen remainder keeps stale
// positions and the global layout is no longer self-consistent (plainly
// visible on the minimap). Zooming changed membership too, so the graph
// shoved itself around on every zoom. Capping is a real optimization for a
// scene too big to simulate; it is a bad trade for one that isn't.
const ACTIVE_CAP_ENTITY_THRESHOLD = 1500;
// Screen-space margin kept active just outside the visible canvas, so a pan
// reveals an already-moving entity instead of a frozen one for a full recheck
// cycle.
const VIEWPORT_PADDING_PX = 400;
// An entity already being simulated is only dropped once it leaves this wider
// band, not the band that let it in. With one shared edge, an entity sitting
// near the boundary crosses back and forth under its own simulated motion and
// membership changes on almost every recheck — churn that costs a simulation
// rebuild each time and, while it shared an alpha with real dataset growth,
// showed up as a permanent pulse. The gap is what makes membership stable
// enough that a change means the CAMERA moved.
const VIEWPORT_RELEASE_PADDING_PX = 1000;
const RECHECK_INTERVAL_MS = 800;

function idsKeyOf(ids: string[]): string {
  return ids.slice().sort().join(",");
}

// Bounds the simulated d3-force graph to entities near the current camera
// viewport once the entity count is large enough for that to matter.
// Entities outside the returned set keep whatever position they last had
// (see useBusinessSimulation's activeIds param) instead of paying for a
// force-simulation tick every frame regardless of whether anyone can see
// them. Deliberately polled on an interval rather than read every RAF frame —
// transform lives in a ref precisely so panning/zooming never triggers a
// React re-render (see useBusinessCamera), and recomputing this set is only
// needed a few times a second, not sixty.
export function useViewportActiveIds(
  entities: BusinessEntity[],
  transformRef: RefObject<ZoomTransform>,
  width: number,
  height: number,
): Set<string> | null {
  const [activeIds, setActiveIds] = useState<Set<string> | null>(null);
  const lastKeyRef = useRef<string | null>(null);
  // Read inside recompute to decide which hysteresis band each entity is
  // judged against. Kept in a ref rather than read off `activeIds`: the
  // interval closes over the state value from the render that created it, so
  // it would keep comparing against a stale set.
  const activeRef = useRef<Set<string> | null>(null);
  // Camera identity the current set was derived from, so an unmoved camera
  // can skip the walk entirely (see recompute).
  const lastStampRef = useRef<string | null>(null);

  useEffect(() => {
    if (entities.length <= ACTIVE_CAP_ENTITY_THRESHOLD || width === 0 || height === 0) {
      lastKeyRef.current = null;
      activeRef.current = null;
      lastStampRef.current = null;
      setActiveIds(null);
      return;
    }
    // This effect re-running means the world changed (entities arrived, the
    // canvas resized), which is exactly when membership must be re-derived
    // even though the camera may not have moved.
    lastStampRef.current = null;

    const recompute = (): void => {
      const t = transformRef.current;
      // Membership is a function of the CAMERA, never of simulated motion.
      // Re-deriving it from live positions on every poll closed a feedback
      // loop: the set changing restarts the simulation, the restart's alpha
      // moves entities, entities leaving the set take their collide force
      // with them so the rest expand into the freed space and drift out too
      // — measured as the active count sliding 163 → 137 → 112 → 99 → 92 →
      // 87 → 84 in consecutive 0.8s polls with nobody touching anything,
      // which reads on screen as the graph endlessly shoving itself around.
      // A still camera must therefore produce a frozen set; entity arrivals
      // still get in, because new entities change `entities` and re-run this
      // effect.
      const stamp = `${t.k}|${t.x}|${t.y}`;
      if (stamp === lastStampRef.current) return;
      lastStampRef.current = stamp;
      const bandOf = (padding: number) => {
        const [minX, minY] = t.invert([-padding, -padding]);
        const [maxX, maxY] = t.invert([width + padding, height + padding]);
        return { minX, minY, maxX, maxY };
      };
      const enter = bandOf(VIEWPORT_PADDING_PX);
      const release = bandOf(VIEWPORT_RELEASE_PADDING_PX);
      const wasActive = activeRef.current;
      const ids: string[] = [];
      entities.forEach((n) => {
        // No position yet (brand new this poll, or seeding hasn't run for
        // this refetch's fresh object instances) — keep it active so it
        // gets placed immediately rather than sitting invisible at (0,0)
        // until the next recheck happens to include it.
        if (n.x == null || n.y == null) {
          ids.push(n.id);
          return;
        }
        // Hysteresis: which band applies depends on whether this entity is
        // already being simulated (see VIEWPORT_RELEASE_PADDING_PX).
        const band = wasActive?.has(n.id) ? release : enter;
        if (n.x >= band.minX && n.x <= band.maxX && n.y >= band.minY && n.y <= band.maxY) ids.push(n.id);
      });
      const key = idsKeyOf(ids);
      if (key === lastKeyRef.current) return;
      lastKeyRef.current = key;
      const next = new Set(ids);
      activeRef.current = next;
      setActiveIds(next);
    };

    recompute();
    const interval = setInterval(recompute, RECHECK_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [entities, transformRef, width, height]);

  return activeIds;
}
