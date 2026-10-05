"use client";

import { useEffect, useRef, useState, type RefObject } from "react";
import type { ZoomTransform } from "d3";
import type { BusinessEntity } from "../sceneTypes";

// Below this count d3-force already keeps up (existing datasets run in the
// hundreds) — capping would only add restart churn for no measurable win, so
// it stays off and every caller gets today's behavior (null = simulate
// everyone).
//
// A single dataset can't reach this today: getBrainGraph fetches at most
// MAX_NODES (500) nodes and entities are a subset of those (CLO-597), so the
// fetch cap keeps one dataset's simulation cheap on its own. This threshold
// covers what that cap can't — a scene merging several datasets' graphs
// (CLO-578) — and a raised MAX_NODES.
const ACTIVE_CAP_ENTITY_THRESHOLD = 500;
// Screen-space margin kept active just outside the visible canvas, so a pan
// reveals an already-moving entity instead of a frozen one for a full recheck
// cycle.
const VIEWPORT_PADDING_PX = 400;
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

  useEffect(() => {
    if (entities.length <= ACTIVE_CAP_ENTITY_THRESHOLD || width === 0 || height === 0) {
      lastKeyRef.current = null;
      setActiveIds(null);
      return;
    }

    const recompute = (): void => {
      const t = transformRef.current;
      const [minX, minY] = t.invert([-VIEWPORT_PADDING_PX, -VIEWPORT_PADDING_PX]);
      const [maxX, maxY] = t.invert([width + VIEWPORT_PADDING_PX, height + VIEWPORT_PADDING_PX]);
      const ids: string[] = [];
      entities.forEach((n) => {
        // No position yet (brand new this poll, or seeding hasn't run for
        // this refetch's fresh object instances) — keep it active so it
        // gets placed immediately rather than sitting invisible at (0,0)
        // until the next recheck happens to include it.
        const inView = n.x == null || n.y == null || (n.x >= minX && n.x <= maxX && n.y >= minY && n.y <= maxY);
        if (inView) ids.push(n.id);
      });
      const key = idsKeyOf(ids);
      if (key === lastKeyRef.current) return;
      lastKeyRef.current = key;
      setActiveIds(new Set(ids));
    };

    recompute();
    const interval = setInterval(recompute, RECHECK_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [entities, transformRef, width, height]);

  return activeIds;
}
