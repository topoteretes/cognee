"use client";

import { useEffect, useState } from "react";

const QUERY = "(prefers-reduced-motion: reduce)";

// The exported HTML this view is ported from has zero prefers-reduced-motion
// rules anywhere — flagged explicitly in the ticket as easy to forget.
// Canvas animation (node birth fade/scale) is JS-driven, so a CSS media
// query alone can't reach it; this hook lets useBusinessSimulation /
// businessDraw skip straight to the settled state instead.
export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);

  useEffect(() => {
    const mql = window.matchMedia(QUERY);
    setReduced(mql.matches);
    const onChange = (e: MediaQueryListEvent): void => setReduced(e.matches);
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, []);

  return reduced;
}
