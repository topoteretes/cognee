"use client";

import { useEffect, useState } from "react";

/** Below this, a wide data table has to drop columns rather than scroll. */
export const NARROW_VIEWPORT_PX = 720;

/**
 * Whether the viewport is too narrow for a wide layout.
 *
 * Starts false and corrects on mount rather than reading `window` during
 * render: the server has no viewport, so a first render that guessed would
 * disagree with the client's and throw a hydration mismatch. The one-frame
 * consequence is that a phone paints the wide table before switching, which
 * is why the caller must not depend on this for anything but layout.
 */
export function useIsNarrowViewport(maxWidthPx: number = NARROW_VIEWPORT_PX): boolean {
  const [narrow, setNarrow] = useState(false);

  useEffect(() => {
    if (typeof window.matchMedia !== "function") return;
    const query = window.matchMedia(`(max-width: ${maxWidthPx}px)`);
    setNarrow(query.matches);
    const onChange = (e: MediaQueryListEvent): void => setNarrow(e.matches);
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, [maxWidthPx]);

  return narrow;
}
