"use client";

import { useEffect, useState, type RefObject } from "react";

export interface CanvasSize {
  width: number;
  height: number;
  dpr: number;
}

// Ports resize() (customer_tutorial.html:6320) — device-pixel-ratio-aware
// canvas backing size, kept in sync with the wrapping element via
// ResizeObserver instead of a window resize listener, so it also reacts to
// layout changes (rail collapse, panel open) that don't fire window resize.
export function useCanvasSize(containerRef: RefObject<HTMLElement | null>): CanvasSize {
  const [size, setSize] = useState<CanvasSize>({ width: 0, height: 0, dpr: 1 });

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const update = (): void => {
      setSize({
        width: el.clientWidth,
        height: el.clientHeight,
        dpr: window.devicePixelRatio || 1,
      });
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(el);
    return () => observer.disconnect();
  }, [containerRef]);

  return size;
}
