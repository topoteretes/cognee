"use client";

import { useCallback, useRef, useState } from "react";

export interface NarrationDisplay {
  text: string | null;
  color: string;
  opacity: number;
}

const SWAP_DELAY_MS = 250;
const DIM_DELAY_MS = 9000;
const DIM_OPACITY = 0.55;
const DEFAULT_COLOR = "#7E8CA6";

// Ports narrate() (customer_tutorial.html ~7130-7138) as ONE shared
// narration channel — source has a single narration line and every caller
// (focusKnowledge, cyclePermission, playSearchEvent, the growth check, the
// source-focus toggle) writes through it, rather than each maintaining its
// own text/timer that all fall back to some other "default" text. Calling
// narrate() fades the CURRENT text out, swaps in the new text+color 250ms
// later, then — after 9s — dims to 55% opacity. It never reverts to
// anything else on its own; it just dims and sits there until the next
// call, exactly like source.
export function useNarration(): { display: NarrationDisplay; narrate: (text: string, color?: string) => void } {
  const [display, setDisplay] = useState<NarrationDisplay>({ text: null, color: DEFAULT_COLOR, opacity: 0 });
  const swapTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const dimTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const narrate = useCallback((text: string, color: string = DEFAULT_COLOR) => {
    if (swapTimerRef.current) clearTimeout(swapTimerRef.current);
    if (dimTimerRef.current) clearTimeout(dimTimerRef.current);
    setDisplay((prev) => ({ ...prev, opacity: 0 }));
    swapTimerRef.current = setTimeout(() => {
      setDisplay({ text, color, opacity: 1 });
    }, SWAP_DELAY_MS);
    dimTimerRef.current = setTimeout(() => {
      setDisplay((prev) => ({ ...prev, opacity: DIM_OPACITY }));
    }, DIM_DELAY_MS);
  }, []);

  return { display, narrate };
}
