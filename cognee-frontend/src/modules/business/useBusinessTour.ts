"use client";

import { useCallback, useEffect, useRef, useState, type RefObject } from "react";
import type { ZoomTransform } from "d3";
import type { BusinessCanvasHandle } from "./canvas/BusinessCanvas";
import type { BrainState } from "./sceneTypes";

const STEP_DWELL_MS = 5000;

const STEP_TEXTS: Array<(brain: BrainState | null) => string> = [
  (brain) => (brain
    ? `this is your business — ${brain.typeNodes.length} kind${brain.typeNodes.length === 1 ? "" : "s"} of things across ${brain.sourceNames.length} source${brain.sourceNames.length === 1 ? "" : "s"}, one connected model`
    : "this is your business — one connected model"),
  () => "zoom in, and the people, accounts, and records behind each kind take shape",
  () => "every line here is a real connection — how one record relates to another",
  () => "at the deepest level: every record your agents can actually search",
];

export interface BusinessTour {
  isPlaying: boolean;
  start: () => void;
  stop: () => void;
}

// A scripted flythrough of all four altimeter levels, narrating each one —
// new in this port (no source equivalent), meant to carry an unattended
// demo through the model: press play and the camera + narration do the
// presenting. goToAltimeterLevel's own transform already dispatches a d3
// zoom event on the canvas (see useBusinessCamera's applyTransform), which
// resets the idle timer other features (auto-insights, the pending-search
// chip) key off of — so the tour naturally keeps them quiet while it plays,
// with no explicit coordination needed here.
export function useBusinessTour(
  canvasRef: RefObject<BusinessCanvasHandle | null>,
  narrate: (text: string, color?: string) => void,
  brainState: BrainState | null,
): BusinessTour {
  const [isPlaying, setIsPlaying] = useState(false);
  const cancelledRef = useRef(false);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const preTourTransformRef = useRef<ZoomTransform | null>(null);

  // Stopping must feel instantaneous — the blur drops on this render and
  // the camera snaps straight back to wherever the user left it before the
  // tour, no animation, no waiting on whatever transition was in flight.
  const stop = useCallback(() => {
    cancelledRef.current = true;
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    setIsPlaying(false);
    const preTour = preTourTransformRef.current;
    preTourTransformRef.current = null;
    if (preTour) canvasRef.current?.setTransformNow(preTour);
  }, [canvasRef]);

  const start = useCallback(() => {
    cancelledRef.current = false;
    preTourTransformRef.current = canvasRef.current?.getTransform() ?? null;
    setIsPlaying(true);
    const runStep = (index: number): void => {
      if (cancelledRef.current || index >= STEP_TEXTS.length) {
        // Finished (or already cancelled): the snapshot is only for
        // interruptions — a completed tour stays where it ended.
        preTourTransformRef.current = null;
        setIsPlaying(false);
        return;
      }
      canvasRef.current?.goToAltimeterLevel(index);
      narrate(STEP_TEXTS[index](brainState), "#43D9E8");
      timeoutRef.current = setTimeout(() => runStep(index + 1), STEP_DWELL_MS);
    };
    runStep(0);
  }, [canvasRef, narrate, brainState]);

  useEffect(() => () => {
    cancelledRef.current = true;
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
  }, []);

  return { isPlaying, start, stop };
}
