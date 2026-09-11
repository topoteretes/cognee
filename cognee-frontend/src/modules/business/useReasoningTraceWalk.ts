"use client";

import { useCallback, useEffect, useRef, useState, type RefObject } from "react";
import type { BusinessCanvasHandle } from "./canvas/BusinessCanvas";
import type { Spotlight } from "./canvas/businessDraw";
import type { BusinessEntity, SemanticLink } from "./sceneTypes";
import { buildReasoningTrace, MAX_TRACE_STEPS } from "./reasoningTrace";

const STEP_DWELL_MS = 900;
// The longest a walk can run before finalizeAnswer gets its turn — the
// asking-agent window (useBusinessQaSurface) has to bridge this, or the
// marker expires while the walk it is annotating is still stepping.
export const MAX_TRACE_DURATION_MS = MAX_TRACE_STEPS * STEP_DWELL_MS;
const TRACE_COLOR = "#F5A83C";

export interface ReasoningTraceWalk {
  isPlaying: boolean;
  play: (
    ids: Set<string>,
    semanticLinks: SemanticLink[],
    entityById: Record<string, BusinessEntity>,
    onDone: () => void,
  ) => void;
  stop: () => void;
}

// Turns a search answer's contributing node ids into a node-by-node camera
// walk ("watch the graph reason") before the existing instant full-set
// highlight lands — same runStep/cancelledRef/setTimeout shape as
// useBusinessTour's altimeter flythrough, just walking a reasoning path
// instead of fixed zoom levels.
export function useReasoningTraceWalk(
  canvasRef: RefObject<BusinessCanvasHandle | null>,
  narrate: (text: string, color?: string) => void,
  setSpotlight: (spotlight: Spotlight | null) => void,
): ReasoningTraceWalk {
  const [isPlaying, setIsPlaying] = useState(false);
  const cancelledRef = useRef(false);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const stop = useCallback(() => {
    cancelledRef.current = true;
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    setIsPlaying(false);
  }, []);

  const play = useCallback(
    (
      ids: Set<string>,
      semanticLinks: SemanticLink[],
      entityById: Record<string, BusinessEntity>,
      onDone: () => void,
    ) => {
      const steps = buildReasoningTrace(ids, semanticLinks, entityById);
      if (!steps.length) {
        onDone();
        return;
      }
      cancelledRef.current = false;
      setIsPlaying(true);
      const seenSoFar = new Set<string>();
      const runStep = (index: number): void => {
        if (cancelledRef.current || index >= steps.length) {
          setIsPlaying(false);
          if (!cancelledRef.current) onDone();
          return;
        }
        const step = steps[index];
        seenSoFar.add(step.id);
        const startedAt = performance.now();
        setSpotlight({ ids: new Set(seenSoFar), startedAt, until: startedAt + STEP_DWELL_MS * 2, source: "trace" });
        canvasRef.current?.focusOnIds(new Set([step.id]));
        narrate(step.narration, TRACE_COLOR);
        timeoutRef.current = setTimeout(() => runStep(index + 1), STEP_DWELL_MS);
      };
      runStep(0);
    },
    [canvasRef, narrate, setSpotlight],
  );

  useEffect(
    () => () => {
      cancelledRef.current = true;
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    },
    [],
  );

  return { isPlaying, play, stop };
}
