"use client";

import { useEffect, useRef, type RefObject } from "react";
import type { BusinessCanvasHandle } from "./canvas/BusinessCanvas";
import type { Spotlight } from "./canvas/businessDraw";
import type { BrainState } from "./sceneTypes";
import { truncate } from "./textUtils";

const IDLE_THRESHOLD_MS = 25000;
const CHECK_INTERVAL_MS = 5000;
const INSIGHT_SPOTLIGHT_MS = 9000;
const SEEN_CAP = 20;
// Distinct from the cyan (#43D9E8) real state-change events use elsewhere
// (cognify complete, source focus, dataset switch) — an ambient "did you
// know" tip is passive, not a state change, and sharing a color with real
// events made them indistinguishable in the narration bar (COG-6233 UX
// audit).
const INSIGHT_COLOR = "#9B8CD9";
// A chunk/document entity's "name" sometimes IS its raw content (a whole
// ingested file, occasionally including secrets like an API key) rather
// than a short human label — narrate() has no length cap of its own, so an
// untruncated name here put raw record content, potentially sensitive,
// directly into the bottom bar.
const INSIGHT_NAME_MAX_CHARS = 60;

interface Insight {
  key: string;
  ids: Set<string>;
  text: string;
}

// A cross-source bridge is the more interesting pick when one exists (it's
// the one relationship type the source-focus lens can't show on its own);
// otherwise the highest-importance record not yet surfaced. `seen` avoids
// repeating the same pick until the pool of options is exhausted.
function pickInsight(brainState: BrainState, seen: Set<string>): Insight | null {
  const bridges = brainState.semanticLinks.filter((l) => l._bridge);
  const freshBridges = bridges.filter((l) => !seen.has(`${l._sid}|${l._tid}`));
  const pool = freshBridges.length ? freshBridges : bridges;
  if (pool.length) {
    const l = pool[Math.floor(Math.random() * pool.length)];
    const a = brainState.entityById[l._sid], b = brainState.entityById[l._tid];
    if (a?.name && b?.name) {
      return {
        key: `${l._sid}|${l._tid}`,
        ids: new Set([a.id, b.id]),
        text: `${truncate(String(a.name), INSIGHT_NAME_MAX_CHARS)} connects to ${truncate(String(b.name), INSIGHT_NAME_MAX_CHARS)} across sources — a bridge this model already sees`,
      };
    }
  }
  const ranked = [...brainState.entities]
    .filter((e) => e.name && !e.is_unnamed)
    .sort((x, y) => (y.importance || 0) - (x.importance || 0));
  const entity = ranked.find((e) => !seen.has(e.id)) || ranked[0];
  if (!entity?.name) return null;
  return {
    key: entity.id,
    ids: new Set([entity.id]),
    text: `${truncate(String(entity.name), INSIGHT_NAME_MAX_CHARS)} is one of the most connected records in this model`,
  };
}

// Self-guided "did you know" moments — new in this port (no source
// equivalent). When nothing has touched the camera in a while (the same
// idle signal a live search event's pending-chip gate already uses, see
// useBusinessQaSurface), briefly surface a real bridge or a highly-
// connected record with a spotlight+narration — reviewer feedback flagged
// the camera-fly this originally also did as "the graph jumps" after any
// period of inactivity, so this stays passive: the spotlight only lights up
// what's already on screen, it never moves the camera to chase it.
export function useBusinessAutoInsights(
  canvasRef: RefObject<BusinessCanvasHandle | null>,
  brainState: BrainState | null,
  narrate: (text: string, color?: string) => void,
  setSpotlight: (spotlight: Spotlight | null) => void,
  enabled: boolean,
): void {
  const seenRef = useRef<Set<string>>(new Set());
  // A ref, not a local inside the effect: `brainState` comes from
  // useBusinessScene, memoized off React Query data that refetches every 8s
  // (the focused dataset's graph, useBrainGraph) — if that response isn't
  // perfectly structurally identical
  // every time, `brainState` gets a new reference each poll, re-running this
  // effect every ~8s. A local "started watching at" timestamp would reset
  // on every one of those re-runs and never accumulate the real 25s this
  // needs, silently killing the feature entirely. Setting it only once (and
  // clearing it when this effect turns off) keeps it stable across those
  // incidental restarts while still resetting for a genuine re-enable
  // (tour ending, a focus lens clearing).
  const startedAtRef = useRef<number | null>(null);

  useEffect(() => {
    if (!enabled || !brainState) {
      startedAtRef.current = null;
      return;
    }
    // getIdleMs() reports Infinity before the camera has EVER been touched
    // (BusinessCanvas.tsx's own comment: that sentinel means "always
    // auto-focus," for the live-event camera-steal check it was ported
    // for). Reused here for a different purpose, that same sentinel made
    // this effect treat a just-loaded page as maximally idle on its very
    // first 5s check — the first tip could fire within ~5s of landing on
    // the page instead of after a real 25s of inactivity. Gating on the
    // smaller of "real camera idle time" and "time since this effect first
    // started watching" means a never-touched camera still has to wait out
    // a genuine IDLE_THRESHOLD_MS, while a real interaction later still
    // gates on getIdleMs() as before.
    if (startedAtRef.current === null) startedAtRef.current = performance.now();
    const interval = setInterval(() => {
      const idleMs = Math.min(canvasRef.current?.getIdleMs() ?? 0, performance.now() - (startedAtRef.current ?? performance.now()));
      if (idleMs < IDLE_THRESHOLD_MS) return;
      const insight = pickInsight(brainState, seenRef.current);
      if (!insight) return;
      seenRef.current.add(insight.key);
      if (seenRef.current.size > SEEN_CAP) seenRef.current.clear();
      const startedAt = performance.now();
      setSpotlight({ ids: insight.ids, startedAt, until: startedAt + INSIGHT_SPOTLIGHT_MS, source: "insight" });
      narrate(`tip: ${insight.text}`, INSIGHT_COLOR);
    }, CHECK_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [enabled, brainState, canvasRef, narrate, setSpotlight]);
}
