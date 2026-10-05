"use client";

import { useCallback, useEffect, useRef, useState, type RefObject } from "react";
import type { BusinessCanvasHandle } from "./canvas/BusinessCanvas";
import type { Spotlight } from "./canvas/businessDraw";
import type { BrainState } from "./sceneTypes";
import type { BusinessGraphNode, SessionEvent } from "./types";
import { truncate } from "./textUtils";
import { setsOf } from "./computeBrainState";
import { useReasoningTraceWalk, MAX_TRACE_DURATION_MS } from "./useReasoningTraceWalk";

// Below this many contributing facts, walking node-by-node adds ceremony
// without a story to tell — go straight to the combined highlight, same as
// before this feature existed.
const MIN_IDS_FOR_TRACE_WALK = 3;

const SPOTLIGHT_DURATION_MS = 9000;
// The provisional asking window a search event opens, before the walk's
// real length is known: long enough that the longest possible walk still
// hands over to finalizeAnswer inside it (finalizeAnswer then shortens or
// extends it to the answer window it actually opened).
const MAX_ASKING_WINDOW_MS = MAX_TRACE_DURATION_MS + SPOTLIGHT_DURATION_MS;
const ANSWER_DURATION_MS = 14000;
const SESSION_MEMORY_DURATION_MS = 20000;
// _bvLiveEvent's threshold for "the presenter touched anything recently" —
// below it, a live search event docks a quiet chip instead of grabbing the
// camera (customer_tutorial.html ~7048).
const IDLE_THRESHOLD_MS = 8000;

// The window an agent is credited with a live answer for — one continuous
// envelope from the search event to the end of the answer's own spotlight,
// NOT a fixed timer off the event. A trace walk (up to 8 steps x 900ms)
// runs before finalizeAnswer opens the 9s answer window, so a 9s timer
// started at the event expired ~1.8s into the very window the marker exists
// to annotate (CLO-606). `until` is extended once the answer lands; the
// marker's fade envelope reads both bounds off this, not off whichever
// spotlight happens to be active.
export interface AskingWindow {
  principalId: string;
  startedAt: number;
  until: number;
}

export interface BusinessQaSurface {
  askingPrincipalId: string | null;
  asking: AskingWindow | null;
  answerEvent: SessionEvent | null;
  sessionMemoryPrincipalName: string | null;
  pendingSearchEvent: SessionEvent | null;
  distilledSets: string[];
  dismissAnswer: () => void;
  dismissSessionMemory: () => void;
  playPendingSearchEvent: () => void;
  dismissPendingSearchEvent: () => void;
  onOpenSessionMemory: (principalId: string, principalName: string) => void;
  // Backs the live search bar (new in this port, no source equivalent) —
  // /v1/search doesn't return node_ids the way the live-events stream does,
  // so a manually-typed question can show its real answer text but can't
  // drive the spotlight/camera-fly the way an agent-attributed search can.
  showManualAnswer: (question: string, answer: string) => void;
}

// Owns the live Q&A surface: playSearchEvent (customer_tutorial.html
// ~7020-7075 — spotlight, agent glow, camera-fit, amber narration, floating
// answer card) and showSessionMemory's click trigger (~7080). Pulled out of
// BusinessView because this concern has its own timers, its own idle-gating
// effect, and its own shared-surface bookkeeping (an answer and a session-
// memory card both draw into the same slot, so opening one clears the
// other) — enough moving parts to be its own hook rather than inline state.
export function useBusinessQaSurface(
  canvasRef: RefObject<BusinessCanvasHandle | null>,
  latestSearchEvent: SessionEvent | null,
  consumeLatestSearchEvent: () => void,
  agents: BusinessGraphNode[],
  brainState: BrainState | null,
  setSpotlight: (spotlight: Spotlight | null) => void,
  setFocusSets: (sets: Set<string> | null) => void,
  narrate: (text: string, color?: string) => void,
  datasetId: string | null,
): BusinessQaSurface {
  const [asking, setAsking] = useState<AskingWindow | null>(null);
  const askingTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // A live search's answer and an agent's session-memory history are the
  // same floating surface source draws into one shared DOM node — opening
  // one must clear the other, not stack two cards at the same spot.
  const [answerEvent, setAnswerEvent] = useState<SessionEvent | null>(null);
  const [sessionMemoryPrincipalName, setSessionMemoryPrincipalName] = useState<string | null>(null);
  const [pendingSearchEvent, setPendingSearchEvent] = useState<SessionEvent | null>(null);
  const qaSurfaceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reasoningTraceWalk = useReasoningTraceWalk(canvasRef, narrate, setSpotlight);

  const distilledSets = Object.keys(brainState?.setColor ?? {}).filter((name) => brainState?.isSessionSet(name));

  // Opens the asking window (a fresh search event) or extends its end
  // (finalizeAnswer, once the real answer window is known), always keeping
  // the original startedAt so the marker's fade-in only ever plays once.
  const holdAskingWindow = useCallback((principalId: string | null, untilMs: number) => {
    if (askingTimerRef.current) clearTimeout(askingTimerRef.current);
    if (!principalId) return;
    // Both bounds read from one clock sample, and taken OUTSIDE the updater:
    // React may invoke an updater more than once, which would otherwise move
    // startedAt and restart the marker's fade-in mid-window.
    const now = performance.now();
    const until = now + untilMs;
    setAsking((current) =>
      current && current.principalId === principalId
        ? { ...current, until }
        : { principalId, startedAt: now, until },
    );
    askingTimerRef.current = setTimeout(
      () => setAsking((current) => (current && current.principalId === principalId ? null : current)),
      untilMs,
    );
  }, []);

  // The instant, whole-set reveal — either the answer to a small handful of
  // facts (no story to walk), or what a reasoning-trace walk lands on once
  // it's done stepping through the facts one by one.
  const finalizeAnswer = useCallback(
    (evt: SessionEvent, ids: Set<string>, principalId: string | null) => {
      const startedAt = performance.now();
      setSpotlight({
        ids,
        startedAt,
        until: startedAt + SPOTLIGHT_DURATION_MS,
        source: "answer",
        question: evt.question,
      });
      // The answer window is the one the marker actually annotates, so the
      // agent's own window ends with it rather than with a timer that
      // started back at the search event, before the walk had even run.
      holdAskingWindow(principalId, SPOTLIGHT_DURATION_MS);
      const text = `agent asked: "${truncate(String(evt.question || ""), 70)}" — ${ids.size} connected facts produced the answer`;
      narrate(text, "#F5A83C");
      requestAnimationFrame(() => canvasRef.current?.focusOnIds(ids));
      setSessionMemoryPrincipalName(null);
      setAnswerEvent(evt);
      if (qaSurfaceTimerRef.current) clearTimeout(qaSurfaceTimerRef.current);
      qaSurfaceTimerRef.current = setTimeout(() => setAnswerEvent(null), ANSWER_DURATION_MS);
    },
    [canvasRef, setSpotlight, narrate, holdAskingWindow],
  );

  // Source always resolves to "the first agent" in practice (its own
  // agentId argument is never actually passed at the call site) — this
  // ports that literal behavior rather than inventing a more precise
  // per-event attribution the source itself doesn't have.
  const playSearchEvent = useCallback(
    (evt: SessionEvent) => {
      const ids = new Set((evt.node_ids as string[] | undefined) || []);
      if (!ids.size) return;
      reasoningTraceWalk.stop();

      const agentId = agents.length > 0 ? agents[0].id : null;
      // Provisional end: enough to cover a walk-free answer on its own, and
      // long enough to bridge the longest possible walk (MAX_TRACE_STEPS x
      // STEP_DWELL_MS) into finalizeAnswer, which then extends it to the
      // real answer window.
      holdAskingWindow(agentId, MAX_ASKING_WINDOW_MS);

      if (ids.size >= MIN_IDS_FOR_TRACE_WALK && brainState) {
        reasoningTraceWalk.play(ids, brainState.semanticLinks, brainState.entityById, () =>
          finalizeAnswer(evt, ids, agentId),
        );
      } else {
        finalizeAnswer(evt, ids, agentId);
      }
    },
    [agents, brainState, reasoningTraceWalk, finalizeAnswer, holdAskingWindow],
  );

  // A pending/playing search event carries node_ids from whatever dataset
  // was active when it arrived — surviving a dataset switch, playing it
  // filters to ids that don't exist in the new dataset, recentering the
  // camera on an empty focus set and dimming everything (COG-6233).
  useEffect(() => {
    setPendingSearchEvent(null);
    setAnswerEvent(null);
    setSessionMemoryPrincipalName(null);
    setAsking(null);
    if (qaSurfaceTimerRef.current) clearTimeout(qaSurfaceTimerRef.current);
    if (askingTimerRef.current) clearTimeout(askingTimerRef.current);
    reasoningTraceWalk.stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetId]);

  useEffect(
    () => () => {
      if (qaSurfaceTimerRef.current) clearTimeout(qaSurfaceTimerRef.current);
      if (askingTimerRef.current) clearTimeout(askingTimerRef.current);
    },
    [],
  );

  useEffect(() => {
    if (!latestSearchEvent) return;
    const idleMs = canvasRef.current?.getIdleMs() ?? Infinity;
    if (idleMs > IDLE_THRESHOLD_MS) {
      playSearchEvent(latestSearchEvent);
    } else {
      setPendingSearchEvent(latestSearchEvent);
    }
    consumeLatestSearchEvent();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latestSearchEvent, playSearchEvent]);

  // Ports showSessionMemory's click trigger — reuses isSessionSet
  // (computeBrainState) to find which session-memory sets, if any, the
  // active brain already has, then lenses the canvas to them the same way
  // source's setFocusSets(sessionSets, 'distilled session memory (agent-made,
  // in amber)') does, narration included (customer_tutorial.html ~6563).
  // Triggered from a user's own operator card, not an agent's — session
  // memory is the user's conversation history, so it belongs to them.
  const onOpenSessionMemory = useCallback(
    (_principalId: string, principalName: string) => {
      setAnswerEvent(null);
      setSessionMemoryPrincipalName(principalName);
      if (qaSurfaceTimerRef.current) clearTimeout(qaSurfaceTimerRef.current);
      qaSurfaceTimerRef.current = setTimeout(() => setSessionMemoryPrincipalName(null), SESSION_MEMORY_DURATION_MS);
      if (distilledSets.length) {
        setFocusSets(new Set(distilledSets));
        const entityCount = brainState?.entities.filter((e) => setsOf(e).some((x) => distilledSets.includes(x))).length ?? 0;
        narrate(
          `showing only distilled session memory (user-made, in amber) — ${entityCount} entities · click again for everything`,
          "#43D9E8",
        );
      }
    },
    [distilledSets, setFocusSets, brainState, narrate],
  );

  const showManualAnswer = useCallback(
    (question: string, answer: string) => {
      setSessionMemoryPrincipalName(null);
      setAnswerEvent({ kind: "search", question, answer });
      if (qaSurfaceTimerRef.current) clearTimeout(qaSurfaceTimerRef.current);
      qaSurfaceTimerRef.current = setTimeout(() => setAnswerEvent(null), ANSWER_DURATION_MS);
      narrate(`answered: "${truncate(question, 70)}"`, "#43D9E8");
    },
    [narrate],
  );

  return {
    askingPrincipalId: asking?.principalId ?? null,
    asking,
    answerEvent,
    sessionMemoryPrincipalName,
    pendingSearchEvent,
    distilledSets,
    dismissAnswer: () => setAnswerEvent(null),
    dismissSessionMemory: () => setSessionMemoryPrincipalName(null),
    playPendingSearchEvent: () => {
      setPendingSearchEvent((evt) => {
        if (evt) playSearchEvent(evt);
        return null;
      });
    },
    dismissPendingSearchEvent: () => setPendingSearchEvent(null),
    onOpenSessionMemory,
    showManualAnswer,
  };
}
