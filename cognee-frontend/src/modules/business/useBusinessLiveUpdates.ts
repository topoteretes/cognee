"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { addBreadcrumb, captureMessage } from "@/utils/monitoring";
import { reportClientLog } from "@/services/http";
import type { CogneeInstance } from "@/modules/instances/types";
import { openLiveSocket, type LiveSocketFrame } from "./liveSocket";
import { reportDroppedSessionEvents } from "./sessionEvent";
import { pollLiveEvents, type CursorRef } from "./pollLiveEvents";
import { brainGraphQueryKey } from "./useBrainGraph";
import { createGraphRefreshThrottle } from "./graphRefreshThrottle";
import type { SessionEvent } from "./types";

const MAX_EVENTS = 200;

function eventKey(e: SessionEvent): string {
  return `${e.kind || "search"}|${e.qa_id || ""}|${e.time || ""}|${e.question || ""}`;
}

export interface LiveUpdatesOptions {
  // Called with `true` once the socket has completed a handshake and is
  // therefore delivering `graph_grew`, and with `false` whenever it is not:
  // no socket at all, reconnecting, or given up. The graph query's own poll
  // is what this gates (see useBrainGraph), so the flag has to track the
  // socket's HEALTH rather than the instance's capability to open one - an
  // instance that can build a socket URL for an endpoint the pod does not
  // serve yet would otherwise silently lose its growth signal entirely.
  onGraphPushChange?: (active: boolean) => void;
}

export interface LiveUpdates {
  events: SessionEvent[];
  live: boolean;
  latestSearchEvent: SessionEvent | null;
  consumeLatestSearchEvent: () => void;
  // Every entity id that has EVER appeared in a search event's node_ids
  // across this session's accumulated event log — not just the current 9s
  // spotlight — so the canvas can mark "this has answered a question
  // before" as a lasting fact rather than a moment.
  answeredIds: Set<string>;
}

// Prefers the dataset-growth/session-events socket (see liveSocket.ts) and
// falls back to the 1.5s live-events poll (pollLiveEvents.ts) only when the
// socket can't be used at all: no `socketTarget` on this instance (local
// mode, test doubles), a rejected handshake (close 1008 — retrying would
// only replay the same rejection), or 5 consecutive reconnect failures. Both
// transports append through the same dedupe path below so a mid-session
// handover from socket to poll can't double up an event the socket had
// already delivered.
export function useBusinessLiveUpdates(
  datasetId: string | null,
  cogniInstance: CogneeInstance,
  options: LiveUpdatesOptions = {},
): LiveUpdates {
  const [events, setEvents] = useState<SessionEvent[]>([]);
  const [live, setLive] = useState(true);
  const [latestSearchEvent, setLatestSearchEvent] = useState<SessionEvent | null>(null);
  const cursorRef = useRef<string | null>(null);
  const seenRef = useRef<Set<string>>(new Set());
  const queryClient = useQueryClient();
  // Held in a ref, not read from `options` inside the effect: a caller passing
  // an inline arrow would otherwise re-run the effect on every render, tearing
  // down and reopening the socket each time.
  const onGraphPushChangeRef = useRef(options.onGraphPushChange);
  useEffect(() => {
    onGraphPushChangeRef.current = options.onGraphPushChange;
  });

  useEffect(() => {
    // Guard first, then alias: narrowing datasetId to `string` before binding
    // `id` gives `id` a real, non-nullable static type — assigning it the
    // other way around (aliasing first, guarding second) types `id` as the
    // original `string | null` forever, since TypeScript doesn't carry
    // control-flow narrowing into the closures declared below that read it.
    if (!datasetId) return;
    const id = datasetId;
    let cancelled = false;
    // Two stoppers, not one. The socket no longer dies permanently on
    // "unreachable" — it keeps retrying on a slow timer — so the poll it
    // fell back to and the socket still trying behind it are live at the same
    // time, and cleanup has to close both. Collapsing them into a single
    // `stopTransport` leaked whichever one was overwritten.
    let stopSocket: (() => void) | null = null;
    let stopPoll: (() => void) | null = null;
    // Captured once: reading `seenRef.current` in cleanup makes the deps
    // lint (correctly) unsure it is looking at the same Set the effect used.
    const seen = seenRef.current;
    // Whether this mount has already completed a handshake. The first `ready`
    // needs no graph refetch (the query fetched on mount, moments earlier);
    // every later one is a RE-handshake, and those are the ones that can have
    // missed a push.
    let hadReady = false;
    // Every graph refetch this socket asks for goes through one throttle, so
    // a burst of pushes during a large ingest cannot stack reads on the pod
    // (see graphRefreshThrottle).
    const graphKey = brainGraphQueryKey(cogniInstance.instanceId, id);
    const graphRefresh = createGraphRefreshThrottle({
      refresh: () => queryClient.invalidateQueries({ queryKey: graphKey }),
      // Exact, because isFetching prefix-matches by default: a future query
      // keyed under this one would otherwise hold this dataset's refresh back.
      isFetching: () => queryClient.isFetching({ queryKey: graphKey, exact: true }) > 0,
      lastReadFailed: () => queryClient.getQueryState(graphKey)?.status === "error",
      onStuck: (waitedMs) =>
        captureMessage("Mindmap graph refresh stuck behind an in-flight read", {
          level: "warning",
          tags: { context: "mindmap-graph-refresh" },
          extra: { waitedMs, datasetId: id, instanceId: cogniInstance.instanceId },
        }),
    });

    function ingest(fresh: SessionEvent[]): void {
      if (cancelled || !fresh.length) return;
      const deduped = fresh.filter((e) => {
        const key = eventKey(e);
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });
      if (!deduped.length) return;
      setEvents((prev) => [...prev, ...deduped].slice(-MAX_EVENTS));
      const lastSearch = deduped.filter((e) => (e.kind || "search") === "search").at(-1);
      if (lastSearch) setLatestSearchEvent(lastSearch);
    }

    function stopPolling(): void {
      stopPoll?.();
      stopPoll = null;
    }

    function startPoll(cursorHolder: CursorRef): void {
      // Idempotent: `onGiveUp` can be reached again after a recovered socket
      // dies a second time, and two concurrent polls would double every
      // event's request without changing what the user sees.
      if (stopPoll) return;
      // The poll carries session events only. Nothing on this path delivers
      // `graph_grew`, so the graph query has to go back to polling for growth
      // itself - both when the socket gave up and when there never was one.
      onGraphPushChangeRef.current?.(false);
      stopPoll = pollLiveEvents(id, cogniInstance, cursorHolder, {
        onFresh: ingest,
        onLiveChange: setLive,
        onDroppedEvents: reportDroppedSessionEvents("poll", id),
      });
    }

    function handleFrame(frame: LiveSocketFrame): void {
      if (cancelled) return;
      switch (frame.kind) {
        case "ready":
          setLive(true);
          // A `ready` is also the recovery signal: the socket may have given
          // up earlier and reconnected on its slow timer, in which case the
          // poll it fell back to is still running and now redundant.
          stopPolling();
          // The server takes its `graph_grew` baseline fresh on every
          // connection and deliberately does not announce a run that was
          // already complete at connect time. So any cognify finishing inside
          // a reconnect gap (a watchdog fire, or the gateway's periodic
          // stream recycle) produces no push at all — the event cursor
          // resumes correctly but graph growth would just be missing. One
          // refetch per RE-handshake closes that gap.
          //
          // Skipped on the first handshake of a mount, and that is not a
          // `staleTime` optimisation: `invalidateQueries` refetches an active
          // observer unconditionally, because staleTime gates the automatic
          // triggers (mount, focus) and not an explicit invalidation. Without
          // this flag every Mindmap mount fetched the whole graph twice, for
          // every user, on the happy path — on a payload that runs to
          // megabytes (see getBrainGraph's MAX_NODES).
          if (hadReady) graphRefresh.request();
          hadReady = true;
          // A completed handshake, not merely an opened connection, is what
          // proves this pod actually serves the endpoint - so this is the one
          // place the graph query's poll may be switched off.
          onGraphPushChangeRef.current?.(true);
          if (frame.cursor) cursorRef.current = frame.cursor;
          break;
        case "live_events":
          setLive(true);
          cursorRef.current = frame.cursor;
          ingest(frame.events);
          break;
        case "graph_grew":
          graphRefresh.request();
          break;
        case "heartbeat":
          break;
      }
    }

    const socketTarget = cogniInstance.socketTarget;
    if (socketTarget) {
      const handle = openLiveSocket(socketTarget, id, cursorRef.current, {
        onFrame: handleFrame,
        onDroppedEvents: reportDroppedSessionEvents("socket", id),
        onReconnecting: () => {
          if (cancelled) return;
          setLive(false);
          onGraphPushChangeRef.current?.(false);
          // Breadcrumb only, no alert: one reconnect is a blip, and the
          // give-up below is the event worth waking someone for. Mirrors how
          // pod.ts separates a slow request from a sustained degradation.
          addBreadcrumb({
            category: "mindmap.live-socket",
            level: "warning",
            message: `live socket reconnecting for dataset ${id}`,
          });
        },
        // Resumes from cursorRef, which the socket kept current on every
        // `ready`/`live_events` frame it delivered — the poll picks up
        // exactly where the socket left off instead of re-fetching from
        // scratch.
        onGiveUp: (reason) => {
          if (cancelled) return;
          setLive(false);
          // Loud on purpose. The fallback keeps session events flowing, so
          // from the UI this failure looks like nothing at all - the only
          // visible symptom is that the graph stops growing, and without
          // this nobody would learn that a whole environment's sockets are
          // failing (e.g. a pod that does not serve the endpoint yet) until
          // a user reported it.
          reportClientLog("MINDMAP-WS", "error", `live socket ${reason} for dataset ${id}, falling back to polling`);
          captureMessage("Mindmap live socket unavailable", {
            level: "warning",
            tags: { context: "mindmap-live-socket" },
            extra: { reason, datasetId: id, instanceId: cogniInstance.instanceId },
          });
          startPoll(cursorRef);
        },
      });
      stopSocket = handle.close;
    } else {
      startPoll(cursorRef);
    }

    return () => {
      cancelled = true;
      graphRefresh.dispose();
      stopSocket?.();
      stopPolling();
      // The next dataset's socket has to prove itself from scratch, so the
      // graph query polls again until its `ready` arrives.
      onGraphPushChangeRef.current?.(false);
      // Without this, switching datasets (or any remount) kept the old
      // dataset's cursor and every event key it had ever seen — seenRef
      // grows unbounded across mount/unmount cycles, and a stale cursor
      // would ask the new dataset's poll for events "since" a timestamp
      // that means nothing for it.
      cursorRef.current = null;
      seen.clear();
      // Also drop the accumulated event log and pending chip — without
      // this, the "▶ new answer" chip survived a dataset switch and
      // playing it filtered node_ids that don't exist in the new dataset,
      // recentering the camera on an empty focus set (COG-6233).
      setEvents([]);
      setLatestSearchEvent(null);
    };
  }, [datasetId, cogniInstance, queryClient]);

  const answeredIds = useMemo(() => {
    const ids = new Set<string>();
    events.forEach((e) => {
      if ((e.kind || "search") !== "search") return;
      (e.node_ids as string[] | undefined)?.forEach((id) => ids.add(id));
    });
    return ids;
  }, [events]);

  return {
    events,
    live,
    latestSearchEvent,
    consumeLatestSearchEvent: () => setLatestSearchEvent(null),
    answeredIds,
  };
}
