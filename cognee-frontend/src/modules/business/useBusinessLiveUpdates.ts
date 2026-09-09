"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { CogneeInstance } from "@/modules/instances/types";
import { HttpError } from "@/services/http/errors";
import getLiveEvents from "./getLiveEvents";
import type { SessionEvent } from "./types";

const POLL_MS = 1500;
const MAX_BACKOFF_MS = 10000;
const MAX_EVENTS = 200;

function eventKey(e: SessionEvent): string {
  return `${e.kind || "search"}|${e.qa_id || ""}|${e.time || ""}|${e.question || ""}`;
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

// Ports the live-events poller (customer_tutorial.html:7086-7194) minus its
// location.reload() growth path. That reload existed only because the
// original is a static export that cannot re-fetch; here, the focused
// dataset's graph already refreshes in place on its own interval (see
// useBrainGraph, CLO-597), and the simulation's position cache
// (useBusinessSimulation) keeps existing nodes
// from resetting when it does — so "graph grew" needs no special handling,
// just this dataset's own accumulated event log for the spotlight and the
// node panel's reverse "which answers used this node" lookup.
export function useBusinessLiveUpdates(
  datasetId: string | null,
  cogniInstance: CogneeInstance,
): LiveUpdates {
  const [events, setEvents] = useState<SessionEvent[]>([]);
  const [live, setLive] = useState(true);
  const [latestSearchEvent, setLatestSearchEvent] = useState<SessionEvent | null>(null);
  const cursorRef = useRef<string | null>(null);
  const seenRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    const id = datasetId;
    if (!id) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let failures = 0;

    async function poll(datasetIdForPoll: string): Promise<void> {
      try {
        const { events: fresh, cursor } = await getLiveEvents(datasetIdForPoll, cursorRef.current, cogniInstance);
        if (cancelled) return;
        failures = 0;
        cursorRef.current = cursor;
        setLive(true);
        const deduped = fresh.filter((e) => {
          const key = eventKey(e);
          if (seenRef.current.has(key)) return false;
          seenRef.current.add(key);
          return true;
        });
        if (deduped.length) {
          setEvents((prev) => [...prev, ...deduped].slice(-MAX_EVENTS));
          const lastSearch = deduped.filter((e) => (e.kind || "search") === "search").at(-1);
          if (lastSearch) setLatestSearchEvent(lastSearch);
        }
      } catch (error) {
        if (!cancelled) {
          failures++;
          setLive(false);
          // A 409 means the backend rejected this exact cursor (e.g. it aged
          // out) — retrying with the same `since` value only produces the
          // same 409 forever, which is exactly what left the indicator
          // stuck on "reconnecting" indefinitely (COG-6412: traced via a
          // live network capture, same stale cursor repeating with no
          // backoff-driven change). Dropping the cursor makes the next poll
          // ask for a fresh snapshot instead of repeating a rejected delta.
          if (error instanceof HttpError && error.status === 409) {
            cursorRef.current = null;
          }
        }
      } finally {
        // Back off (max 10s) while the backend is unreachable; recover
        // instantly once a poll succeeds (customer_tutorial.html ~7170).
        if (!cancelled) {
          const delay = failures ? Math.min(POLL_MS * (failures + 1), MAX_BACKOFF_MS) : POLL_MS;
          timer = setTimeout(() => poll(datasetIdForPoll), delay);
        }
      }
    }

    poll(id);
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
      // Without this, switching datasets (or any remount) kept the old
      // dataset's cursor and every event key it had ever seen — seenRef
      // grows unbounded across mount/unmount cycles, and a stale cursor
      // would ask the new dataset's poll for events "since" a timestamp
      // that means nothing for it.
      cursorRef.current = null;
      seenRef.current.clear();
      // Also drop the accumulated event log and pending chip — without
      // this, the "▶ new answer" chip survived a dataset switch and
      // playing it filtered node_ids that don't exist in the new dataset,
      // recentering the camera on an empty focus set (COG-6233).
      setEvents([]);
      setLatestSearchEvent(null);
    };
  }, [datasetId, cogniInstance]);

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
