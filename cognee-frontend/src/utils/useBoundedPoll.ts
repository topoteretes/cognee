"use client";

import { useEffect, useRef, useState } from "react";

/**
 * Runs a fresh check immediately, then again `intervalMs` after each one
 * settles, for as long as `active` and `runKey` stay the same, up to
 * `maxPolls` calls. Two callers (onboarding's connection detection, the
 * tutorial's live agent-connection state) each poll a pod endpoint on a timer
 * that must stop itself once nobody is watching and must baseline itself
 * fresh per run, which is exactly the shape that was previously written out
 * by hand in each one.
 *
 * Checks never overlap: the next one is scheduled only after the current
 * one's promise resolves, not on a fixed setInterval tick. A pod slow enough
 * that one poll is still in flight when the next was due used to fire it
 * anyway, and two responses for the same "is it there yet" question could then
 * land out of order, and whichever arrived first became the baseline even if
 * it was the one carrying the new registration, which read as "already
 * there before we started watching" and never flipped to connected. Piling
 * up concurrent requests against an already-slow pod made it slower still.
 * The cost is that a slow first request keeps the whole run waiting for it
 * rather than racing a second one in parallel, which is the correct trade:
 * a caller with a real "is it stuck" timeout wraps its own fetch, this hook
 * just never contributes extra concurrent load to that being true.
 *
 * `makeCheck` is called once per run, on mount while active, and again
 * whenever `active` or `runKey` changes, and its return value is the
 * function actually polled. This is the point of the factory: a check that
 * needs per-run state (a baseline snapshot, "have we found it yet") can hold
 * that state in plain local variables closed over by the returned function,
 * the same way a hand-written effect would, instead of reaching for a ref
 * that this hook resets on the caller's behalf.
 *
 * `runKey` exists because `active` alone isn't always the right thing to
 * restart on, a caller polling on behalf of "this instance" or "this agent"
 * needs a fresh baseline when that identity changes too, even while `active`
 * never flips. Passed as a single comparable value (not a dependency array)
 * so this hook's own dependency list stays a plain, statically-checkable
 * array; callers that don't need one can pass any constant.
 *
 * The check receives `isStopped`, checked deliberately mid-function rather
 * than only at the top: a poll that is awaiting a slow fetch when the run
 * tears down (unmount, `active` flipping false, or `runKey` changing) must
 * not act on a stale result once it resolves.
 *
 * A baseline callers take off the first response is still a race in one
 * narrow window: a registration that lands after that first request left
 * the pod but before its response arrived is invisible to it and will not
 * itself flip a caller's state (it becomes part of the *next* poll's own
 * baseline read, same as this hook's `makeCheck` never sees anything before
 * its own first call). This is honest, bounded staleness, not the unbounded,
 * order-dependent one overlapping requests produced: a slow first response
 * makes the run wait longer to see anything new, it does not make it miss a
 * registration that arrives afterward.
 *
 * Returns `expired`: false for the whole run, true once it reaches `maxPolls`
 * without being stopped first. This is the one bound in the system, so a caller
 * that wants to show "gave up waiting" reads this rather than standing up a
 * second timer that would drift from the one actually governing the poll.
 * Resets to false at the start of every run, same as everything else here.
 */
export function useBoundedPoll(
  active: boolean,
  runKey: string | number,
  makeCheck: () => (isStopped: () => boolean) => void | Promise<void>,
  intervalMs: number,
  maxPolls: number,
): boolean {
  const makeCheckRef = useRef(makeCheck);
  makeCheckRef.current = makeCheck;
  const [expired, setExpired] = useState(false);

  useEffect(() => {
    setExpired(false);
    if (!active) return;
    const check = makeCheckRef.current();
    let stopped = false;
    const isStopped = () => stopped;
    let polls = 0;
    let timeoutId: ReturnType<typeof setTimeout>;

    // One in-flight check at a time: the next run is scheduled only once
    // `check` settles, so the interval is measured from completion, not from
    // when this one started.
    async function runCheck(): Promise<void> {
      await check(isStopped);
      if (stopped) return;
      timeoutId = setTimeout(() => {
        if (stopped) return;
        if (++polls > maxPolls) {
          setExpired(true);
          return;
        }
        void runCheck();
      }, intervalMs);
    }

    void runCheck();

    return () => {
      stopped = true;
      clearTimeout(timeoutId);
    };
  }, [active, runKey, intervalMs, maxPolls]);

  return expired;
}
