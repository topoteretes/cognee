"use client";

import { useCallback, useEffect, useState } from "react";
import type { CogneeInstance } from "@/modules/instances/types";
import getAgentConnections, {
  connectionIntegrationKey,
  isLaterTimestamp,
  lastSeenByAgent,
  runningAgents,
} from "@/modules/agents/getAgentConnections";
import type { AgentConnection, AgentIntegrationKey } from "@/modules/agents/getAgentConnections";
import { useAgentConnectionDetection } from "@/app/(setup)/onboarding/hooks/useAgentConnectionDetection";
import type { AgentActivitySignal } from "@/app/(setup)/onboarding/hooks/useAgentConnectionDetection";
import { useBoundedPoll } from "@/utils/useBoundedPoll";

// The endpoint is cheap and this only ever runs while a tutorial step is on
// screen, so one constant cadence, no faster-then-slower switch, which used
// to force the poll's run to restart (see git history on this file for what
// that cost).
const POLL_INTERVAL_MS = 3000;
// ~10 min at 3s, same reasoning as useAgentConnectionDetection: a tutorial
// nobody ever completes must not poll forever while its tab stays open.
const MAX_POLLS = 200;

export interface AgentConnectionState {
  /** The newest matching connection that was not there on the first
   *  (baseline) poll of this run, or null until one shows up. Re-evaluated
   *  every poll, so it moves on to a newer registration on its own, e.g. a
   *  second terminal opened after the first one closed. */
  connected: AgentConnection | null;
  /** True once a new registration for this agent shows up since this run's
   *  baseline — the raw signal `connected` carries. Kept as its own boolean
   *  alongside `connected` so a reader that only needs "did this happen" does
   *  not have to null-check the object. */
  registeredThisRun: boolean;
  /** True once a new session belonging to this agent shows up since this
   *  run's baseline (useAgentConnectionDetection, now filtered to this
   *  agent's own sessionIntegrationKey). The session-side twin of
   *  `registeredThisRun`: a pod old enough to have no `/agents/register`
   *  endpoint never produces a registration at all, only a session, so this
   *  is what proves the connection on that pod. */
  agentSessionObserved: boolean;
  /** Which signal proved the connection, or null while neither has. The one
   *  field the "Connected" copy and the `agent_connected` event both read —
   *  before this, the copy read `connected` (registration only) while the
   *  event fired off `connected || firstMemorySignal`, so a session-only pod
   *  reported "connected" to analytics while the screen kept saying
   *  "Waiting…" forever. Registration wins when both are true: it is the
   *  faster signal (the SessionStart hook, not a memory write), so it is
   *  never "behind" a session that already flipped this. */
  connectedVia: "registration" | "session" | null;
  /** True while one of this agent's sessions is running, by the server's own
   *  effective_status, the same rule agentStatus uses for the memory graph.
   *  Never read off the connection row's `status`: "active" there is set at
   *  registration and is never cleared when a session crashes or its
   *  terminal window is closed, so it cannot tell a live agent from a dead
   *  one (see agentNodes.ts). */
  running: boolean;
  /** Newest known activity for this agent: the connected registration's own
   *  timestamp, or a session's, whichever is newer. See lastSeenByAgent.
   *  Null until either exists. */
  lastSeenAt: string | null;
  /** True once a new session appears after this hook went active, see
   *  useAgentConnectionDetection, which this composes rather than
   *  reimplementing. */
  firstMemorySaved: boolean;
  /** Same value as `firstMemorySaved`, as the signal that produced it rather
   *  than a boolean, kept distinct because analytics reads which one fired.
   *  Null until it does. */
  firstMemorySignal: AgentActivitySignal;
  /** True once the bounded poll gives up without ever finding a registration.
   *  Meaningless once `connected` is set, a caller should check that first. */
  expired: boolean;
  /** Starts a fresh run: new baselines and a new poll budget for both the
   *  registration and the session signal. What an expired step offers
   *  instead of making the user close and reopen it. */
  retry: () => void;
}

function newest(agents: AgentConnection[]): AgentConnection | undefined {
  return agents.reduce<AgentConnection | undefined>(
    (best, a) => (!best || isLaterTimestamp(a.last_active_at, best.last_active_at) ? a : best),
    undefined,
  );
}

/**
 * Live connection state for one agent (claude-code | codex) while a tutorial
 * step is on screen, feeds the shared AgentConnectionStatus component.
 *
 * Polls GET /v1/agents/connections directly: the plugin's SessionStart hook
 * registers there at launch, seconds before a session or memory write would
 * show up. `firstMemorySaved` still needs that slower signal, so it delegates
 * to useAgentConnectionDetection instead of polling a second, parallel
 * baseline for the same thing.
 *
 * Detects a fresh registration the same way useAgentConnectionDetection
 * detects a fresh session: by id, not by comparing timestamps. Each plugin
 * launch registers under a new connection id, so "connected" is a matching id
 * that was not there on the first poll of this run, not a `last_active_at`
 * newer than "now". Wall-clock comparison would depend on the browser's clock
 * agreeing with the pod's, which it has no reason to, and a browser even a
 * few seconds ahead would make every registration look stale forever.
 *
 * Deliberately not "locked onto" one connection once found: `connected` is
 * recomputed from scratch every poll, always the newest id outside the
 * baseline. A user who opens a second terminal and closes the first is still
 * reported as connected, locking onto the first id would instead have kept
 * reporting it "offline" forever once it unregistered.
 */
export function useAgentConnectionState(
  agent: AgentIntegrationKey,
  cogniInstance: CogneeInstance | null,
  active: boolean,
): AgentConnectionState {
  const [connected, setConnected] = useState<AgentConnection | null>(null);
  const [attempt, setAttempt] = useState(0);
  const retry = useCallback(() => setAttempt((n) => n + 1), []);
  const runKey = `${cogniInstance?.instanceId ?? ""}:${agent}:${attempt}`;

  // A fresh window whenever the run identity changes: active turning on, the
  // instance/agent this hook is watching changing while it stays on, or a
  // retry. Without that, a stale "connected" from the previous agent or
  // tenant would keep rendering until the next registration overwrote it.
  useEffect(() => {
    setConnected(null);
  }, [active, runKey]);

  const makeCheck = useCallback(() => {
    // null = no baseline poll has completed yet. Populated on the first
    // check of this run with every matching connection id that already
    // existed, so those don't read as "just connected" the instant the step
    // opens, the same baseline-then-diff shape useAgentConnectionDetection
    // uses for sessions. Never reassigned after that, so a run never
    // restarts mid-flight the way a cadence switch used to force it to.
    //
    // useBoundedPoll never overlaps checks, so this baseline is always the
    // first request's own response, not whichever of several concurrent
    // requests happened to resolve first (that used to let a registration
    // that landed before a slow first response resolved get baselined in as
    // "already there", and connected would never flip). The one honest gap
    // left: a registration that lands after the first request has already
    // left the pod, but before its response arrives, is invisible to this
    // baseline. It is not lost, the next poll's own comparison sees it as
    // new, just delayed by however long that first response took.
    let baselineIds: Set<string> | null = null;

    return async (isStopped: () => boolean) => {
      if (!cogniInstance) return;
      const page = await getAgentConnections(cogniInstance, {
        range: "24h",
        activeOnly: false,
        includeSources: false,
      }).catch((err) => {
        console.warn("[useAgentConnectionState] connections poll failed:", err instanceof Error ? err.message : err);
        return null;
      });
      if (isStopped() || !page) return;

      const matches = page.agents.filter((a) => connectionIntegrationKey(a) === agent);

      if (baselineIds === null) {
        baselineIds = new Set(matches.map((a) => a.id));
        return;
      }

      const target = newest(matches.filter((a) => !baselineIds?.has(a.id)));
      if (target) setConnected(target);
    };
  }, [agent, cogniInstance]);

  const expired = useBoundedPoll(active && !!cogniInstance, runKey, makeCheck, POLL_INTERVAL_MS, MAX_POLLS);

  // Reuses the sessions useAgentConnectionDetection already polls rather than
  // standing up a second sessions fetch just for "last seen". See its own
  // doc comment for why a connection's own timestamp alone reads a session
  // open since yesterday as "last seen at launch".
  const activity = useAgentConnectionDetection(cogniInstance, active, agent, attempt);
  const lastSeenAt = lastSeenByAgent(connected ? [connected] : [], activity.sessions)[agent] ?? null;
  // Once the sessions poll has run out its bound, its last answer is history,
  // so "running" is no longer claimed; lastSeenAt still stands, it is a past
  // timestamp either way.
  const running = !activity.stale && runningAgents(activity.sessions).has(agent);

  const registeredThisRun = connected !== null;
  const agentSessionObserved = activity.signal !== null;
  // Registration wins when both are true, see connectedVia's own doc comment.
  const connectedVia: AgentConnectionState["connectedVia"] = registeredThisRun
    ? "registration"
    : agentSessionObserved
      ? "session"
      : null;

  return {
    connected,
    registeredThisRun,
    agentSessionObserved,
    connectedVia,
    running,
    lastSeenAt,
    firstMemorySaved: activity.signal !== null,
    firstMemorySignal: activity.signal,
    expired: expired && connectedVia === null,
    retry,
  };
}
