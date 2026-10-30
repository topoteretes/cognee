"use client";

import { useCallback, useEffect, useState } from "react";
import type { CogneeInstance } from "@/modules/instances/types";
import { listSessions, SEARCH_SESSION_PREFIX } from "@/modules/sessions/getSessions";
import type { SessionRow } from "@/modules/sessions/getSessions";
import { sessionIntegrationKey } from "@/modules/agents/getAgentConnections";
import type { AgentIntegrationKey } from "@/modules/agents/getAgentConnections";
import { useBoundedPoll } from "@/utils/useBoundedPoll";

// Detects agent activity while either the Upload or Recall step is active,
// by watching for a NEW session (a session_id proves an agent conversation).
// Baselined when the step opens so only activity AFTER that counts.
//
// Used to also watch dataset doc counts, as graph-direct uploads carry no
// session_id. Dropped (CLO-989): GET /v1/datasets/{id}/data/count 405s on at
// least one live pod, and the fallback it triggers (getDatasetDataCount's own
// "list up to 1000 rows and count them" path) ran every poll on top of it,
// twice the pod load of the sessions check for a signal the plugin's own
// SessionStart hook makes redundant anyway, it always opens a session, so
// "first memory saved" never needed the docs path to be reachable.
const POLL_INTERVAL_MS = 7000;
const MAX_POLLS = 85; // ~10 min at 7s

function realSessionIds(rows: { session_id: string }[]): string[] {
  return rows.map((s) => s.session_id).filter((id) => !id.startsWith(SEARCH_SESSION_PREFIX));
}

/** `realSessionIds`, narrowed to sessions this agent's own plugin opened.
 *  Without this, a Codex session opened while the Claude Code tutorial step
 *  is on screen (or a Cursor/Search session with no agent at all) satisfied
 *  the "a new session appeared" check and confirmed the wrong tutorial. */
function ownAgentSessionIds(rows: { session_id: string }[], agent: AgentIntegrationKey): string[] {
  return realSessionIds(rows).filter((id) => sessionIntegrationKey(id) === agent);
}

/** Which signal proved the agent is talking to Cognee. A plain boolean until
 *  there is a second signal worth telling apart in analytics again. */
export type AgentActivitySignal = "session" | null;

export interface AgentActivityState {
  signal: AgentActivitySignal;
  /** Sessions from this hook's own poll, exposed so a caller that also wants
   *  "last seen" or "is it running" (see useAgentConnectionState) doesn't
   *  have to stand up a second sessions poll next to this one. Empty until
   *  the first poll resolves. Keeps refreshing after `signal` is set, until
   *  the poll's bound runs out. */
  sessions: SessionRow[];
  /** True once the bounded poll has run out: `sessions` is then the last
   *  answer, no longer a live one, so nothing should present it as "now". */
  stale: boolean;
}

export function useAgentConnectionDetection(
  cogniInstance: CogneeInstance | null,
  active: boolean,
  agent: AgentIntegrationKey,
  attempt = 0,
): AgentActivityState {
  const [signal, setSignal] = useState<AgentActivitySignal>(null);
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const runKey = `${cogniInstance?.instanceId ?? ""}:${agent}:${attempt}`;

  useEffect(() => {
    setSignal(null);
    setSessions([]);
  }, [active, runKey]);

  // A fresh baseline per run, held in a plain closure the way a hand-written
  // poll loop would, useBoundedPoll calls this once when a run starts and
  // reuses the function it returns for every poll in that run.
  const makeCheck = useCallback(() => {
    // Baselined off the first (and, thanks to useBoundedPoll never
    // overlapping checks, only ever one in flight at a time) response for
    // this run. A session that starts after that request left the pod but
    // before it answered is not in this baseline, it becomes "new" on the
    // very next poll instead, which is a bounded, known delay rather than a
    // coin flip between two concurrent requests answering out of order.
    let primed = false;
    const baselineSessionIds = new Set<string>();

    return async (isStopped: () => boolean) => {
      if (!cogniInstance) return;
      const page = await listSessions(cogniInstance, { range: "24h", limit: 50 }).catch((err) => {
        console.warn("Onboarding progress check: sessions fetch failed", err);
        return null;
      });
      if (isStopped() || !page) return;
      // A swallowed failure (see FAILED_PAGE) answers with an empty list, not
      // "there really is nothing here" — priming the baseline off it would
      // baseline as empty, and the very next successful poll would then read
      // every pre-existing session as brand new. Nor may it replace the last
      // good `sessions`, which running/lastSeenAt still read while this poll
      // keeps trying: swapping them for [] would blank both over one blip.
      if (page.failed) return;

      setSessions(page.sessions);
      const ids = ownAgentSessionIds(page.sessions, agent);
      if (!primed) {
        ids.forEach((id) => baselineSessionIds.add(id));
        primed = true;
        return;
      }
      if (ids.some((id) => !baselineSessionIds.has(id))) setSignal("session");
    };
  }, [cogniInstance, agent]);

  // Bounded, not gated on `signal`: the sessions it keeps refreshing are what
  // "running" and "last seen" read while the step stays open, so stopping at
  // the first session would freeze both at that moment. ~10 minutes at 7s
  // still caps the load for a tab nobody closes; `stale` says when it ran out.
  const stale = useBoundedPoll(active && !!cogniInstance, runKey, makeCheck, POLL_INTERVAL_MS, MAX_POLLS);

  return { signal, sessions, stale };
}
