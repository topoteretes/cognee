"use client";

import { useEffect, useState } from "react";
import getAgentConnectionDetail from "@/modules/agents/getAgentConnectionDetail";
import type { RecentQa, RecentTrace } from "@/modules/agents/getAgentConnectionDetail";
import type { CogneeInstance } from "@/modules/instances/types";

export interface AgentConnectionDetailState {
  qas: RecentQa[];
  traces: RecentTrace[];
  loading: boolean;
  failed: boolean;
}

const IDLE: AgentConnectionDetailState = { qas: [], traces: [], loading: false, failed: false };

/**
 * Recent memory for one connection, fetched once per (instance, agentId)
 * pair when the modal opens — never polled, since the detail call itself
 * takes seconds on a loaded pod. `agentId` null means the caller has no
 * connection row to ask about (see latestConnectionForAgent), and this
 * reports the idle state rather than guessing at a fetch.
 *
 * Aborted on unmount (the modal closing) and on `agentId` changing before it
 * resolves, so a slow response for an agent the user already switched away
 * from can never land in state for the wrong card.
 */
export function useAgentConnectionDetail(
  cogniInstance: CogneeInstance | null,
  agentId: string | null,
): AgentConnectionDetailState {
  const [state, setState] = useState<AgentConnectionDetailState>(IDLE);

  useEffect(() => {
    if (!cogniInstance || !agentId) {
      setState(IDLE);
      return;
    }
    let cancelled = false;
    const controller = new AbortController();
    setState({ qas: [], traces: [], loading: true, failed: false });
    getAgentConnectionDetail(cogniInstance, agentId, { signal: controller.signal })
      .then((detail) => {
        if (cancelled) return;
        setState({ qas: detail.qas, traces: detail.traces, loading: false, failed: false });
      })
      .catch((e) => {
        if (cancelled || controller.signal.aborted) return;
        console.warn("[useAgentConnectionDetail] fetch failed:", e instanceof Error ? e.message : e);
        setState({ qas: [], traces: [], loading: false, failed: true });
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [cogniInstance, agentId]);

  return state;
}
