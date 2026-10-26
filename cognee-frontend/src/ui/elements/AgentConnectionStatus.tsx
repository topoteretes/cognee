"use client";

import { useEffect, useRef, type ReactElement } from "react";
import type { CogneeInstance } from "@/modules/instances/types";
import { agentDisplayName } from "@/modules/agents/getAgentConnections";
import type { AgentIntegrationKey } from "@/modules/agents/getAgentConnections";
import { trackEvent } from "@/modules/analytics";
import { timeAgo } from "@/utils/timeAgo";
import { useAgentConnectionState } from "./useAgentConnectionState";

type TrackEventParams = Parameters<typeof trackEvent>[0];

const TEXT = "rgba(237,236,234,0.75)";
const MUTED = "rgba(237,236,234,0.5)";
const GREEN = "#22C55E";

/**
 * Live setup progress for one plugin agent, shared by all three tutorial
 * surfaces (onboarding, the integrations wizard, the dashboard connect
 * panel) so a user reading two of them at once never sees two different
 * answers to "is it connected yet".
 *
 * Feeds entirely off useAgentConnectionState, this component owns no
 * polling or detection logic of its own, only how the four states it
 * produces (waiting, connected, first memory saved, expired) are worded and
 * laid out. `agent` is looked up in AGENT_INTEGRATION_DEFS for its display
 * name, so a third agent needs a row there and nothing here.
 *
 * `track` defaults to the page-agnostic `trackEvent`; pass a page's own
 * wrapper (e.g. useOnboardingTrackEvent) to get that page's usual enrichment
 * on these two events too.
 */
export function AgentConnectionStatus({
  agent,
  cogniInstance,
  pageName,
  track = trackEvent,
  onConnectedChange,
}: {
  agent: AgentIntegrationKey;
  cogniInstance: CogneeInstance | null;
  pageName: string;
  track?: (params: TrackEventParams) => void;
  onConnectedChange?: (connected: boolean) => void;
}): ReactElement {
  const name = agentDisplayName(agent);
  const state = useAgentConnectionState(agent, cogniInstance, true);

  useEffect(() => {
    onConnectedChange?.(state.connectedVia !== null);
  }, [onConnectedChange, state.connectedVia]);

  // Fired once per mount, on connectedVia's null-to-non-null edge — the same
  // field the "Connected" copy below reads, so the event and the screen can
  // never disagree about whether this happened. A tutorial step is one "run",
  // and re-showing it (closing and reopening the wizard) is a new run worth
  // counting again. Was two separate reads (`connected` for the copy, a
  // fallback chain for the event) that disagreed on a pod with no
  // registration endpoint: the event fired "connected" off the session signal
  // while the screen kept showing "Waiting for … to start…" forever.
  const connectedTracked = useRef(false);
  useEffect(() => {
    if (!state.connectedVia || connectedTracked.current) return;
    connectedTracked.current = true;
    track({ pageName, eventName: "agent_connected", additionalProperties: { agent, signal: state.connectedVia } });
  }, [state.connectedVia, agent, pageName, track]);

  const firstMemoryTracked = useRef(false);
  useEffect(() => {
    if (!state.firstMemorySignal || firstMemoryTracked.current) return;
    firstMemoryTracked.current = true;
    track({
      pageName,
      eventName: "first_memory_saved",
      additionalProperties: { agent, signal: state.firstMemorySignal },
    });
  }, [state.firstMemorySignal, agent, pageName, track]);

  return (
    <div role="status" aria-live="polite" style={{ marginTop: 12, fontSize: 13, lineHeight: 1.6 }}>
      {state.connectedVia ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          <span style={{ color: GREEN }}>✓ Connected</span>
          {state.firstMemorySaved && <span style={{ color: GREEN }}>✓ First memory saved</span>}
          {state.running ? (
            <span style={{ color: MUTED }}>
              Running{state.lastSeenAt && `, last activity ${timeAgo(state.lastSeenAt)}`}
            </span>
          ) : (
            state.lastSeenAt && <span style={{ color: MUTED }}>Last seen {timeAgo(state.lastSeenAt)}</span>
          )}
        </div>
      ) : state.expired ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 4, alignItems: "flex-start" }}>
          <span style={{ color: TEXT }}>Not seeing {name} yet</span>
          <span style={{ color: MUTED }}>
            Common causes: {name} was not restarted after installing the plugin, or the base URL / API key in
            ~/.cognee/.env is wrong.
          </span>
          <button
            type="button"
            onClick={state.retry}
            style={{ marginTop: 4, padding: 0, background: "none", border: "none", color: TEXT, textDecoration: "underline", cursor: "pointer", font: "inherit" }}
          >
            Check again
          </button>
        </div>
      ) : (
        <span style={{ color: MUTED }}>Waiting for {name} to start...</span>
      )}
    </div>
  );
}
