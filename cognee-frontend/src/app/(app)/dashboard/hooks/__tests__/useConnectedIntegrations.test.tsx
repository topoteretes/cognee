import React from "react";
import { render } from "@testing-library/react";
import { useConnectedIntegrations } from "@/app/(app)/dashboard/hooks/useConnectedIntegrations";
import type { SessionRow } from "@/modules/sessions/getSessions";
import type { AgentConnection } from "@/modules/agents/getAgentConnections";

function session(sessionId: string): SessionRow {
  return {
    session_id: sessionId,
    user_id: "agent-1",
    dataset_id: null,
    status: "completed",
    effective_status: "completed",
    started_at: "2026-09-01T00:00:00Z",
    last_activity_at: "2026-09-01T00:01:00Z",
    ended_at: null,
    tokens_in: 1,
    tokens_out: 1,
    cost_usd: 0,
    error_count: 0,
    last_model: null,
  };
}

function connection(overrides: Partial<AgentConnection> = {}): AgentConnection {
  return {
    id: "conn-1",
    agent_session_name: "goran-laptop",
    type: "claude_code",
    session_id: "",
    last_active_at: "2026-09-01T00:00:00Z",
    status: "active",
    user_id: "u1",
    memory_mode: "hybrid",
    datasets: [],
    ...overrides,
  };
}

function detect(
  sessions: SessionRow[],
  tenantId: string | null = "t1",
  connections: AgentConnection[] = [],
): Record<string, boolean> {
  let seen: Record<string, boolean> = {};
  function Probe(): React.ReactElement {
    seen = useConnectedIntegrations(sessions, tenantId, connections);
    return <span />;
  }
  render(<Probe />);
  return seen;
}

beforeEach(() => {
  localStorage.clear();
});

describe("useConnectedIntegrations", () => {
  it("marks the Claude Code card connected for either shipped prefix", () => {
    // Detection used to carry its own prefix table naming only "cc_", so a
    // Claude Code session with the plugin's other prefix left the card reading
    // "Connect" while its sessions were in the feed.
    expect(detect([session("claude_abc")])["claude-code"]).toBe(true);
    expect(detect([session("cc_abc")])["claude-code"]).toBe(true);
  });

  it("does not mark Claude Code connected for a Claude Desktop session", () => {
    expect(detect([session("claude_desktop_abc")])["claude-code"]).toBeUndefined();
  });

  it("records the agent channel alongside the card key", () => {
    // The memory graph reads channel names from this same record, which is why
    // one detection pass now writes both.
    const state = detect([session("codex_1")]);

    expect(state["Codex"]).toBe(true);
    expect(state["codex"]).toBe(true);
  });

  it("records a channel that has no integration card of its own", () => {
    expect(detect([session("cursor_1")])["Cursor"]).toBe(true);
  });

  it("claims nothing for a session from no recognized client", () => {
    expect(detect([session("some-sdk-caller")])).toEqual({});
  });

  it("keeps an agent connected after its sessions leave the polled window", () => {
    detect([session("cc_abc")]);

    // Same tenant, empty feed: the persisted answer is what the graph reads.
    expect(detect([])["Claude Code"]).toBe(true);
  });

  it("keeps one tenant's agents out of another's", () => {
    detect([session("cc_abc")], "t1");

    expect(detect([], "t2")).toEqual({});
  });

  it("reports nothing before a tenant is known", () => {
    expect(detect([session("cc_abc")], null)).toEqual({});
  });

  it("marks a card connected from a registration alone, with no session yet", () => {
    // The whole point of CLO-989: the plugin registers at launch, before it
    // has run a recall or remember that would show up as a session.
    const state = detect([], "t1", [connection({ type: "claude_code" })]);

    expect(state["claude-code"]).toBe(true);
    expect(state["Claude Code"]).toBe(true);
  });

  it("resolves a plugin-1.3.1-shaped connection (type: api) off its session_id", () => {
    const state = detect([], "t1", [connection({ type: "api", session_id: "codex_abc" })]);

    expect(state["codex"]).toBe(true);
  });

  it("ignores a connection that matches no known integration", () => {
    expect(detect([], "t1", [connection({ type: "cursor", session_id: "cursor_1" })])).toEqual({});
  });

  it("persists a registration-only connection the same way a session-based one is", () => {
    detect([], "t1", [connection({ type: "codex" })]);

    expect(detect([])["Codex"]).toBe(true);
  });
});
