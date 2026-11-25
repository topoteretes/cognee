const mockUseAgentConnectionState = jest.fn();
jest.mock("../useAgentConnectionState", () => ({
  useAgentConnectionState: (...args: unknown[]) => mockUseAgentConnectionState(...args),
}));

import { StrictMode } from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { AgentConnectionStatus } from "../AgentConnectionStatus";
import type { AgentConnection } from "@/modules/agents/getAgentConnections";
import type { AgentConnectionState } from "../useAgentConnectionState";
import type { CogneeInstance } from "@/modules/instances/types";

const INSTANCE = { name: "CloudCognee", instanceId: "tenant-1", fetch: jest.fn() } as CogneeInstance;

function connection(overrides: Partial<AgentConnection> = {}): AgentConnection {
  return {
    id: "conn-1",
    agent_session_name: "goran-laptop",
    type: "claude_code",
    session_id: "claude_abc",
    last_active_at: "2026-09-29T12:00:00.000Z",
    status: "active",
    user_id: "u1",
    memory_mode: "hybrid",
    datasets: [],
    ...overrides,
  };
}

function state(overrides: Partial<AgentConnectionState> = {}): AgentConnectionState {
  const connected = overrides.connected ?? null;
  const firstMemorySignal = overrides.firstMemorySignal ?? null;
  // Derived the same way the real hook derives it (registration wins), so a
  // test that only sets `connected` or `firstMemorySignal` gets a consistent
  // `connectedVia` without every call site having to spell it out too.
  const connectedVia: AgentConnectionState["connectedVia"] =
    connected !== null ? "registration" : firstMemorySignal !== null ? "session" : null;
  return {
    connected,
    registeredThisRun: connected !== null,
    agentSessionObserved: firstMemorySignal !== null,
    connectedVia,
    running: false,
    lastSeenAt: null,
    firstMemorySaved: false,
    firstMemorySignal,
    expired: false,
    retry: jest.fn(),
    ...overrides,
  };
}

beforeEach(() => {
  jest.useFakeTimers().setSystemTime(new Date("2026-09-29T12:05:00.000Z"));
  mockUseAgentConnectionState.mockReset().mockReturnValue(state());
});

afterEach(() => {
  jest.useRealTimers();
});

describe("AgentConnectionStatus", () => {
  it("reports the same connection signal used by its status text", () => {
    const onConnectedChange = jest.fn();
    const { rerender } = render(<AgentConnectionStatus agent="claude-code" cogniInstance={INSTANCE} pageName="Onboarding" onConnectedChange={onConnectedChange} />);
    expect(onConnectedChange).toHaveBeenLastCalledWith(false);

    mockUseAgentConnectionState.mockReturnValue(state({ connected: connection() }));
    rerender(<AgentConnectionStatus agent="claude-code" cogniInstance={INSTANCE} pageName="Onboarding" onConnectedChange={onConnectedChange} />);
    expect(onConnectedChange).toHaveBeenLastCalledWith(true);
    expect(screen.getByText("✓ Connected")).toBeInTheDocument();
  });

  it("shows a waiting message before any registration is seen", () => {
    render(<AgentConnectionStatus agent="claude-code" cogniInstance={INSTANCE} pageName="Onboarding" />);

    expect(screen.getByText("Waiting for Claude Code to start...")).toBeInTheDocument();
  });

  it("shows connected and running while a session of the agent is running", () => {
    mockUseAgentConnectionState.mockReturnValue(
      state({ connected: connection(), running: true, lastSeenAt: "2026-09-29T12:05:00.000Z" }),
    );

    render(<AgentConnectionStatus agent="claude-code" cogniInstance={INSTANCE} pageName="Onboarding" />);

    expect(screen.getByText("✓ Connected")).toBeInTheDocument();
    expect(screen.queryByText("✓ First memory saved")).not.toBeInTheDocument();
    expect(screen.getByText("Running, last activity just now")).toBeInTheDocument();
  });

  it("never says Running off a registration alone, however active its row reads", () => {
    mockUseAgentConnectionState.mockReturnValue(
      state({ connected: connection({ status: "active" }), running: false, lastSeenAt: "2026-09-29T12:00:00.000Z" }),
    );

    render(<AgentConnectionStatus agent="claude-code" cogniInstance={INSTANCE} pageName="Dashboard" />);

    expect(screen.queryByText(/Running/)).not.toBeInTheDocument();
    expect(screen.getByText("Last seen 5m ago")).toBeInTheDocument();
  });

  it("adds the first memory saved checkpoint once it fires", () => {
    mockUseAgentConnectionState.mockReturnValue(
      state({
        connected: connection(),
        running: true,
        lastSeenAt: "2026-09-29T12:05:00.000Z",
        firstMemorySaved: true,
        firstMemorySignal: "session",
      }),
    );

    render(<AgentConnectionStatus agent="claude-code" cogniInstance={INSTANCE} pageName="Onboarding" />);

    expect(screen.getByText("✓ First memory saved")).toBeInTheDocument();
  });

  it("shows no activity line when nothing dates the connection", () => {
    mockUseAgentConnectionState.mockReturnValue(state({ connected: connection(), lastSeenAt: null }));

    render(<AgentConnectionStatus agent="codex" cogniInstance={INSTANCE} pageName="Dashboard" />);

    expect(screen.getByText("✓ Connected")).toBeInTheDocument();
    expect(screen.queryByText(/Last seen|Running/)).not.toBeInTheDocument();
  });

  it("shows Connected off the session signal alone, on a pod with no registration endpoint", () => {
    // CLO-989 review finding: the copy used to read `connected` only, so a
    // session-only pod fired agent_connected while the screen kept showing
    // "Waiting for Claude Code to start…" forever.
    mockUseAgentConnectionState.mockReturnValue(state({ firstMemorySaved: true, firstMemorySignal: "session" }));

    render(<AgentConnectionStatus agent="claude-code" cogniInstance={INSTANCE} pageName="Onboarding" />);

    expect(screen.getByText("✓ Connected")).toBeInTheDocument();
    expect(screen.queryByText("Waiting for Claude Code to start...")).not.toBeInTheDocument();
  });

  it("shows the timeout state with causes once the bound expires with nothing found", () => {
    mockUseAgentConnectionState.mockReturnValue(state({ expired: true }));

    render(<AgentConnectionStatus agent="claude-code" cogniInstance={INSTANCE} pageName="Onboarding" />);

    expect(screen.getByText("Not seeing Claude Code yet")).toBeInTheDocument();
    expect(screen.getByText(/not restarted after installing the plugin/)).toBeInTheDocument();
  });

  it("offers Check again once expired, which starts a fresh run", () => {
    const retry = jest.fn();
    mockUseAgentConnectionState.mockReturnValue(state({ expired: true, retry }));

    render(<AgentConnectionStatus agent="claude-code" cogniInstance={INSTANCE} pageName="Onboarding" />);
    fireEvent.click(screen.getByRole("button", { name: "Check again" }));

    expect(retry).toHaveBeenCalledTimes(1);
  });

  it("prefers connected state over expired if both are somehow true", () => {
    mockUseAgentConnectionState.mockReturnValue(state({ connected: connection(), running: true, expired: true }));

    render(<AgentConnectionStatus agent="claude-code" cogniInstance={INSTANCE} pageName="Onboarding" />);

    expect(screen.getByText("✓ Connected")).toBeInTheDocument();
    expect(screen.queryByText("Not seeing Claude Code yet")).not.toBeInTheDocument();
  });

  it("marks the status region for assistive tech", () => {
    render(<AgentConnectionStatus agent="claude-code" cogniInstance={INSTANCE} pageName="Onboarding" />);

    expect(screen.getByRole("status")).toHaveAttribute("aria-live", "polite");
  });

  describe("analytics", () => {
    it("fires agent_connected with signal registration exactly once when connected appears", () => {
      const track = jest.fn();
      const { rerender } = render(
        <AgentConnectionStatus agent="claude-code" cogniInstance={INSTANCE} pageName="Onboarding" track={track} />,
      );
      expect(track).not.toHaveBeenCalled();

      mockUseAgentConnectionState.mockReturnValue(state({ connected: connection(), running: true }));
      rerender(<AgentConnectionStatus agent="claude-code" cogniInstance={INSTANCE} pageName="Onboarding" track={track} />);
      rerender(<AgentConnectionStatus agent="claude-code" cogniInstance={INSTANCE} pageName="Onboarding" track={track} />);

      expect(track).toHaveBeenCalledTimes(1);
      expect(track).toHaveBeenCalledWith({
        pageName: "Onboarding",
        eventName: "agent_connected",
        additionalProperties: { agent: "claude-code", signal: "registration" },
      });
    });

    it("fires agent_connected exactly once even when StrictMode double-invokes effects", () => {
      const track = jest.fn();
      mockUseAgentConnectionState.mockReturnValue(state({ connected: connection(), running: true }));

      render(
        <StrictMode>
          <AgentConnectionStatus agent="claude-code" cogniInstance={INSTANCE} pageName="Onboarding" track={track} />
        </StrictMode>,
      );

      const connectedCalls = track.mock.calls.filter((c) => c[0].eventName === "agent_connected");
      expect(connectedCalls).toHaveLength(1);
    });

    it("fires agent_connected with the session signal when no registration ever appears", () => {
      const track = jest.fn();
      mockUseAgentConnectionState.mockReturnValue(state({ firstMemorySaved: true, firstMemorySignal: "session" }));

      render(<AgentConnectionStatus agent="claude-code" cogniInstance={INSTANCE} pageName="Onboarding" track={track} />);

      expect(track).toHaveBeenCalledWith({
        pageName: "Onboarding",
        eventName: "agent_connected",
        additionalProperties: { agent: "claude-code", signal: "session" },
      });
    });

    it("does not fire agent_connected twice for both signals on the same run", () => {
      const track = jest.fn();
      mockUseAgentConnectionState.mockReturnValue(
        state({ connected: connection(), running: true, firstMemorySaved: true, firstMemorySignal: "session" }),
      );

      render(<AgentConnectionStatus agent="claude-code" cogniInstance={INSTANCE} pageName="Onboarding" track={track} />);

      const connectedCalls = track.mock.calls.filter((c) => c[0].eventName === "agent_connected");
      expect(connectedCalls).toHaveLength(1);
      expect(connectedCalls[0][0].additionalProperties.signal).toBe("registration");
    });

    it("fires agent_connected once when the session signal lands a poll before the registration", () => {
      const track = jest.fn();
      mockUseAgentConnectionState.mockReturnValue(state({ firstMemorySaved: true, firstMemorySignal: "session" }));
      const { rerender } = render(
        <AgentConnectionStatus agent="claude-code" cogniInstance={INSTANCE} pageName="Onboarding" track={track} />,
      );

      mockUseAgentConnectionState.mockReturnValue(
        state({ connected: connection(), firstMemorySaved: true, firstMemorySignal: "session" }),
      );
      rerender(<AgentConnectionStatus agent="claude-code" cogniInstance={INSTANCE} pageName="Onboarding" track={track} />);

      const connectedCalls = track.mock.calls.filter((c) => c[0].eventName === "agent_connected");
      expect(connectedCalls).toHaveLength(1);
      expect(connectedCalls[0][0].additionalProperties.signal).toBe("session");
    });

    it("fires first_memory_saved exactly once", () => {
      const track = jest.fn();
      const { rerender } = render(
        <AgentConnectionStatus agent="claude-code" cogniInstance={INSTANCE} pageName="Onboarding" track={track} />,
      );

      mockUseAgentConnectionState.mockReturnValue(state({ firstMemorySaved: true, firstMemorySignal: "session" }));
      rerender(<AgentConnectionStatus agent="claude-code" cogniInstance={INSTANCE} pageName="Onboarding" track={track} />);
      rerender(<AgentConnectionStatus agent="claude-code" cogniInstance={INSTANCE} pageName="Onboarding" track={track} />);

      const memoryCalls = track.mock.calls.filter((c) => c[0].eventName === "first_memory_saved");
      expect(memoryCalls).toHaveLength(1);
      expect(memoryCalls[0][0]).toEqual({
        pageName: "Onboarding",
        eventName: "first_memory_saved",
        additionalProperties: { agent: "claude-code", signal: "session" },
      });
    });
  });
});
