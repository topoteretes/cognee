const mockGetAgentConnections = jest.fn();
jest.mock("@/modules/agents/getAgentConnections", () => {
  const actual = jest.requireActual("@/modules/agents/getAgentConnections");
  return {
    __esModule: true,
    ...actual,
    default: (...args: unknown[]) => mockGetAgentConnections(...args),
  };
});

const mockUseAgentConnectionDetection = jest.fn();
jest.mock("@/app/(setup)/onboarding/hooks/useAgentConnectionDetection", () => ({
  useAgentConnectionDetection: (...args: unknown[]) => mockUseAgentConnectionDetection(...args),
}));

import { act, renderHook } from "@testing-library/react";
import { useAgentConnectionState } from "../useAgentConnectionState";
import type { AgentConnection } from "@/modules/agents/getAgentConnections";
import type { CogneeInstance } from "@/modules/instances/types";

const INSTANCE = { name: "CloudCognee", instanceId: "tenant-1", fetch: jest.fn() } as CogneeInstance;

function connection(overrides: Partial<AgentConnection> = {}): AgentConnection {
  return {
    id: "conn-1",
    agent_session_name: "goran-laptop",
    type: "claude_code",
    session_id: "claude_abc",
    last_active_at: "2026-09-29T12:00:05.000Z",
    status: "active",
    user_id: "u1",
    memory_mode: "hybrid",
    datasets: [],
    ...overrides,
  };
}

function page(agents: AgentConnection[]) {
  return { agents, memory_sources: [], total: agents.length, limit: 50, offset: 0, has_more: false };
}

const IDLE = {
  connected: null,
  registeredThisRun: false,
  agentSessionObserved: false,
  connectedVia: null,
  running: false,
  lastSeenAt: null,
  firstMemorySaved: false,
  firstMemorySignal: null,
  expired: false,
  retry: expect.any(Function),
};

function session(sessionId: string, overrides: Record<string, unknown> = {}) {
  return {
    session_id: sessionId,
    status: "running",
    effective_status: "running",
    started_at: "2026-09-29T12:00:00.000Z",
    last_activity_at: "2026-09-29T12:00:30.000Z",
    ...overrides,
  };
}

beforeEach(() => {
  jest.useFakeTimers().setSystemTime(new Date("2026-09-29T12:00:00.000Z"));
  mockGetAgentConnections.mockReset().mockResolvedValue(page([]));
  mockUseAgentConnectionDetection.mockReset().mockReturnValue({ signal: null, sessions: [], stale: false });
});

afterEach(() => {
  jest.useRealTimers();
});

async function flush(): Promise<void> {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}

async function tick(ms: number): Promise<void> {
  await act(async () => {
    jest.advanceTimersByTime(ms);
    await flush();
  });
}

describe("useAgentConnectionState", () => {
  it("reports nothing while inactive and never polls", async () => {
    const { result } = renderHook(() => useAgentConnectionState("claude-code", INSTANCE, false));
    await flush();

    expect(result.current).toEqual(IDLE);
    expect(mockGetAgentConnections).not.toHaveBeenCalled();
  });

  it("ignores a registration that already existed on the first (baseline) poll", async () => {
    // A registration made before the tutorial step opened must not read as
    // "just connected" the instant the hook starts watching, same rule
    // useAgentConnectionDetection applies to sessions, applied here to ids.
    mockGetAgentConnections.mockResolvedValue(page([connection({ id: "pre-existing" })]));

    const { result } = renderHook(() => useAgentConnectionState("claude-code", INSTANCE, true));
    await flush();
    expect(result.current.connected).toBeNull();

    // Still there on the next poll, unchanged, must keep reading as
    // pre-existing rather than "new" on every subsequent check too.
    await tick(3000);
    expect(result.current.connected).toBeNull();
    expect(result.current.running).toBe(false);
  });

  it("is immune to the browser clock disagreeing with the pod's (no time comparison at all)", async () => {
    // A `last_active_at` far in the past by wall-clock terms is still a NEW
    // id relative to the baseline snapshot, and must count as connected ,
    // detection is purely "was this id here before", never a timestamp
    // compare, so browser/pod clock skew cannot hide a real connection.
    mockGetAgentConnections.mockResolvedValueOnce(page([]));
    const { result } = renderHook(() => useAgentConnectionState("claude-code", INSTANCE, true));
    await flush();

    mockGetAgentConnections.mockResolvedValue(
      page([connection({ id: "conn-skewed", last_active_at: "2020-01-01T00:00:00.000Z" })]),
    );
    await tick(3000);

    expect(result.current.connected?.id).toBe("conn-skewed");
  });

  it("takes the baseline from the first request's own response even when it is slow, and still flips connected once the registration shows up", async () => {
    // Models the real bug: the pod is slow (4s+) answering the baseline
    // request. A registration that lands on the pod while that request is
    // still in flight cannot be in its response (it already left before the
    // registration happened), it must not be lost, just picked up by the
    // next poll instead.
    const released: { release: ((value: ReturnType<typeof page>) => void) | null } = { release: null };
    mockGetAgentConnections.mockImplementationOnce(
      () => new Promise((resolve) => { released.release = resolve; }),
    );
    const { result } = renderHook(() => useAgentConnectionState("claude-code", INSTANCE, true));
    await flush();
    expect(mockGetAgentConnections).toHaveBeenCalledTimes(1);

    // Still pending 4s later, nothing overlapping fired a second request.
    await tick(4000);
    expect(mockGetAgentConnections).toHaveBeenCalledTimes(1);

    // The slow baseline finally answers: empty, because the registration
    // had not happened yet when this request left the pod.
    released.release?.(page([]));
    await flush();
    expect(result.current.connected).toBeNull();

    // Next poll, a full interval after the baseline settled, now sees it.
    mockGetAgentConnections.mockResolvedValue(page([connection({ status: "active" })]));
    await tick(3000);

    expect(result.current.connected?.id).toBe("conn-1");
  });

  it("marks a fresh registration connected, without claiming it is running", async () => {
    mockGetAgentConnections.mockResolvedValueOnce(page([]));
    const { result } = renderHook(() => useAgentConnectionState("claude-code", INSTANCE, true));
    await flush();

    mockGetAgentConnections.mockResolvedValue(page([connection({ status: "active" })]));
    await tick(3000);

    expect(result.current.connected?.id).toBe("conn-1");
    // An "active" row is set at registration and never cleared on a crash,
    // so it alone must never read as running.
    expect(result.current.running).toBe(false);
    expect(result.current.lastSeenAt).toBe("2026-09-29T12:00:05.000Z");
  });

  it("reports running only from a running session of this agent", async () => {
    mockUseAgentConnectionDetection.mockReturnValue({
      signal: "session",
      sessions: [session("codex_other"), session("claude_dead", { status: "abandoned", effective_status: "abandoned" })],
      stale: false,
    });
    const { result, rerender } = renderHook(() => useAgentConnectionState("claude-code", INSTANCE, true));
    await flush();
    expect(result.current.running).toBe(false);

    mockUseAgentConnectionDetection.mockReturnValue({ signal: "session", sessions: [session("claude_live")], stale: false });
    rerender();
    expect(result.current.running).toBe(true);
  });

  it("stops claiming running once the sessions poll has gone stale", async () => {
    mockUseAgentConnectionDetection.mockReturnValue({ signal: "session", sessions: [session("claude_live")], stale: true });
    const { result } = renderHook(() => useAgentConnectionState("claude-code", INSTANCE, true));
    await flush();

    expect(result.current.running).toBe(false);
    expect(result.current.lastSeenAt).toBe("2026-09-29T12:00:30.000Z");
  });

  it("orders timestamps as instants, not strings, when picking the newest", async () => {
    // Python's isoformat drops a zero fraction, so one pod emits both forms;
    // as strings ".5Z" sorts before "Z" and would pick the older row.
    mockGetAgentConnections.mockResolvedValueOnce(page([]));
    const { result } = renderHook(() => useAgentConnectionState("claude-code", INSTANCE, true));
    await flush();

    mockGetAgentConnections.mockResolvedValue(
      page([
        connection({ id: "older", last_active_at: "2026-09-29T12:00:05Z" }),
        connection({ id: "newer", last_active_at: "2026-09-29T12:00:05.500Z" }),
      ]),
    );
    await tick(3000);

    expect(result.current.connected?.id).toBe("newer");
  });

  it("starts a fresh run on retry after expiring, and can then find a registration", async () => {
    mockGetAgentConnections.mockResolvedValue(page([]));
    const { result } = renderHook(() => useAgentConnectionState("claude-code", INSTANCE, true));
    await flush();
    for (let i = 0; i < 205; i++) {
      await tick(3000);
    }
    expect(result.current.expired).toBe(true);
    const callsBeforeRetry = mockGetAgentConnections.mock.calls.length;

    act(() => result.current.retry());
    await flush();
    expect(result.current.expired).toBe(false);
    expect(mockGetAgentConnections.mock.calls.length).toBe(callsBeforeRetry + 1);

    mockGetAgentConnections.mockResolvedValue(page([connection({ id: "late" })]));
    await tick(3000);
    expect(result.current.connected?.id).toBe("late");
    expect(mockUseAgentConnectionDetection).toHaveBeenLastCalledWith(INSTANCE, true, "claude-code", 1);
  });

  it("hands off to a newer second registration after the first one goes inactive", async () => {
    // Opening a second terminal and closing the first must not leave the
    // hook reporting "offline" forever off the id it happened to find first
    //, connected is recomputed every poll, not locked onto one id.
    mockGetAgentConnections.mockResolvedValueOnce(page([]));
    const { result } = renderHook(() => useAgentConnectionState("claude-code", INSTANCE, true));
    await flush();

    mockGetAgentConnections.mockResolvedValue(
      page([connection({ id: "first", status: "active", last_active_at: "2026-09-29T12:00:01.000Z" })]),
    );
    await tick(3000);
    expect(result.current.connected?.id).toBe("first");

    mockGetAgentConnections.mockResolvedValue(
      page([
        connection({ id: "first", status: "inactive", last_active_at: "2026-09-29T12:00:01.000Z" }),
        connection({ id: "second", status: "active", last_active_at: "2026-09-29T12:00:09.000Z" }),
      ]),
    );
    await tick(3000);

    expect(result.current.connected?.id).toBe("second");
  });

  it("ignores a connection for the other agent", async () => {
    mockGetAgentConnections.mockResolvedValueOnce(page([]));
    const { result } = renderHook(() => useAgentConnectionState("claude-code", INSTANCE, true));
    await flush();

    mockGetAgentConnections.mockResolvedValue(page([connection({ type: "codex", session_id: "codex_abc" })]));
    await tick(3000);

    expect(result.current.connected).toBeNull();
  });

  it("picks the newest of several simultaneously-fresh connections", async () => {
    mockGetAgentConnections.mockResolvedValueOnce(page([]));
    const { result } = renderHook(() => useAgentConnectionState("claude-code", INSTANCE, true));
    await flush();

    mockGetAgentConnections.mockResolvedValue(
      page([
        connection({ id: "older", last_active_at: "2026-09-29T12:00:01.000Z" }),
        connection({ id: "newer", last_active_at: "2026-09-29T12:00:09.000Z" }),
      ]),
    );
    await tick(3000);

    expect(result.current.connected?.id).toBe("newer");
  });

  it("degrades to no state on a failed poll, without throwing", async () => {
    mockGetAgentConnections.mockRejectedValue(new Error("network down"));
    const warn = jest.spyOn(console, "warn").mockImplementation(() => {});

    const { result } = renderHook(() => useAgentConnectionState("claude-code", INSTANCE, true));
    await flush();

    expect(result.current.connected).toBeNull();
    warn.mockRestore();
  });

  it("keeps the connected state on a poll that fails after a prior success", async () => {
    mockGetAgentConnections.mockResolvedValueOnce(page([]));
    const { result } = renderHook(() => useAgentConnectionState("claude-code", INSTANCE, true));
    await flush();

    mockGetAgentConnections.mockResolvedValueOnce(page([connection()]));
    await tick(3000);
    expect(result.current.connected?.id).toBe("conn-1");

    mockGetAgentConnections.mockRejectedValueOnce(new Error("timeout"));
    const warn = jest.spyOn(console, "warn").mockImplementation(() => {});
    await tick(3000);

    expect(result.current.connected?.id).toBe("conn-1");
    warn.mockRestore();
  });

  it("stops polling once it hits the bounded cap, and reports expired", async () => {
    mockGetAgentConnections.mockResolvedValue(page([]));
    const { result } = renderHook(() => useAgentConnectionState("claude-code", INSTANCE, true));
    await flush();

    // 1 immediate poll + 200 bounded interval polls = 201, never a 202nd.
    for (let i = 0; i < 205; i++) {
      await tick(3000);
    }

    expect(mockGetAgentConnections).toHaveBeenCalledTimes(201);
    expect(result.current.expired).toBe(true);
  });

  it("never reports expired while a connection has been found", async () => {
    mockGetAgentConnections.mockResolvedValueOnce(page([]));
    const { result } = renderHook(() => useAgentConnectionState("claude-code", INSTANCE, true));
    await flush();

    mockGetAgentConnections.mockResolvedValue(page([connection()]));
    for (let i = 0; i < 205; i++) {
      await tick(3000);
    }

    expect(result.current.connected).not.toBeNull();
    expect(result.current.expired).toBe(false);
  });

  it("reports connectedVia 'registration' once a registration lands, even with a session signal also present", async () => {
    mockUseAgentConnectionDetection.mockReturnValue({ signal: "session", sessions: [], stale: false });
    mockGetAgentConnections.mockResolvedValueOnce(page([]));
    const { result } = renderHook(() => useAgentConnectionState("claude-code", INSTANCE, true));
    await flush();

    mockGetAgentConnections.mockResolvedValue(page([connection()]));
    await tick(3000);

    // Registration wins when both signals are true — see connectedVia's own
    // doc comment for why.
    expect(result.current.registeredThisRun).toBe(true);
    expect(result.current.agentSessionObserved).toBe(true);
    expect(result.current.connectedVia).toBe("registration");
  });

  it("reports connectedVia 'session' on a pod with no registration endpoint (session signal only)", async () => {
    mockUseAgentConnectionDetection.mockReturnValue({ signal: "session", sessions: [], stale: false });
    mockGetAgentConnections.mockResolvedValue(page([]));

    const { result } = renderHook(() => useAgentConnectionState("claude-code", INSTANCE, true));
    await flush();

    expect(result.current.registeredThisRun).toBe(false);
    expect(result.current.agentSessionObserved).toBe(true);
    expect(result.current.connectedVia).toBe("session");
  });

  it("only reports expired once connectedVia is null, even if only the session signal ever fires", async () => {
    // Regression: expired used to read `connected === null`, so a session-only
    // pod (no registration endpoint) that had already proven the connection
    // via a session would still flip to "expired" the moment the bounded poll
    // ran out, and the UI would show "Not seeing Claude Code yet" for an agent
    // that had, in fact, connected.
    mockUseAgentConnectionDetection.mockReturnValue({ signal: "session", sessions: [], stale: false });
    mockGetAgentConnections.mockResolvedValue(page([]));

    const { result } = renderHook(() => useAgentConnectionState("claude-code", INSTANCE, true));
    await flush();
    for (let i = 0; i < 205; i++) {
      await tick(3000);
    }

    expect(result.current.connectedVia).toBe("session");
    expect(result.current.expired).toBe(false);
  });

  it("composes firstMemorySaved and firstMemorySignal from useAgentConnectionDetection", async () => {
    mockUseAgentConnectionDetection.mockReturnValue({ signal: "session", sessions: [] });

    const { result } = renderHook(() => useAgentConnectionState("claude-code", INSTANCE, true));
    await flush();

    expect(result.current.firstMemorySaved).toBe(true);
    expect(result.current.firstMemorySignal).toBe("session");
    expect(mockUseAgentConnectionDetection).toHaveBeenCalledWith(INSTANCE, true, "claude-code", 0);
  });

  it("reads last seen from a matching session when it is newer than the registration", async () => {
    mockGetAgentConnections.mockResolvedValueOnce(page([]));
    const { result } = renderHook(() => useAgentConnectionState("claude-code", INSTANCE, true));
    await flush();

    mockGetAgentConnections.mockResolvedValue(page([connection({ last_active_at: "2026-09-29T12:00:05.000Z" })]));
    mockUseAgentConnectionDetection.mockReturnValue({
      signal: null,
      sessions: [{
        session_id: "claude_abc",
        user_id: "u1",
        dataset_id: null,
        status: "running",
        effective_status: "running",
        started_at: null,
        last_activity_at: "2026-09-29T13:00:00.000Z",
        ended_at: null,
        tokens_in: null,
        tokens_out: null,
        cost_usd: null,
        error_count: 0,
        last_model: null,
      }],
    });
    await tick(3000);

    expect(result.current.lastSeenAt).toBe("2026-09-29T13:00:00.000Z");
  });

  it("clears state and stops polling when it goes inactive", async () => {
    mockGetAgentConnections.mockResolvedValueOnce(page([]));
    const { result, rerender } = renderHook(
      ({ active }: { active: boolean }) => useAgentConnectionState("claude-code", INSTANCE, active),
      { initialProps: { active: true } },
    );
    await flush();
    mockGetAgentConnections.mockResolvedValue(page([connection()]));
    await tick(3000);
    expect(result.current.connected).not.toBeNull();

    rerender({ active: false });

    expect(result.current).toEqual(IDLE);

    mockGetAgentConnections.mockClear();
    await tick(10_000);
    expect(mockGetAgentConnections).not.toHaveBeenCalled();
  });

  it("does not warn about updating unmounted state when a poll resolves after unmount", async () => {
    let resolvePoll: ((value: ReturnType<typeof page>) => void) | null = null;
    mockGetAgentConnections.mockImplementation(
      () => new Promise((resolve) => { resolvePoll = resolve; }),
    );
    const consoleError = jest.spyOn(console, "error").mockImplementation(() => {});

    const { unmount } = renderHook(() => useAgentConnectionState("claude-code", INSTANCE, true));
    unmount();

    // The poll was in flight when the component unmounted; resolving it now
    // must not touch state that no longer has anywhere to render to.
    await act(async () => {
      resolvePoll?.(page([connection()]));
      await flush();
    });

    expect(consoleError).not.toHaveBeenCalled();
    consoleError.mockRestore();
  });

  it("resets to idle and re-baselines when the agent it's watching changes while active", async () => {
    // A stale "connected" from claude-code must not leak into the codex
    // tutorial step if the same hook instance gets reused across steps.
    mockGetAgentConnections.mockResolvedValueOnce(page([]));
    const initialAgent: { agent: "claude-code" | "codex" } = { agent: "claude-code" };
    const { result, rerender } = renderHook(
      ({ agent }: { agent: "claude-code" | "codex" }) => useAgentConnectionState(agent, INSTANCE, true),
      { initialProps: initialAgent },
    );
    await flush();
    mockGetAgentConnections.mockResolvedValue(page([connection({ type: "claude_code" })]));
    await tick(3000);
    expect(result.current.connected).not.toBeNull();

    mockGetAgentConnections.mockClear().mockResolvedValue(page([connection({ type: "claude_code" })]));
    rerender({ agent: "codex" });

    // Cleared immediately, synchronously with the switch, not left showing
    // the old agent's connection until the next poll resolves.
    expect(result.current.connected).toBeNull();

    // The stray claude-code row from the still-in-flight response must not
    // reappear as "connected" for the codex step just because it's a
    // "new" id relative to codex's fresh (empty) baseline.
    await flush();
    expect(result.current.connected).toBeNull();
  });

  it("resets to idle and re-baselines when the instance it's watching changes while active", async () => {
    const otherInstance = { name: "CloudCognee", instanceId: "tenant-2", fetch: jest.fn() } as CogneeInstance;
    mockGetAgentConnections.mockResolvedValueOnce(page([]));
    const { result, rerender } = renderHook(
      ({ instance }: { instance: CogneeInstance }) => useAgentConnectionState("claude-code", instance, true),
      { initialProps: { instance: INSTANCE } },
    );
    await flush();
    mockGetAgentConnections.mockResolvedValue(page([connection()]));
    await tick(3000);
    expect(result.current.connected).not.toBeNull();

    rerender({ instance: otherInstance });

    expect(result.current.connected).toBeNull();
  });

  it("ignores a stale poll's result once the active window has moved on", async () => {
    // Baseline poll for window 1, resolves immediately, empty.
    mockGetAgentConnections.mockResolvedValueOnce(page([]));
    const { result, rerender } = renderHook(
      ({ active }: { active: boolean }) => useAgentConnectionState("claude-code", INSTANCE, active),
      { initialProps: { active: true } },
    );
    await flush();

    // Window 1's second poll, the one that would report a match, is left
    // in flight.
    let resolveStalePoll: ((value: ReturnType<typeof page>) => void) | null = null;
    mockGetAgentConnections.mockImplementationOnce(
      () => new Promise((resolve) => { resolveStalePoll = resolve; }),
    );
    await tick(3000);

    // Deactivate and reactivate while that poll is still in flight, a fresh
    // window with its own baseline starts, and the in-flight poll now
    // belongs to a window that no longer exists.
    rerender({ active: false });
    mockGetAgentConnections.mockResolvedValue(page([]));
    rerender({ active: true });
    await flush();

    // The stale poll resolves with what would otherwise read as a brand new
    // connection relative to window 1's baseline, without the isStopped
    // guard this would still be applied to window 2's state.
    await act(async () => {
      resolveStalePoll?.(page([connection()]));
      await flush();
    });

    expect(result.current.connected).toBeNull();
  });
});
