const mockListSessions = jest.fn();
jest.mock("@/modules/sessions/getSessions", () => {
  const actual = jest.requireActual("@/modules/sessions/getSessions");
  return { __esModule: true, ...actual, listSessions: (...args: unknown[]) => mockListSessions(...args) };
});

import { act, renderHook } from "@testing-library/react";
import { useAgentConnectionDetection } from "../useAgentConnectionDetection";
import type { CogneeInstance } from "@/modules/instances/types";

const INSTANCE = { name: "CloudCognee", instanceId: "tenant-1", fetch: jest.fn() } as CogneeInstance;

function sessionsPage(sessionIds: string[], failed = false) {
  return {
    sessions: sessionIds.map((id) => ({
      session_id: id,
      user_id: "u1",
      dataset_id: null,
      status: "running",
      effective_status: "running",
      started_at: null,
      last_activity_at: null,
      ended_at: null,
      tokens_in: null,
      tokens_out: null,
      cost_usd: null,
      error_count: 0,
      last_model: null,
    })),
    total: sessionIds.length,
    limit: 50,
    offset: 0,
    has_more: false,
    ...(failed ? { failed: true as const } : {}),
  };
}

beforeEach(() => {
  jest.useFakeTimers();
  mockListSessions.mockReset().mockResolvedValue(sessionsPage([]));
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

describe("useAgentConnectionDetection", () => {
  it("stays null on the priming poll even if sessions already exist", async () => {
    mockListSessions.mockResolvedValue(sessionsPage(["cc_1"]));

    const { result } = renderHook(() => useAgentConnectionDetection(INSTANCE, true, "claude-code"));
    await flush();

    expect(result.current.signal).toBeNull();
  });

  it("reports 'session' once a session appears after the baseline poll", async () => {
    mockListSessions.mockResolvedValueOnce(sessionsPage([]));
    const { result } = renderHook(() => useAgentConnectionDetection(INSTANCE, true, "claude-code"));
    await flush();
    expect(result.current.signal).toBeNull();

    mockListSessions.mockResolvedValue(sessionsPage(["cc_1"]));
    await act(async () => {
      jest.advanceTimersByTime(7000);
      await flush();
    });

    expect(result.current.signal).toBe("session");
  });

  it("keeps refreshing sessions after the signal fires, so running and last seen stay current", async () => {
    mockListSessions.mockResolvedValueOnce(sessionsPage([]));
    const { result } = renderHook(() => useAgentConnectionDetection(INSTANCE, true, "claude-code"));
    await flush();

    mockListSessions.mockResolvedValue(sessionsPage(["cc_1"]));
    await tick(7000);
    expect(result.current.signal).toBe("session");

    mockListSessions.mockResolvedValue(sessionsPage(["cc_1", "cc_2"]));
    await tick(7000);
    expect(result.current.sessions.map((s) => s.session_id)).toEqual(["cc_1", "cc_2"]);
    expect(result.current.stale).toBe(false);
  });

  it("reports stale once the bounded poll runs out", async () => {
    const { result } = renderHook(() => useAgentConnectionDetection(INSTANCE, true, "claude-code"));
    await flush();
    for (let i = 0; i < 90; i++) {
      await tick(7000);
    }

    expect(result.current.stale).toBe(true);
  });

  it("starts a fresh run with a new baseline when attempt changes", async () => {
    mockListSessions.mockResolvedValueOnce(sessionsPage([]));
    const { result, rerender } = renderHook(({ attempt }) => useAgentConnectionDetection(INSTANCE, true, "claude-code", attempt), {
      initialProps: { attempt: 0 },
    });
    await flush();
    mockListSessions.mockResolvedValue(sessionsPage(["cc_1"]));
    await tick(7000);
    expect(result.current.signal).toBe("session");

    rerender({ attempt: 1 });
    await flush();

    // cc_1 is in the new run's baseline, so it no longer counts as new.
    expect(result.current.signal).toBeNull();
  });

  it("does nothing while inactive", async () => {
    renderHook(() => useAgentConnectionDetection(INSTANCE, false, "claude-code"));
    await flush();

    expect(mockListSessions).not.toHaveBeenCalled();
  });

  it("exposes the sessions from its own poll, so a caller doesn't need a second sessions fetch", async () => {
    mockListSessions.mockResolvedValue(sessionsPage(["cc_1"]));

    const { result } = renderHook(() => useAgentConnectionDetection(INSTANCE, true, "claude-code"));
    await flush();

    expect(result.current.sessions.map((s) => s.session_id)).toEqual(["cc_1"]);
  });

  it("ignores a new session belonging to a different agent", async () => {
    mockListSessions.mockResolvedValueOnce(sessionsPage([]));
    const { result } = renderHook(() => useAgentConnectionDetection(INSTANCE, true, "claude-code"));
    await flush();

    // codex_1 is a real, new session — just not this agent's.
    mockListSessions.mockResolvedValue(sessionsPage(["codex_1"]));
    await tick(7000);

    expect(result.current.signal).toBeNull();
    // Still surfaced for running/lastSeen, which read every session, not just
    // this agent's own.
    expect(result.current.sessions.map((s) => s.session_id)).toEqual(["codex_1"]);
  });

  it("does not prime the baseline off a swallowed sessions failure", async () => {
    mockListSessions.mockResolvedValueOnce(sessionsPage([], true));
    const { result } = renderHook(() => useAgentConnectionDetection(INSTANCE, true, "claude-code"));
    await flush();

    // A real answer next, carrying a session that pre-dates this run. If the
    // failed page above had primed the baseline as empty, this would read as
    // "new" and fire a false signal the instant the pod recovered.
    mockListSessions.mockResolvedValue(sessionsPage(["cc_1"]));
    await tick(7000);
    expect(result.current.signal).toBeNull();

    // The first real poll is now this run's baseline; cc_1 belongs to it.
    mockListSessions.mockResolvedValue(sessionsPage(["cc_1", "cc_2"]));
    await tick(7000);
    expect(result.current.signal).toBe("session");
  });

  it("keeps the last good sessions list across a swallowed failure", async () => {
    mockListSessions.mockResolvedValueOnce(sessionsPage(["cc_1"]));
    const { result } = renderHook(() => useAgentConnectionDetection(INSTANCE, true, "claude-code"));
    await flush();
    expect(result.current.sessions.map((s) => s.session_id)).toEqual(["cc_1"]);

    mockListSessions.mockResolvedValue(sessionsPage([], true));
    await tick(7000);

    expect(result.current.sessions.map((s) => s.session_id)).toEqual(["cc_1"]);
  });

  it("never touches the instance's own fetch (no dataset or doc-count call)", async () => {
    // listSessions is mocked at the module boundary above, so the only path
    // left that would reach the pod directly is a leftover dataset/doc-count
    // call, this hook no longer imports getDatasets or getDatasetDataCount
    // at all, so instance.fetch must stay untouched for the whole poll.
    renderHook(() => useAgentConnectionDetection(INSTANCE, true, "claude-code"));
    await flush();
    await act(async () => {
      jest.advanceTimersByTime(7000);
      await flush();
    });

    expect(INSTANCE.fetch).not.toHaveBeenCalled();
  });
});
