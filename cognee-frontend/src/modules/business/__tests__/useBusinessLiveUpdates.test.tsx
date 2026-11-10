import type { ReactNode } from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useBusinessLiveUpdates, type LiveUpdatesOptions } from "../useBusinessLiveUpdates";
import { brainGraphQueryKey } from "../useBrainGraph";
import { IN_FLIGHT_RECHECK_MS, MAX_REFRESH_GAP_MS, MIN_REFRESH_GAP_MS } from "../graphRefreshThrottle";
import type { CogneeInstance } from "@/modules/instances/types";
import type { LiveSocketFrame, LiveSocketHandlers } from "../liveSocket";
import type { SessionEvent } from "../types";

const mockReportClientLog = jest.fn();
jest.mock("@/services/http", () => ({
  ...jest.requireActual("@/services/http"),
  reportClientLog: (...args: unknown[]) => mockReportClientLog(...args),
}));

const mockCaptureMessage = jest.fn();
const mockAddBreadcrumb = jest.fn();
jest.mock("@/utils/monitoring", () => ({
  captureMessage: (...args: unknown[]) => mockCaptureMessage(...args),
  addBreadcrumb: (...args: unknown[]) => mockAddBreadcrumb(...args),
}));

const mockGetLiveEvents = jest.fn();
jest.mock("../getLiveEvents", () => ({
  __esModule: true,
  default: (...args: unknown[]) => mockGetLiveEvents(...args),
}));

let openedHandlers: LiveSocketHandlers | null = null;
const mockCloseSocket = jest.fn();
const mockOpenLiveSocket = jest.fn((_target: unknown, _datasetId: string, _since: string | null, handlers: LiveSocketHandlers) => {
  openedHandlers = handlers;
  return { close: mockCloseSocket };
});
jest.mock("../liveSocket", () => ({
  __esModule: true,
  openLiveSocket: (...args: Parameters<typeof mockOpenLiveSocket>) => mockOpenLiveSocket(...args),
}));

function socketInstance(): CogneeInstance {
  return {
    name: "CloudCognee",
    instanceId: "inst-1",
    fetch: jest.fn(),
    socketTarget: (path: string) => ({ url: `wss://tenant-x.example.com/api${path}`, apiKey: "key" }),
  };
}

function pollOnlyInstance(): CogneeInstance {
  return { name: "test", instanceId: "inst-2", fetch: jest.fn() };
}

function readyFrame(cursor: string | null): LiveSocketFrame {
  return { kind: "ready", dataset_id: "dataset-1", cursor };
}

function liveEventsFrame(cursor: string, events: SessionEvent[]): Extract<LiveSocketFrame, { kind: "live_events" }> {
  return { kind: "live_events", events, cursor };
}

function Wrapper({ children, client }: { children: ReactNode; client: QueryClient }) {
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

function renderWithClient(datasetId: string | null, instance: CogneeInstance, options?: LiveUpdatesOptions) {
  const client = new QueryClient();
  const invalidateSpy = jest.spyOn(client, "invalidateQueries");
  const view = renderHook(() => useBusinessLiveUpdates(datasetId, instance, options), {
    wrapper: ({ children }) => <Wrapper client={client}>{children}</Wrapper>,
  });
  return { ...view, client, invalidateSpy };
}

afterEach(() => {
  jest.restoreAllMocks();
  openedHandlers = null;
});

describe("useBusinessLiveUpdates — transport selection", () => {
  it("polls from the start for an instance with no socketTarget", async () => {
    mockGetLiveEvents.mockResolvedValue({ events: [], cursor: null });
    renderWithClient("dataset-1", pollOnlyInstance());

    await waitFor(() => expect(mockGetLiveEvents).toHaveBeenCalled());
    expect(mockOpenLiveSocket).not.toHaveBeenCalled();
  });

  it("uses the socket instead of polling when socketTarget is available", async () => {
    renderWithClient("dataset-1", socketInstance());

    await waitFor(() => expect(mockOpenLiveSocket).toHaveBeenCalled());
    expect(mockGetLiveEvents).not.toHaveBeenCalled();
  });
});

describe("useBusinessLiveUpdates — socket give-up hands over to the poll", () => {
  it("resumes the poll from the socket's last cursor with no duplicated events", async () => {
    mockGetLiveEvents.mockResolvedValue({ events: [], cursor: null });
    const { result } = renderWithClient("dataset-1", socketInstance());

    await waitFor(() => expect(openedHandlers).not.toBeNull());
    const handlers = openedHandlers as LiveSocketHandlers;

    const sharedEvent = { kind: "search", qa_id: "q1", time: "t1", question: "why" };
    act(() => {
      handlers.onFrame(readyFrame("cursor-a"));
      handlers.onFrame(liveEventsFrame("cursor-b", [sharedEvent]));
    });
    await waitFor(() => expect(result.current.events).toHaveLength(1));

    // The socket gives up (1008 or unreachable) after already having
    // delivered `sharedEvent` — the poll fallback must resume from
    // cursor-b, not replay events the socket already appended.
    act(() => handlers.onGiveUp("unreachable"));
    mockGetLiveEvents.mockResolvedValueOnce({ events: [sharedEvent], cursor: "cursor-c" });

    await waitFor(() => expect(mockGetLiveEvents.mock.calls[0]?.slice(0, 2)).toEqual(["dataset-1", "cursor-b"]));
    await waitFor(() => expect(result.current.events).toHaveLength(1));
  });

  it("reports live:false while reconnecting and live:true again on the next ready frame", async () => {
    const { result } = renderWithClient("dataset-1", socketInstance());
    await waitFor(() => expect(openedHandlers).not.toBeNull());
    const handlers = openedHandlers as LiveSocketHandlers;

    act(() => handlers.onFrame(readyFrame(null)));
    expect(result.current.live).toBe(true);

    act(() => handlers.onReconnecting());
    expect(result.current.live).toBe(false);

    act(() => handlers.onFrame(readyFrame(null)));
    expect(result.current.live).toBe(true);
  });
});

describe("useBusinessLiveUpdates — graph_grew", () => {
  it("invalidates the focused dataset's graph query instead of polling it", async () => {
    const { invalidateSpy } = renderWithClient("dataset-1", socketInstance());
    await waitFor(() => expect(openedHandlers).not.toBeNull());
    const handlers = openedHandlers as LiveSocketHandlers;

    act(() => handlers.onFrame({ kind: "graph_grew", pipeline_run_id: "run-1" }));

    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: brainGraphQueryKey("inst-1", "dataset-1") });
  });
});

describe("useBusinessLiveUpdates — graph_grew bursts during a large ingest", () => {
  const graphKey = { queryKey: brainGraphQueryKey("inst-1", "dataset-1") };

  function graphInvalidations(spy: jest.SpyInstance): number {
    return spy.mock.calls.filter((c) => JSON.stringify(c[0]) === JSON.stringify(graphKey)).length;
  }

  async function openSocket(): Promise<LiveSocketHandlers> {
    await waitFor(() => expect(openedHandlers).not.toBeNull());
    jest.useFakeTimers();
    return openedHandlers as LiveSocketHandlers;
  }

  afterEach(() => {
    jest.useRealTimers();
  });

  it("collapses a burst of pushes into one refetch now and one after the gap", async () => {
    const { invalidateSpy } = renderWithClient("dataset-1", socketInstance());
    const handlers = await openSocket();

    act(() => {
      for (let i = 0; i < 10; i += 1) handlers.onFrame({ kind: "graph_grew", pipeline_run_id: "run-1" });
    });
    expect(graphInvalidations(invalidateSpy)).toBe(1);

    act(() => {
      jest.advanceTimersByTime(MIN_REFRESH_GAP_MS);
    });
    expect(graphInvalidations(invalidateSpy)).toBe(2);
  });

  it("holds a push back while a graph read is in flight, then refreshes a gap after it lands", async () => {
    const { client, invalidateSpy } = renderWithClient("dataset-1", socketInstance());
    const handlers = await openSocket();
    let finishRead: (value: null) => void = () => {};
    const read = client.fetchQuery({
      queryKey: graphKey.queryKey,
      queryFn: () => new Promise<null>((resolve) => (finishRead = resolve)),
    });

    act(() => handlers.onFrame({ kind: "graph_grew", pipeline_run_id: "run-1" }));
    expect(graphInvalidations(invalidateSpy)).toBe(0);

    await act(async () => {
      finishRead(null);
      await read;
    });
    act(() => {
      jest.advanceTimersByTime(IN_FLIGHT_RECHECK_MS);
    });
    expect(graphInvalidations(invalidateSpy)).toBe(0);

    act(() => {
      jest.advanceTimersByTime(MAX_REFRESH_GAP_MS);
    });
    expect(graphInvalidations(invalidateSpy)).toBe(1);
  });

  it("does not hold a push back behind a different query keyed under the graph's", async () => {
    const { client, invalidateSpy } = renderWithClient("dataset-1", socketInstance());
    const handlers = await openSocket();
    void client.fetchQuery({
      queryKey: [...graphKey.queryKey, "child"],
      queryFn: () => new Promise<null>(() => {}),
    });

    act(() => handlers.onFrame({ kind: "graph_grew", pipeline_run_id: "run-1" }));

    expect(graphInvalidations(invalidateSpy)).toBe(1);
  });

  it("spaces the refetch after a failed graph read at the ceiling", async () => {
    const { client, invalidateSpy } = renderWithClient("dataset-1", socketInstance());
    const handlers = await openSocket();
    await act(async () => {
      await client
        .fetchQuery({ queryKey: graphKey.queryKey, queryFn: () => Promise.reject(new Error("409")), retry: false })
        .catch(() => undefined);
    });

    await act(async () => handlers.onFrame({ kind: "graph_grew", pipeline_run_id: "run-1" }));
    act(() => handlers.onFrame({ kind: "graph_grew", pipeline_run_id: "run-2" }));
    act(() => {
      jest.advanceTimersByTime(MAX_REFRESH_GAP_MS - 1);
    });
    expect(graphInvalidations(invalidateSpy)).toBe(1);

    act(() => {
      jest.advanceTimersByTime(1);
    });
    expect(graphInvalidations(invalidateSpy)).toBe(2);
  });

  it("drops a pending trailing refetch when the dataset view unmounts", async () => {
    const { invalidateSpy, unmount } = renderWithClient("dataset-1", socketInstance());
    const handlers = await openSocket();

    act(() => {
      handlers.onFrame({ kind: "graph_grew", pipeline_run_id: "run-1" });
      handlers.onFrame({ kind: "graph_grew", pipeline_run_id: "run-1" });
    });
    unmount();
    act(() => {
      jest.advanceTimersByTime(MIN_REFRESH_GAP_MS * 2);
    });

    expect(graphInvalidations(invalidateSpy)).toBe(1);
  });
});

describe("useBusinessLiveUpdates — dataset switch", () => {
  it("resets the cursor, seen events, and accumulated log on unmount (COG-6233)", async () => {
    mockGetLiveEvents.mockResolvedValue({ events: [], cursor: null });
    const { result, unmount } = renderWithClient("dataset-1", pollOnlyInstance());

    await waitFor(() => expect(mockGetLiveEvents).toHaveBeenCalled());
    unmount();

    expect(result.current.events).toEqual([]);
  });

  it("closes the socket on unmount", async () => {
    const { unmount } = renderWithClient("dataset-1", socketInstance());
    await waitFor(() => expect(mockOpenLiveSocket).toHaveBeenCalled());

    unmount();
    expect(mockCloseSocket).toHaveBeenCalled();
  });
});

// The graph query polls for growth unless something else is delivering it
// (see useBrainGraph). Getting this signal wrong is invisible on screen — the
// session-event stream keeps working either way — so every transition it can
// make is pinned down here.
describe("useBusinessLiveUpdates — graph-push signal", () => {
  it("reports pushes active only once the handshake completes", async () => {
    const onGraphPushChange = jest.fn();
    renderWithClient("dataset-1", socketInstance(), { onGraphPushChange });

    await waitFor(() => expect(openedHandlers).not.toBeNull());
    // Opening a socket proves nothing: a pod that does not serve the endpoint
    // still accepts the connection attempt.
    expect(onGraphPushChange).not.toHaveBeenCalledWith(true);

    act(() => (openedHandlers as LiveSocketHandlers).onFrame(readyFrame(null)));
    expect(onGraphPushChange).toHaveBeenCalledWith(true);
  });

  it("reports pushes inactive again while reconnecting", async () => {
    const onGraphPushChange = jest.fn();
    renderWithClient("dataset-1", socketInstance(), { onGraphPushChange });
    await waitFor(() => expect(openedHandlers).not.toBeNull());
    const handlers = openedHandlers as LiveSocketHandlers;

    act(() => handlers.onFrame(readyFrame(null)));
    act(() => handlers.onReconnecting());

    expect(onGraphPushChange).toHaveBeenLastCalledWith(false);
  });

  it("reports pushes inactive when the socket gives up", async () => {
    mockGetLiveEvents.mockResolvedValue({ events: [], cursor: null });
    const onGraphPushChange = jest.fn();
    renderWithClient("dataset-1", socketInstance(), { onGraphPushChange });
    await waitFor(() => expect(openedHandlers).not.toBeNull());
    const handlers = openedHandlers as LiveSocketHandlers;

    act(() => handlers.onFrame(readyFrame(null)));
    act(() => handlers.onGiveUp("unreachable"));

    expect(onGraphPushChange).toHaveBeenLastCalledWith(false);
  });

  it("reports pushes inactive for an instance that has no socket at all", async () => {
    mockGetLiveEvents.mockResolvedValue({ events: [], cursor: null });
    const onGraphPushChange = jest.fn();
    renderWithClient("dataset-1", pollOnlyInstance(), { onGraphPushChange });

    await waitFor(() => expect(mockGetLiveEvents).toHaveBeenCalled());
    expect(onGraphPushChange).toHaveBeenCalledWith(false);
    expect(onGraphPushChange).not.toHaveBeenCalledWith(true);
  });

  it("reports pushes inactive on unmount, so the next dataset re-proves its socket", async () => {
    const onGraphPushChange = jest.fn();
    const { unmount } = renderWithClient("dataset-1", socketInstance(), { onGraphPushChange });
    await waitFor(() => expect(openedHandlers).not.toBeNull());

    act(() => (openedHandlers as LiveSocketHandlers).onFrame(readyFrame(null)));
    unmount();

    expect(onGraphPushChange).toHaveBeenLastCalledWith(false);
  });
});

describe("useBusinessLiveUpdates — telemetry", () => {
  it("logs and alerts when the socket gives up, since the fallback hides it from the UI", async () => {
    mockGetLiveEvents.mockResolvedValue({ events: [], cursor: null });
    renderWithClient("dataset-1", socketInstance());
    await waitFor(() => expect(openedHandlers).not.toBeNull());

    act(() => (openedHandlers as LiveSocketHandlers).onGiveUp("rejected"));

    expect(mockReportClientLog).toHaveBeenCalledWith(
      "MINDMAP-WS",
      "error",
      expect.stringContaining("rejected"),
    );
    expect(mockCaptureMessage).toHaveBeenCalledWith(
      "Mindmap live socket unavailable",
      expect.objectContaining({ extra: expect.objectContaining({ reason: "rejected", datasetId: "dataset-1" }) }),
    );
  });

  it("leaves a breadcrumb rather than an alert for a single reconnect", async () => {
    renderWithClient("dataset-1", socketInstance());
    await waitFor(() => expect(openedHandlers).not.toBeNull());

    act(() => (openedHandlers as LiveSocketHandlers).onReconnecting());

    expect(mockAddBreadcrumb).toHaveBeenCalled();
    expect(mockCaptureMessage).not.toHaveBeenCalled();
  });
});

describe("useBusinessLiveUpdates — a socket that comes back", () => {
  it("stops the fallback poll once a ready proves the socket is serving again", async () => {
    mockGetLiveEvents.mockResolvedValue({ events: [], cursor: null });
    renderWithClient("dataset-1", socketInstance());

    // Give up: the caller falls back to the poll.
    await act(async () => {
      openedHandlers?.onGiveUp("unreachable");
    });
    await waitFor(() => expect(mockGetLiveEvents).toHaveBeenCalled());
    const callsWhilePolling = mockGetLiveEvents.mock.calls.length;

    // The socket keeps retrying behind the poll and eventually lands.
    await act(async () => {
      openedHandlers?.onFrame(readyFrame("c9"));
    });
    mockGetLiveEvents.mockClear();
    // The poll's own next tick is what would betray a still-running loop, so
    // wait past one interval of real time rather than advancing a clock this
    // suite does not fake.
    await new Promise((resolve) => setTimeout(resolve, 1700));
    expect(callsWhilePolling).toBeGreaterThan(0);
    expect(mockGetLiveEvents).not.toHaveBeenCalled();
  });

  it("refetches the graph on a RE-handshake but not on the first one", async () => {
    const { invalidateSpy } = renderWithClient("dataset-1", socketInstance());
    const graphKey = { queryKey: brainGraphQueryKey("inst-1", "dataset-1") };

    await act(async () => {
      openedHandlers?.onFrame(readyFrame(null));
    });
    // The query already fetched on mount moments earlier, and
    // invalidateQueries refetches an active observer regardless of staleTime,
    // so invalidating here would fetch the whole graph a second time for
    // every user on the happy path.
    expect(invalidateSpy).not.toHaveBeenCalledWith(graphKey);

    // A second handshake is a reconnect, and the server does not replay a
    // `graph_grew` for a run that finished while the socket was away.
    await act(async () => {
      openedHandlers?.onFrame(readyFrame("c2"));
    });
    expect(invalidateSpy).toHaveBeenCalledWith(graphKey);
    expect(invalidateSpy.mock.calls.filter((c) => JSON.stringify(c[0]) === JSON.stringify(graphKey))).toHaveLength(1);
  });

  it("does not start a second poll when give-up is reached twice", async () => {
    mockGetLiveEvents.mockResolvedValue({ events: [], cursor: null });
    renderWithClient("dataset-1", socketInstance());
    await act(async () => {
      openedHandlers?.onGiveUp("unreachable");
      openedHandlers?.onGiveUp("unreachable");
    });
    await waitFor(() => expect(mockGetLiveEvents).toHaveBeenCalled());
    // pollLiveEvents fires its first request immediately, so a second loop
    // shows up as a second request with no timer advance at all.
    expect(mockGetLiveEvents.mock.calls.length).toBe(1);
  });

  it("closes the socket as well as the poll on unmount, after a give-up", async () => {
    mockGetLiveEvents.mockResolvedValue({ events: [], cursor: null });
    const view = renderWithClient("dataset-1", socketInstance());
    await act(async () => {
      openedHandlers?.onGiveUp("unreachable");
    });
    mockCloseSocket.mockClear();
    view.unmount();
    // The socket is still retrying on its slow timer at this point, so
    // forgetting its stopper would leak a reconnecting socket per mount.
    expect(mockCloseSocket).toHaveBeenCalled();
  });
});

describe("useBusinessLiveUpdates — invariants the module header claims", () => {
  const sharedEvent: SessionEvent = { kind: "search", qa_id: "q-shared", time: "2026-01-01T00:00:00Z", question: "why" };

  it("does not double an event delivered by both transports", async () => {
    // Since the socket keeps retrying after a give-up, a live socket and its
    // fallback poll now overlap as a NORMAL state rather than a one-way
    // handover — so the same event arriving twice is a routine event, not an
    // exotic one. This is the invariant the module header advertises.
    mockGetLiveEvents.mockResolvedValue({ events: [sharedEvent], cursor: "c1" });
    const { result } = renderWithClient("dataset-1", socketInstance());

    await act(async () => {
      openedHandlers?.onFrame(liveEventsFrame("c1", [sharedEvent]));
    });
    expect(result.current.events).toHaveLength(1);

    await act(async () => {
      openedHandlers?.onGiveUp("unreachable");
    });
    await waitFor(() => expect(mockGetLiveEvents).toHaveBeenCalled());

    // The poll replays the same event from the shared cursor. One event, still.
    await waitFor(() => expect(result.current.events).toHaveLength(1));

    // And the socket recovering does not resurrect it either.
    await act(async () => {
      openedHandlers?.onFrame(liveEventsFrame("c1", [sharedEvent]));
    });
    expect(result.current.events).toHaveLength(1);
  });

  // Both of these switch dataset on ONE mount rather than remounting. The
  // seen-key set and the cursor live in refs, so a remount gets fresh ones for
  // free and a test built that way passes whether or not the cleanup resets
  // anything. An in-place dataset switch is the only case where the reset is
  // observable, and it is also the real user action.
  it("forgets the previous dataset's events when the dataset switches under it", async () => {
    const client = new QueryClient();
    const instance = socketInstance();
    const view = renderHook(({ id }: { id: string }) => useBusinessLiveUpdates(id, instance), {
      initialProps: { id: "dataset-1" },
      wrapper: ({ children }) => <Wrapper client={client}>{children}</Wrapper>,
    });

    await act(async () => {
      openedHandlers?.onFrame(liveEventsFrame("c1", [sharedEvent]));
    });
    expect(view.result.current.events).toHaveLength(1);

    view.rerender({ id: "dataset-2" });
    // The same event id can legitimately appear in the new dataset's stream.
    // If the seen-keys carried over it would be swallowed as a repeat and the
    // new dataset would render as idle.
    await act(async () => {
      openedHandlers?.onFrame({ kind: "live_events", cursor: "c1", events: [sharedEvent] });
    });
    expect(view.result.current.events).toHaveLength(1);
    expect(view.result.current.events[0]).toEqual(sharedEvent);
  });

  it("does not carry the old dataset's cursor into the new dataset's poll", async () => {
    mockGetLiveEvents.mockResolvedValue({ events: [], cursor: null });
    const client = new QueryClient();
    const instance = socketInstance();
    const view = renderHook(({ id }: { id: string }) => useBusinessLiveUpdates(id, instance), {
      initialProps: { id: "dataset-1" },
      wrapper: ({ children }) => <Wrapper client={client}>{children}</Wrapper>,
    });

    await act(async () => {
      openedHandlers?.onFrame(readyFrame("cursor-from-dataset-1"));
    });

    view.rerender({ id: "dataset-2" });
    mockGetLiveEvents.mockClear();
    await act(async () => {
      openedHandlers?.onGiveUp("unreachable");
    });
    // A `since` from another dataset means nothing to this one, so the poll
    // has to start from scratch rather than asking for events after a
    // timestamp that belongs to a different stream.
    await waitFor(() => expect(mockGetLiveEvents.mock.calls[0]?.slice(0, 2)).toEqual(["dataset-2", null]));
  });

  it("wires a drop reporter into both transports, not just the socket", async () => {
    mockGetLiveEvents.mockResolvedValue({ events: [], cursor: null });
    renderWithClient("dataset-1", socketInstance());

    // Socket half: liveSocket hands the count to whatever the hook supplied.
    mockAddBreadcrumb.mockClear();
    openedHandlers?.onDroppedEvents?.(2, 3);
    expect(mockAddBreadcrumb).toHaveBeenCalledWith(
      expect.objectContaining({ category: "mindmap.live-events", data: expect.objectContaining({ transport: "socket" }) }),
    );

    // Poll half: the callback has to reach getLiveEvents, or a drift over the
    // fallback transport is invisible exactly when the socket is already gone.
    await act(async () => {
      openedHandlers?.onGiveUp("unreachable");
    });
    await waitFor(() => expect(mockGetLiveEvents).toHaveBeenCalled());
    const pollReporter = mockGetLiveEvents.mock.calls.at(-1)?.[3];
    expect(typeof pollReporter).toBe("function");
    mockAddBreadcrumb.mockClear();
    (pollReporter as (d: number, t: number) => void)(1, 4);
    expect(mockAddBreadcrumb).toHaveBeenCalledWith(
      expect.objectContaining({ data: expect.objectContaining({ transport: "poll" }) }),
    );
  });
});
