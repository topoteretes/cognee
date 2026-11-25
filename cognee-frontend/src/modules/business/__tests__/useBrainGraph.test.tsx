import type { ReactNode } from "react";
import { renderHook, act } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useBrainGraph } from "../useBrainGraph";
import type { CogneeInstance } from "@/modules/instances/types";

// Mirrors the hook's own constant (GRAPH_REFETCH_INTERVAL_MS) and the circuit
// breaker's (CIRCUIT_BREAKER_THRESHOLD / CIRCUIT_BREAKER_BACKOFF_MS) — kept
// local so these assert the observable cadence, not the implementation.
const POLL_MS = 8_000;
const BACKOFF_MS = 60_000;
const FAILURES_TO_OPEN = 3;
// React Query re-arms the interval from the moment a tick settles, and the
// helpers below burn a few milliseconds per settle — negligible against an 8s
// cadence, but enough that "the next tick fired" needs a little slack.
const TICK_SLACK_MS = 50;

const mockFetch = jest.fn();

function instance(): CogneeInstance {
  return { name: "CloudCognee", instanceId: "inst-1", fetch: mockFetch };
}

// Every cloud tenant looks like this: createPodClient always builds a
// socketTarget, whether or not the pod behind it actually serves the socket.
function socketCapableInstance(): CogneeInstance {
  return {
    ...instance(),
    socketTarget: (path: string) => ({ url: `wss://tenant-x.example.com/api${path}`, apiKey: "key" }),
  };
}

function Wrapper({ children }: { children: ReactNode }) {
  // No retry override here on purpose: the hook must bring its own
  // `retry: false`, and a wrapper-level one would hide a missing one.
  const client = new QueryClient();
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

// The shared Wrapper mints a QueryClient per render, which is fine for a
// hook that never re-renders with new props but would wipe the cache under
// rerender() — this keeps one client across a test's renders.
function stableWrapper() {
  const client = new QueryClient();
  return function StableWrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  };
}

const payload = { nodes: [], links: [] };

// A pod that answers with JSON; the streamed transport is getBrainGraph's own
// concern and is covered in its test.
function respondOk() {
  mockFetch.mockResolvedValue({ headers: { get: () => "application/json" }, json: async () => payload });
}

// The pod http client rejects on non-2xx (it never hands back a non-ok
// Response), so that's what a failed tick looks like here.
function respondError() {
  mockFetch.mockRejectedValue(new Error("500 Internal Server Error"));
}

// Moves the fake clock and drains the microtasks each fired timer produces —
// the async variant does that between timers, the sync one does not, which
// leaves a just-started fetch unsettled.
async function advance(ms: number) {
  await act(async () => {
    await jest.advanceTimersByTimeAsync(ms);
  });
}

// React Query batches its result notifications onto a zero-delay timer, which
// a zero-length advance does not run — so making a settled tick observable
// means nudging the clock, not just flushing microtasks. RTL's waitFor is
// avoided throughout: under fake timers it advances the clock by its own
// polling interval, which silently shifts the cadence measured here.
async function settle() {
  await advance(1);
}

beforeEach(() => {
  jest.useFakeTimers();
  respondOk();
});

afterEach(() => {
  jest.useRealTimers();
});

describe("useBrainGraph", () => {
  it("fetches only the focused dataset's graph", async () => {
    const { result } = renderHook(() => useBrainGraph(instance(), "ds-1"), { wrapper: Wrapper });

    await settle();
    expect(result.current.data).toBe(payload);
    // Asserting the URL alone: the request options are getBrainGraph's own
    // concern and are covered in its test, so this stays about which dataset
    // gets fetched.
    expect(mockFetch.mock.calls[0][0]).toBe("/v1/visualize/json?dataset_id=ds-1&max_nodes=1000");
  });

  it("hands the read React Query's abort signal, so a cancelled refetch frees its connection", async () => {
    renderHook(() => useBrainGraph(instance(), "ds-1"), { wrapper: Wrapper });

    await settle();
    expect(mockFetch.mock.calls[0][1].signal).toBeInstanceOf(AbortSignal);
  });

  it("stays idle while no dataset is focused", async () => {
    renderHook(() => useBrainGraph(instance(), null), { wrapper: Wrapper });

    await advance(POLL_MS * 2);
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("polls the focused dataset on the normal cadence", async () => {
    renderHook(() => useBrainGraph(instance(), "ds-1"), { wrapper: Wrapper });

    await settle();
    expect(mockFetch).toHaveBeenCalledTimes(1);

    await advance(POLL_MS + TICK_SLACK_MS);
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });

  it("does not retry inside a failed tick — the next tick is the retry", async () => {
    respondError();
    renderHook(() => useBrainGraph(instance(), "ds-1"), { wrapper: Wrapper });

    await settle();
    // React Query's default retry (3 attempts, backing off from 1s) would fire
    // again well inside one poll interval.
    await advance(POLL_MS - TICK_SLACK_MS);
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it("surfaces a failed tick as `error` instead of throwing it at an error boundary", async () => {
    respondError();
    const { result } = renderHook(() => useBrainGraph(instance(), "ds-1"), { wrapper: Wrapper });

    await settle();
    expect(result.current.error).toBeInstanceOf(Error);
  });

  it("keeps the last good graph alongside the error when a background tick fails", async () => {
    // This pairing is why BusinessView must gate its error state on "and there
    // is nothing to show": a rendered graph survives a failed poll tick, so an
    // error alone must not put a full-bleed overlay over it.
    const { result } = renderHook(() => useBrainGraph(instance(), "ds-1"), { wrapper: Wrapper });
    await settle();

    respondError();
    await advance(POLL_MS + TICK_SLACK_MS);
    await settle();

    expect(result.current.error).toBeInstanceOf(Error);
    expect(result.current.data).toBe(payload);
  });

  it("backs the cadence off once consecutive ticks keep failing", async () => {
    respondError();
    renderHook(() => useBrainGraph(instance(), "ds-1"), { wrapper: Wrapper });

    await settle();
    for (let tick = 1; tick < FAILURES_TO_OPEN; tick++) {
      await advance(POLL_MS + TICK_SLACK_MS);
      await settle();
    }
    expect(mockFetch).toHaveBeenCalledTimes(FAILURES_TO_OPEN);

    // Breaker open: the normal interval must no longer fire, only the backoff.
    await advance(POLL_MS);
    expect(mockFetch).toHaveBeenCalledTimes(FAILURES_TO_OPEN);
    await advance(BACKOFF_MS);
    expect(mockFetch).toHaveBeenCalledTimes(FAILURES_TO_OPEN + 1);
  });

  it("returns to the normal cadence as soon as a tick succeeds again", async () => {
    respondError();
    renderHook(() => useBrainGraph(instance(), "ds-1"), { wrapper: Wrapper });

    await settle();
    for (let tick = 1; tick < FAILURES_TO_OPEN; tick++) {
      await advance(POLL_MS + TICK_SLACK_MS);
      await settle();
    }

    respondOk();
    await advance(BACKOFF_MS);
    await settle();
    expect(mockFetch).toHaveBeenCalledTimes(FAILURES_TO_OPEN + 1);

    await advance(POLL_MS + TICK_SLACK_MS);
    expect(mockFetch).toHaveBeenCalledTimes(FAILURES_TO_OPEN + 2);
  });

  // The growth signal is either the socket's `graph_grew` or this poll, never
  // neither. Gating the poll on the instance being socket-CAPABLE rather than
  // on the socket actually working left a pod that does not serve the
  // endpoint with no growth signal at all, silently, for the whole session.
  describe("growth-signal handover", () => {
    it("keeps polling a dataset that holds rows on the stream, handing the previous payload to the read", async () => {
      mockFetch.mockResolvedValue({
        headers: { get: () => "application/json" },
        json: async () => ({ nodes: [{ id: "row", type: "DltRow", text: "Row Data:\n  person: Ana" }], links: [] }),
      });
      renderHook(() => useBrainGraph(instance(), "csv"), { wrapper: stableWrapper() });
      await settle();

      await advance(POLL_MS + TICK_SLACK_MS);
      expect(mockFetch).toHaveBeenCalledTimes(2);
      expect(mockFetch.mock.calls[1][1].headers.Accept).toContain("text/event-stream");
    });

    it("keeps polling a socket-capable instance until pushes are confirmed", async () => {
      renderHook(() => useBrainGraph(socketCapableInstance(), "ds-1"), { wrapper: Wrapper });

      await settle();
      expect(mockFetch).toHaveBeenCalledTimes(1);

      await advance(POLL_MS + TICK_SLACK_MS);
      expect(mockFetch).toHaveBeenCalledTimes(2);
    });

    it("stops polling while the socket is pushing growth", async () => {
      renderHook(() => useBrainGraph(socketCapableInstance(), "ds-1", true), { wrapper: Wrapper });

      await settle();
      expect(mockFetch).toHaveBeenCalledTimes(1);

      await advance(POLL_MS * 2);
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });

    it("resumes polling as soon as the pushes stop", async () => {
      const { rerender } = renderHook(({ push }) => useBrainGraph(socketCapableInstance(), "ds-1", push), {
        initialProps: { push: true },
        wrapper: stableWrapper(),
      });

      await settle();
      await advance(POLL_MS * 2);
      expect(mockFetch).toHaveBeenCalledTimes(1);

      rerender({ push: false });
      await advance(POLL_MS + TICK_SLACK_MS);
      expect(mockFetch).toHaveBeenCalledTimes(2);
    });
  });
});
