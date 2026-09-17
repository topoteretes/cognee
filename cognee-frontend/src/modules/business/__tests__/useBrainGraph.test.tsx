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

function Wrapper({ children }: { children: ReactNode }) {
  // No retry override here on purpose: the hook must bring its own
  // `retry: false`, and a wrapper-level one would hide a missing one.
  const client = new QueryClient();
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

const payload = { nodes: [], links: [] };

function respondOk() {
  mockFetch.mockResolvedValue({ json: async () => payload });
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
    expect(mockFetch).toHaveBeenCalledWith("/v1/visualize/json?dataset_id=ds-1&max_nodes=500");
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
});
