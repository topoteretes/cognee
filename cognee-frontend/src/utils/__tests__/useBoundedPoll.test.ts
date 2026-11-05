import { act, renderHook } from "@testing-library/react";
import { useBoundedPoll } from "../useBoundedPoll";

// Flushes the microtask queue so a check's `await` (even on a value that's
// already resolved, e.g. a synchronous jest.fn()'s return) has a chance to
// continue before the next assertion — advancing fake timers alone only
// fires due macrotasks, it does not drain pending promise continuations.
async function flush(): Promise<void> {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}

async function tick(ms: number): Promise<void> {
  await act(async () => {
    jest.advanceTimersByTime(ms);
    await Promise.resolve();
    await Promise.resolve();
  });
}

describe("useBoundedPoll", () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it("does nothing while inactive", async () => {
    const check = jest.fn();
    renderHook(() => useBoundedPoll(false, "k", () => check, 1000, 10));

    await tick(5000);

    expect(check).not.toHaveBeenCalled();
  });

  it("checks immediately, then again on every interval", async () => {
    const check = jest.fn();
    renderHook(() => useBoundedPoll(true, "k", () => check, 1000, 10));
    await flush();

    expect(check).toHaveBeenCalledTimes(1);

    // One interval hop at a time: each hop is itself a timer -> microtask ->
    // new timer chain, and a single bulk advance across multiple hops can't
    // interleave the microtask flush each hop depends on.
    await tick(1000);
    await tick(1000);
    await tick(1000);

    expect(check).toHaveBeenCalledTimes(4);
  });

  it("stops polling once maxPolls is reached", async () => {
    const check = jest.fn();
    renderHook(() => useBoundedPoll(true, "k", () => check, 1000, 2));
    await flush();

    for (let i = 0; i < 10; i++) {
      await tick(1000);
    }

    // 1 immediate call + 2 bounded interval calls, never a third.
    expect(check).toHaveBeenCalledTimes(3);
  });

  it("reports expired once maxPolls is reached, false until then", async () => {
    const check = jest.fn();
    const { result } = renderHook(() => useBoundedPoll(true, "k", () => check, 1000, 1));
    await flush();
    expect(result.current).toBe(false);

    // First interval tick: polls becomes 1, still within maxPolls.
    await tick(1000);
    expect(result.current).toBe(false);

    // Second interval tick: polls would become 2, over maxPolls — expires
    // instead of checking again.
    await tick(1000);
    expect(result.current).toBe(true);
  });

  it("resets expired to false when a fresh run starts", async () => {
    const check = jest.fn();
    const { result, rerender } = renderHook(
      ({ runKey }: { runKey: string }) => useBoundedPoll(true, runKey, () => check, 1000, 1),
      { initialProps: { runKey: "a" } },
    );
    await flush();
    await tick(1000);
    await tick(1000);
    expect(result.current).toBe(true);

    rerender({ runKey: "b" });
    expect(result.current).toBe(false);
  });

  it("marks the poll stopped once the effect tears down mid-check", async () => {
    const captured: { isStopped: (() => boolean) | null } = { isStopped: null };
    const check = jest.fn((isStopped: () => boolean) => {
      captured.isStopped = isStopped;
    });
    const { unmount } = renderHook(() => useBoundedPoll(true, "k", () => check, 1000, 10));
    await flush();

    expect(captured.isStopped?.()).toBe(false);
    unmount();
    expect(captured.isStopped?.()).toBe(true);
  });

  it("restarts the interval when intervalMs changes", async () => {
    const check = jest.fn();
    const { rerender } = renderHook(
      ({ intervalMs }: { intervalMs: number }) => useBoundedPoll(true, "k", () => check, intervalMs, 10),
      { initialProps: { intervalMs: 1000 } },
    );
    await flush();
    check.mockClear();

    rerender({ intervalMs: 5000 });
    await flush();
    check.mockClear();

    await tick(4000);
    // The old 1000ms cadence would have fired by now; the new 5000ms one hasn't.
    expect(check).not.toHaveBeenCalled();

    await tick(1000);
    expect(check).toHaveBeenCalledTimes(1);
  });

  it("calls makeCheck again, fresh, whenever runKey changes", async () => {
    const makeCheck = jest.fn(() => jest.fn());
    const { rerender } = renderHook(
      ({ runKey }: { runKey: string }) => useBoundedPoll(true, runKey, makeCheck, 1000, 10),
      { initialProps: { runKey: "a" } },
    );
    await flush();
    expect(makeCheck).toHaveBeenCalledTimes(1);

    rerender({ runKey: "a" });
    expect(makeCheck).toHaveBeenCalledTimes(1);

    rerender({ runKey: "b" });
    expect(makeCheck).toHaveBeenCalledTimes(2);
  });

  it("gives each run its own check instance — a run's local state doesn't leak into the next", async () => {
    // Models the real use case: makeCheck closes over a baseline that should
    // reset every time runKey changes, without this hook needing to know
    // anything about what a "baseline" is.
    const makeCheck = jest.fn(() => {
      let calls = 0;
      return () => {
        calls += 1;
        seenCallCounts.push(calls);
      };
    });
    const seenCallCounts: number[] = [];

    const { rerender } = renderHook(
      ({ runKey }: { runKey: string }) => useBoundedPoll(true, runKey, makeCheck, 1000, 10),
      { initialProps: { runKey: "a" } },
    );
    await flush();
    await tick(1000);
    expect(seenCallCounts).toEqual([1, 2]);

    rerender({ runKey: "b" });
    // A fresh run: its check starts counting from 1 again, not 3.
    expect(seenCallCounts).toEqual([1, 2, 1]);
  });

  describe("sequential polling (no overlap)", () => {
    it("does not start a second check while the first is still pending past its own interval", async () => {
      const released: { release: (() => void) | null } = { release: null };
      const check = jest.fn().mockImplementationOnce(
        () => new Promise<void>((resolve) => { released.release = resolve; }),
      );
      renderHook(() => useBoundedPoll(true, "k", () => check, 1000, 10));
      await flush();
      expect(check).toHaveBeenCalledTimes(1);

      // The interval would have fired twice by now if checks overlapped —
      // the first one is still unresolved, so nothing new should start.
      await tick(5000);
      expect(check).toHaveBeenCalledTimes(1);

      released.release?.();
      await flush();
      // Now that it settled, exactly one more check starts (after its own
      // interval elapses, not immediately).
      expect(check).toHaveBeenCalledTimes(1);
      await tick(1000);
      expect(check).toHaveBeenCalledTimes(2);
    });

    it("measures the interval from when the previous check settles, not from when it started", async () => {
      const released: { release: (() => void) | null } = { release: null };
      const check = jest.fn().mockImplementationOnce(
        () => new Promise<void>((resolve) => { released.release = resolve; }),
      );
      renderHook(() => useBoundedPoll(true, "k", () => check, 1000, 10));
      await flush();

      // First check has been "in flight" for 3s when it finally settles.
      await tick(3000);
      released.release?.();
      await flush();
      expect(check).toHaveBeenCalledTimes(1);

      // The next one must wait a fresh 1000ms from settlement, not fire
      // immediately just because 3s already passed since it started.
      await tick(999);
      expect(check).toHaveBeenCalledTimes(1);
      await tick(1);
      expect(check).toHaveBeenCalledTimes(2);
    });

    it("does not schedule a next check once stopped while the current one was still pending", async () => {
      const released: { release: (() => void) | null } = { release: null };
      const check = jest.fn().mockImplementationOnce(
        () => new Promise<void>((resolve) => { released.release = resolve; }),
      );
      const { unmount } = renderHook(() => useBoundedPoll(true, "k", () => check, 1000, 10));
      await flush();
      expect(check).toHaveBeenCalledTimes(1);

      unmount();
      released.release?.();
      await flush();

      // The pending check's own promise settled after teardown — without the
      // post-await stopped guard this would still schedule (and later fire)
      // another one.
      await tick(10_000);
      expect(check).toHaveBeenCalledTimes(1);
    });
  });
});
