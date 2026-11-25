import { HttpError } from "@/services/http/errors";
import { pollLiveEvents, type CursorRef } from "../pollLiveEvents";
import type { CogneeInstance } from "@/modules/instances/types";

const mockGetLiveEvents = jest.fn();
jest.mock("../getLiveEvents", () => ({
  __esModule: true,
  default: (...args: unknown[]) => mockGetLiveEvents(...args),
}));

const instance: CogneeInstance = { name: "test", instanceId: "test-instance", fetch: jest.fn() };

function callbacks() {
  const fresh: Array<{ events: unknown[]; cursor: string | null }> = [];
  const liveChanges: boolean[] = [];
  return {
    fresh,
    liveChanges,
    onFresh: (events: unknown[], cursor: string | null) => fresh.push({ events, cursor }),
    onLiveChange: (live: boolean) => liveChanges.push(live),
  };
}

beforeEach(() => jest.useFakeTimers());
afterEach(() => {
  jest.useRealTimers();
  jest.restoreAllMocks();
});

// This is a moved, not rewritten, version of what used to live inline in
// useBusinessLiveUpdates.ts — same COG-6412 409-drops-the-cursor fix, same
// backoff. See useBusinessLiveUpdates.test.ts for the transport-handover
// coverage this extraction exists to enable.
describe("pollLiveEvents", () => {
  it("drops the cursor after a 409 so the next poll starts a fresh snapshot", async () => {
    mockGetLiveEvents
      .mockResolvedValueOnce({ events: [], cursor: "cursor-1" })
      .mockRejectedValueOnce(new HttpError(409, "Conflict", "stale cursor"))
      .mockResolvedValueOnce({ events: [], cursor: "cursor-2" });

    const cursorRef: CursorRef = { current: null };
    const cb = callbacks();
    const stop = pollLiveEvents("dataset-1", instance, cursorRef, cb);

    await jest.advanceTimersByTimeAsync(0);
    expect(mockGetLiveEvents.mock.calls[0].slice(0, 3)).toEqual(["dataset-1", null, instance]);
    expect(cursorRef.current).toBe("cursor-1");

    await jest.advanceTimersByTimeAsync(1500);
    expect(mockGetLiveEvents.mock.calls[1].slice(0, 3)).toEqual(["dataset-1", "cursor-1", instance]);
    // The 409 must drop the cursor rather than keep repeating it.
    expect(cursorRef.current).toBeNull();

    // Backoff after one failure is POLL_MS * (failures + 1) = 3000ms.
    await jest.advanceTimersByTimeAsync(3000);
    expect(mockGetLiveEvents.mock.calls[2].slice(0, 3)).toEqual(["dataset-1", null, instance]);

    stop();
  });

  it("reports live:false on a failure and live:true again once a poll succeeds", async () => {
    mockGetLiveEvents
      .mockRejectedValueOnce(new HttpError(500, "Internal Server Error", "boom"))
      .mockResolvedValueOnce({ events: [], cursor: null });

    const cb = callbacks();
    const stop = pollLiveEvents("dataset-1", instance, { current: null }, cb);

    await jest.advanceTimersByTimeAsync(0);
    expect(cb.liveChanges).toEqual([false]);

    await jest.advanceTimersByTimeAsync(3000);
    expect(cb.liveChanges).toEqual([false, true]);

    stop();
  });

  it("resumes from a cursor seeded by the caller (a socket-to-poll handover)", async () => {
    mockGetLiveEvents.mockResolvedValueOnce({ events: [], cursor: "cursor-3" });
    const cursorRef: CursorRef = { current: "socket-cursor" };

    const stop = pollLiveEvents("dataset-1", instance, cursorRef, callbacks());
    await jest.advanceTimersByTimeAsync(0);

    expect(mockGetLiveEvents.mock.calls[0].slice(0, 3)).toEqual(["dataset-1", "socket-cursor", instance]);
    stop();
  });

  it("stops polling once the returned cleanup function is called", async () => {
    mockGetLiveEvents.mockResolvedValue({ events: [], cursor: null });
    const stop = pollLiveEvents("dataset-1", instance, { current: null }, callbacks());

    await jest.advanceTimersByTimeAsync(0);
    expect(mockGetLiveEvents).toHaveBeenCalledTimes(1);

    stop();
    await jest.advanceTimersByTimeAsync(10_000);
    expect(mockGetLiveEvents).toHaveBeenCalledTimes(1);
  });
});

describe("pollLiveEvents — the cursor policy it shares with the socket", () => {
  it("leaves the resume point alone when a tick returns no usable cursor", async () => {
    // A null cursor means "nothing to advance to", not "start over". Assigning
    // it made the next request drop `since` and replay the entire history,
    // every 1.5s, for as long as the pod kept omitting it.
    mockGetLiveEvents
      .mockResolvedValueOnce({ events: [], cursor: "cursor-1" })
      .mockResolvedValueOnce({ events: [], cursor: null })
      .mockResolvedValueOnce({ events: [], cursor: "cursor-2" });

    const cursorRef: CursorRef = { current: null };
    const stop = pollLiveEvents("dataset-1", instance, cursorRef, callbacks());

    await jest.advanceTimersByTimeAsync(0);
    expect(cursorRef.current).toBe("cursor-1");

    await jest.advanceTimersByTimeAsync(1500);
    expect(cursorRef.current).toBe("cursor-1");

    // And the next request still resumes from it rather than from scratch.
    await jest.advanceTimersByTimeAsync(1500);
    expect(mockGetLiveEvents.mock.calls[2]?.slice(0, 2)).toEqual(["dataset-1", "cursor-1"]);

    stop();
  });

  it("hands the caller's drop reporter through to the fetcher", async () => {
    // Without this the fallback transport silently eats malformed events at
    // exactly the moment the socket is already gone.
    mockGetLiveEvents.mockResolvedValue({ events: [], cursor: null });
    const onDroppedEvents = jest.fn();
    const stop = pollLiveEvents("dataset-1", instance, { current: null }, { ...callbacks(), onDroppedEvents });

    await jest.advanceTimersByTimeAsync(0);
    expect(mockGetLiveEvents.mock.calls[0]?.[3]).toBe(onDroppedEvents);

    stop();
  });
});
