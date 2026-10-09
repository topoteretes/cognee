import { renderHook, waitFor } from "@testing-library/react";
import { HttpError } from "@/services/http/errors";
import { useBusinessLiveUpdates } from "../useBusinessLiveUpdates";
import type { CogneeInstance } from "@/modules/instances/types";

const mockGetLiveEvents = jest.fn();
jest.mock("../getLiveEvents", () => ({
  __esModule: true,
  default: (...args: unknown[]) => mockGetLiveEvents(...args),
}));

const instance: CogneeInstance = { name: "test", instanceId: "test-instance", fetch: jest.fn() };

// COG-6412: a live network capture showed the poll repeating the exact same
// `since` cursor after a 409, because the cursor was only ever advanced on
// success — so a single rejected cursor left the "live" indicator stuck on
// "reconnecting" forever, since every retry re-sent the same value and got
// the same 409 back.
describe("useBusinessLiveUpdates", () => {
  afterEach(() => jest.restoreAllMocks());

  it("drops the cursor after a 409 so the next poll starts a fresh snapshot", async () => {
    mockGetLiveEvents
      .mockResolvedValueOnce({ events: [], cursor: "2026-08-20T06:12:17.499028+00:00" })
      .mockRejectedValueOnce(new HttpError(409, "Conflict", "stale cursor"))
      .mockResolvedValueOnce({ events: [], cursor: "2026-08-20T06:12:19.000000+00:00" });

    renderHook(() => useBusinessLiveUpdates("dataset-1", instance));

    await waitFor(() => expect(mockGetLiveEvents).toHaveBeenCalledTimes(2), { timeout: 3000 });
    // Second call carries the cursor the first call returned.
    expect(mockGetLiveEvents.mock.calls[1][1]).toBe("2026-08-20T06:12:17.499028+00:00");

    // The retry after a 409 backs off to POLL_MS * (failures + 1) = 3000ms
    // (see useBusinessLiveUpdates.ts), so a 3000ms waitFor timeout raced the
    // scheduled poll exactly at the wire and flaked on slower CI runners.
    // Give real margin above the scheduled delay instead of matching it.
    await waitFor(() => expect(mockGetLiveEvents).toHaveBeenCalledTimes(3), { timeout: 8000 });
    // Third call must NOT repeat the cursor that just 409'd.
    expect(mockGetLiveEvents.mock.calls[2][1]).toBeNull();
  });

  it("reports live: true again once a poll succeeds after a failure", async () => {
    mockGetLiveEvents
      .mockRejectedValueOnce(new HttpError(500, "Internal Server Error", "boom"))
      .mockResolvedValueOnce({ events: [], cursor: null });

    const { result } = renderHook(() => useBusinessLiveUpdates("dataset-1", instance));

    await waitFor(() => expect(result.current.live).toBe(false));
    await waitFor(() => expect(result.current.live).toBe(true), { timeout: 3000 });
  });
});
