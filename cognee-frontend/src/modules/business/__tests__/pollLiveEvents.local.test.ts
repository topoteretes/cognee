import fetchMock from "jest-fetch-mock";
import localFetch from "@/modules/instances/localFetch";
import { pollLiveEvents } from "../pollLiveEvents";

it("recovers a rejected local cursor through the real fetch adapter", async () => {
  jest.useFakeTimers();
  fetchMock.resetMocks();
  jest.spyOn(console, "log").mockImplementation(() => {});
  jest.spyOn(console, "error").mockImplementation(() => {});
  fetchMock
    .mockResponseOnce(JSON.stringify({ error: "Failed to fetch live events" }), { status: 409 })
    .mockResponseOnce(JSON.stringify({ events: [], cursor: "fresh-cursor" }));
  const cursor = { current: "stale-cursor" as string | null };
  const onLiveChange = jest.fn();
  const stop = pollLiveEvents("dataset-1", { name: "local", instanceId: "local", fetch: localFetch }, cursor, {
    onFresh: jest.fn(), onLiveChange,
  });
  try {
    await jest.advanceTimersByTimeAsync(0);
    expect(cursor.current).toBeNull();
    await jest.advanceTimersByTimeAsync(3000);
    const retryUrl = String(fetchMock.mock.calls[1][0]);
    expect(retryUrl).toContain("dataset_id=dataset-1");
    expect(retryUrl).not.toContain("since=");
    expect(cursor.current).toBe("fresh-cursor");
    expect(onLiveChange.mock.calls).toEqual([[false], [true]]);
    expect(console.error).not.toHaveBeenCalled();
  } finally {
    stop();
    jest.useRealTimers();
    jest.restoreAllMocks();
  }
});
