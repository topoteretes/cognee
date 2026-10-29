import fetchMock from "jest-fetch-mock";
import localFetch from "../localFetch";
import { HttpError } from "@/services/http/errors";

describe("localFetch errors", () => {
  beforeEach(() => {
    fetchMock.resetMocks();
    jest.spyOn(console, "log").mockImplementation(() => {});
    jest.spyOn(console, "error").mockImplementation(() => {});
  });
  afterEach(() => jest.restoreAllMocks());

  it("preserves cancellation without reporting a network failure", async () => {
    const error = new DOMException("Aborted", "AbortError");
    fetchMock.mockRejectOnce(error);
    await expect(localFetch("/v1/datasets")).rejects.toBe(error);
    expect(console.error).not.toHaveBeenCalled();
  });

  it("still reports a real connection failure", async () => {
    fetchMock.mockRejectOnce(new TypeError("Failed to fetch"));
    await expect(localFetch("/v1/datasets")).rejects.toThrow("No connection to the server.");
    expect(console.error).toHaveBeenCalled();
  });

  it("preserves the server's error message", async () => {
    fetchMock.mockResponseOnce(JSON.stringify({ detail: "Service unavailable" }), { status: 503 });
    await expect(localFetch("/v1/datasets")).rejects.toThrow("Service unavailable");
    expect(console.error).not.toHaveBeenCalled();
  });

  it("preserves HTTP status for the live-events cursor recovery", async () => {
    fetchMock.mockResponseOnce(JSON.stringify({ error: "Failed to fetch live events" }), { status: 409 });
    const error = await localFetch("/v1/visualize/live-events?since=stale").catch((error) => error);
    expect(error).toBeInstanceOf(HttpError);
    expect(error.status).toBe(409);
    expect(console.error).not.toHaveBeenCalled();
  });
});
