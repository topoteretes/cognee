/** @jest-environment-options {"url": "https://ui.example.test/dashboard"} */
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

describe("localFetch backend selection", () => {
  const originalUrl = process.env.NEXT_PUBLIC_LOCAL_API_URL;
  beforeEach(() => {
    fetchMock.resetMocks();
    jest.spyOn(console, "log").mockImplementation(() => {});
  });
  afterEach(() => {
    document.getElementById("cognee-runtime-config")?.remove();
    if (originalUrl === undefined) delete process.env.NEXT_PUBLIC_LOCAL_API_URL;
    else process.env.NEXT_PUBLIC_LOCAL_API_URL = originalUrl;
    jest.restoreAllMocks();
  });

  it("sends requests to the runtime URL even when a different URL was baked in", async () => {
    process.env.NEXT_PUBLIC_LOCAL_API_URL = "http://build-time.test:8000";
    const config = document.createElement("script");
    config.id = "cognee-runtime-config";
    config.type = "application/json";
    config.textContent = JSON.stringify({ backendUrl: "https://runtime.test/" });
    document.head.appendChild(config);
    fetchMock.mockResponseOnce("{}");
    await localFetch("/v1/datasets");
    expect(fetchMock.mock.calls[0][0]).toBe("https://runtime.test/api/v1/datasets");
  });

  it("uses the browser host when no backend URL is configured", async () => {
    delete process.env.NEXT_PUBLIC_LOCAL_API_URL;
    fetchMock.mockResponseOnce("{}");
    await localFetch("/health");
    expect(fetchMock.mock.calls[0][0]).toBe("https://ui.example.test:8000/health");
  });
});
