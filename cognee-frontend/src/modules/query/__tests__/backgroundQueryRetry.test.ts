import { backgroundQueryRetry, backgroundQueryRetryDelay, BACKGROUND_QUERY_RETRY_COUNT } from "../backgroundQueryRetry";

describe("backgroundQueryRetry", () => {
  it("gives up immediately on a 404, which a pod without the route answers forever", () => {
    // The case this exists for: GET /v1/activity/agents on a pod image that
    // has not allowlisted it. Four requests and ~7s of backoff per page load
    // bought nothing.
    expect(backgroundQueryRetry(0, { status: 404 })).toBe(false);
  });

  it("gives up on the other client errors too", () => {
    for (const status of [400, 401, 403, 422]) {
      expect(backgroundQueryRetry(0, { status })).toBe(false);
    }
  });

  it("keeps retrying a 429, which is a not-now rather than a no", () => {
    expect(backgroundQueryRetry(0, { status: 429 })).toBe(true);
  });

  it("keeps the full budget for a server error", () => {
    expect(backgroundQueryRetry(0, { status: 500 })).toBe(true);
    expect(backgroundQueryRetry(0, { status: 503 })).toBe(true);
  });

  it("keeps the full budget for a network error with no status at all", () => {
    expect(backgroundQueryRetry(0, new Error("network"))).toBe(true);
    expect(backgroundQueryRetry(0, null)).toBe(true);
    expect(backgroundQueryRetry(0, undefined)).toBe(true);
  });

  it("does not read a non-numeric status as a client error", () => {
    // A thrown object carrying status: "404" is not a status we understand,
    // so it gets the retryable default rather than a silent give-up.
    expect(backgroundQueryRetry(0, { status: "404" })).toBe(true);
  });

  it("stops at the budget once the failures are retryable", () => {
    expect(backgroundQueryRetry(BACKGROUND_QUERY_RETRY_COUNT - 1, { status: 500 })).toBe(true);
    expect(backgroundQueryRetry(BACKGROUND_QUERY_RETRY_COUNT, { status: 500 })).toBe(false);
  });
});

describe("backgroundQueryRetryDelay", () => {
  it("backs off exponentially", () => {
    expect(backgroundQueryRetryDelay(0)).toBe(1000);
    expect(backgroundQueryRetryDelay(1)).toBe(2000);
    expect(backgroundQueryRetryDelay(2)).toBe(4000);
  });

  it("caps the wait so a slow pod does not stall a page for a minute", () => {
    expect(backgroundQueryRetryDelay(10)).toBe(8000);
  });
});
