import getBrainsSummary from "../getBrainsSummary";
import { SLOW_READ_TIMEOUT_MS } from "@/services/http";
import type { CogneeInstance } from "@/modules/instances/types";

function instanceSpy() {
  const fetch = jest.fn().mockResolvedValue({ json: async () => ({}) });
  return { instance: { fetch } as unknown as CogneeInstance, fetch };
}

afterEach(() => jest.restoreAllMocks());

describe("getBrainsSummary", () => {
  it("asks for every readable dataset without building a graph", async () => {
    const { instance, fetch } = instanceSpy();

    await getBrainsSummary(instance);

    expect(fetch.mock.calls[0][0]).toBe("/v1/visualize/brains-summary");
  });

  it("overrides the default GET budget, which a loaded pod cannot meet", async () => {
    // Served from relational metadata and the count cache, so it is cheap —
    // but it still contends with an in-flight cognify run, and was measured
    // being reset by the client at 9995 and 9999 ms while the same call took
    // 421 ms against an idle pod. On the default budget the dataset list
    // silently disappears at exactly the moment the graph is growing.
    const { instance, fetch } = instanceSpy();

    await getBrainsSummary(instance);

    expect(fetch.mock.calls[0][1]).toEqual({ timeoutMs: SLOW_READ_TIMEOUT_MS });
    expect(SLOW_READ_TIMEOUT_MS).toBeGreaterThan(10_000);
  });
});
