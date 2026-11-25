import getGovernanceGraph from "../getGovernanceGraph";
import { SLOW_READ_TIMEOUT_MS } from "@/services/http";
import type { CogneeInstance } from "@/modules/instances/types";

function instanceSpy() {
  const fetch = jest.fn().mockResolvedValue({ json: async () => ({}) });
  return { instance: { fetch } as unknown as CogneeInstance, fetch };
}

afterEach(() => jest.restoreAllMocks());

describe("getGovernanceGraph", () => {
  it("asks for the provenance graph", async () => {
    const { instance, fetch } = instanceSpy();

    await getGovernanceGraph(instance);

    expect(fetch.mock.calls[0][0]).toBe("/v1/schema/provenance/json");
  });

  it("overrides the default GET budget, which a warm pod cannot meet", async () => {
    // Builds a whole-tenant provenance graph from the relational store, and
    // pod latency climbs with uptime (COG-5722 measured 5.9s to 10.9s over an
    // hour at zero concurrency). On the default budget this failed on a
    // healthy pod and took the governance rails down with it.
    const { instance, fetch } = instanceSpy();

    await getGovernanceGraph(instance);

    expect(fetch.mock.calls[0][1]).toEqual({ timeoutMs: SLOW_READ_TIMEOUT_MS });
    expect(SLOW_READ_TIMEOUT_MS).toBeGreaterThan(10_000);
  });
});
