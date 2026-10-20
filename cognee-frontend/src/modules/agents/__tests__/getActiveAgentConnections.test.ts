import type { CogneeInstance } from "@/modules/instances/types";
import { getActiveAgentConnections } from "../getActiveAgentConnections";

function instanceReturning(body: unknown, status = 200): CogneeInstance {
  return {
    name: "test",
    instanceId: "test-instance",
    fetch: jest.fn().mockResolvedValue(
      new Response(JSON.stringify(body), {
        status,
        headers: { "Content-Type": "application/json" },
      }),
    ),
  };
}

describe("getActiveAgentConnections", () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it("requests active connections and returns their session identifiers", async () => {
    const instance = instanceReturning({
      agents: [
        { session_id: "codex_123", status: "active" },
        { session_id: null, status: "active" },
      ],
    });

    await expect(getActiveAgentConnections(instance)).resolves.toEqual([
      { session_id: "codex_123", status: "active" },
      { session_id: null, status: "active" },
    ]);
    expect(instance.fetch).toHaveBeenCalledWith(
      "/v1/agents/connections?active_only=true&include_sources=false&limit=500",
      { signal: undefined },
    );
  });

  it("fails closed for unsuccessful or malformed responses", async () => {
    const warning = jest.spyOn(console, "warn").mockImplementation(() => undefined);

    await expect(getActiveAgentConnections(instanceReturning({}, 503))).resolves.toEqual([]);
    await expect(
      getActiveAgentConnections(instanceReturning({ agents: "invalid" })),
    ).resolves.toEqual([]);
    expect(warning).toHaveBeenCalledWith("[getActiveAgentConnections] unexpected response shape");
  });
});
