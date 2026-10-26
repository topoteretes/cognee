import getAgentConnections, {
  connectionIntegrationKey,
  isAgentIntegrationKey,
  keyForChannel,
  isLaterTimestamp,
  lastSeenByAgent,
  newestFirst,
  runningAgents,
  sessionIntegrationKey,
} from "../getAgentConnections";
import type { AgentConnection } from "../getAgentConnections";
import type { SessionRow } from "@/modules/sessions/getSessions";
import type { CogneeInstance } from "@/modules/instances/types";

function connection(overrides: Partial<AgentConnection> = {}): AgentConnection {
  return {
    id: "conn-1",
    agent_session_name: "goran-laptop",
    type: "claude_code",
    session_id: "claude_abc123",
    last_active_at: "2026-09-29T00:00:00Z",
    status: "active",
    user_id: "u1",
    memory_mode: "hybrid",
    datasets: [],
    ...overrides,
  };
}

function session(overrides: Partial<SessionRow> = {}): SessionRow {
  return {
    session_id: "claude_abc123",
    user_id: "u1",
    dataset_id: null,
    status: "running",
    effective_status: "running",
    started_at: "2026-09-29T00:00:00Z",
    last_activity_at: "2026-09-29T00:00:00Z",
    ended_at: null,
    tokens_in: null,
    tokens_out: null,
    cost_usd: null,
    error_count: 0,
    last_model: null,
    ...overrides,
  };
}

function instance(fetchImpl: jest.Mock): CogneeInstance {
  return { name: "CloudCognee", instanceId: "tenant-1", fetch: fetchImpl };
}

function jsonResponse(payload: unknown): { json: () => Promise<unknown> } {
  return { json: async () => payload };
}

describe("sessionIntegrationKey", () => {
  it("resolves a claude-code session prefix", () => {
    expect(sessionIntegrationKey("claude_abc")).toBe("claude-code");
    expect(sessionIntegrationKey("cc_abc")).toBe("claude-code");
  });

  it("resolves a codex session prefix", () => {
    expect(sessionIntegrationKey("codex_abc")).toBe("codex");
  });

  it("does not mistake a Claude Desktop session for Claude Code", () => {
    expect(sessionIntegrationKey("claude_desktop_abc")).toBeNull();
  });

  it("returns null for an empty session_id", () => {
    expect(sessionIntegrationKey("")).toBeNull();
  });
});

describe("keyForChannel", () => {
  it("resolves a plugin agent's own channel name", () => {
    expect(keyForChannel("Claude Code")).toBe("claude-code");
    expect(keyForChannel("Codex")).toBe("codex");
  });

  it("returns null for a channel no AGENT_INTEGRATION_DEFS row names", () => {
    // The memory graph's Cursor/VS Code/OpenClaw nodes all resolve to this —
    // clicking one falls back to navigating to Integrations rather than
    // opening a modal meant only for the two plugin agents.
    expect(keyForChannel("Cursor")).toBeNull();
    expect(keyForChannel("Company Dataset")).toBeNull();
  });
});

describe("connectionIntegrationKey", () => {
  it("resolves a modern claude_code row directly off type", () => {
    expect(connectionIntegrationKey(connection({ type: "claude_code", session_id: "" }))).toBe("claude-code");
  });

  it("resolves a codex row directly off type", () => {
    expect(connectionIntegrationKey(connection({ type: "codex", session_id: "" }))).toBe("codex");
  });

  it("falls back to the session_id prefix for the plugin 1.3.1 shape (type: api)", () => {
    expect(connectionIntegrationKey(connection({ type: "api", session_id: "claude_e7c5b154" }))).toBe("claude-code");
    expect(connectionIntegrationKey(connection({ type: "api", session_id: "codex_e7c5b154" }))).toBe("codex");
  });

  it("does not mistake a Claude Desktop session for Claude Code", () => {
    expect(connectionIntegrationKey(connection({ type: "api", session_id: "claude_desktop_abc" }))).toBeNull();
  });

  it("ignores an unrecognized type with no matching session_id prefix", () => {
    expect(connectionIntegrationKey(connection({ type: "cursor", session_id: "cursor_abc" }))).toBeNull();
  });

  it("ignores a row with no session_id to fall back on", () => {
    expect(connectionIntegrationKey(connection({ type: "unknown-client", session_id: "" }))).toBeNull();
  });
});

describe("getAgentConnections", () => {
  it("throws on a 200 with no agents array, rather than reading it as zero connections", async () => {
    const fetchImpl = jest.fn().mockResolvedValue(jsonResponse({ error: "upstream failure" }));

    await expect(getAgentConnections(instance(fetchImpl))).rejects.toThrow(
      "agents/connections returned a payload with no agents array",
    );
  });

  it("throws when agents is present but not an array", async () => {
    const fetchImpl = jest.fn().mockResolvedValue(jsonResponse({ agents: "nope" }));

    await expect(getAgentConnections(instance(fetchImpl))).rejects.toThrow();
  });

  it("returns the envelope with normalized rows on a well-formed 200", async () => {
    const fetchImpl = jest.fn().mockResolvedValue(
      jsonResponse({
        agents: [connection()],
        memory_sources: [],
        total: 1,
        limit: 50,
        offset: 0,
        has_more: false,
      }),
    );

    const page = await getAgentConnections(instance(fetchImpl));

    expect(page.agents).toEqual([connection()]);
    expect(page.total).toBe(1);
  });

  it("drops a malformed row instead of failing the whole page", async () => {
    const fetchImpl = jest.fn().mockResolvedValue(
      jsonResponse({ agents: [null, "not an object", connection({ id: "conn-2" })] }),
    );

    const page = await getAgentConnections(instance(fetchImpl));

    expect(page.agents).toHaveLength(1);
    expect(page.agents[0].id).toBe("conn-2");
  });

  it("defaults an unrecognized status to unknown rather than passing it through raw", async () => {
    const fetchImpl = jest.fn().mockResolvedValue(
      jsonResponse({ agents: [{ ...connection(), status: "flerbing" }] }),
    );

    const page = await getAgentConnections(instance(fetchImpl));

    expect(page.agents[0].status).toBe("unknown");
  });

  it("normalizes a datasets entry that is a bare string, role unknown", async () => {
    const fetchImpl = jest.fn().mockResolvedValue(
      jsonResponse({ agents: [{ ...connection(), datasets: ["handbook", "contracts"] }] }),
    );

    const page = await getAgentConnections(instance(fetchImpl));

    expect(page.agents[0].datasets).toEqual([
      { name: "handbook", role: null },
      { name: "contracts", role: null },
    ]);
  });

  it("normalizes a datasets entry shaped as an object with a name and role field", async () => {
    const fetchImpl = jest.fn().mockResolvedValue(
      jsonResponse({ agents: [{ ...connection(), datasets: [{ name: "handbook", id: "d1", role: "read_write" }] }] }),
    );

    const page = await getAgentConnections(instance(fetchImpl));

    expect(page.agents[0].datasets).toEqual([{ name: "handbook", role: "read_write" }]);
  });

  it("defaults a named dataset's role to null when the row sends none", async () => {
    const fetchImpl = jest.fn().mockResolvedValue(
      jsonResponse({ agents: [{ ...connection(), datasets: [{ name: "handbook", id: "d1" }] }] }),
    );

    const page = await getAgentConnections(instance(fetchImpl));

    expect(page.agents[0].datasets).toEqual([{ name: "handbook", role: null }]);
  });

  it("drops a datasets entry that is neither a string nor a named object, rather than guessing", async () => {
    const fetchImpl = jest.fn().mockResolvedValue(
      jsonResponse({ agents: [{ ...connection(), datasets: ["handbook", 42, null, { id: "d1" }] }] }),
    );

    const page = await getAgentConnections(instance(fetchImpl));

    expect(page.agents[0].datasets).toEqual([{ name: "handbook", role: null }]);
  });

  it("keeps memory_mode and user_id, defaulting to an empty string when absent", async () => {
    const withoutFields: Partial<AgentConnection> = { ...connection() };
    delete withoutFields.memory_mode;
    delete withoutFields.user_id;
    const fetchImpl = jest.fn().mockResolvedValue(jsonResponse({ agents: [{ ...connection(), memory_mode: "hybrid", user_id: "user-1" }, withoutFields] }));

    const page = await getAgentConnections(instance(fetchImpl));

    expect(page.agents[0]).toMatchObject({ memory_mode: "hybrid", user_id: "user-1" });
    expect(page.agents[1]).toMatchObject({ memory_mode: "", user_id: "" });
  });

  it("defaults datasets to an empty list when the field is absent", async () => {
    const withoutDatasets: Partial<AgentConnection> = { ...connection() };
    delete withoutDatasets.datasets;
    const fetchImpl = jest.fn().mockResolvedValue(jsonResponse({ agents: [withoutDatasets] }));

    const page = await getAgentConnections(instance(fetchImpl));

    expect(page.agents[0].datasets).toEqual([]);
  });

  it("sends active_only=true and include_sources=false by default", async () => {
    const fetchImpl = jest.fn().mockResolvedValue(jsonResponse({ agents: [] }));

    await getAgentConnections(instance(fetchImpl));

    const [path] = fetchImpl.mock.calls[0] as [string];
    expect(path).toContain("active_only=true");
    expect(path).toContain("include_sources=false");
  });

  it("lets a caller ask for inactive connections too", async () => {
    const fetchImpl = jest.fn().mockResolvedValue(jsonResponse({ agents: [] }));

    await getAgentConnections(instance(fetchImpl), { range: "24h", activeOnly: false, includeSources: false });

    const [path] = fetchImpl.mock.calls[0] as [string];
    expect(path).toContain("range=24h");
    expect(path).toContain("active_only=false");
  });

  it("sends no limit by default, so a freshness-only caller keeps the endpoint's own default page", async () => {
    const fetchImpl = jest.fn().mockResolvedValue(jsonResponse({ agents: [] }));

    await getAgentConnections(instance(fetchImpl));

    const [path] = fetchImpl.mock.calls[0] as [string];
    expect(path).not.toContain("limit=");
  });

  it("sends the requested limit when a caller asks for the widest page", async () => {
    const fetchImpl = jest.fn().mockResolvedValue(jsonResponse({ agents: [] }));

    await getAgentConnections(instance(fetchImpl), { limit: 500 });

    const [path] = fetchImpl.mock.calls[0] as [string];
    expect(path).toContain("limit=500");
  });

  it("drops a row with no id, rather than persisting it as connected forever", async () => {
    // A row shaped like { type: "claude_code" } with no id is enough, on its
    // own, to make useConnectedIntegrations and useAgentConnectionState treat
    // Claude Code as connected forever: both key a connection off `id`, and a
    // shared "" would match every baseline and every diff.
    const fetchImpl = jest.fn().mockResolvedValue(
      jsonResponse({ agents: [{ type: "claude_code" }, connection({ id: "conn-2" })] }),
    );

    const page = await getAgentConnections(instance(fetchImpl));

    expect(page.agents).toHaveLength(1);
    expect(page.agents[0].id).toBe("conn-2");
  });

  it("drops a row whose id is not a string", async () => {
    const fetchImpl = jest.fn().mockResolvedValue(
      jsonResponse({ agents: [{ ...connection(), id: 12345 }] }),
    );

    const page = await getAgentConnections(instance(fetchImpl));

    expect(page.agents).toHaveLength(0);
  });
});

describe("isAgentIntegrationKey", () => {
  it("accepts a key AGENT_INTEGRATION_DEFS lists", () => {
    expect(isAgentIntegrationKey("claude-code")).toBe(true);
    expect(isAgentIntegrationKey("codex")).toBe(true);
  });

  it("rejects an MCP connector key", () => {
    expect(isAgentIntegrationKey("cursor")).toBe(false);
    expect(isAgentIntegrationKey("openclaw")).toBe(false);
  });
});

describe("lastSeenByAgent", () => {
  it("prefers the session's last_activity_at when it is newer than the registration", () => {
    // The whole point of this fix: a session open since yesterday must not
    // read as "last seen at launch" just because that is all the connection
    // row's last_active_at (set once, at registration) can say.
    const result = lastSeenByAgent(
      [connection({ last_active_at: "2026-09-29T18:17:09Z" })],
      [session({ last_activity_at: "2026-09-29T18:26:44Z" })],
    );

    expect(result["claude-code"]).toBe("2026-09-29T18:26:44Z");
  });

  it("falls back to the registration's last_active_at when there is no session yet", () => {
    const result = lastSeenByAgent([connection({ last_active_at: "2026-09-29T18:17:09Z" })], []);

    expect(result["claude-code"]).toBe("2026-09-29T18:17:09Z");
  });

  it("keeps the registration's newer last_active_at over a stale session", () => {
    // The reverse of the "session is newer" case: a session considered after
    // the connection must not blindly win just because it was checked last.
    const result = lastSeenByAgent(
      [connection({ last_active_at: "2026-09-29T19:00:00Z" })],
      [session({ last_activity_at: "2026-09-29T10:00:00Z" })],
    );

    expect(result["claude-code"]).toBe("2026-09-29T19:00:00Z");
  });

  it("reads from a session alone when there is no connection row for it", () => {
    const result = lastSeenByAgent([], [session({ last_activity_at: "2026-09-29T18:26:44Z" })]);

    expect(result["claude-code"]).toBe("2026-09-29T18:26:44Z");
  });

  it("keeps each agent's newest timestamp independent of the others", () => {
    const result = lastSeenByAgent(
      [
        connection({ type: "claude_code", last_active_at: "2026-09-29T00:00:00Z" }),
        connection({ type: "codex", session_id: "", last_active_at: "2026-09-29T02:00:00Z" }),
      ],
      [],
    );

    expect(result["claude-code"]).toBe("2026-09-29T00:00:00Z");
    expect(result["codex"]).toBe("2026-09-29T02:00:00Z");
  });

  it("ignores a connection or session that matches no known integration", () => {
    const result = lastSeenByAgent(
      [connection({ type: "cursor", session_id: "cursor_1" })],
      [session({ session_id: "cursor_1" })],
    );

    expect(result).toEqual({});
  });

  it("ignores a session with no last_activity_at rather than treating it as newest", () => {
    const result = lastSeenByAgent(
      [connection({ last_active_at: "2026-09-29T18:17:09Z" })],
      [session({ last_activity_at: null })],
    );

    expect(result["claude-code"]).toBe("2026-09-29T18:17:09Z");
  });

  it("returns an empty map for two empty lists", () => {
    expect(lastSeenByAgent([], [])).toEqual({});
  });

  it("skips an unparseable last_active_at rather than rendering NaNd ago", () => {
    const result = lastSeenByAgent([connection({ last_active_at: "not-a-date" })], []);

    expect(result["claude-code"]).toBeUndefined();
  });

  it("skips an unparseable last_activity_at even as the only signal for that agent", () => {
    const result = lastSeenByAgent([], [session({ last_activity_at: "not-a-date" })]);

    expect(result["claude-code"]).toBeUndefined();
  });

  it("prefers a parseable registration timestamp over an unparseable session one", () => {
    const result = lastSeenByAgent(
      [connection({ last_active_at: "2026-09-29T18:17:09Z" })],
      [session({ last_activity_at: "not-a-date" })],
    );

    expect(result["claude-code"]).toBe("2026-09-29T18:17:09Z");
  });
});

describe("isLaterTimestamp", () => {
  it("compares instants, so a dropped zero fraction does not reorder them", () => {
    expect(isLaterTimestamp("2026-09-30T06:46:48.1Z", "2026-09-30T06:46:48Z")).toBe(true);
    expect(isLaterTimestamp("2026-09-30T06:46:48Z", "2026-09-30T06:46:48.1Z")).toBe(false);
    expect(isLaterTimestamp("2026-09-30T06:46:48+00:00", "2026-09-30T06:46:47.999999Z")).toBe(true);
  });

  it("never lets an unparseable value win over a parseable one", () => {
    expect(isLaterTimestamp("garbage", "2026-09-30T06:46:48Z")).toBe(false);
    expect(isLaterTimestamp("2026-09-30T06:46:48Z", "garbage")).toBe(true);
  });
});

describe("runningAgents", () => {
  it("counts only running sessions, per agent, by the server's effective status", () => {
    const running = runningAgents([
      { session_id: "claude_live", status: "running", effective_status: "running" },
      { session_id: "codex_dead", status: "running", effective_status: "abandoned" },
      { session_id: "openclaw_live", status: "running", effective_status: "running" },
    ]);

    expect([...running]).toEqual(["claude-code"]);
  });
});

describe("newestFirst", () => {
  it("sorts newest first as instants, with missing values last", () => {
    const sorted = ["2026-09-30T06:46:48Z", null, "2026-09-30T06:46:48.1Z", "2026-09-30T06:46:47Z"].sort(newestFirst);

    expect(sorted).toEqual(["2026-09-30T06:46:48.1Z", "2026-09-30T06:46:48Z", "2026-09-30T06:46:47Z", null]);
  });
});
