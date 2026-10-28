import {
  ACTIVE_NOW_MS,
  DYNAMIC_AGENT_DEFS,
  PERSISTENT_AGENT_DEFS,
  agentStatus,
  agentStatusText,
  agentStatusTooltip,
  hasRegistered,
} from "@/app/(app)/dashboard/agentNodes";
import { AGENT_CHANNEL_NAMES } from "@/modules/sessions/getSessions";
import type { SessionRow } from "@/modules/sessions/getSessions";

function session(sessionId: string, status = "completed"): SessionRow {
  return {
    session_id: sessionId,
    user_id: "agent-1",
    dataset_id: null,
    status,
    effective_status: status,
    started_at: "2026-09-01T00:00:00Z",
    last_activity_at: "2026-09-01T00:01:00Z",
    ended_at: null,
    tokens_in: 10,
    tokens_out: 20,
    cost_usd: 0.1,
    error_count: 0,
    last_model: null,
  };
}

const NO_REGISTRY: Record<string, boolean> = {};

describe("agentStatus", () => {
  it("reads an agent with a session in flight as live", () => {
    expect(agentStatus([session("cc_abc", "running")], NO_REGISTRY, "Claude Code", false)).toBe("live");
  });

  it("reads a registered agent between sessions as connected, not disconnected", () => {
    // A session is minutes long. Requiring a running one to look like anything
    // other than "never heard of" made a working agent read as absent nearly
    // all the time, which is what the graph's third state is for.
    expect(agentStatus([session("cc_abc")], NO_REGISTRY, "Claude Code", false)).toBe("connected");
  });

  it("reads an agent nothing has ever seen as disconnected", () => {
    expect(agentStatus([session("cc_abc", "running")], NO_REGISTRY, "Cursor", false)).toBe("disconnected");
  });

  it("does not light up Claude Code for a Claude Desktop session", () => {
    // "claude_desktop_" also starts with "claude_", and Claude Code used to
    // carry "claude_" in its own prefix list, so both nodes matched one session.
    const sessions = [session("claude_desktop_xyz", "running")];

    expect(agentStatus(sessions, NO_REGISTRY, "Claude Desktop", false)).toBe("live");
    expect(agentStatus(sessions, NO_REGISTRY, "Claude Code", false)).toBe("disconnected");
  });

  it("still recognizes Claude Code's other shipped prefix", () => {
    expect(agentStatus([session("claude_abc", "running")], NO_REGISTRY, "Claude Code", false)).toBe("live");
  });

  it("falls back to the stored status when the pod sends no effective_status", () => {
    // `effective_status` is typed non-optional but nothing validates it at the
    // wire; comparing it directly left a live agent permanently not-running.
    const partial = { ...session("cc_abc", "running"), effective_status: "" };

    expect(agentStatus([partial], NO_REGISTRY, "Claude Code", false)).toBe("live");
  });

  it("treats the status case-insensitively", () => {
    const upper = { ...session("cc_abc"), effective_status: "RUNNING" };

    expect(agentStatus([upper], NO_REGISTRY, "Claude Code", false)).toBe("live");
  });

  it("prefers the pod's derived status over the stored one", () => {
    const ended = { ...session("cc_abc", "running"), effective_status: "abandoned" };

    expect(agentStatus([ended], NO_REGISTRY, "Claude Code", false)).toBe("connected");
  });

  it("reads a registered agent as connected, never live, off registration alone", () => {
    // The pod's connection status is only ever set "active" at registration
    // and cleared by the plugin's own unregister call, a crashed session
    // never gets there, so this can stay "active" long after the agent is
    // gone. "live" must come only from the running session's own
    // server-computed, timed-out status, or a crash reads as live forever.
    expect(agentStatus([], { "Claude Code": true }, "Claude Code", false)).toBe("connected");
  });

  it("reads an unconfirmed agent as unknown while telemetry is still loading", () => {
    // The false claim this exists to prevent: a fresh session with nothing in
    // the registry yet must not read as "Not connected yet" before the first
    // poll has actually said so.
    expect(agentStatus([], NO_REGISTRY, "Claude Code", true)).toBe("unknown");
  });

  it("still reads live off a running session even while loading", () => {
    expect(agentStatus([session("cc_abc", "running")], NO_REGISTRY, "Claude Code", true)).toBe("live");
  });

  it("still reads connected off the sticky registry even while loading", () => {
    // Loading only ever downgrades an unconfirmed disconnected reading — it
    // must never mask an answer the registry already has.
    expect(agentStatus([], { "Claude Code": true }, "Claude Code", true)).toBe("connected");
  });

  it("reads disconnected, not unknown, once loading has finished with nothing found", () => {
    expect(agentStatus([], NO_REGISTRY, "Claude Code", false)).toBe("disconnected");
  });
});

describe("hasRegistered", () => {
  it("remembers an agent whose sessions have aged out of the polled page", () => {
    // The feed is scoped to the range toggle and capped at 50 rows, and the pod
    // caps it again, so a busy workspace loses an agent's last session within
    // the hour. Without the sticky answer the graph called it never-connected.
    expect(hasRegistered([], { "Claude Code": true }, "Claude Code")).toBe(true);
  });

  it("registers a first-ever session before anything has been persisted", () => {
    expect(hasRegistered([session("codex_1")], NO_REGISTRY, "Codex")).toBe(true);
  });

  it("does not accept a non-boolean left in the persisted registry", () => {
    const corrupt = { "Claude Code": "yes" } as unknown as Record<string, boolean>;

    expect(hasRegistered([], corrupt, "Claude Code")).toBe(false);
  });

  it("says nothing about an agent absent from both the feed and the registry", () => {
    expect(hasRegistered([session("cc_abc")], { Codex: true }, "Cursor")).toBe(false);
  });
});

describe("agentStatusTooltip", () => {
  const NOW = new Date("2026-09-30T00:00:00.000Z");

  beforeEach(() => {
    jest.useFakeTimers().setSystemTime(NOW);
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it("reads 'Live, running' for a live node with no last-seen time, never claiming 'now' without one", () => {
    expect(agentStatusTooltip("live", null, NOW)).toBe("Live, running");
  });

  it("reads 'Connected, last seen X ago' for a connected node with a known last-seen time", () => {
    expect(agentStatusTooltip("connected", "2026-09-29T18:00:00.000Z", NOW)).toBe("Connected, last seen 6h ago");
  });

  it("reads bare 'Connected' for a connected node with no last-seen time (registry only)", () => {
    expect(agentStatusTooltip("connected", null, NOW)).toBe("Connected");
  });

  it("reads 'Not connected yet' for a disconnected node", () => {
    expect(agentStatusTooltip("disconnected", null, NOW)).toBe("Not connected yet");
  });

  it("reads 'Checking…' for an unknown node, not a confident Not connected", () => {
    expect(agentStatusTooltip("unknown", null, NOW)).toBe("Checking…");
  });

  it("shares its wording with agentStatusText rather than deriving its own copy", () => {
    const { badge, detail } = agentStatusText("connected", "2026-09-29T18:00:00.000Z", NOW);
    expect(agentStatusTooltip("connected", "2026-09-29T18:00:00.000Z", NOW)).toBe(`${badge}, ${detail}`);
  });
});

describe("agentStatusText live freshness", () => {
  const NOW = new Date("2026-09-30T00:00:00.000Z");

  // The stale branch's wording goes through timeAgo, which reads the real
  // clock rather than taking `now` as a parameter — fake timers keep it in
  // step with the fixture dates above instead of the test run's wall clock.
  beforeEach(() => {
    jest.useFakeTimers().setSystemTime(NOW);
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it("reads 'active now' when the last activity is within the threshold", () => {
    const lastSeenAt = new Date(NOW.getTime() - ACTIVE_NOW_MS).toISOString();

    expect(agentStatusText("live", lastSeenAt, NOW).detail).toBe("active now");
  });

  it("reads 'running · last activity X ago' the instant the last activity crosses the threshold", () => {
    const lastSeenAt = new Date(NOW.getTime() - ACTIVE_NOW_MS - 1).toISOString();

    expect(agentStatusText("live", lastSeenAt, NOW).detail).toBe("running · last activity 2m ago");
  });

  it("does not claim 'active now' off a session whose effective_status is still running but has been quiet for 24 minutes", () => {
    const lastSeenAt = new Date(NOW.getTime() - 24 * 60 * 1000).toISOString();

    expect(agentStatusText("live", lastSeenAt, NOW).detail).toBe("running · last activity 24m ago");
  });

  it("reads just 'running' with no last-seen time at all: 'now' needs a fresh timestamp behind it", () => {
    expect(agentStatusText("live", null, NOW).detail).toBe("running");
  });

  it("reads just 'running' off an unparseable last-seen value, claiming neither fresh nor stale", () => {
    expect(agentStatusText("live", "not-a-date", NOW).detail).toBe("running");
  });

  it("keeps the Live badge in both the fresh and stale case", () => {
    const stale = new Date(NOW.getTime() - ACTIVE_NOW_MS - 1).toISOString();

    expect(agentStatusText("live", stale, NOW).badge).toBe("Live");
  });
});

describe("the graph's agent roster", () => {
  it("names channels getSessions can actually resolve", () => {
    // The defs no longer carry prefixes: a node is tied to its session by the
    // channel name getSessions resolves, so a name that is not in that table
    // would leave the node permanently disconnected with nothing to show why.
    const named = [...PERSISTENT_AGENT_DEFS, ...DYNAMIC_AGENT_DEFS].map((d) => d.name);

    expect(named.filter((n) => !AGENT_CHANNEL_NAMES.includes(n))).toEqual([]);
  });

  it("covers every channel getSessions can name", () => {
    // The other direction: a channel with no def is memory arriving from a
    // client the graph never draws.
    const named = new Set([...PERSISTENT_AGENT_DEFS, ...DYNAMIC_AGENT_DEFS].map((d) => d.name));

    expect(AGENT_CHANNEL_NAMES.filter((n) => !named.has(n))).toEqual([]);
  });
});
