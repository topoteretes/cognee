import { CogneeInstance } from "../instances/types";
import { channelForSessionId, isSessionRunning } from "../sessions/getSessions";
import type { SessionRow } from "../sessions/getSessions";

/**
 * One row of `GET /v1/agents/connections` — a plugin registration, written by
 * the SessionStart hook's `POST /agents/register` at launch and updated by
 * `POST /agents/unregister` when the session ends cleanly.
 *
 * Trimmed to the fields callers actually read. The wire row also carries
 * tenant_id/source/origin_function/metadata — add one back here the day
 * something reads it, rather than modelling the whole envelope up front.
 *
 * `last_active_at` is set at registration only and never refreshed, so it
 * cannot be read as "still talking to us" — that is what `status` is for. A
 * session that crashes instead of unregistering can stay "active" forever.
 */
export interface AgentConnection {
  id: string;
  agent_session_name: string;
  /** Wire values observed: "claude_code", "codex", "api", ... — see
   *  connectionIntegrationKey for how an "api"-typed row still resolves. */
  type: string;
  session_id: string;
  last_active_at: string;
  status: "active" | "inactive" | "unknown";
  /** The user_id this connection row was registered under — the id the
   *  detail endpoint (`GET /v1/agents/connections/{agent_id}`) takes as its
   *  path parameter. Empty when the row doesn't carry one, which just means
   *  no detail fetch is possible for it (see latestConnectionUserId). */
  user_id: string;
  /** "session" | "cognee" | "hybrid" | "none" | "unknown" on the wire, plus
   *  whatever else a future pod build adds — kept as a bare string rather
   *  than a union so an unrecognized value degrades to "say nothing" at the
   *  display layer (memoryModePhrase) instead of failing to normalize. */
  memory_mode: string;
  /** Datasets this connection writes to, when the pod sends them. Wire item
   *  shape isn't pinned down by any spec this frontend has seen, only that
   *  it's "an array of datasets" — normalizeAgentDatasets accepts a bare
   *  string entry (role unknown) or an object entry with a string `name`
   *  and optional string `role`, and drops anything else, rather than
   *  guessing at a shape. */
  datasets: AgentDataset[];
}

export interface AgentDataset {
  name: string;
  /** Wire values observed: "read_write" — kept as a bare string, same
   *  reasoning as `memory_mode`, and null when the row didn't send one. */
  role: string | null;
}

export interface AgentConnectionsPage {
  agents: AgentConnection[];
  memory_sources: unknown;
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
}

export type AgentConnectionRange = "24h" | "7d" | "30d" | "all";

/**
 * The two plugin integrations this frontend recognizes off a connection row,
 * in one place. `channel` must match a name in getSessions.ts's CHANNEL_DEFS
 * exactly — connectionIntegrationKey falls back to it for a row shaped like
 * plugin 1.3.1's (`type: "api"`, real client on session_id instead). Adding a
 * third agent is one more row here; nothing else in this module or
 * useConnectedIntegrations.ts needs a matching branch.
 */
export interface AgentIntegrationDef {
  /** Integration card key, as already persisted in users' browsers. */
  key: "claude-code" | "codex";
  /** Modern `type` value the plugin sends at registration. */
  wireType: string;
  channel: string;
}

export const AGENT_INTEGRATION_DEFS: readonly AgentIntegrationDef[] = [
  { key: "claude-code", wireType: "claude_code", channel: "Claude Code" },
  { key: "codex", wireType: "codex", channel: "Codex" },
];

export type AgentIntegrationKey = AgentIntegrationDef["key"];

/** `channel` doubles as the human-facing display name — both are already
 *  "Claude Code"/"Codex". Add a separate `displayName` field the day one of
 *  them needs to diverge from the other. */
export function agentDisplayName(key: AgentIntegrationKey): string {
  return AGENT_INTEGRATION_BY_KEY.get(key)?.channel ?? key;
}

/** `AGENT_INTEGRATION_DEFS` keyed for lookup — the one place that reverse
 *  mapping exists, so useConnectedIntegrations.ts and agentNodes.ts read the
 *  same table instead of each building their own copy of it. */
export const AGENT_INTEGRATION_BY_KEY: ReadonlyMap<AgentIntegrationKey, AgentIntegrationDef> = new Map(
  AGENT_INTEGRATION_DEFS.map((def) => [def.key, def]),
);

/** The def whose `channel` matches, if any. The one place that reverse
 *  lookup happens, shared by connectionIntegrationKey and sessionIntegrationKey
 *  so a session and a connection resolve a session_id to the same agent —
 *  also exported for the memory graph, which only has a node's channel name
 *  (e.g. clicking a node to decide which modal it opens). */
export function keyForChannel(channel: string): AgentIntegrationKey | null {
  return AGENT_INTEGRATION_DEFS.find((def) => def.channel === channel)?.key ?? null;
}

/**
 * A connection row's integration key, or null if it matches neither def in
 * AGENT_INTEGRATION_DEFS.
 *
 * `type` is checked first because it is the modern, direct signal. A row that
 * fails it falls back to the session_id prefix table channelForSessionId
 * already owns, rather than keeping a second prefix table here that could
 * disagree with it.
 */
export function connectionIntegrationKey(
  connection: Pick<AgentConnection, "type" | "session_id">,
): AgentIntegrationKey | null {
  const byType = AGENT_INTEGRATION_DEFS.find((def) => def.wireType === connection.type);
  if (byType) return byType.key;
  if (!connection.session_id) return null;
  return keyForChannel(channelForSessionId(connection.session_id));
}

/** A session's integration key off its session_id prefix, or null. The
 *  session-side twin of connectionIntegrationKey's fallback path: sessions
 *  carry no `type`, only a session_id, so this is the whole of the match. */
export function sessionIntegrationKey(sessionId: string): AgentIntegrationKey | null {
  if (!sessionId) return null;
  return keyForChannel(channelForSessionId(sessionId));
}

/** Whether a plain string names one of AGENT_INTEGRATION_DEFS's rows, the
 *  one place a card component checks "is this a plugin agent, not an MCP
 *  connector" before it can offer registration-derived copy ("Set up",
 *  "Last seen ...") instead of a bare "Connected". */
export function isAgentIntegrationKey(key: string): key is AgentIntegrationKey {
  return AGENT_INTEGRATION_DEFS.some((def) => def.key === key);
}

/**
 * Whether ISO timestamp `a` is later than `b`, compared as instants, not as
 * strings. The pod serializes with Python's isoformat, which drops the
 * fraction when microseconds are zero, so one server emits both
 * "...:48Z" and "...:48.1Z"; as strings "." sorts before "Z" and those two
 * order backwards. An unparseable value never wins over a parseable one.
 */
export function isLaterTimestamp(a: string, b: string): boolean {
  const ta = Date.parse(a);
  const tb = Date.parse(b);
  if (Number.isNaN(ta)) return false;
  if (Number.isNaN(tb)) return true;
  return ta > tb;
}

/** Sort comparator, newest timestamp first, by the same instant comparison
 *  as isLaterTimestamp. A missing or unparseable value sorts last. */
export function newestFirst(a: string | null, b: string | null): number {
  if (isLaterTimestamp(a ?? "", b ?? "")) return -1;
  if (isLaterTimestamp(b ?? "", a ?? "")) return 1;
  return 0;
}

/**
 * Newest known activity per integration key, from lists the caller already
 * has, no fetch of its own. The one function every "Last seen" line reads,
 * so the Integrations cards, the dashboard card, its tooltip, and the
 * tutorial's live line cannot drift into disagreeing answers.
 *
 * Combines two signals because neither alone is "last seen": a connection's
 * `last_active_at` is set once at registration and never refreshed (see
 * AgentConnection), so a session that has been open for hours reads as
 * "last seen at launch" if that is all this looked at. A session's own
 * `last_activity_at` is the pod's actual per-request activity timestamp for
 * that session, matched the same way `connectionIntegrationKey` matches a
 * connection with no `type` to go on: off its session_id prefix.
 */
export function lastSeenByAgent(
  connections: readonly Pick<AgentConnection, "type" | "session_id" | "last_active_at">[],
  sessions: readonly Pick<SessionRow, "session_id" | "last_activity_at">[],
): Partial<Record<AgentIntegrationKey, string>> {
  const result: Partial<Record<AgentIntegrationKey, string>> = {};
  const consider = (key: AgentIntegrationKey | null, at: string | null | undefined): void => {
    // An unparseable value must never become the first (and therefore only)
    // entry for a key: `!current` alone used to let it through unconditionally
    // on the very first call, before isLaterTimestamp's own NaN guard ever ran,
    // and the badge went on to render "NaNd ago" for that agent forever.
    if (!key || !at || Number.isNaN(Date.parse(at))) return;
    const current = result[key];
    if (!current || isLaterTimestamp(at, current)) result[key] = at;
  };
  for (const connection of connections) consider(connectionIntegrationKey(connection), connection.last_active_at);
  for (const session of sessions) consider(sessionIntegrationKey(session.session_id), session.last_activity_at);
  return result;
}

/**
 * Integration keys with at least one running session, by the server's own
 * effective_status (isSessionRunning), the same rule agentStatus uses for
 * the memory graph's "live". The one place every card, tutorial line and
 * tooltip asks "is it running", so none of them can say "Connected" while
 * another says "Live" for the same agent.
 *
 * Sessions only, never connection rows: a connection's `status: "active"`
 * is set at registration and never cleared when a session crashes or its
 * terminal window closes, so it cannot tell a live agent from a dead one.
 */
export function runningAgents(
  sessions: readonly Pick<SessionRow, "session_id" | "status" | "effective_status">[],
): ReadonlySet<AgentIntegrationKey> {
  const running = new Set<AgentIntegrationKey>();
  for (const session of sessions) {
    const key = sessionIntegrationKey(session.session_id);
    if (key && isSessionRunning(session)) running.add(key);
  }
  return running;
}

/**
 * This agent's most recently registered connection row, or null if it has
 * none. The one place the connected modal decides which row's `user_id` to
 * ask the detail endpoint about, its `memory_mode` to describe, and its
 * datasets to list roles for — so those three can't each pick a different
 * connection and disagree.
 */
export function latestConnectionForAgent(
  agent: AgentIntegrationKey,
  connections: readonly AgentConnection[],
): AgentConnection | null {
  return connections
    .filter((c) => connectionIntegrationKey(c) === agent)
    .reduce<AgentConnection | null>((latest, c) => {
      if (!latest || isLaterTimestamp(c.last_active_at, latest.last_active_at)) return c;
      return latest;
    }, null);
}

/**
 * The earliest `last_active_at` among this agent's connection rows — "first
 * connected", not "last seen". Only meaningful when the caller already knows
 * the connections page it read from wasn't truncated (see
 * `connectionsTruncated` on useDashboardTelemetry): a registration older than
 * the page's cutoff would make this wrong in the direction that matters,
 * since it would understate how long ago the agent first connected.
 */
export function earliestConnectionAt(
  agent: AgentIntegrationKey,
  connections: readonly AgentConnection[],
): string | null {
  return connections
    .filter((c) => connectionIntegrationKey(c) === agent && c.last_active_at)
    .reduce<string | null>((earliest, c) => {
      if (!earliest || isLaterTimestamp(earliest, c.last_active_at)) return c.last_active_at;
      return earliest;
    }, null);
}

const KNOWN_STATUSES: readonly AgentConnection["status"][] = ["active", "inactive", "unknown"];

function str(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function normalizeAgentDatasets(value: unknown): AgentDataset[] {
  if (!Array.isArray(value)) return [];
  const datasets: AgentDataset[] = [];
  for (const item of value) {
    if (typeof item === "string") {
      datasets.push({ name: item, role: null });
    } else if (item !== null && typeof item === "object" && typeof (item as { name?: unknown }).name === "string") {
      const role = (item as { role?: unknown }).role;
      datasets.push({ name: (item as { name: string }).name, role: typeof role === "string" ? role : null });
    }
  }
  return datasets;
}

/**
 * Connection rows, field-normalized — same reasoning as normalizeSessionRows:
 * `isConnectionsEnvelope` only proves `agents` is an array, so every field on
 * a row arrives unvalidated. A row that isn't an object at all is dropped
 * rather than costing the whole page.
 */
function normalizeAgentConnections(rows: unknown[]): AgentConnection[] {
  const out: AgentConnection[] = [];
  for (const raw of rows) {
    if (raw === null || typeof raw !== "object") continue;
    const r = raw as Partial<AgentConnection>;
    // A row with no id (or an id that isn't a string) is dropped rather than
    // kept with id: "" — the sticky per-tenant registry (useConnectedIntegrations)
    // and useAgentConnectionState both key a connection off `id`, and a shared
    // "" would let one such row match every baseline and every diff forever,
    // which is enough on its own to persist an agent as connected for good.
    const id = str(r.id);
    if (!id) continue;
    const status = KNOWN_STATUSES.includes(r.status as AgentConnection["status"])
      ? (r.status as AgentConnection["status"])
      : "unknown";
    out.push({
      id,
      agent_session_name: str(r.agent_session_name),
      type: str(r.type),
      session_id: str(r.session_id),
      last_active_at: str(r.last_active_at),
      status,
      user_id: str(r.user_id),
      memory_mode: str(r.memory_mode),
      datasets: normalizeAgentDatasets(r.datasets),
    });
  }
  return out;
}

function isConnectionsEnvelope(
  value: unknown,
): value is { agents: unknown[] } & Omit<AgentConnectionsPage, "agents"> {
  return typeof value === "object" && value !== null && Array.isArray((value as { agents?: unknown }).agents);
}

/**
 * Registered agent connections for the current tenant.
 *
 * Throws on a payload with no `agents` array rather than coercing it to []:
 * a 200 carrying an error envelope would otherwise be indistinguishable from
 * a tenant with nothing registered — the same failure getAgents.ts closes
 * for the roster endpoint.
 */
export default function getAgentConnections(
  instance: CogneeInstance,
  params: { range?: AgentConnectionRange; activeOnly?: boolean; includeSources?: boolean; limit?: number } = {},
  opts: { signal?: AbortSignal; timeoutMs?: number } = {},
): Promise<AgentConnectionsPage> {
  const q = new URLSearchParams();
  if (params.range) q.set("range", params.range);
  q.set("active_only", String(params.activeOnly ?? true));
  q.set("include_sources", String(params.includeSources ?? false));
  // No default here: the endpoint's own default (50) is fine for a caller that
  // only wants a freshness signal, and the widest page (500, the documented
  // maximum, `le=500`) is opt-in for a caller asking "has this ever
  // registered" — see useDashboardTelemetry, the one place that question is
  // asked against a truncatable page.
  if (params.limit !== undefined) q.set("limit", String(params.limit));
  const init: RequestInit & { timeoutMs?: number } = { signal: opts.signal, timeoutMs: opts.timeoutMs };
  return instance
    .fetch(`/v1/agents/connections?${q.toString()}`, init)
    .then((response) => response.json())
    .then((payload: unknown) => {
      if (!isConnectionsEnvelope(payload)) {
        throw new Error("agents/connections returned a payload with no agents array");
      }
      return { ...payload, agents: normalizeAgentConnections(payload.agents) };
    });
}
