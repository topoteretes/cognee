import { connectionIntegrationKey, sessionIntegrationKey } from "@/modules/agents/getAgentConnections";
import type { AgentConnection, AgentDataset, AgentIntegrationKey } from "@/modules/agents/getAgentConnections";
import type { SessionRow } from "@/modules/sessions/getSessions";
import { formatClockTime } from "@/utils/formatDate";

/**
 * One name plus how many more, for a surface too small to list them all:
 * "agent_sessions" / "agent_sessions +2". The full list belongs in the same
 * element's tooltip or a detail view, never dropped.
 */
export function compactNames(names: readonly string[]): string {
  const [first, ...rest] = names;
  if (first === undefined) return "";
  return rest.length > 0 ? `${first} +${rest.length}` : first;
}

/** This agent's own sessions, out of the shared telemetry feed. */
export function matchingSessions(agent: AgentIntegrationKey, sessions: SessionRow[]): SessionRow[] {
  return sessions.filter((s) => sessionIntegrationKey(s.session_id) === agent);
}

/** Dataset names this agent writes to, deduplicated across its connections —
 *  a dataset repeated on two registrations is one fact, not two. */
export function writesToDatasets(agent: AgentIntegrationKey, connections: AgentConnection[]): string[] {
  return Array.from(
    new Set(
      connections
        .filter((c) => connectionIntegrationKey(c) === agent)
        .flatMap((c) => c.datasets.map((d) => d.name)),
    ),
  ).sort();
}

/** Same dataset set as `writesToDatasets`, keeping each one's role — for a
 *  detail view that needs to say what an agent does with a dataset, not
 *  only that it writes to it. Deduplicated by name; the first role seen for
 *  a name wins, since two connections agreeing on a dataset are describing
 *  the one same access, not two different ones. */
export function writesToDatasetsDetailed(agent: AgentIntegrationKey, connections: AgentConnection[]): AgentDataset[] {
  const byName = new Map<string, AgentDataset>();
  for (const c of connections) {
    if (connectionIntegrationKey(c) !== agent) continue;
    for (const dataset of c.datasets) {
      if (!byName.has(dataset.name)) byName.set(dataset.name, dataset);
    }
  }
  return Array.from(byName.values()).sort((a, b) => a.name.localeCompare(b.name));
}

const MEMORY_MODE_PHRASES: Record<string, string> = {
  hybrid: "session + knowledge graph",
  session: "session only",
  cognee: "knowledge graph",
};

/** Human wording for a connection's `memory_mode`, or null for "none"/
 *  "unknown"/anything this frontend doesn't recognize — omitted rather than
 *  guessed at, same reasoning as `memory_mode`'s own type. */
export function memoryModePhrase(memoryMode: string): string | null {
  return MEMORY_MODE_PHRASES[memoryMode] ?? null;
}

const DATASET_ROLE_PHRASES: Record<string, string> = {
  read_write: "reads and writes",
  read: "reads from",
  write: "writes to",
};

/** Human wording for a dataset's `role`, or null for anything this frontend
 *  doesn't recognize. */
export function datasetRolePhrase(role: string): string | null {
  return DATASET_ROLE_PHRASES[role] ?? null;
}

/**
 * "Memory: session + knowledge graph · reads and writes agent_sessions" — the
 * one line the connected modal shows for memory mode and dataset access,
 * replacing what used to be a separate "Writes to" sentence merged into it.
 *
 * The role half only names actual dataset(s) when every dataset shares the
 * same recognized role: roles that differ are left for the per-chip label
 * instead of one sentence trying to average them into something untrue, and
 * an unrecognized or absent role omits that half entirely rather than
 * guessing.
 */
export function memoryLineText(memoryMode: string, datasets: readonly AgentDataset[]): string | null {
  const modePhrase = memoryModePhrase(memoryMode);
  const roles = new Set(datasets.map((d) => d.role));
  const uniformRole = roles.size === 1 ? datasets[0]?.role : null;
  const roleWord = uniformRole ? datasetRolePhrase(uniformRole) : null;
  const rolePhrase = roleWord && datasets.length > 0
    ? `${roleWord} ${datasets.length === 1 ? datasets[0].name : `${datasets.length} datasets`}`
    : null;
  const parts = [modePhrase, rolePhrase].filter((p): p is string => p !== null);
  return parts.length > 0 ? `Memory: ${parts.join(" · ")}` : null;
}

/**
 * How long a session ran, from `started_at` to `ended_at` when the pod sent
 * one, else to its last known activity — a session still running has no
 * `ended_at`, so `last_activity_at` is the only honest end point for it.
 * Null when either end is missing or fails to parse: a NaN math result used
 * to render as the literal string "NaNm ago" (see isLaterTimestamp's own
 * history of the same bug) and a negative span (clock skew, bad data) is
 * just as untrustworthy.
 */
export function sessionDurationPhrase(
  session: Pick<SessionRow, "started_at" | "last_activity_at" | "ended_at">,
): string | null {
  const end = session.ended_at ?? session.last_activity_at;
  if (!session.started_at || !end) return null;
  const startMs = Date.parse(session.started_at);
  const endMs = Date.parse(end);
  if (Number.isNaN(startMs) || Number.isNaN(endMs)) return null;
  const mins = Math.round((endMs - startMs) / 60000);
  if (mins < 0) return null;
  if (mins < 1) return "under a minute";
  if (mins < 60) return `${mins}m`;
  const hours = Math.floor(mins / 60);
  const rest = mins % 60;
  return rest > 0 ? `${hours}h ${rest}m` : `${hours}h`;
}

const SESSION_STATUS_LABELS: Record<string, string> = {
  running: "Running",
  completed: "Ended",
  abandoned: "Abandoned",
  failed: "Failed",
};

/** "Running" / "Ended" / "Abandoned" / "Failed" for a session row, off the
 *  same running/completed/abandoned/failed values the Activity page reads.
 *  Null for unknown statuses. */
export function sessionStatusLabel(session: Pick<SessionRow, "status" | "effective_status">): string | null {
  const status = (session.effective_status || session.status || "").toLowerCase();
  return SESSION_STATUS_LABELS[status] ?? null;
}

/** "Started 1:27 PM" for a session that started today, "Started Sep 30, 1:27
 *  PM" otherwise (see formatClockTime — the one clock-time formatter this
 *  feature uses, so a session row and "First connected" can't render the
 *  same instant two different ways). Null when `startedAt` is missing or
 *  fails to parse. */
export function sessionStartedLabel(startedAt: string | null, now: Date): string | null {
  const when = formatClockTime(startedAt, now);
  return when ? `Started ${when}` : null;
}
