import { channelForSessionId, isSessionRunning } from "@/modules/sessions/getSessions";
import type { SessionRow } from "@/modules/sessions/getSessions";
import { timeAgo } from "@/utils/timeAgo";
import type { NodeStatus } from "@/app/(app)/dashboard/partials/redesign/MemoryFlowDiagram";

// `name` is the channel `channelForSessionId` resolves, which is what ties a
// session to a node. No prefixes here on purpose: they live once, in
// getSessions' CHANNEL_DEFS, which applies them in precedence order. Matching
// each def against its own prefix list independently made "claude_" swallow
// "claude_desktop_", so every Claude Desktop session also lit up Claude Code.
export interface AgentDef { name: string; logo: string }

/** Always listed in the memory graph, connected or not. */
export const PERSISTENT_AGENT_DEFS: AgentDef[] = [
  { name: "Claude Code", logo: "claude" },
  { name: "Codex", logo: "codex" },
  { name: "Claude Desktop", logo: "claude" },
  { name: "OpenClaw", logo: "openclaw" },
];

// Dynamic agents — NOT shown by default. They appear once they have registered,
// then follow the same three-state rule as the persistent ones.
export const DYNAMIC_AGENT_DEFS: AgentDef[] = [
  { name: "Hermes Agent", logo: "hermes" },
  { name: "VS Code", logo: "vscode" },
  { name: "Cursor", logo: "cursor" },
  { name: "Gemini CLI", logo: "gemini" },
  { name: "Cline", logo: "cline" },
  // getSessions' CHANNEL_DEFS added this channel; without a matching def here
  // an opencode_ session resolved to a real channel name that named no node,
  // so the roster-completeness test below is what would have caught it.
  // /visuals/logos/opencode.svg does not exist yet — until it is added this
  // node renders with a missing mask icon, same as any other logo key that
  // does not have an asset.
  { name: "OpenCode", logo: "opencode" },
];

function matches(session: SessionRow, name: string): boolean {
  return channelForSessionId(session.session_id) === name;
}

/**
 * Has this agent ever reached the workspace?
 *
 * Not answerable from `sessions` alone. That feed is scoped to this page's
 * range toggle and capped at 50 rows, and the pod caps it again, so on a busy
 * workspace an agent that ran an hour ago is simply not in it. `registry` is
 * the sticky per-tenant answer that survives the window; the live feed is
 * still consulted so a first-ever session registers immediately.
 */
export function hasRegistered(sessions: SessionRow[], registry: Record<string, boolean>, name: string): boolean {
  return registry[name] === true || sessions.some((s) => matches(s, name));
}

/**
 * Four states: the three the graph always drew, plus "unknown" for the
 * window between mount and the first telemetry answer.
 *
 * A session is minutes long, so requiring a running one to look anything other
 * than disconnected made a working agent read as absent nearly all the time.
 * "live" is a session in flight, "connected" is registered and idle.
 *
 * `loading` only ever downgrades an unconfirmed "disconnected" reading to
 * "unknown" — it never masks a real answer. The sticky registry can still
 * say "connected" (and a running session still says "live") before the first
 * poll resolves; what it cannot do yet is rule connection out, so this must
 * not claim it has.  Before this, a workspace whose registry had not seen this
 * browser yet — a fresh signup, a different device — rendered a confident red
 * "Not connected yet" for the several seconds telemetry takes to answer, then
 * flipped to green the moment it did. That flash was the false claim, not the
 * settled state on either side of it.
 *
 * Deliberately NOT also "live" for an active plugin registration: the pod's
 * `status: "active"` on a connection row is only ever set at registration and
 * cleared by the plugin's own unregister call, so a session that crashes or
 * loses network before it can unregister leaves that row "active" forever.
 * Verified live: a local tenant had three claude_code rows reading "active"
 * simultaneously, at least two of them from dead sessions. A registration is
 * evidence the agent is installed, not that it is running right now, which is
 * exactly the distinction "connected" vs "live" exists to draw. The running
 * session's own effective_status, computed server-side with a real timeout,
 * is still the only thing this graph treats as "live".
 */
export function agentStatus(sessions: SessionRow[], registry: Record<string, boolean>, name: string, loading: boolean): NodeStatus {
  if (sessions.some((s) => isSessionRunning(s) && matches(s, name))) return "live";
  if (hasRegistered(sessions, registry, name)) return "connected";
  return loading ? "unknown" : "disconnected";
}

/**
 * The words behind a node's status, so the graph's tooltip and the
 * Integrations/dashboard cards can never say something different about the
 * same agent: a badge ("Live" / "Connected" / "Checking…" / "Not connected
 * yet") and, when there is one, a detail phrase to go with it ("active now",
 * or "last seen X ago" off the same lastSeenByAgent this whole feature reads
 * everywhere else). `reconnect` is not a state agentStatus ever produces,
 * folded into "disconnected" here rather than left unhandled.
 */
/** How fresh `lastSeenAt` has to be for a "live" agent to read "active now"
 *  rather than a plain last-activity time. `effective_status: "running"` on
 *  the session row only means the pod hasn't timed it out yet — verified
 *  live, a session can sit "running" for 20+ minutes with nothing actually
 *  happening, and "active now" next to "last activity 24m ago" was this
 *  frontend claiming freshness the server never reported. */
export const ACTIVE_NOW_MS = 2 * 60 * 1000;

export function agentStatusText(
  status: NodeStatus,
  lastSeenAt: string | null,
  now: Date,
): { badge: string; detail: string | null } {
  if (status === "live") {
    const lastSeenMs = lastSeenAt ? Date.parse(lastSeenAt) : Number.NaN;
    // "Now" is a claim that needs a fresh timestamp behind it. With none, or
    // an unparseable one, all that is known is that the server still reports
    // a running session, so that is all this says.
    if (!lastSeenAt || Number.isNaN(lastSeenMs)) return { badge: "Live", detail: "running" };
    const detail =
      now.getTime() - lastSeenMs <= ACTIVE_NOW_MS ? "active now" : `running · last activity ${timeAgo(lastSeenAt)}`;
    return { badge: "Live", detail };
  }
  if (status === "connected") return { badge: "Connected", detail: lastSeenAt ? `last seen ${timeAgo(lastSeenAt)}` : null };
  if (status === "unknown") return { badge: "Checking…", detail: null };
  return { badge: "Not connected yet", detail: null };
}

/** `agentStatusText`, joined into the one-line sentence the graph's node
 *  tooltip shows. */
export function agentStatusTooltip(status: NodeStatus, lastSeenAt: string | null, now: Date): string {
  const { badge, detail } = agentStatusText(status, lastSeenAt, now);
  return detail ? `${badge}, ${detail}` : badge;
}
