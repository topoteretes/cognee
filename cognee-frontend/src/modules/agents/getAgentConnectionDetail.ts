import { CogneeInstance } from "../instances/types";

/**
 * One remembered question from a session's recent history. `time` is kept
 * as the raw ISO string (or null) rather than pre-formatted, so the caller
 * decides how stale to consider it — same reasoning as AgentConnection's
 * `last_active_at`.
 */
export interface RecentQa {
  qaId: string;
  question: string;
  time: string | null;
}

/**
 * One recent tool-call trace, used only as a fallback when a session has no
 * recent_qas yet. `time` is optional: the wire shape this frontend has seen
 * (SessionAgentTraceEntry) carries no timestamp field, unlike SessionQAEntry
 * — kept anyway in case a future pod build adds one, rather than a second
 * normalizer the day it does.
 */
export interface RecentTrace {
  traceId: string;
  /** memory_query when the pod sent one, else origin_function — never both,
   *  never neither (an entry with neither is dropped, see normalizeTraces). */
  label: string;
  status: string;
  errorMessage: string | null;
  time: string | null;
}

export interface AgentConnectionDetail {
  qas: RecentQa[];
  traces: RecentTrace[];
}

function str(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function optionalStr(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

/** Drops an entry with no question or no qa_id — nothing renderable is left
 *  of it either way (see ConnectedAgentModal's recent-memory list). */
function normalizeQas(value: unknown): RecentQa[] {
  if (!Array.isArray(value)) return [];
  const out: RecentQa[] = [];
  for (const raw of value) {
    if (raw === null || typeof raw !== "object") continue;
    const r = raw as Record<string, unknown>;
    const qaId = str(r.qa_id);
    const question = str(r.question);
    if (!qaId || !question) continue;
    out.push({ qaId, question, time: optionalStr(r.time) });
  }
  return out;
}

/** Drops an entry with no trace_id, no status, or neither memory_query nor
 *  origin_function to label itself with. */
function normalizeTraces(value: unknown): RecentTrace[] {
  if (!Array.isArray(value)) return [];
  const out: RecentTrace[] = [];
  for (const raw of value) {
    if (raw === null || typeof raw !== "object") continue;
    const r = raw as Record<string, unknown>;
    const traceId = str(r.trace_id);
    const status = str(r.status);
    const label = str(r.memory_query) || str(r.origin_function);
    if (!traceId || !status || !label) continue;
    out.push({ traceId, label, status, errorMessage: optionalStr(r.error_message), time: optionalStr(r.time) });
  }
  return out;
}

function isDetailEnvelope(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

/**
 * Recent memory for a single connection, keyed by its `user_id` (the wire
 * calls the path parameter `agent_id`, but it is the connection row's
 * user_id, not the plugin's integration key) — see latestConnectionForAgent
 * for how the caller picks which connection to ask about.
 *
 * The pod hydrates recent_qas/recent_traces for one session only (the first
 * matching connection's own session_id) and swallows hydration failures
 * into an empty list server-side, so an empty result here is not evidence
 * of "nothing happened", only "nothing more than this to show" — the modal's
 * copy says so explicitly rather than implying full history.
 *
 * Throws on a non-object payload rather than coercing it to empty lists,
 * same reasoning as getAgentConnections: a 200 carrying an error envelope
 * must not read as "no memory yet".
 */
export default function getAgentConnectionDetail(
  instance: CogneeInstance,
  agentId: string,
  opts: { signal?: AbortSignal } = {},
): Promise<AgentConnectionDetail> {
  const init: RequestInit = { signal: opts.signal };
  return instance
    .fetch(`/v1/agents/connections/${encodeURIComponent(agentId)}`, init)
    .then((response) => response.json())
    .then((payload: unknown) => {
      if (!isDetailEnvelope(payload)) {
        throw new Error("agents/connections/{id} returned a non-object payload");
      }
      return {
        qas: normalizeQas(payload.recent_qas),
        traces: normalizeTraces(payload.recent_traces),
      };
    });
}
