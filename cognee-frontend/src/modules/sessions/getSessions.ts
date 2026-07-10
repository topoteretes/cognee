import { CogneeInstance } from "../instances/types";

// Sessions created by the Search page's chat UI carry this prefix — it is how
// user search conversations are told apart from agent sessions everywhere.
export const SEARCH_SESSION_PREFIX = "search-ui-";

export interface SessionRow {
  session_id: string;
  user_id: string;
  dataset_id: string | null;
  status: string;
  effective_status: string;
  started_at: string | null;
  last_activity_at: string | null;
  ended_at: string | null;
  tokens_in: number;
  tokens_out: number;
  cost_usd: number;
  error_count: number;
  last_model: string | null;
}

export interface SessionsPage {
  sessions: SessionRow[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
}

export interface SessionStats {
  range: string;
  sessions: number;
  total_spend_usd: number;
  avg_spend_per_session_usd: number;
  tokens_in: number;
  tokens_out: number;
  tokens_total: number;
  agent_time_s: number;
  avg_session_s: number;
  success_rate: number;
  completed: number;
  failed: number;
  abandoned: number;
  running: number;
}

export type TimeRange = "24h" | "7d" | "30d" | "all";

const EMPTY_PAGE: SessionsPage = {
  sessions: [],
  total: 0,
  limit: 50,
  offset: 0,
  has_more: false,
};

// Cache per instance base URL so we probe at most once per page load.
// Uses a Promise cache so concurrent callers share the same in-flight request.
const _sessionsProbe = new Map<string, Promise<boolean>>();

function isSessionsAvailable(instance: CogneeInstance): Promise<boolean> {
  const key = (instance as { baseUrl?: string }).baseUrl ?? "default";
  const existing = _sessionsProbe.get(key);
  if (existing) return existing;
  const probe = instance.fetch("/v1/sessions?limit=1")
    .then((r) => r.ok)
    .catch(() => false);
  _sessionsProbe.set(key, probe);
  return probe;
}

export async function listSessions(
  instance: CogneeInstance,
  params: { range?: TimeRange; limit?: number; offset?: number; status?: string } = {},
): Promise<SessionsPage> {
  if (!(await isSessionsAvailable(instance))) return EMPTY_PAGE;
  const q = new URLSearchParams();
  if (params.range) q.set("range", params.range);
  if (params.limit !== undefined) q.set("limit", String(params.limit));
  if (params.offset !== undefined) q.set("offset", String(params.offset));
  if (params.status) q.set("status", params.status);
  return instance
    .fetch(`/v1/sessions?${q.toString()}`)
    .then((r) => (r.ok ? r.json() : EMPTY_PAGE))
    .catch(() => EMPTY_PAGE);
}

export async function getSessionStats(
  instance: CogneeInstance,
  range: TimeRange = "24h",
): Promise<SessionStats | null> {
  if (!(await isSessionsAvailable(instance))) return null;
  return instance
    .fetch(`/v1/sessions/stats?range=${range}`)
    .then((r) => (r.ok ? r.json() : null))
    .catch(() => null);
}

export interface TraceEntry {
  trace_id?: string;
  origin_function?: string;
  status?: "success" | "error" | string;
  memory_query?: string;
  memory_context?: string;
  method_params?: Record<string, unknown> | null;
  method_return_value?: unknown;
  error_message?: string;
  session_feedback?: string;
  time?: string;
}

export interface SessionDetail extends SessionRow {
  label: string | null;
  msg_count: number;
  tool_calls: number;
  qas: Record<string, unknown>[];
  traces: TraceEntry[];
}

export async function getSessionDetail(
  instance: CogneeInstance,
  sessionId: string,
): Promise<SessionDetail | null> {
  if (!(await isSessionsAvailable(instance))) return null;
  return instance
    .fetch(`/v1/sessions/${encodeURIComponent(sessionId)}`)
    .then((r) => (r.ok ? r.json() : null))
    .catch(() => null);
}
