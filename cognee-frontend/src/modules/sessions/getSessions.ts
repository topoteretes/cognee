import { CogneeInstance } from "../instances/types";

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

export function listSessions(
  instance: CogneeInstance,
  params: { range?: TimeRange; limit?: number; offset?: number; status?: string } = {},
): Promise<SessionsPage> {
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

export function getSessionStats(
  instance: CogneeInstance,
  range: TimeRange = "24h",
): Promise<SessionStats | null> {
  return instance
    .fetch(`/v1/sessions/stats?range=${range}`)
    .then((r) => (r.ok ? r.json() : null))
    .catch(() => null);
}
