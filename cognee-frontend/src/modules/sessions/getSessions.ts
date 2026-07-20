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
    .catch((err) => {
      console.warn("[getSessions] sessions-availability probe failed:", err instanceof Error ? err.message : err);
      return false;
    });
  _sessionsProbe.set(key, probe);
  return probe;
}

export async function listSessions(
  instance: CogneeInstance,
  params: { range?: TimeRange; limit?: number; offset?: number; status?: string } = {},
  opts: { signal?: AbortSignal; timeoutMs?: number } = {},
): Promise<SessionsPage> {
  if (!(await isSessionsAvailable(instance))) return EMPTY_PAGE;
  const q = new URLSearchParams();
  if (params.range) q.set("range", params.range);
  if (params.limit !== undefined) q.set("limit", String(params.limit));
  if (params.offset !== undefined) q.set("offset", String(params.offset));
  if (params.status) q.set("status", params.status);
  const fetchInit: RequestInit & { timeoutMs?: number } = { signal: opts.signal, timeoutMs: opts.timeoutMs };
  return instance
    .fetch(`/v1/sessions?${q.toString()}`, fetchInit)
    .then((r) => (r.ok ? r.json() : EMPTY_PAGE))
    .catch((err) => {
      console.warn("[getSessions] listSessions failed, returning empty page:", err instanceof Error ? err.message : err);
      return EMPTY_PAGE;
    });
}

export async function getSessionStats(
  instance: CogneeInstance,
  range: TimeRange = "24h",
): Promise<SessionStats | null> {
  if (!(await isSessionsAvailable(instance))) return null;
  return instance
    .fetch(`/v1/sessions/stats?range=${range}`)
    .then((r) => (r.ok ? r.json() : null))
    .catch((err) => {
      console.warn("[getSessions] getSessionStats failed:", err instanceof Error ? err.message : err);
      return null;
    });
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

export interface EnrichmentRun {
  id: string | null;
  created_at: string | null;
  status: "completed" | "running" | "failed";
  dataset_name: string | null;
  // Pipeline runs coalesced into this entry — one improve() emits several
  // memify sub-pipeline runs in a burst.
  count: number;
  // Errored sub-runs within the burst. improve()'s stages are best-effort,
  // so the burst only counts as failed when nothing completed at all.
  // Not shown to users on partial success — kept for internal KPIs.
  error_count: number;
  // Error of the newest errored sub-run — only rendered when the whole
  // burst failed. Null until the pod exposes run_info errors.
  failure_reason: string | null;
  // Oldest run in the burst — with created_at (newest) this bounds the
  // burst's wall-clock duration.
  started_at: string | null;
}

// improve()'s graph stages (feedback weights, session Q&A persist, enrichment)
// all record as memify runs — the closest queryable signal for "graph
// enrichment" until the pod exposes per-session improve state directly.
const ENRICHMENT_PIPELINE = "memify_pipeline";
// Runs closer together than this are one improve() burst.
const ENRICHMENT_COALESCE_MS = 5 * 60_000;

interface ActivityRun {
  id?: string;
  pipeline_name?: string;
  status?: string;
  dataset_id?: string | null;
  dataset_name?: string | null;
  created_at?: string | null;
  pipeline_run_id?: string | null;
  error?: string | null;
}

type RunStatus = EnrichmentRun["status"];

function runStatus(raw: string | undefined): RunStatus {
  const s = raw ?? "";
  return s.includes("COMPLETED") ? "completed" : s.includes("ERRORED") ? "failed" : "running";
}

// Naive ISO timestamps from the pod are UTC.
function isoToMs(iso: string | null | undefined): number {
  if (!iso) return 0;
  const hasTz = /Z$|[+-]\d{2}:?\d{2}$/.test(iso);
  return Date.parse(hasTz ? iso : iso + "Z") || 0;
}

// The activity endpoint returns one row when a pipeline run starts and another
// when it finishes, sharing a pipeline_run_id — keep only the terminal row
// (or the newest, while still running).
function dedupeByRunId(rows: ActivityRun[]): ActivityRun[] {
  const byRun = new Map<string, ActivityRun>();
  for (const row of rows) {
    const key = row.pipeline_run_id ?? row.id ?? String(byRun.size);
    const prev = byRun.get(key);
    if (!prev) { byRun.set(key, row); continue; }
    const prevTerminal = runStatus(prev.status) !== "running";
    const rowTerminal = runStatus(row.status) !== "running";
    if ((rowTerminal && !prevTerminal) || (rowTerminal === prevTerminal && isoToMs(row.created_at) > isoToMs(prev.created_at))) {
      byRun.set(key, row);
    }
  }
  return [...byRun.values()];
}

// Newest burst first.
export async function getGraphEnrichmentRuns(
  instance: CogneeInstance,
  datasetId: string,
): Promise<EnrichmentRun[]> {
  try {
    const r = await instance.fetch("/v1/activity/pipeline-runs");
    if (!r.ok) return [];
    const data: unknown = await r.json();
    const rows = dedupeByRunId(
      (Array.isArray(data) ? (data as ActivityRun[]) : [])
        .filter((run) => run.dataset_id === datasetId && run.pipeline_name === ENRICHMENT_PIPELINE),
    ).sort((a, b) => isoToMs(b.created_at) - isoToMs(a.created_at));

    const bursts: EnrichmentRun[] = [];
    let prevTs = 0;
    for (const row of rows) {
      const ts = isoToMs(row.created_at);
      const status = runStatus(row.status);
      const current = bursts[bursts.length - 1];
      if (current && prevTs - ts <= ENRICHMENT_COALESCE_MS) {
        current.count += 1;
        if (status === "failed") {
          current.error_count += 1;
          current.failure_reason = current.failure_reason ?? row.error ?? null;
        }
        // improve() runs its stages best-effort, so an errored stage does not
        // fail the burst: running while any stage runs, completed as long as
        // any stage completed, failed only when every stage errored.
        if (status === "running") current.status = "running";
        else if (status === "completed" && current.status === "failed") current.status = "completed";
        current.started_at = row.created_at ?? current.started_at;
      } else {
        bursts.push({
          id: row.pipeline_run_id ?? row.id ?? null,
          created_at: row.created_at ?? null,
          status,
          dataset_name: row.dataset_name ?? null,
          count: 1,
          error_count: status === "failed" ? 1 : 0,
          failure_reason: status === "failed" ? (row.error ?? null) : null,
          started_at: row.created_at ?? null,
        });
      }
      prevTs = ts;
    }
    return bursts;
  } catch (err) {
    console.warn("[getSessions] getGraphEnrichmentRuns failed:", err instanceof Error ? err.message : err);
    return [];
  }
}

export async function getSessionDetail(
  instance: CogneeInstance,
  sessionId: string,
  scope: { datasetId?: string | null; ownerUserId?: string } = {},
): Promise<SessionDetail | null> {
  if (!(await isSessionsAvailable(instance))) return null;
  const query = new URLSearchParams();
  if (scope.datasetId) query.set("dataset_id", scope.datasetId);
  if (scope.ownerUserId) query.set("owner_user_id", scope.ownerUserId);
  const suffix = query.size ? `?${query.toString()}` : "";
  return instance
    .fetch(`/v1/sessions/${encodeURIComponent(sessionId)}${suffix}`)
    .then((r) => (r.ok ? r.json() : null))
    .catch((err) => {
      console.warn("[getSessions] getSessionDetail failed:", err instanceof Error ? err.message : err);
      return null;
    });
}
