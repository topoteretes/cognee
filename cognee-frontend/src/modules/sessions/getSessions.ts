import { CogneeInstance } from "../instances/types";

// Sessions created by the Search page's chat UI carry this prefix — it is how
// user search conversations are told apart from agent sessions everywhere.
export const SEARCH_SESSION_PREFIX = "search-ui-";

// The pod reports cost_usd/total_spend_usd as 0: it can't price the LiteLLM
// gateway alias ("litellm") so its per-call estimate always falls through to
// $0. Cost is a cloud-only concern, so we derive it here from the token counts
// the pod *does* report, using the gateway's flat billing rate — the same
// $2.50 / 1M tokens (in and out) that LiteLLM actually charges per token
// (see cloud-backend litellm-gateway input/output_cost_per_token). Overridable
// per-deploy so it can track a rate change without a code release.
export const LLM_COST_PER_1M_TOKENS = Number(
  process.env.NEXT_PUBLIC_LLM_COST_PER_1M_TOKENS ?? 2.5,
);

// Estimated USD cost for a session from its token usage. Both in/out bill at the
// same flat rate, so total tokens is enough.
export function estimateCostUsd(tokensIn: number, tokensOut: number): number {
  return ((tokensIn + tokensOut) / 1_000_000) * LLM_COST_PER_1M_TOKENS;
}

// ── "Money saved" baseline ──
// What the same work would cost with no cognee in the loop: without recall the
// agent re-sends the context it can't remember, burning an estimated 7× the
// tokens cognee routes, and pays frontier-model list price for them instead of
// the gateway's flat rate. Both figures are estimates, so both are overridable
// per-deploy and labelled as estimated wherever they surface.
export const NO_COGNEE_TOKEN_MULTIPLIER = Number(
  process.env.NEXT_PUBLIC_NO_COGNEE_TOKEN_MULTIPLIER ?? 7,
);
/** Claude Opus 5 list price ($ / 1M tokens) — the no-cognee comparison model. */
export const BASELINE_COST_PER_1M_TOKENS = Number(
  process.env.NEXT_PUBLIC_BASELINE_COST_PER_1M_TOKENS ?? 15,
);

/** No-cognee cost for a token count cognee actually routed. */
export function estimateNoCogneeCostUsd(tokens: number): number {
  return ((tokens * NO_COGNEE_TOKEN_MULTIPLIER) / 1_000_000) * BASELINE_COST_PER_1M_TOKENS;
}

/**
 * Tokens the same work would have burned without cognee, minus the ones it did
 * burn — the tokens never sent. This is the figure that holds on any plan: on
 * usage-based billing it converts to cash, on a subscription it is limit
 * headroom that simply went unspent.
 */
export function tokensAvoided(tokens: number): number {
  return tokens * Math.max(0, NO_COGNEE_TOKEN_MULTIPLIER - 1);
}

/** Token count behind a cognee spend figure — inverts the flat gateway rate. */
export function tokensFromSpendUsd(spendUsd: number): number {
  if (LLM_COST_PER_1M_TOKENS <= 0) return 0;
  return (spendUsd / LLM_COST_PER_1M_TOKENS) * 1_000_000;
}

/**
 * Same baseline, reached from a cognee spend figure rather than raw tokens —
 * needed for the billing endpoint, which reports dollars only. Safe because
 * that spend is itself the flat per-token gateway rate, so dividing by the rate
 * recovers the token count it was priced from.
 */
export function noCogneeCostFromSpendUsd(spendUsd: number): number {
  return estimateNoCogneeCostUsd(tokensFromSpendUsd(spendUsd));
}

// A session's access channel is read from its session_id prefix — the same
// signal the dashboard's Agents panel uses to tell integrations apart (see
// OverviewPage's PERSISTENT_AGENT_DEFS/DYNAMIC_AGENT_DEFS). Claude Desktop is
// checked before Claude Code since "claude_desktop_" also starts with
// "claude_" — order here is significant, unlike in OverviewPage where each
// def is matched independently.
const CHANNEL_DEFS: Array<{ name: string; prefixes: string[] }> = [
  { name: "Claude Desktop", prefixes: ["claude_desktop_"] },
  { name: "Claude Code", prefixes: ["claude_", "cc_"] },
  { name: "Codex", prefixes: ["codex_"] },
  { name: "OpenClaw", prefixes: ["openclaw_"] },
  { name: "Hermes Agent", prefixes: ["hermes_"] },
  { name: "VS Code", prefixes: ["vscode_"] },
  { name: "Cursor", prefixes: ["cursor_"] },
  { name: "Gemini CLI", prefixes: ["gemini_"] },
  { name: "Cline", prefixes: ["cline_"] },
];

// "Via" — how a session reached Cognee: the web UI, a known agent/IDE
// integration, or bare API/SDK access (no recognized client prefix).
export function channelForSessionId(sessionId: string): string {
  if (sessionId.startsWith(SEARCH_SESSION_PREFIX)) return "UI";
  return CHANNEL_DEFS.find((d) => d.prefixes.some((p) => sessionId.startsWith(p)))?.name ?? "API";
}

/**
 * Brain column for a session that carries no dataset scope.
 *
 * Says only what a null `dataset_id` proves: the operation was not confined to
 * one brain. It deliberately does NOT claim every brain was queried — that
 * holds for in-app chat (see the datasetIds fallback in
 * AgentActivityTerminal.handleSearch) but not for agents reaching the pod over
 * MCP/SDK, whose scoping this frontend never sees, and it is plainly false for
 * a session abandoned before it queried anything. Still not the unknown dash:
 * the value is absent by design, not missing, and most memory traffic is
 * cross-brain.
 */
export const UNSCOPED_BRAIN_LABEL = "unscoped";

/** Brain column for a session row, given a dataset id → name lookup. */
export function sessionBrainLabel(datasetId: string | null, nameById: Map<string, string>): string {
  if (!datasetId) return UNSCOPED_BRAIN_LABEL;
  return nameById.get(datasetId) ?? datasetId;
}

/** Structural param, not FilterContext's `Dataset`: a module must not import from `@/ui`. */
export function datasetNameById(datasets: { id: string; name: string }[]): Map<string, string> {
  return new Map(datasets.map((d) => [d.id, d.name]));
}

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

// Cache per instance base URL so we probe at most once per page load — but
// only a positive result is durable. The pod's per-request latency climbs
// under load (see the pod-latency comment on the shared http client), and a
// single slow/failed probe during a heavy cognify run must not brand a
// perfectly healthy instance "unavailable" for the rest of the tab: that
// made real sessions vanish from the UI, indistinguishable from having none,
// until a full page reload happened to reset this module. A negative result
// is retried after a short cooldown instead of being cached forever.
// Uses a Promise cache so concurrent callers share the same in-flight request.
const _sessionsProbe = new Map<string, { promise: Promise<boolean>; expiresAt: number }>();
const PROBE_NEGATIVE_TTL_MS = 30_000;

function isSessionsAvailable(instance: CogneeInstance): Promise<boolean> {
  const key = (instance as { baseUrl?: string }).baseUrl ?? "default";
  const cached = _sessionsProbe.get(key);
  if (cached && cached.expiresAt > Date.now()) return cached.promise;
  const probe = instance.fetch("/v1/sessions?limit=1")
    .then((r) => r.ok)
    .catch((err) => {
      console.warn("[getSessions] sessions-availability probe failed:", err instanceof Error ? err.message : err);
      return false;
    });
  const entry = { promise: probe, expiresAt: Infinity };
  _sessionsProbe.set(key, entry);
  probe.then((ok) => {
    entry.expiresAt = ok ? Infinity : Date.now() + PROBE_NEGATIVE_TTL_MS;
  });
  return probe;
}

// The pod reports the LiteLLM gateway route ("litellm_proxy/litellm", or
// bare "litellm" depending on deployment) as the model for calls that went
// through the proxy — an infrastructure alias, not a model name a user
// would recognize. LiteLLM must never surface anywhere in the UI, so any
// value mentioning it is hidden rather than rendered.
function displayModel(model: string | null | undefined): string | null {
  if (!model || model.toLowerCase().includes("litellm")) return null;
  return model;
}

// A 200 is no guarantee of shape — a proxy error envelope or a pod mid-deploy
// can return anything, and mapping over `sessions` that isn't an array threw
// a TypeError that the network-error catch silently turned into an empty
// page. Checked explicitly so a malformed payload is at least distinguishable
// from "no sessions" in the console.
function isSessionsPage(value: unknown): value is SessionsPage {
  return (
    typeof value === "object" &&
    value !== null &&
    Array.isArray((value as { sessions?: unknown }).sessions)
  );
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
    .then((r) => (r.ok ? (r.json() as Promise<unknown>) : EMPTY_PAGE))
    .then((page: unknown) => {
      if (!isSessionsPage(page)) {
        console.warn("[getSessions] listSessions got a 200 with an unexpected shape, returning empty page");
        return EMPTY_PAGE;
      }
      return {
        ...page,
        sessions: page.sessions.map((s) => ({ ...s, last_model: displayModel(s.last_model) })),
      };
    })
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

export type RunStatus = EnrichmentRun["status"];

export function runStatus(raw: string | undefined): RunStatus {
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
): Promise<SessionDetail | null> {
  if (!(await isSessionsAvailable(instance))) return null;
  return instance
    .fetch(`/v1/sessions/${encodeURIComponent(sessionId)}`)
    .then((r) => (r.ok ? r.json() : null))
    .then((detail: SessionDetail | null) =>
      detail ? { ...detail, last_model: displayModel(detail.last_model) } : null,
    )
    .catch((err) => {
      console.warn("[getSessions] getSessionDetail failed:", err instanceof Error ? err.message : err);
      return null;
    });
}
