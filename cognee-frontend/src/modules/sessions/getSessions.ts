import { CogneeInstance } from "../instances/types";

// Sessions created by the Search page's chat UI carry this prefix — it is how
// user search conversations are told apart from agent sessions everywhere.
export const SEARCH_SESSION_PREFIX = "search-ui-";

/**
 * A per-deploy numeric override, or the default when it is not usable.
 *
 * `Number(process.env.X ?? fallback)` was the pattern here, and it fails in two
 * silent ways that both surface as a wrong claim rather than an error. An unset
 * variable that is present but empty (`X=` in an env file) parses to 0, so the
 * token multiplier became 0 and the Cost Savings panel confidently reported "0
 * tokens saved" while its baseline line collapsed onto actual spend. A typo
 * parses to NaN and the same panel renders "NaN tokens saved".
 *
 * Bounded rather than merely finite: a multiplier below 1 says the same work
 * costs less without cognee, which is not an estimate this product is making,
 * and it inverted the chart's savings band. Enforcing the bound here is what
 * lets the arithmetic downstream stay plain, instead of each consumer carrying
 * its own clamp and disagreeing about the shape of it.
 */
function envNumber(raw: string | undefined, fallback: number, min: number, name: string): number {
  if (raw === undefined || raw.trim() === "") return fallback;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed < min) {
    console.warn(`[getSessions] ${name}=${JSON.stringify(raw)} is not a number >= ${min}; using ${fallback}`);
    return fallback;
  }
  return parsed;
}

// The pod reports cost_usd/total_spend_usd as 0: it can't price the LiteLLM
// gateway alias ("litellm") so its per-call estimate always falls through to
// $0. Cost is a cloud-only concern, so we derive it here from the token counts
// the pod *does* report, using the gateway's flat billing rate — the same
// $1.00 / 1M tokens (in and out) that LiteLLM actually charges per token
// (see cloud-backend litellm-gateway input/output_cost_per_token). Overridable
// per-deploy so it can track a rate change without a code release.
export const LLM_COST_PER_1M_TOKENS = envNumber(
  process.env.NEXT_PUBLIC_LLM_COST_PER_1M_TOKENS,
  1.0,
  // Strictly above zero: `tokensFromSpendUsd` divides by it.
  Number.MIN_VALUE,
  "NEXT_PUBLIC_LLM_COST_PER_1M_TOKENS",
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
export const NO_COGNEE_TOKEN_MULTIPLIER = envNumber(
  process.env.NEXT_PUBLIC_NO_COGNEE_TOKEN_MULTIPLIER,
  7,
  // At least 1: below that the baseline sits under actual spend and every
  // figure built on it flips sign.
  1,
  "NEXT_PUBLIC_NO_COGNEE_TOKEN_MULTIPLIER",
);
/** Claude Opus 5 list price ($ / 1M tokens) — the no-cognee comparison model. */
export const BASELINE_COST_PER_1M_TOKENS = envNumber(
  process.env.NEXT_PUBLIC_BASELINE_COST_PER_1M_TOKENS,
  15,
  Number.MIN_VALUE,
  "NEXT_PUBLIC_BASELINE_COST_PER_1M_TOKENS",
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
  // No clamp. `NO_COGNEE_TOKEN_MULTIPLIER` is validated at >= 1 where it is
  // read, so this cannot go negative — and the clamp that used to be here was
  // half a guard: `estimateNoCogneeCostUsd` had none, so a multiplier under 1
  // reported zero tokens saved while the chart drew the baseline below actual
  // spend. One bound at the boundary, not two that disagree.
  return tokens * (NO_COGNEE_TOKEN_MULTIPLIER - 1);
}

/** Token count behind a cognee spend figure — inverts the flat gateway rate. */
export function tokensFromSpendUsd(spendUsd: number): number {
  // No zero check: the rate is validated above zero where it is read. The
  // check that was here returned 0, which silently collapsed the no-cognee
  // baseline to nothing and made the panel report no savings at all.
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

// A session's access channel is read from its session_id prefix. This table is
// the only place that mapping lives: the dashboard's memory graph and the
// Integrations cards both resolve a session through `channelForSessionId`
// rather than carrying their own copy of these prefixes, because a second copy
// is a second chance to get the precedence wrong. Order is significant, and it
// is the whole reason for the single source: "claude_desktop_" also starts with
// "claude_", so matching each def independently lights up Claude Code for
// every Claude Desktop session.
const CHANNEL_DEFS: Array<{ name: string; prefixes: string[] }> = [
  { name: "Claude Desktop", prefixes: ["claude_desktop_"] },
  { name: "Claude Code", prefixes: ["claude_", "cc_"] },
  { name: "Codex", prefixes: ["codex_"] },
  // cognee's KNOWN_PLUGINS ships an `opencode` key; without an entry here an
  // opencode_ session fell through to "API" while the roster said "Opencode".
  { name: "OpenCode", prefixes: ["opencode_"] },
  { name: "OpenClaw", prefixes: ["openclaw_"] },
  { name: "Hermes Agent", prefixes: ["hermes_"] },
  { name: "VS Code", prefixes: ["vscode_"] },
  { name: "Cursor", prefixes: ["cursor_"] },
  { name: "Gemini CLI", prefixes: ["gemini_"] },
  { name: "Cline", prefixes: ["cline_"] },
];

// "Via" — how a session reached Cognee: the Cloud Dashboard's own Search
// page, a known agent/IDE integration, or bare API/SDK access (no recognized
// client prefix). "Cloud", not "UI" — "UI" doesn't say whose, and every other
// channel here is named after the actual product it came from.
export function channelForSessionId(sessionId: string): string {
  if (sessionId.startsWith(SEARCH_SESSION_PREFIX)) return "Cloud";
  return CHANNEL_DEFS.find((d) => d.prefixes.some((p) => sessionId.startsWith(p)))?.name ?? "API";
}

/** Every channel `channelForSessionId` can name, for callers that enumerate them. */
export const AGENT_CHANNEL_NAMES: readonly string[] = CHANNEL_DEFS.map((d) => d.name);

/**
 * A session's status, lowercased, preferring the pod's derived
 * `effective_status` over the stored `status`.
 *
 * `effective_status` is typed non-optional on `SessionRow` but nothing
 * validates it at the wire, and a pod that omits it left every reader that
 * compared it directly permanently unequal to "running". Both fields are
 * treated as possibly absent here so that one answer serves every caller.
 */
export function sessionEffectiveStatus(session: {
  effective_status?: string | null;
  status?: string | null;
}): string {
  return (session.effective_status || session.status || "").toLowerCase();
}

/** True while the session is still open, by the status above. */
export function isSessionRunning(session: { effective_status?: string | null; status?: string | null }): boolean {
  return sessionEffectiveStatus(session) === "running";
}

/**
 * The canonical `CHANNEL_DEFS` name for an already-de-slugged agent type, if
 * it names one of them.
 *
 * The roster and the Via column describe the same clients from two different
 * sources (an `agent_type` string, a session-id prefix) and must not disagree
 * on spelling. Title-casing alone gets "Openclaw", "Vs Code" and "Gemini Cli"
 * where this table says OpenClaw, VS Code and Gemini CLI.
 */
export function channelNames(): string[] {
  return CHANNEL_DEFS.map((d) => d.name);
}

export function channelDisplayName(deSlugged: string): string | undefined {
  const want = deSlugged.trim().toLowerCase().replace(/[\s_-]+/g, " ");
  return CHANNEL_DEFS.find((d) => d.name.toLowerCase() === want)?.name;
}

// The pod's origin values (`_stamp_operation_origin` in
// cognee-saas-pod/api/server.py, over cognee's modules/operations/origin.py
// vocabulary) mapped into the same names the session prefixes produce — so
// the Via column and its filter dropdown never offer `cloud` and "Cloud", or
// `api` and "API", as two separate values for one channel.
const ORIGIN_LABELS: Record<string, string> = {
  // This dashboard, sending POD_ORIGIN_HEADER. Same label a search-ui-
  // session resolves to, because it is the same surface.
  cloud: "Cloud",
  api: "API",
  cli: "CLI",
  mcp: "MCP",
  background: "Background",
};

/**
 * The Via column for an activity row.
 *
 * The session prefix is consulted first because it names the actual client —
 * Claude Code, Cursor, Cursor's IDE integration — which is finer-grained than
 * `origin`, whose whole vocabulary is a handful of entry points.
 *
 * Rows with no session fall back to `origin`, which is where an upload or a
 * cognify lands: ingestion is not session work, so it carries no session. The
 * pipelines a session-bound improve launches do carry the session (cognee PR
 * 6160) and name its client. Reading "Cloud" there depends on the pod running the
 * origin middleware — a pod image without it records "sdk" for every HTTP
 * request, and "sdk" is discarded below rather than rendered, because it is
 * origin.py's default-when-unset and printing "SDK" for a request that
 * plainly arrived over HTTP would be a claim the data does not support.
 */
export function viaLabel(row: Pick<PipelineRun, "origin" | "session_ids">): string {
  return viaChannel(row) ?? UNKNOWN_VIA_LABEL;
}

/** The dash `viaLabel` renders when nothing identifies the surface. A
 *  table-cell string, never a grouping key. See viaChannel. */
export const UNKNOWN_VIA_LABEL = "—";

/**
 * Same derivation as `viaLabel`, but `undefined` rather than a rendered dash
 * when the surface is unknown.
 *
 * Anything that GROUPS by channel must use this. Bucketing on the dash files
 * every unidentified row under a category literally named "—", which is what
 * the Analytics breakdown did on every pod with no origin middleware, i.e.
 * all of them today, collapsing the whole panel to a single dash row.
 */
export function viaChannel(row: Pick<PipelineRun, "origin" | "session_ids">): string | undefined {
  // Several sessions name a client only when they agree. A run that bridges
  // a Claude Code session and a Cursor session has no single client, so it
  // falls through to origin like a row with no session at all.
  const channels = new Set(row.session_ids.map(channelForSessionId));
  if (channels.size === 1) return [...channels][0];
  if (!row.origin || row.origin === "sdk") return undefined;
  return ORIGIN_LABELS[row.origin] ?? row.origin;
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
// Parenthesized for the same reason UNKNOWN_DATASET_LABEL is, and it was not
// until the Analytics breakdown turned this column into a grouping key and a
// filter link: a dataset can be named anything, so a workspace with a dataset
// actually called "unscoped" had its runs merged into this bucket and selected
// by its drill-through. The hazard was diagnosed for one of the two sentinels
// in this column and missed for the other.
export const UNSCOPED_BRAIN_LABEL = "(no dataset)";

/**
 * A pipeline run with no dataset_id.
 *
 * Deliberately not "unscoped": dataset_id is nullable at the source, so a run
 * without one carries no evidence either way, while a session without one is
 * positively known to be unscoped. Two facts, two words.
 *
 * A readable label rather than the dash it used to be, and that is the fix: the
 * Activity page drops "—" from its Dataset filter options, because you cannot
 * filter on "absent". But a readable dataset is missing for every recall, so the
 * dash was the majority of rows and the filter had no option for any of them.
 *
 * Parenthesized because a bare word cannot be a sentinel here: a dataset can be
 * named anything, and "unknown"/"unscoped" are plausible names, so a real
 * dataset would have shared one filter option with the no-dataset rows and
 * neither could be selected on its own.
 */
export const UNKNOWN_DATASET_LABEL = "(not recorded)";

/**
 * Dataset column for a pipeline run, given a dataset id → name lookup.
 *
 * Shared, because the Analytics breakdown's bucket name and the Activity
 * table's cell for the same row have to be one string: the drill-through link
 * carries the bucket name, and a page that labels the row differently filters
 * every one of them out. That is what broke, and on the largest bucket in the
 * table — `ActivityPage` rendered "—" where the row model rendered
 * `UNKNOWN_DATASET_LABEL`, so the link selected nothing and the page had no
 * option matching the value it had been handed.
 *
 * Byte-equal to the expression `activityRowModel.buildActivityRows` inlines,
 * which is left alone so it stays identical to the sibling branch's copy.
 */
export function runDatasetLabel(
  row: Pick<PipelineRun, "dataset_name" | "dataset_id">,
  nameById: Map<string, string>,
): string {
  return row.dataset_name || (row.dataset_id ? nameById.get(row.dataset_id) : undefined) || UNKNOWN_DATASET_LABEL;
}

/** Brain column for a session row, given a dataset id → name lookup. */
export function sessionBrainLabel(datasetId: string | null, nameById: Map<string, string>): string {
  if (!datasetId) return UNSCOPED_BRAIN_LABEL;
  return nameById.get(datasetId) ?? datasetId;
}

/** Structural param, not FilterContext's `Dataset`: a module must not import from `@/ui`. */
export function datasetNameById(datasets: { id: string; name: string }[]): Map<string, string> {
  return new Map(datasets.map((d) => [d.id, d.name]));
}

// `kind` is the backend's own discriminator between a pipeline run
// (pipeline_name set) and a single memory operation — recall/search/remember/
// forget/improve/delete/prune — which reports pipeline_name: null instead.
// Every consumer that only cares about pipelines (the onboarding checklist,
// the "ran <pipeline>" log line) must branch on `kind`, not on pipeline_name
// truthiness, so it keeps working if the backend ever adds a third kind.
//
// Lives here rather than in the component that renders it: this is the wire
// shape of GET /v1/activity/pipeline-runs, and a module must not import from
// `@/ui`. AgentActivityTerminal re-exports it for its existing importers.
export interface PipelineRun {
  id: string;
  pipeline_name: string | null;
  status: string;
  dataset_id: string | null;
  dataset_name: string | null;
  owner_id: string | null;
  owner_email: string | null;
  created_at: string | null;
  pipeline_run_id: string | null;
  kind: "pipeline" | "operation";
  operation_name: string | null;
  origin: string | null;
  outcome: string | null;
  // The exception class name on a failed row, e.g. "DatasetNotFoundError".
  // This is the ONLY error field `/v1/activity/pipeline-runs` returns; there
  // is no `error` key, which is why `EnrichmentRun.failure_reason` read null
  // for every failed burst until this was modelled.
  error_class: string | null;
  // Independently nullable: null means "not measured", 0 means "measured
  // zero" — never conflate the two when summing or displaying these. Rows
  // from a pod predating operation records omit them entirely; see
  // normalizeActivityRows, which is where that becomes an explicit null.
  tokens_in: number | null;
  tokens_out: number | null;
  started_at: string | null;
  ended_at: string | null;
  user_id: string | null;
  // Every session the row belongs to, from the pod's `session_ids` list. A
  // pod from before cognee PR 6191 sends one `session_id` instead, which
  // normalizeActivityRows folds into a one-element list. Empty for most
  // rows; one entry for a session-bound search or recall; several for an
  // improve over several sessions, and for every pipeline such a call
  // launches.
  session_ids: string[];
  parent_operation_id: string | null;
  background: boolean | null;
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
  // Nullable for the same reason PipelineRun's are, and normalized the same
  // way: null is "not measured", 0 is "measured zero". These were typed
  // `number` while nothing validated them at the wire, so `|| 0` turned an
  // absent column into a session that read as costing nothing.
  tokens_in: number | null;
  tokens_out: number | null;
  cost_usd: number | null;
  error_count: number;
  last_model: string | null;
}

export interface SessionsPage {
  sessions: SessionRow[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
  /** Set when this empty page is a swallowed failure rather than a real
   *  answer — see FAILED_PAGE. Absent on every successful response. */
  failed?: boolean;
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

/**
 * The same empty page, marked as a swallow rather than a real answer.
 *
 * `listSessions` deliberately never rejects, because most callers want the
 * leniency. That made a 500 on /v1/sessions indistinguishable from a tenant
 * with no sessions, so a surface rendering both halves of the activity feed
 * showed the run rows only with no banner and no hint. Callers that care read
 * `failed`; the rest keep ignoring it.
 */
const FAILED_PAGE: SessionsPage = { ...EMPTY_PAGE, failed: true };

// Cache per instance identity so we probe at most once per page load — but
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

/**
 * Whether this pod answers /v1/sessions at all.
 *
 * Deliberately takes NO per-caller options. The promise is cached at module
 * scope and shared by every caller, so anything caller-specific inside it
 * leaks: threading one caller's AbortSignal through here meant that when
 * react-query cancelled that caller's query on unmount — which it does on
 * last-observer removal, i.e. navigating away mid-probe — the shared promise
 * rejected, the negative result was cached for 30s, and Search, Sessions,
 * Integrations, onboarding detection, the dashboard and getSessionStats all
 * read "sessions unavailable" for half a minute off one navigation.
 *
 * The caller's timeout is out for the same reason plus one more: caller B
 * would wait on a promise built with caller A's budget. So this runs on the
 * client's central GET timeout, per .claude/rules/api.md.
 *
 * That leaves the probe stricter than the 25s call it guards, which is the
 * original complaint. It is survivable now in a way it was not before: a
 * failed probe returns FAILED_PAGE, so a slow pod surfaces as "sessions
 * couldn't be loaded" rather than as an empty workspace. Reporting the wrong
 * thing was the bug; being conservative about a slow pod is not.
 */
function isSessionsAvailable(instance: CogneeInstance): Promise<boolean> {
  // instanceId, not a `baseUrl` the pod client does not set: that fell back to
  // the literal "default" for every instance, so one tenant probing false
  // suppressed the sessions half for every other tenant in the same tab —
  // exactly what COG-6233 documents this id to prevent.
  const key = instance.instanceId ?? instance.name;
  const cached = _sessionsProbe.get(key);
  if (cached && cached.expiresAt > Date.now()) return cached.promise;
  // Same 25s ceiling as everything downstream of this probe: on the client's
  // 10s GET default a merely busy pod fails the probe and is branded
  // unavailable for the cooldown, which reads as "no sessions".
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
function isSessionsPage(value: unknown): value is { sessions: unknown[] } & Omit<SessionsPage, "sessions"> {
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
  // FAILED_PAGE, not EMPTY_PAGE. A failed probe is a failure, and answering
  // with a page that carries no `failed` flag made every caller read "the
  // sessions endpoint is unreachable" as "this workspace has no sessions" —
  // the one outcome FAILED_PAGE was introduced to prevent. It was also the
  // likelier of the two failures, since the probe runs first and, until now,
  // on a shorter timeout than the call it guards.
  if (!(await isSessionsAvailable(instance))) return FAILED_PAGE;
  const q = new URLSearchParams();
  if (params.range) q.set("range", params.range);
  if (params.limit !== undefined) q.set("limit", String(params.limit));
  if (params.offset !== undefined) q.set("offset", String(params.offset));
  if (params.status) q.set("status", params.status);
  const fetchInit: RequestInit & { timeoutMs?: number } = { signal: opts.signal, timeoutMs: opts.timeoutMs };
  return instance
    .fetch(`/v1/sessions?${q.toString()}`, fetchInit)
    // FAILED_PAGE survives the shape guard below and the spread that follows
    // it, so a non-200 stays marked as a failure without a second branch.
    .then((r) => (r.ok ? (r.json() as Promise<unknown>) : FAILED_PAGE))
    .then((page: unknown) => {
      if (!isSessionsPage(page)) {
        console.warn("[getSessions] listSessions got a 200 with an unexpected shape, returning empty page");
        return FAILED_PAGE;
      }
      return {
        ...page,
        // Normalized before anything reads a field: `isSessionsPage` proves
        // only that `sessions` is an array, so every field behind it arrived
        // unchecked. A row that isn't an object at all (null, a number, a
        // proxy's error envelope) is dropped there rather than costing the
        // whole page.
        sessions: normalizeSessionRows(page.sessions).map((s) => ({ ...s, last_model: displayModel(s.last_model) })),
      };
    })
    .catch((err) => {
      if (opts.signal?.aborted || (err instanceof Error && err.name === "AbortError")) throw err;
      console.warn("[getSessions] listSessions failed, returning empty page:", err instanceof Error ? err.message : err);
      return FAILED_PAGE;
    });
}

// The pod pages sessions by last_activity_at before anyone can filter them,
// and it has no session_id filter — so asking for one page and keeping the
// search-UI rows finds nothing on any account with agent traffic, because
// MCP/CLI sessions fill the page first. Instead scan pages until enough
// search-UI sessions are found, bounded so a huge account can't spin forever.
//
// Page size is a compromise: bigger pages mean fewer round trips but a heavier
// payload, and range "all" drops the indexed last_activity_at predicate so the
// pod runs an unfiltered COUNT(*) per page. 200 keeps the common case at one
// request without asking for a megabyte of rows to throw away.
const SESSION_SCAN_PAGE_SIZE = 200;
const MAX_SESSION_SCAN_PAGES = 5;
// The client's 10s GET default is below this pod's documented single-request
// latency under load, and a timeout here reads as "no conversations". The two
// heaviest comparable callers raise it the same way.
const SESSION_SCAN_TIMEOUT_MS = 25_000;

export interface SearchSessionsResult {
  sessions: SessionRow[];
  // False when a page failed or the scan hit its bound with more to read, so
  // the caller can say "couldn't load" instead of asserting "none exist".
  complete: boolean;
}

// A 200 whose rows are missing a string session_id would throw inside the
// prefix filter, outside listSessions' own catch — real, and previously
// untested.
//
// The user_id check is the other half: the pod admits a session on
// `user_id IN visible_ids OR dataset_id IN permitted_dataset_ids`, and
// UI-created datasets carry tenant-wide read, so the prefix alone lets a
// teammate's conversation into your sidebar — and a follow-up would then be
// written under their session id.
function isOwnSearchSession(row: SessionRow, userId: string): boolean {
  return (
    typeof row?.session_id === "string" &&
    row.session_id.startsWith(SEARCH_SESSION_PREFIX) &&
    row.user_id === userId
  );
}

/**
 * Sessions this user started from the Search page, newest first.
 *
 * Scans the whole history rather than the last 30 days: a conversation the
 * user can still see in the sidebar must not vanish because they didn't search
 * for a month.
 *
 * Pass a signal so navigating away mid-scan stops the remaining pages instead
 * of leaving up to five 200-row requests running against the pod.
 */
export async function listSearchSessions(
  instance: CogneeInstance,
  limit: number,
  userId: string,
  signal?: AbortSignal,
): Promise<SearchSessionsResult> {
  const found: SessionRow[] = [];
  for (let page = 0; page < MAX_SESSION_SCAN_PAGES; page++) {
    // Checked before each page as well as passed down: an abort between pages
    // must not start the next request.
    if (signal?.aborted) return { sessions: found.slice(0, limit), complete: false };
    const result = await listSessions(
      instance,
      { range: "all", limit: SESSION_SCAN_PAGE_SIZE, offset: page * SESSION_SCAN_PAGE_SIZE },
      { timeoutMs: SESSION_SCAN_TIMEOUT_MS, signal },
    );
    // A swallowed failure carries has_more: false, so without this the loop
    // would read a 502 as "that was the last page" and report a confident
    // empty sidebar.
    if (result.failed) return { sessions: found.slice(0, limit), complete: false };
    found.push(...result.sessions.filter((row) => isOwnSearchSession(row, userId)));
    if (found.length >= limit) return { sessions: found.slice(0, limit), complete: true };
    if (!result.has_more) return { sessions: found, complete: true };
  }
  // Ran out of pages with the list still unexhausted.
  return { sessions: found.slice(0, limit), complete: false };
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
  // Error of the newest errored sub-run, only rendered when the whole burst
  // failed. Carries `error_class` (the exception class name), which is the
  // only error field the endpoint returns; it read null for every burst while
  // this looked for an `error` key the endpoint has never sent.
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
  pipeline_name?: string | null;
  status?: string;
  dataset_id?: string | null;
  dataset_name?: string | null;
  created_at?: string | null;
  pipeline_run_id?: string | null;
  error_class?: string | null;
  // Never sent by the current endpoint. Kept because a pod could start
  // surfacing a run_info error under this name, and it costs nothing to
  // prefer it when present.
  error?: string | null;
  // Raw session fields, read through rowSessionIds: the list from cognee PR
  // 6191, or the single id a pod from before it sends.
  session_ids?: unknown;
  session_id?: unknown;
}

/** The three states the graph-enrichment card renders. This was the whole of
 *  `RunStatus` until the runs feed gained states that card cannot express -
 *  see enrichmentBurstStatus for how they fold into these three. */
export type EnrichmentStatus = EnrichmentRun["status"];

/**
 * What a pipeline row's `status` column says, in the feed's vocabulary.
 *
 * Five states, not three. The pod's stored `PipelineRunStatus` has four
 * members (INITIATED / STARTED / COMPLETED / ERRORED) and
 * `/v1/activity/pipeline-runs` adds a fifth computed at read time: a row
 * stuck at STARTED for longer than `PIPELINE_RUN_ABANDON_AFTER_SECONDS`
 * (default 30 min) is reported as `"ABANDONED"` (cognee SDK-591).
 *
 * Both of those used to fall through the else branch and read as "running":
 * an INITIATED row has no `started_at` at all, and an ABANDONED row is one
 * whose worker died. The
 * feed showed each as in flight forever, with the duration cell counting up
 * from a start that never happened, which is what made SDK-591's fix
 * invisible in the UI. `"running"` stays the fallback for a status this
 * frontend has not been taught, since that is the one state a live pod can
 * legitimately report with a value we do not recognise yet.
 *
 * INITIATED rows are neither rare nor transient, which is worth knowing before
 * reading a feed full of them: cognee's `add` resets the dataset's pipeline
 * status on every call (add.py calls reset_dataset_pipeline_run_status for
 * add_pipeline and cognify_pipeline, forget.py for cognify_pipeline), and each
 * reset writes an INITIATED row through log_pipeline_run_initiated, which mints
 * its own uuid4. No later STARTED or terminal row shares that id, so nothing
 * ever dedupes it away or moves it on. An add therefore leaves nought, one or
 * two permanent queued rows describing work that finished minutes later: the
 * reset inspects only the latest row per pipeline_name and skips a pipeline
 * already sitting at INITIATED, so a quiet dataset accumulates them slower
 * than one row per add. Reporting
 * them as queued is honest about what the row says; it does not make them
 * meaningful. The fix is upstream, in the id that writer mints.
 */
export type RunStatus = EnrichmentStatus | "queued" | "abandoned";
/** `RunStatus` plus the two states only an operation row can be in: "unknown"
 *  (no terminal signal recorded at all) and "accepted" (background work handed
 *  off). See activityRowStatus. */
export type ActivityRowStatus = RunStatus | "unknown" | "accepted";

export function runStatus(raw: string | undefined): RunStatus {
  const s = raw ?? "";
  if (s.includes("COMPLETED")) return "completed";
  if (s.includes("ERRORED")) return "failed";
  // Not a stored enum member, so it carries no DATASET_PROCESSING_ prefix.
  if (s.includes("ABANDONED")) return "abandoned";
  if (s.includes("INITIATED")) return "queued";
  return "running";
}

/**
 * `runStatus` narrowed to the three states the enrichment card renders.
 *
 * That card coalesces memify runs into bursts and distinguishes only
 * running / completed / failed, so the two new states need a home: a queued
 * run has not started yet, and an abandoned one is a stage that will never
 * finish. Folding abandoned into "running" instead would leave the card
 * claiming enrichment is in progress forever, which is the same bug this
 * change removes from the activity table.
 *
 * The queued arm has no producer today and is here for exhaustiveness over
 * `runStatus`'s return type, not as evidence such rows occur: the only writer
 * of INITIATED rows is a dataset reset, and that is only ever called for
 * add_pipeline and cognify_pipeline, never for memify_pipeline.
 */
function enrichmentBurstStatus(raw: string | undefined): EnrichmentStatus {
  const status = runStatus(raw);
  if (status === "queued") return "running";
  if (status === "abandoned") return "failed";
  return status;
}

// The activity endpoint's operation-record fields (`kind`, `operation_name`,
// `outcome`, `tokens_in`/`tokens_out`, `session_ids`, `parent_operation_id`)
// rolled out to the pod after this frontend shipped. An older pod OMITS them
// rather than sending null, so `PipelineRun` declaring them non-optional is a
// contract the wire does not always honour — and `x === null` does not catch
// an absent key, which is how an unmeasured row renders a fabricated 0.
//
// Normalize once, here, at the boundary: every consumer downstream then only
// ever sees the modern shape, with `null` meaning "not measured" and 0 meaning
// "measured zero".
//
// `kind` is DERIVED, not defaulted. cognee has written operation rows since
// v1.5.0.dev3 but only started returning `kind` in 1.5.3.dev1, so a pod that
// omits the field is still sending operation rows and defaulting them to
// "pipeline" makes a `forget` render as "remember". Mirror the server's own
// derivation (`get_activity_router.py`: pipeline if pipeline_name is not null,
// else operation) and reject anything outside the union, so a third kind the
// backend adds later cannot silently land in the pipeline arm.
function activityRowKind(r: Partial<PipelineRun>): PipelineRun["kind"] {
  if (r.kind === "operation" || r.kind === "pipeline") return r.kind;
  // `== null`, not `=== null`: a pod that omits `kind` omits nothing else
  // about pipeline_name, which is null on exactly the operation rows.
  return r.pipeline_name == null ? "operation" : "pipeline";
}

/** A number, or null for anything that is not one. Absent, a string and NaN
 *  all mean the same thing here: not measured. */
function finiteOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/**
 * Total tokens for a row, or null when the row does not say.
 *
 * Null if EITHER side is missing, not only if both are. The column is a total,
 * and a total that silently omits one measured-but-absent half understates the
 * row while claiming to be complete: `tokens_in: 0, tokens_out: null` rendered
 * as a confident `0`. "Does not say" is the honest reading, and it is the same
 * rule `durationOf` already applies to a run with no recorded end.
 */
export function sumTokens(tokensIn: number | null, tokensOut: number | null): number | null {
  if (tokensIn === null || tokensOut === null) return null;
  return tokensIn + tokensOut;
}

/**
 * Session rows, field-normalized.
 *
 * `isSessionsPage` only ever checked that `sessions` is an array, so every
 * field below arrived unvalidated behind a non-nullable type. That is what let
 * an absent token column reach `(s.tokens_in || 0)` and render as a measured
 * zero, and it is the same boundary hole `normalizeActivityRows` was written
 * to close for the activity endpoint.
 */
export function normalizeSessionRows(rows: unknown[]): SessionRow[] {
  const out: SessionRow[] = [];
  for (const raw of rows) {
    if (raw === null || typeof raw !== "object") continue;
    const r = raw as Partial<SessionRow>;
    out.push({
      session_id: typeof r.session_id === "string" ? r.session_id : "",
      user_id: typeof r.user_id === "string" ? r.user_id : "",
      dataset_id: r.dataset_id ?? null,
      status: typeof r.status === "string" ? r.status : "",
      effective_status: typeof r.effective_status === "string" ? r.effective_status : "",
      started_at: r.started_at ?? null,
      last_activity_at: r.last_activity_at ?? null,
      ended_at: r.ended_at ?? null,
      tokens_in: finiteOrNull(r.tokens_in),
      tokens_out: finiteOrNull(r.tokens_out),
      cost_usd: finiteOrNull(r.cost_usd),
      error_count: finiteOrNull(r.error_count) ?? 0,
      last_model: r.last_model ?? null,
    });
  }
  return out;
}

/**
 * The sessions a raw activity row names.
 *
 * cognee PR 6191 lists them in `session_ids`. A pod from before it sends one
 * `session_id` (null for most rows) and no list, which reads as a list of
 * one. Anything else is "no sessions": a row never half-carries them.
 */
export function rowSessionIds(r: { session_ids?: unknown; session_id?: unknown }): string[] {
  if (Array.isArray(r.session_ids)) {
    return r.session_ids.filter((id): id is string => typeof id === "string" && id !== "");
  }
  return typeof r.session_id === "string" && r.session_id !== "" ? [r.session_id] : [];
}

export function normalizeActivityRows(payload: unknown): PipelineRun[] {
  // A non-array 200 is the reachable failure on this endpoint — a gateway or
  // a mid-deploy pod answers with an error envelope, and quietly coercing that
  // to [] is what makes a broken endpoint read as an empty workspace. The
  // client already rejects any non-ok status before this runs.
  if (!Array.isArray(payload)) {
    throw new Error("activity/pipeline-runs returned a non-array payload");
  }
  const rows: PipelineRun[] = [];
  for (const raw of payload) {
    if (raw === null || typeof raw !== "object") continue;
    const r = raw as Partial<PipelineRun>;
    rows.push({
      id: r.id ?? "",
      pipeline_name: r.pipeline_name ?? null,
      // typeof, not `?? ""`: the field is `string | null` from the documented
      // producer, but `.includes` three call sites deep throws on any other
      // JSON type and takes the whole page down rather than one row, and
      // normalizeSessionRows already guards its own status this way.
      status: typeof r.status === "string" ? r.status : "",
      dataset_id: r.dataset_id ?? null,
      dataset_name: r.dataset_name ?? null,
      owner_id: r.owner_id ?? null,
      owner_email: r.owner_email ?? null,
      created_at: r.created_at ?? null,
      pipeline_run_id: r.pipeline_run_id ?? null,
      kind: activityRowKind(r),
      operation_name: r.operation_name ?? null,
      origin: r.origin ?? null,
      outcome: r.outcome ?? null,
      error_class: typeof r.error_class === "string" ? r.error_class : null,
      // Coerced, not just defaulted. `?? null` let a non-number through a
      // boundary whose stated job is normalization, and a string then survived
      // into `(tokens_in ?? 0) + (tokens_out ?? 0)` as concatenation: a footer
      // reading "0100001" and a cost off by an order of magnitude, all without
      // throwing, because the field is typed `number | null`.
      tokens_in: finiteOrNull(r.tokens_in),
      tokens_out: finiteOrNull(r.tokens_out),
      started_at: r.started_at ?? null,
      ended_at: r.ended_at ?? null,
      user_id: r.user_id ?? null,
      session_ids: rowSessionIds(raw as Record<string, unknown>),
      parent_operation_id: r.parent_operation_id ?? null,
      background: r.background ?? null,
    });
  }
  return rows;
}

/**
 * Session ids that already have an operation row in this feed.
 *
 * Recall/search used to be visible only as a session row; on a pod that emits
 * operation records the same event also arrives as an operation row, and
 * rendering both double-counts it. Suppressing session rows wholesale the
 * moment a pod *can* emit operations is wrong in the other direction: rows
 * written before SDK-399 were never backfilled, so a pod upgrade would erase
 * that history from the table. Match per row instead — a session row is
 * redundant only when an operation row actually references it.
 *
 * An operation over several sessions (an improve given two) lists every one
 * of them (cognee PR 6191), so each of those session rows is covered too. A
 * pod from before that lists at most one, and the rest keep their own rows.
 */
export function sessionIdsWithOperationRows(runs: PipelineRun[]): Set<string> {
  const ids = new Set<string>();
  for (const r of runs) {
    if (r.kind !== "operation") continue;
    for (const id of r.session_ids) ids.add(id);
  }
  return ids;
}

/**
 * Terminal state of an activity row.
 *
 * Operation rows are written once, from `record_operation`'s `finally`, with
 * `status` left null by construction — so `runStatus` alone renders every
 * completed or failed operation as a permanent "running". `outcome`
 * ("succeeded"/"failed") is the real signal for those rows. `background: true`
 * with `outcome: "succeeded"` means the work was accepted and started, not
 * finished, so it reports "accepted".
 *
 * "accepted" rather than "running", which is what this returned until it was
 * found to be a promise the data never keeps: the table is append-only and no
 * writer ever revisits an operation row, so every background `remember`
 * (cognee calls `set_background(True)` whenever it launches background work)
 * sat at "running" forever. The launched work is separately visible as the
 * pipeline rows it produced, each with its own status, so nothing is lost by
 * this row admitting it only knows the hand-off succeeded.
 *
 * "unknown" covers the shape a released pod actually returns: an operation row
 * with neither `status` nor `outcome`, because the router omits `outcome` in
 * the same versions that omit `kind`. Calling a finished `forget` "running"
 * forever is a worse lie than admitting the row does not say.
 */
export function activityRowStatus(row: PipelineRun): ActivityRowStatus {
  if (row.outcome === "failed") return "failed";
  if (row.outcome === "succeeded") return row.background === true ? "accepted" : "completed";
  if (row.kind === "operation" && !row.status) return "unknown";
  return runStatus(row.status);
}

/**
 * Whether this row's tokens may be added to a total.
 *
 * A parent's token columns already include every nested child's spend
 * (cognee's `UsageAccumulator.add` recurses into `self.parent`), and the
 * endpoint's own docstring says to sum one level only.
 *
 * This is a summation rule, not a visibility rule, and the difference matters:
 * pipeline runs carry `parent_operation_id` as well (`log_pipeline_run_complete`
 * parents a cognify run to the memify run it ran inside), so filtering the row
 * list on it would hide real nested runs from the activity log rather than
 * just from its total.
 *
 * `== null`, not `=== null`: a pod that predates operation records omits the
 * field entirely, and treating that as "has a parent" would zero every total.
 */
export function countsTowardTotals(row: Pick<PipelineRun, "parent_operation_id">): boolean {
  return row.parent_operation_id == null;
}

/**
 * A set-level version of the rule above, for an aggregator that sums a whole
 * feed rather than rendering one row at a time.
 *
 * `countsTowardTotals` alone is only safe when every row's declared parent is
 * guaranteed to also be in the same sum — true for the activity table, which
 * promotes an orphaned child (parent absent, filtered out, or off the page)
 * to depth 0 so its tokens land somewhere. An aggregator that instead drops
 * every row with ANY parent id, present or not, undercounts by exactly that
 * child's tokens whenever its parent didn't survive into the same feed —
 * ordinary on a pod that caps the runs feed at 50 rows, not an edge case.
 *
 * `present` is the set of `pipeline_run_id`s actually in the rows being
 * summed. A row counts unless it declares a parent AND that parent is one of
 * them — the same "is the parent actually here" test the activity table's own
 * tree walk performs via `depth`, expressed for a caller that has no tree.
 */
export function countsTowardTotalsInFeed(row: Pick<PipelineRun, "parent_operation_id">, present: ReadonlySet<string>): boolean {
  return row.parent_operation_id == null || !present.has(row.parent_operation_id);
}

/** `present` set for `countsTowardTotalsInFeed`: every `pipeline_run_id` this
 *  feed actually carries. */
export function presentRunIds(rows: readonly Pick<PipelineRun, "pipeline_run_id">[]): Set<string> {
  const ids = new Set<string>();
  for (const r of rows) {
    if (r.pipeline_run_id) ids.add(r.pipeline_run_id);
  }
  return ids;
}

// Naive ISO timestamps from the pod are UTC.
function isoToMs(iso: string | null | undefined): number {
  if (!iso) return 0;
  const hasTz = /Z$|[+-]\d{2}:?\d{2}$/.test(iso);
  return Date.parse(hasTz ? iso : iso + "Z") || 0;
}

/**
 * How final a row's status is, for picking one row per `pipeline_run_id`.
 *
 * A rank rather than the terminal/not-terminal boolean this replaced, and the
 * reason is narrower than it looks. Not the straggler-COMPLETED case: a worker
 * that finishes past the abandon threshold writes a COMPLETED row sharing the
 * id, and the old boolean already preferred it, because it read the unmapped
 * "ABANDONED" as non-terminal. What the rank actually buys is correctness once
 * "queued" and "abandoned" exist as mapped values, where a plain
 * `!== "running"` test would read BOTH as terminal:
 *
 * - an INITIATED row would beat the STARTED row it precedes,
 * - and an ABANDONED row would beat the COMPLETED row that supersedes it
 *   whenever the abandoned verdict is the newer of the two.
 *
 * Both of those pairs need the row that is NOT terminal to be the NEWER of the
 * two, and that ordering is what makes them legacy-only. A current pod mints a
 * fresh uuid4 per execution, so the only two rows sharing an id are one run's
 * STARTED row and its own terminal row, and a run always starts before it
 * finishes: the abandoned verdict sits on the older row and both the rank and
 * the boolean pick the terminal one. A pod that derived the id from
 * pipeline_id + dataset_id (cognee before eb841afd9, 2026-05-11) shared it
 * across every run of that pipeline on that dataset, so run N's stuck row is
 * newer than run N-1's completion and the ordering inverts.
 *
 * Enumerated rather than argued, because reasoning about it got the answer
 * wrong twice: over all four reachable pairs the rank and the boolean agree
 * except on those two legacy orderings. It is legacy-data correctness, not a
 * live fix, and the test above pins the one pair that distinguishes rank 1
 * from rank 2.
 *
 * That expires with SDK-595. Once the reset writer takes the run id instead of
 * minting one, an INITIATED row shares an id with the run it resets on every
 * pod, so the first bullet stops being a legacy curiosity and becomes the
 * mainline case. CLO-756 tracks rewriting this block then.
 */
function statusFinality(status: string | undefined): number {
  const s = runStatus(status);
  if (s === "completed" || s === "failed") return 2;
  if (s === "abandoned") return 1;
  return 0; // running, queued
}

/**
 * One row per `pipeline_run_id`: the most final one, newest breaking a tie.
 *
 * A pipeline run emits a STARTED row and a terminal row under one id, and an
 * operation writes a single row, so this collapses the feed to one row per
 * thing that happened. See `statusFinality` for which row wins and why that is
 * a rank rather than a terminal/not-terminal test.
 *
 * Generic so a richer caller-side shape (e.g. PipelineRun, which adds
 * kind/tokens/operation_name on top of these fields) survives the round trip
 * instead of being narrowed to ActivityRun.
 */
export function dedupeByRunId<T extends ActivityRun>(rows: T[]): T[] {
  const byRun = new Map<string, T>();
  for (const row of rows) {
    const key = row.pipeline_run_id || row.id || String(byRun.size);
    const prev = byRun.get(key);
    if (!prev) { byRun.set(key, row); continue; }
    const prevRank = statusFinality(prev.status);
    const rowRank = statusFinality(row.status);
    if (rowRank > prevRank || (rowRank === prevRank && isoToMs(row.created_at) > isoToMs(prev.created_at))) {
      byRun.set(key, row);
    }
  }
  return [...byRun.values()];
}

// Newest burst first.
/** Page size for the enrichment card. Bounded, because this only ever needs
 *  enough rows to coalesce the recent bursts it renders. */
const ENRICHMENT_FETCH_LIMIT = 200;

export async function getGraphEnrichmentRuns(
  instance: CogneeInstance,
  datasetId: string,
  sessionId: string | null = null,
): Promise<EnrichmentRun[]> {
  try {
    // Filter server-side. This used to fetch the endpoint's default page and
    // filter in the browser, which was fine while that table held only
    // pipeline runs — it now also holds one row per recall/search, so on a
    // chatty tenant every memify row was crowded off the page and the card
    // said "No graph enrichment yet" while enrichment was running.
    // `pipeline_name` also excludes operation records server-side, which is
    // exactly what this caller wants.
    const params = new URLSearchParams({
      pipeline_name: ENRICHMENT_PIPELINE,
      dataset_id: datasetId,
      limit: String(ENRICHMENT_FETCH_LIMIT),
    });
    // The session page wants the improve runs that bridged ITS session, not
    // every improve on the dataset (CLO-1105). The pod stamps those runs
    // since cognee PR 6160 and filters on the stamp.
    if (sessionId) params.set("session_id", sessionId);
    const r = await instance.fetch(`/v1/activity/pipeline-runs?${params}`);
    if (!r.ok) return [];
    const data: unknown = await r.json();
    // Still filtered client-side: a pod that predates these query params
    // ignores them and answers with the unfiltered newest page.
    const datasetRows = (Array.isArray(data) ? (data as ActivityRun[]) : [])
      .filter((run) => run.dataset_id === datasetId && run.pipeline_name === ENRICHMENT_PIPELINE);
    // A pod from before the stamp ignores `session_id` and names no session on
    // any memify row. Keep the dataset-wide view there, as the card showed
    // before, rather than nothing; once any row is stamped, another session's
    // run is dropped even where the pod did not filter.
    const anyStamped = datasetRows.some((run) => rowSessionIds(run).length > 0);
    const sessionRows = sessionId && anyStamped
      ? datasetRows.filter((run) => rowSessionIds(run).includes(sessionId))
      : datasetRows;
    const rows = dedupeByRunId(sessionRows).sort((a, b) => isoToMs(b.created_at) - isoToMs(a.created_at));

    const bursts: EnrichmentRun[] = [];
    let prevTs = 0;
    for (const row of rows) {
      const ts = isoToMs(row.created_at);
      const status = enrichmentBurstStatus(row.status);
      const current = bursts[bursts.length - 1];
      if (current && prevTs - ts <= ENRICHMENT_COALESCE_MS) {
        current.count += 1;
        if (status === "failed") {
          current.error_count += 1;
          current.failure_reason = current.failure_reason ?? row.error ?? row.error_class ?? null;
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
          failure_reason: status === "failed" ? (row.error ?? row.error_class ?? null) : null,
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
    // Same pod, same load as the session scan: on the client's 10s GET default
    // a slow detail silently becomes a conversation missing from the sidebar.
    .fetch(`/v1/sessions/${encodeURIComponent(sessionId)}`, { timeoutMs: SESSION_SCAN_TIMEOUT_MS })
    .then((r) => (r.ok ? r.json() : null))
    .then((detail: SessionDetail | null) =>
      detail ? { ...detail, last_model: displayModel(detail.last_model) } : null,
    )
    .catch((err) => {
      console.warn("[getSessions] getSessionDetail failed:", err instanceof Error ? err.message : err);
      return null;
    });
}
