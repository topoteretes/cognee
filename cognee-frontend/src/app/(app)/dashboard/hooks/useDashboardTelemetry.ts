"use client";

import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { notifications } from "@mantine/notifications";
import { useCogniInstance, useTenant } from "@/modules/tenant/TenantProvider";
import { useFilter } from "@/ui/layout/FilterContext";
import { listSessions, normalizeActivityRows } from "@/modules/sessions/getSessions";
import type { PipelineRun, SessionRow, TimeRange } from "@/modules/sessions/getSessions";
import getAgentConnections from "@/modules/agents/getAgentConnections";
import type { AgentConnection } from "@/modules/agents/getAgentConnections";
import { useCircuitBreaker } from "@/modules/query/useCircuitBreaker";

const TELEMETRY_POLL_INTERVAL_MS = 15_000;
// Background polls get extra headroom because local/loaded pods can take several
// seconds per request (COG-5722) — we don't want a slow pod surfacing as a false
// error while it's still working.
const BACKGROUND_POLL_TIMEOUT_MS = 25_000;
// CostPanel and the Analytics breakdown *aggregate* this feed across the whole
// selected range, so a page-size cap silently truncates their totals rather
// than just their lists, and operation rows now compete with pipeline runs for
// the same slots. 500 is the documented maximum.
//
// It is inert on every released cognee up to and including v1.5.3, which
// hardcodes LIMIT 50 with no `limit` parameter — FastAPI ignores an unknown
// query param, so today all three callers cap at 50. A range busier than 500
// would still need server-side aggregation, which the endpoint does not offer
// (it returns a bare array with no total).
const RUNS_LIMIT = 500;
const SESSIONS_LIMIT = 50;
// Same reasoning as RUNS_LIMIT: `connections` answers "has this agent EVER
// registered" for the sticky registry, and the endpoint's own default (50,
// `le=500`) silently truncated that to the newest 50 registrations. On a
// workspace with more churn than that, an agent's one-and-only registration
// falls off the page and useConnectedIntegrations reads it as never connected
// — the registry has no other way to find out. This is the endpoint's
// documented maximum, not a page we then loop past; see connectionsTruncated
// below for what happens on a tenant busy enough to still overflow it.
const CONNECTIONS_LIMIT = 500;
/** The widest window the endpoint understands. The page size is the real bound
 *  either way, so asking for less only narrows what the local slices can see. */
const SESSIONS_RANGE: TimeRange = "all";
/**
 * Page size every released cognee up to and including 1.5.3 hardcodes, ignoring
 * the `limit` we send. Verified against the router source: 1.5.3 does
 * `.limit(50)` with no `limit` parameter, and 1.5.4 is the first build to accept
 * `limit: int = Query(50, ge=1, le=500)`.
 *
 * Both numbers matter, because truncation has to be measured against the page
 * the pod actually returned rather than against what we asked for. Comparing
 * `runs.length` to RUNS_LIMIT alone made the flag permanently false on every pod
 * in production, and with it the Analytics "Partial totals" banner and the Cost
 * panel's "at least" qualifier — the two disclosures that exist for exactly this
 * cap. 1.5.4's own docstring states the rule: `len(results) == limit` means
 * another page may exist.
 */
const RUNS_POD_HARDCODED_LIMIT = 50;

export interface DashboardTelemetry {
  runs: PipelineRun[];
  sessions: SessionRow[];
  /** Plugin registrations, polled on the same cadence as runs/sessions so the
   *  sticky per-tenant registry (useConnectedIntegrations) never disagrees
   *  with a tutorial reading the same endpoint about whether an agent has
   *  ever registered. Registration only, not liveness: the pod's connection
   *  `status` sticks at "active" after a crash (nothing ever clears it but
   *  the plugin's own unregister call), so agentStatus's "live" state is
   *  session-only and does not read this. Empty on a failed fetch; there is
   *  no separate `connectionsFailed` flag because nothing here downgrades the
   *  page for it, the registry just falls back to sessions alone. */
  connections: AgentConnection[];
  /** The connections endpoint returned a full page (has_more), so a
   *  registration older than the newest CONNECTIONS_LIMIT may not be in
   *  `connections`. Absent evidence, not negative evidence: nothing here (or
   *  in useConnectedIntegrations, whose sticky registry answers the same
   *  question) infers "never connected" from a row's absence on a truncated
   *  page. Exposed for a caller that needs to say so rather than invent copy
   *  no design has asked for yet. */
  connectionsTruncated: boolean;
  loading: boolean;
  /** The fetch failed. Exposed because an empty runs/sessions after a failure
   *  is indistinguishable from an empty workspace, and every panel that reads
   *  this feed was reporting the second when it meant the first. */
  error: boolean;
  /** The sessions half was unreachable while the runs half answered.
   *
   *  Separate from `error` because listSessions never rejects — it swallows and
   *  flags — so `isError` is false on this path and a full /v1/sessions outage
   *  read as an empty workspace. That is the exact outcome FAILED_PAGE exists
   *  to prevent, and until now this hook was throwing the flag away. */
  sessionsFailed: boolean;
  /** The mirror image: sessions answered but the runs half did not. Both halves
   *  now degrade the same way — see useActivityFeed, which this now matches. A
   *  runs failure used to discard a perfectly good sessions half via
   *  Promise.all and report a total outage: every memory-graph agent read
   *  "disconnected" though /v1/sessions had just answered. */
  runsFailed: boolean;
  /** The runs endpoint returned a full page, so there is very likely more
   *  activity in this range than these totals account for.
   *
   *  Every consumer of this feed AGGREGATES it, so the page-size cap truncates
   *  their numbers and not just their lists — and until now nothing reported
   *  that, which made a capped total indistinguishable from a complete one. */
  runsTruncated: boolean;
  /** Rows the endpoint actually returned. What the notice should quote: the
   *  limit we request is ignored by every pod before 1.5.4. */
  runsReturned: number;
  /** Same for the sessions half. `sessionsTotal` is null when the pod did not
   *  say how many it holds — older images return a bare array. */
  sessionsTruncated: boolean;
  sessionsShown: number;
  sessionsTotal: number | null;
  /** Epoch ms of the last successful fetch, 0 before the first one. What an
   *  "Updated Xs ago" label reads, so a stale panel can be recognised as one. */
  updatedAt: number;
  /** A refetch over data already on screen. `loading` is only the first load,
   *  so without this a Refresh control can report nothing. */
  refreshing: boolean;
  refetch: () => void;
}

/**
 * The shared activity feed, always at its widest.
 *
 * It took a range, and that parameter was the whole of CLO-722: passing a
 * panel's toggle into it re-scoped every other consumer of the feed, and the
 * Analytics page then reintroduced the same mistake in a new shape, blanking
 * itself to a skeleton on each flip to refetch a payload that cannot differ
 * (the runs endpoint takes no time filter at all).
 *
 * So there is no range to pass. Every consumer slices this locally, one cache
 * entry per tenant serves the whole app, and a toggle flip costs nothing. That
 * is a stronger guarantee than any test of the old call sites could give,
 * because the mistake is no longer expressible.
 */
export function useDashboardTelemetry(): DashboardTelemetry {
  const { cogniInstance, isInitializing } = useCogniInstance();
  const { tenant, tenantReady } = useTenant();
  const { loading: filterLoading } = useFilter();

  const telemetryBreaker = useCircuitBreaker(TELEMETRY_POLL_INTERVAL_MS);

  const telemetryQuery = useQuery({
    queryKey: ["dashboard-telemetry", tenant?.tenant_id ?? null],
    queryFn: async ({ signal }) => {
      if (!cogniInstance) throw new Error("cogniInstance unavailable");
      const init: RequestInit & { timeoutMs?: number } = { signal, timeoutMs: BACKGROUND_POLL_TIMEOUT_MS };
      // allSettled, not all — matching useActivityFeed. A rejected runs half
      // used to take the whole query down via Promise.all and throw away a
      // sessions half that had already answered.
      const [runsResult, sessionsResult, connectionsResult] = await Promise.allSettled([
        // The client rejects on any non-ok status before this resolves; the
        // failure left to guard is a 200 carrying an error envelope, which
        // normalizeActivityRows throws on rather than coercing to [].
        cogniInstance.fetch(`/v1/activity/pipeline-runs?limit=${RUNS_LIMIT}`, init).then((r) => r.json() as Promise<unknown>),
        listSessions(cogniInstance, { range: SESSIONS_RANGE, limit: SESSIONS_LIMIT }, { signal, timeoutMs: BACKGROUND_POLL_TIMEOUT_MS }),
        // range "all" and activeOnly false, not a freshness window: this feed
        // only ever answers "has this agent EVER registered" (for the sticky
        // registry, see the `connections` field above), so a connection that
        // is old or already unregistered still has to count. A narrower range
        // or active-only filter would suit "is it live right now", which is
        // exactly the question this feed does not answer.
        getAgentConnections(
          cogniInstance,
          { range: "all", activeOnly: false, includeSources: false, limit: CONNECTIONS_LIMIT },
          { signal, timeoutMs: BACKGROUND_POLL_TIMEOUT_MS },
        ),
      ]);

      signal.throwIfAborted();

      // Read rather than discarded. normalizeActivityRows throws a named
      // diagnostic for a 200 carrying an error envelope, and react-query holds
      // that rejection in state nothing reads — so the one failure the shared
      // client logs as a SUCCESS was invisible everywhere.
      let runs: PipelineRun[] = [];
      let runsFailed = false;
      if (runsResult.status === "fulfilled") {
        try {
          // Normalized at the boundary: rows from a pod predating operation
          // records omit `kind` and the token columns, and every consumer that
          // compared against `null` or `"pipeline"` silently mishandled them.
          runs = normalizeActivityRows(runsResult.value);
        } catch (e) {
          runsFailed = true;
          console.warn("[useDashboardTelemetry] pipeline-runs returned an unusable payload:", e instanceof Error ? e.message : e);
        }
      } else {
        runsFailed = true;
        console.warn("[useDashboardTelemetry] pipeline-runs failed:", runsResult.reason instanceof Error ? runsResult.reason.message : runsResult.reason);
      }

      const sessionsPage = sessionsResult.status === "fulfilled" ? sessionsResult.value : undefined;
      const sessions = sessionsPage?.sessions ?? [];

      // Purely additive: a failed fetch degrades to an empty list rather
      // than affecting `error`/the circuit breaker, since every reader of
      // this feed already has a session-only fallback for "no connections
      // known".
      if (connectionsResult.status === "rejected") {
        console.warn(
          "[useDashboardTelemetry] connections fetch failed:",
          connectionsResult.reason instanceof Error ? connectionsResult.reason.message : connectionsResult.reason,
        );
      }
      const connections = connectionsResult.status === "fulfilled" ? connectionsResult.value.agents : [];
      // false on a failed fetch, same as every other truncation flag here: a
      // fetch that never answered proves nothing about whether the page it
      // would have returned was full.
      const connectionsTruncated = connectionsResult.status === "fulfilled" && connectionsResult.value.has_more === true;

      // Only a total outage is a query error now. Either half alone degrades
      // to a flag over the data that did arrive, same as useActivityFeed —
      // the circuit breaker below reads `isError`, so a real total outage
      // still has to surface there.
      if (runsFailed && (!sessionsPage || sessionsPage.failed === true)) {
        throw new Error("dashboard telemetry unavailable: both halves failed");
      }

      // `total`/`has_more` are typed as present but only `sessions` is shape-
      // checked at the boundary, so neither can be trusted to be a number —
      // hence the length comparison as the fallback signal rather than
      // has_more alone.
      const sessionsTotal = typeof sessionsPage?.total === "number" ? sessionsPage.total : null;
      return {
        runs,
        sessions,
        connections,
        connectionsTruncated,
        sessionsFailed: !sessionsPage || sessionsPage.failed === true,
        runsFailed,
        // A full page under either regime. `runsReturned` travels with it so
        // the banner can name the page it actually got instead of the ceiling
        // we asked for: saying "the newest 500" over a 50-row page would be a
        // ten-fold overstatement where there is currently only silence.
        runsTruncated: runs.length >= RUNS_LIMIT || runs.length === RUNS_POD_HARDCODED_LIMIT,
        runsReturned: runs.length,
        sessionsShown: sessions.length,
        sessionsTotal,
        // `has_more` and a stated total are evidence. A full page on its own is
        // not: with exactly 50 sessions in range, has_more false and total 50,
        // the length arm fired and the page rendered "the newest 50 of 50
        // sessions … therefore a floor, not the whole history", which refutes
        // itself. The length is only a signal when the pod said nothing.
        sessionsTruncated: sessionsPage?.has_more === true
          || (sessionsTotal !== null && sessionsTotal > sessions.length)
          || (sessionsTotal === null && sessions.length >= SESSIONS_LIMIT),
      };
    },
    // tenantReady, not just cogniInstance: a freshly-created workspace gets a
    // cogniInstance immediately (optimistic), but the pod itself can take up
    // to a minute to actually answer — firing this against it early just
    // produces a burst of failed requests (dashboard now mounts before the
    // pod is ready, since it's excluded from TenantProvider's isInitializing
    // wait; see TenantProvider.tsx).
    enabled: !!cogniInstance && !isInitializing && !filterLoading && tenantReady,
    refetchInterval: telemetryBreaker.refetchInterval,
    refetchIntervalInBackground: false,
    // No per-fetch retry: this is a poll, so the next tick IS the retry, and
    // the circuit breaker above already tracks consecutive tick failures —
    // an extra retry here would just delay that signal.
    retry: false,
  });

  useEffect(() => {
    telemetryBreaker.report(telemetryQuery);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [telemetryQuery.isError, telemetryQuery.isSuccess, telemetryQuery.dataUpdatedAt, telemetryQuery.errorUpdatedAt]);

  // Surface the breaker opening once per episode — not on every failed tick.
  useEffect(() => {
    if (!telemetryBreaker.isOpen) return;
    notifications.show({
      title: "Workspace is having trouble responding",
      message: "We'll keep trying in the background — some data may be out of date.",
      color: "orange",
      autoClose: 8000,
    });
  }, [telemetryBreaker.isOpen]);

  // Read the query directly. There was a useState mirror here, kept on the
  // stated grounds that it "survives a failed poll tick so one bad request
  // doesn't blank a populated dashboard" — react-query already does that, it
  // holds `data` across a failed refetch on the same key. What the mirror
  // actually added was a window where `data` is undefined (any fresh key, on
  // success as much as on failure) and the reset blanked runs/sessions while
  // `loading` had already latched false, so every range flip flashed the
  // panels through their empty states with no spinner. That also fired two
  // false `checklist_item_completed` events per flip from OverviewPage.
  //
  // isPending, not isLoading: it stays true while the query is disabled (no
  // cogniInstance yet), which is what the panels need to keep showing a
  // skeleton rather than an empty state.
  return {
    runs: telemetryQuery.data?.runs ?? [],
    sessions: telemetryQuery.data?.sessions ?? [],
    connections: telemetryQuery.data?.connections ?? [],
    connectionsTruncated: telemetryQuery.data?.connectionsTruncated ?? false,
    loading: telemetryQuery.isPending,
    error: telemetryQuery.isError,
    sessionsFailed: telemetryQuery.data?.sessionsFailed ?? false,
    runsFailed: telemetryQuery.data?.runsFailed ?? false,
    runsTruncated: telemetryQuery.data?.runsTruncated ?? false,
    runsReturned: telemetryQuery.data?.runsReturned ?? 0,
    sessionsTruncated: telemetryQuery.data?.sessionsTruncated ?? false,
    sessionsShown: telemetryQuery.data?.sessionsShown ?? 0,
    sessionsTotal: telemetryQuery.data?.sessionsTotal ?? null,
    updatedAt: telemetryQuery.dataUpdatedAt,
    // isPending excluded: the very first load is `loading`, and a control that
    // says "Refreshing…" before anything has ever arrived is describing the
    // skeleton next to it.
    refreshing: telemetryQuery.isFetching && !telemetryQuery.isPending,
    refetch: telemetryQuery.refetch,
  };
}
