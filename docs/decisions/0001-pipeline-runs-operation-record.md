# ADR 0001: pipeline_runs as the durable record of all cognee operations

- Status: Accepted
- Ticket: SDK-399
- Related cloud tickets: CLO-395 (activity_log), CLO-451, COG-6073 (tenant_logs)
- Related debt tickets: COG-5359 / COG-6089 (unbounded run_info growth)
- Explicitly out of scope: cloud activity_log/tenant_logs, embedding usage capture
  (CLO-450), cost/pricing calculation (CLO-449), retention policy, frontend.

## Context

Until now, `pipeline_runs` recorded only the four pipeline-backed operations
(add, cognify, memify, improve) as 2-4 append-only rows per run sharing a
`pipeline_run_id`, with a dataset-shaped status enum
(`DATASET_PROCESSING_*`), no triggering user, no end time, no token spend,
and no generic outcome. The non-pipeline operations — forget, recall,
search, remember (session path), delete, prune — left no durable record at
all. The cloud side is building activity/tenant logging (CLO-395,
CLO-451, COG-6073), which raised the question of where the system of
record lives.

## Decision 1: OSS `pipeline_runs` is the system of record

The OSS `pipeline_runs` table is the durable, queryable record of ALL
cognee operations — pipeline-backed and non-pipeline alike. Cloud
`activity_log` and `tenant_logs` are downstream, cloud-side projections:
they may consume, enrich, or mirror `pipeline_runs`, but they never
substitute for it, and OSS features must never depend on them. Anything
the cloud needs about "what ran, who ran it, did it succeed, what did it
cost in tokens" must be answerable from `pipeline_runs` alone.

## Decision 2: append-only rows, self-contained terminal row

We keep the append-only write pattern and reject one-mutable-row-per-run.

Rationale:
- Every existing "current status" consumer derives state from the latest
  row per partition (`row_number() OVER (... ORDER BY created_at DESC)`):
  `get_pipeline_status`, `check_pipeline_run_qualification` (the
  pipeline-cache behavioral gate), `get_pipeline_run(s)_by_dataset`,
  startup stale-run recovery, and the activity feed. All of them keep
  working unchanged with append-only inserts; recovery additionally
  depends on the STARTED row's immutable `created_at` as the run start
  time. A mutable row would require rewriting every writer and revisiting
  recovery semantics for no consumer benefit.
- The single-record requirement (start and end readable without a
  self-join) is met by making the TERMINAL row (COMPLETED / ERRORED)
  self-contained: it carries `started_at` (propagated in memory from the
  run start), `ended_at`, `user_id`, `tenant_id`, `operation_name`,
  `outcome`, `error_class`, `tokens_in`, `tokens_out`.
- Non-pipeline operations write exactly ONE row, at operation end, so
  they are single-record by construction. Consequence (accepted): an
  operation that dies mid-flight (process kill) leaves no row; pipeline
  ops retain their STARTED row for that case.

## Supporting decisions

1. **Status vs outcome.** `PipelineRunStatus` (`DATASET_PROCESSING_*`) is
   frozen: it is a public API response model (`/datasets/status`) and a
   native Postgres enum type. Operation-level result lives in a new
   nullable String column `outcome` ("succeeded" / "failed") plus
   `error_class` (exception class name). Non-pipeline rows have
   `status = NULL` and `pipeline_name = NULL`, which makes them invisible
   to all legacy status readers by their existing filters.
2. **User attribution.** `user_id` and `tenant_id` are typed, indexed,
   nullable columns — never `run_info`. They record the triggering user
   (resolved default user included), not the dataset owner. Old rows stay
   NULL; no backfill.
3. **Token spend.** A session-independent ContextVar accumulator is fed
   from the same choke point as session usage (`record_llm_call`).
   Operations therefore capture tokens even with NO active session scope
   (previously those tokens were dropped). `SessionModelUsage` is
   unchanged — it remains the per-session/per-model grain; the same call
   feeds both, so the numbers reconcile. Accumulators chain to parents:
   `remember`'s row includes the tokens of its nested add/cognify/improve
   rows — do not SUM token columns across nesting levels.
   Counts are **provider-billed** (`response.usage`, including hidden
   reasoning tokens) for structured-output calls on the litellm instructor
   and litellm_native paths: the adapters attach the raw response as
   `_raw_response`, and `LLMGateway._record_session_usage_after` passes
   the exact counts as overrides via `_exact_usage_from_result` (adopted
   from PR #4342 / CLO-434; SDK-399 adds the litellm_native attachment
   that extends it beyond instructor). Cross-checked against
   OpenAI-reported usage on a live run. Fallback to char/4 estimates of
   the user text / output remains for plain-`str` outputs (the
   connectivity test) and BAML; on instructor validation retries only the
   final attempt's usage is counted. Transcript/image LLM calls and all
   embedding calls (CLO-450) are still not counted.
4. **run_info is frozen.** No new keys, ever (COG-5359 / COG-6089). All
   new data is typed columns.
5. **prune is self-erasing.** `prune.prune_system(metadata=True)` drops
   the relational database including `pipeline_runs` itself; its own
   record cannot survive by definition. The recorder swallows the write
   failure and the operation is unaffected. This is accepted, not a bug.
6. **Recording never breaks operations.** The operation recorder logs and
   swallows every internal error; the wrapped operation's own exceptions
   always propagate unchanged.
7. **Background launches record the launch, not the background work.**
   `remember(run_in_background=True)` (and the session-path background
   improve bridge) writes its single operation row when the call returns,
   i.e. when the background task was successfully launched —
   `outcome = "succeeded"` means "accepted and started", not "background
   work finished". The `background` Boolean column makes this explicit in
   data: `background = TRUE` rows carry launch semantics; a later failure
   inside the background task is durably visible on the nested pipeline
   rows (cognify/improve ERRORED rows with `outcome = "failed"`), which
   link back via `parent_operation_id`. Known, accepted limitation of the
   single-row-on-exit design; revisit only if an operation-level terminal
   write from the background task is needed.
8. **Remote (`serve()`) calls are not recorded client-side.** When the SDK
   is connected to a remote cognee instance, the operation executes there
   and the remote instance records it; recording client-side too would
   double-count.
9. **Origin.** Every record carries the surface that initiated it in
   `origin`: `"sdk"` (default), `"api"` (stamped per request by FastAPI
   middleware), `"cli"` (cognee-cli startup), `"mcp"` (in-process MCP
   server; in client mode the remote API records `"api"` instead), and
   `"background"` for system-initiated continuations such as the
   session-bridge improve. Carried by a ContextVar
   (`cognee/modules/operations/origin.py`), so it flows into background
   tasks automatically.
10. **Nesting is an explicit tree whose edges mirror the token chain.**
    Each operation allocates its id at scope entry (persisted as the
    row's `pipeline_run_id`); operations AND pipeline runs push that id
    onto a shared parent chain (`parent_run_scope`), and every child row
    stores the innermost enclosing run in `parent_operation_id`. A
    pipeline started inside another pipeline (the session bridge runs
    cognify inside a memify task) parents to that enclosing pipeline —
    not to the operation above both — so parent edges follow exactly the
    same path the token totals chain along. Consequently the children of
    any row sum to ≤ that row's own tokens (no sibling double-count),
    and top-level spend is `SUM(tokens) WHERE parent_operation_id IS
    NULL`. Never SUM across nesting levels.
11. **Session linkage.** Operation rows carry the session-cache id in
    `session_id` (String, indexed — session ids are strings, not FKs),
    set by search/recall/remember and by improve when bridging exactly
    one session. This is the join key for reconciling per-operation token
    spend with the per-session `SessionModelUsage` grain.
12. **Error messages are scrubbed and bounded.** `error_message` stores
    the exception text after conservative PII/secret redaction (emails,
    bearer/key-shaped tokens, long hex/base64 blobs, home-directory user
    names, long digit runs) and hard truncation to 512 chars
    (`cognee/modules/operations/scrub_error.py`). Raw messages still go
    to logs; only the durable typed column is scrubbed. `run_info` stays
    frozen (its legacy raw `error` key is unchanged).

## Schema delta (migration chained on b8c1d3e5f7a9)

New nullable columns on `pipeline_runs`: `user_id UUID (ix)`,
`tenant_id UUID`, `operation_name String (ix)`,
`started_at / ended_at DateTime(tz)`, `outcome String (ix)`,
`error_class String`, `error_message String` (scrubbed, ≤512 chars),
`tokens_in / tokens_out Integer`, `origin String`,
`session_id String (ix)`, `parent_operation_id UUID (ix)`,
`background Boolean`.
Old rows remain NULL for all of them; no backfill is performed.
