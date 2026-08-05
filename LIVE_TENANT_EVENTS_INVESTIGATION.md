# Live tenant event access — investigation

**Question.** Telemetry already reaches Segment → MotherDuck. How do we *also* make the
events we track available live, on the tenant's own pod?

**Context.** COG-6062 (#4332) proposed a local `postgres` telemetry sink and was closed:
pods already deliver to Segment, so a second delivery path wasn't justified by the
warehouse gap alone. Only the identity fix survived (#4344). But MotherDuck lags **~4h**
(measured: newest `event_timestamp` 04:55 UTC against 08:53 wall clock), so it cannot
answer "what is my memory doing right now". That is the gap this investigates.

Everything below was verified against `dev` and `cloud-backend@main`, not assumed.

---

## 1. What already works, live, today

Three of these are shipped and need no new write path.

### `GET /v1/activity/pipeline-runs` — live, durable, already exposed

Reads `PipelineRun` rows from the tenant's own relational DB, joined to `Dataset` and
owner `User`. Returns the last 50, permission-filtered via
`get_specific_user_permission_datasets`. No lag, survives restart.

Covers: cognify, ingestion, memify — status, dataset, owner, `pipeline_run_id`.

### `queries` + `results` tables — live and durable, but **not exposed**

`search()` writes a `Query` row (`text`, `query_type`, `user_id`, `created_at`) via
`log_query`, and a `Result` row (`value`, `query_id`) via `log_result`
(`cognee/modules/search/methods/search.py:79,144`). Every search a tenant runs is already
durably recorded on their own pod.

**No endpoint reads them.** This is the largest cheap win available: real user activity,
already persisted, currently invisible to the tenant.

Known limitation, same one that bit the memory-score work: `Query` has no `dataset_id`.
Attribution is tenant-wide only until that column is added at log time.

### `GET /v1/activity/spans` — exists, but cannot serve this

In-memory OTEL spans from `CogneeSpanExporter`. Three independent disqualifiers:

- `COGNEE_TRACING_ENABLED: false` in both tenant charts — off everywhere by default
  (`cognee-v2/values.yaml:66`: *"off everywhere; dev traces come from Dash0 injection"*).
- Coverage is **LLM adapter calls**, not user operations — the `@observe` decorators sit
  on the nine `litellm_*` adapters, not on `remember`/`recall`/`improve`.
- Buffer is `_MAX_TRACES = 50` traces, in memory, lost on restart.

It is a debugging aid. Do not build an activity feed on it.

### Also present

`GET /v1/activity/users`, `/agents`, `/export/{dataset_id}`.

---

## 2. The actual gap

After using everything above, what remains unavailable on-pod — Segment-only:

| Missing | Volume signal (3d, warehouse) |
|---|---|
| `cognee.remember`, `.improve`, `.forget`, `.recall`, `.export`, `.push` | `cognee.improve` 273,412 |
| ~28 `* API Endpoint Invoked` events | `Datasets` 737,126, `Remember Entry` 378,393 |
| Per-task pipeline events (`* Task Started/Completed/Errored`) | `Coroutine Task Started` 1,881,473 |

The third row is 54% of all telemetry volume and is execution detail no tenant should see.
It should not be in scope for any option below.

So the real gap is narrow: **`remember` / `improve` / `forget` operations, and
endpoint-level invocations.** Search and cognify are already covered by §1.

---

## 3. Options

### Option 0 — expose what is already persisted

New endpoint over `queries` + `results`, alongside the existing `/pipeline-runs`.

- **Cost:** one read-only endpoint. No write path, no table, no migration.
- **Covers:** search (durably, with the answer text), plus cognify already.
- **Leaves out:** remember / improve / forget, endpoint invocations.
- **Risk:** near zero. Reads data the pod already writes.

### Option 1 — local telemetry sink to the tenant's Postgres

What #4332 built (`postgres_sink.py`, branch `feat/cog-6062-telemetry-tenant-sink`
retained).

- **Covers:** everything, uniformly, with pagination and a retention window.
- **Cost:** a write path on every event, a table, a migration, plus the cloud-side
  migration since tenant pods run cloud-backend's Alembic chain (`COGNEE_ALEMBIC_PATH`).
- **Why it was rejected:** not the storage — the surrounding apparatus. A comma-separated
  `TELEMETRY_SINK` list, an event-name taxonomy, promoted columns, retention config, and
  `origin`/`pipeline_run_id` ContextVars, across 40 files. If this option is revived it
  should be a boolean and one table, scoped to the ~12 user-meaningful operation events
  from §2 — not the 28 endpoint events, not the per-task events.

### Option 2 — in-memory ring buffer

Mirror the `CogneeSpanExporter` pattern: bounded deque of recent operation events, read
back over an endpoint.

- **Cost:** ~60 lines, no schema, no migration, no I/O on the hot path.
- **Good for:** "what is happening right now" — the live ticker case.
- **Fails at:** anything historical. Lost on restart, and pods restart on every deploy.
  Single-replica only (same constraint the CLO-419 reservation ledger accepted).

### Option 3 — SSE / WebSocket push

Best UX for a genuinely live feed, but **not an alternative to 1 or 2** — a stream needs a
buffer behind it for initial load and reconnect. Additive, decide later.

---

## 4. Recommendation

**Do Option 0 now.** It is the only one that delivers real tenant-visible activity with no
new write path, and it closes the search half of the gap immediately. It is also
independently correct — persisting searches and never showing them to the tenant is a
straight omission.

**Then decide Option 1 vs 2 on one product question**, which is not a technical call:

> Is the tenant-facing activity view a **live ticker** (what is running now) or an
> **audit log** (what happened last week, filterable)?

- Ticker → Option 2. Bounded memory, no schema, accept restart loss.
- Audit log → Option 1, scoped as described. MotherDuck's 4h lag and the fact that
  tenants cannot query the warehouse make it the only durable answer.
- Both → Option 1 plus Option 3 on top. Do **not** run 1 and 2 together.

**Do not** try to serve this from MotherDuck. Beyond the 4h lag, tenants have no warehouse
access, and routing tenant reads through the control plane to a shared warehouse would
cross the tenant isolation boundary that NetworkPolicies exist to enforce.

---

## 5. Sequencing note

Option 0 depends on nothing. Option 1 depends on #4344 (identity) being merged first —
without `tenant_id` and a real `user_id` on the event, a locally stored row cannot be
attributed either, which was the same defect that made the warehouse data unusable.

## 6. Open questions

- `dataset_id` on `Query` at log time — needed for per-dataset activity filtering, and
  already an open follow-up from the memory-score work (COG-6067).
- Retention for Option 1: fixed window in code, or configurable? Prefer fixed until
  someone asks.
- 8.4M rows in 3 days carry a **NULL `tracking_event`** in `analytics.main.pipeline_events`
  — the single largest bucket in the table, larger than every named event combined.
  Unrelated to this investigation, but it means warehouse-side event accounting is
  currently unreliable and someone should look.
