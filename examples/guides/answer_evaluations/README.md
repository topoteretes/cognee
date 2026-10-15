# Sampled answer evaluations: local and cloud

The SDK owns trace ingestion, bounded sampling, evaluations and read APIs.
Cloud mounts these same routes; plugins never receive warehouse credentials.

```text
Claude/Codex completed turn (explicit opt-in)
  -> redact -> local private spool (100 entries, 24h)
  -> authenticated POST /api/v1/evaluations/traces
  -> SDK transaction: encrypted source + immutable receipt + sample admission
  -> worker reconciliation -> local DuckDB or MotherDuck (encrypted replica)
  -> daily seeded selection of 20 distinct eligible traces
  -> matched replay without/with captured evidence -> blinded Jev preference
  -> SDK relational DB: selection/version IDs, verdicts, costs, latency, summary
  -> GET /api/v1/evaluations/summary -> plugin cache / cloud UI
```

## Local setup

Install `cognee[answer-evaluations]`. Run migrations using the normal SDK server
startup. Configure the API and worker with the same relational database and:

- `COGNEE_EVALUATIONS_ENABLED=true` (default false).
- `COGNEE_EVALUATION_ENCRYPTION_KEYS`: generated Fernet key(s), comma separated;
  newest first. Keep old keys while retained data needs decrypting.
- `COGNEE_EVALUATION_ID_KEY`: a separate, stable random key of at least 32 bytes.
  Do not rotate this in place: IDs use it for ownership and replay receipts.
- Worker only: `COGNEE_EVALUATION_WAREHOUSE=/absolute/private/path/evaluations.duckdb`.
  A cloud deployment uses `md:<database>` and a worker-only `motherduck_token`.

Generate keys locally, keep them in the deployment secret store, and never
commit them. A POSIX directory for local data should be private (0700).
Encryption keys must not live alongside database backups.

Start a reconciliation-only worker (no provider calls):

```sh
python -m cognee.modules.answer_evaluations.worker --loop
```

For paid daily comparisons also configure `EVAL_MODEL_URL` (complete
OpenAI-compatible completions URL), `EVAL_MODEL_API_KEY`, `EVAL_MODEL_VERSION`,
`EVAL_JUDGE_MODEL`, `TYPESAFE_API_KEY`, and input/output prices per million tokens
for both providers: `EVAL_INPUT_USD_PER_MILLION`, `EVAL_OUTPUT_USD_PER_MILLION`,
`EVAL_JUDGE_INPUT_USD_PER_MILLION`, `EVAL_JUDGE_OUTPUT_USD_PER_MILLION`.
Then run `python -m cognee.modules.answer_evaluations.worker --loop --evaluate`.
Alternatively schedule the command without `--loop` through cron/task scheduler.
Run only one worker against a local DuckDB file. Cloud uses CronJob Forbid plus
transactional per-scope daily admission. Reconciliation must continue even when
capture is disabled so deletions/expiry reach the warehouse.

`EVAL_MAX_COST_USD` defaults to 5 per invocation, with per-request admission
reservations; `EVAL_MAX_OUTPUT_TOKENS` defaults to 1200. An unexpected provider
usage/price response or reservation overrun stops that scope's run. These are
admission controls, not hard provider billing caps; configure provider spending
limits too. No automatic retry of failed/partial paid runs. A crashed run remains
visible as running until marked abandoned on a later sweep; never treat it as a
completed evaluation. The worker uses no production-memory writes or fresh
recall: it evaluates the evidence captured when the original answer was produced.
The production cloud recall deadline remains 45 seconds.

Enable the plugin's `COGNEE_EVALUATION_CAPTURE=true` explicitly and restart its
session. Normal installation leaves capture off. Only completed prompt/answer
pairs are uploaded, with bounded injected Cognee evidence, captured model name
(if available), and source snapshot hash. No hidden reasoning, full repository,
or arbitrary tool transcript is collected. Missing canonical dataset IDs skip
capture. Trace snapshots without evidence are stored but not eligible for the
matched memory comparison. Prior conversation beyond captured context is not
reconstructed; this measures bounded answer replay, not full coding-task success.

## Storage and retention

The application database's `answer_evaluation_scopes` serializes admission,
`answer_evaluation_traces` holds encrypted source bodies and keyed receipts, and
`answer_evaluation_runs` holds selected IDs/versions, selection seed, numeric
verdicts and summaries. Bodies are never duplicated into run history. MotherDuck
(or local DuckDB) holds an encrypted replica in `answer_quality.traces`.

There are at most 100 active source bodies per owning user and dataset, selected
by random priority within a rolling 30-day window. Expired content and runs are
removed by reconciliation. Receipt-only tombstones prevent retry resurrection
until their original expiry; a scope admits at most 10,000 receipts per window.
Different event IDs remain different interactions even with identical text.

The SQL commit is the upload ACK boundary. Warehouse failures leave accepted
traces recoverable; reconciliation atomically replaces each scope's replica.
Each store has a 100-body bound. A replica can temporarily contain evicted bodies
until reconciliation succeeds. Monitor `revision - synced_revision`; deletion
responses say `deletion_pending`, not falsely `deleted everywhere`.

The worker selects 20 distinct eligible traces, records those IDs before provider
calls, and runs at most once daily per scope. Fewer eligible traces stays pending;
unjudgeable or incomplete runs do not replace the last completed score. Providers
and rubric versions are recorded. Evaluation results never become user memory.

Authenticated child plugin agents submit to their owning user's scope and may
read its dataset summary. Detailed run history and privacy controls require the
owning human credential. Dataset read permission is checked in both API and worker;
revoked dataset access removes retained source content on reconciliation.

## API and privacy controls

All read/delete/consent calls require `?dataset_id=<uuid>`:

- `POST /api/v1/evaluations/traces`: typed <=64KB body, event UUID and
  `consent_version=answer-evaluation-v1`; 202 only after durable commit. Identical
  retry returns the same receipt; changed content conflicts with 409. Concurrent
  storage contention may return 503; retry with exactly the same body/event ID.
- `GET /api/v1/evaluations/summary`: personal summary or pending baseline.
- `GET /api/v1/evaluations/runs` and `/runs/{uuid}`: bounded history/provenance,
  not raw conversations.
- `PUT /api/v1/evaluations/consent`, body `{"enabled":false}`: block uploads,
  erase source bodies and run associations, and queue warehouse cleanup.
  Explicit `{"enabled":true}` re-enables future capture; old receipts stay erased.
- `DELETE /api/v1/evaluations/traces/{receipt}`: erase that source; invalidate the
  scope's run history so a deleted contribution cannot silently persist.

Disable plugin capture too when revoking. Local pending files are redacted,
0600 and expire after 24h during the next sweep; they are not claimed anonymous.
SDK content is encrypted using Fernet with separately stored keys. Identifiers
and payload-conflict digests use HMAC, not public hashes of personal text.
Redaction is best-effort: names and indirect identifiers can remain. Treat all
retained content as personal data. No raw bodies should reach application logs.
Review purpose/legal basis, processor DPAs, provider retention, region/transfers,
access controls and backup deletion before production opt-in. Provider requests
already in flight cannot be recalled by revoking consent. EU warehouse placement
alone is not a compliance guarantee.

## Display contract

`Better answers ▰▰▰▰▱ 74%`

This is a normalized product score, not a percentage improvement in correctness:
`round(47 + 53 * clamp((better - worse) / comparisons, 0, 1))`.
Initial baseline: `47% · calibrating`. Completed results update the value; they
never accumulate points just for running. Stale results are marked after 48h.
The stripe shows one bar and no evaluation count. Detailed UI exposes ties,
regressions, provenance and the formula. Jev preference is not objective task
success; no correctness percentage is inferred from its confidence.

Build the worker with the SDK's existing Dockerfile:
`docker build --build-arg COGNEE_EXTRAS=answer-evaluations -t your-registry/evaluator:commit .`.
Override its entrypoint with `python -m cognee.modules.answer_evaluations.worker`.
Validate locally with `uv run --extra answer-evaluations pytest
cognee/tests/unit/answer_evaluations/` before publishing that image.
