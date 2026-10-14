# Usage accounting for controlled trials

Use a fresh authenticated user session and evaluation dataset for every trial.
Typed idempotent capture, `recall(session_id=...)`, and single-session
`improve(session_ids=[...])` bind model work to that session on every surface:
SDK, HTTP, MCP, and the background improve after `remember(session_id=...)`.
A recall without a `session_id` binds nothing beyond its session turn.
LiteLLM embedding requests contribute their provider input tokens and estimated
cost to the session totals, under the embedding model's own per-model row. They
are kept out of the operation counters (`pipeline_runs.tokens_in`/`tokens_out`,
which count LLM tokens only) and never become the session's `last_model`. An
embedding instrumentation failure does not retry the paid request or fail the
operation.

Read `/api/v1/sessions/{session_id}` after synchronous work finishes. Its token and
cost totals include the session's child operations: do not add pipeline totals a
second time. Multi-session improve has no single unambiguous attribution target;
run one session per trial. Background work must finish before taking the snapshot.

Add client/agent usage separately. Session metrics alone do not include client
model calls. Existing provider paths can estimate tokens, and prices are estimates;
unpriced models can still contribute zero cost to the existing aggregate. Treat
combined totals as incomplete unless coverage and pricing for the chosen providers
have been verified. Do not interpret an unavailable counter as a free operation.

The Claude plugin's `plugin_eval.py outcomes` runner joins one server aggregate
with client usage, preserves the accounting basis and reports unknown totals as
null. The harness keeps matched seeds, models, task fixtures and token budgets,
uses isolated datasets, and retains the 45-second cloud recall deadline.
