---
name: cognee-improve-sessions
description: Use when working with cognee's session memory or improve() — storing conversation turns, agent traces and feedback with session_id, bridging sessions into the permanent graph, reading an ImproveResult, understanding why an improve stage was skipped, already_completed or lock_held, or tuning the IMPROVE_* settings.
---

# Session memory and improve()

cognee has two kinds of memory:

- **Session memory**: a fast cache of conversation turns, agent traces, and
  feedback, keyed by `session_id`. Writing is instant, with no LLM
  extraction.
- **The permanent graph**: what `remember()` builds without a session.

`improve()` connects them. It bridges session content into the graph and
enriches the graph itself. `remember()` calls it automatically, so most
users never call it directly.

```python
import cognee

# Session write: returns immediately; improve() bridges it in the background
await cognee.remember("User prefers dark mode.", session_id="chat_1")

# Session-aware query: session cache first, then the graph
results = await cognee.recall("What does the user prefer?", session_id="chat_1")

# Bridge sessions into a dataset's graph explicitly
result = await cognee.improve(dataset="main_dataset", session_ids=["chat_1"])
print(result.status, result.stage_summary())

await cognee.wait_for_background_tasks()   # before a script exits
```

## Use it

### Writing session memory

| What | How |
|---|---|
| A fact or note | `remember(text, session_id=...)` (stored as a Q&A entry with the text as the answer) |
| A Q&A turn | `recall(query, session_id=...)` with a completion search type saves the turn itself; or `remember(cognee.QAEntry(question=..., answer=...), session_id=...)` |
| An agent step | `remember(cognee.TraceEntry(origin_function=..., status="success", ...), session_id=...)`, or the `@cognee.agent_memory(save_session_traces=True)` decorator |
| Feedback on an answer | `remember(cognee.FeedbackEntry(qa_id=..., feedback_score=...), session_id=...)` or `cognee.session.add_feedback(session_id, qa_id, feedback_text=..., feedback_score=...)` |
| Read a session | `cognee.session.get_session(session_id, last_n=...)` |

Requirements: `CACHING=true` (default). The cache backend is
`CACHE_BACKEND`, one of `sqlite` (default), `postgres`, `redis`, `fs`,
`tapes`. Sessions expire after `SESSION_TTL_SECONDS` (default 7 days).

### What improve() does: nine stages, in order

Every run goes through the same ordered stages
(`cognee/modules/improve/registry.py`). Each stage checks a gate before it
spends any LLM or embedding cost, and reports one `StageResult`.

| # | Stage | What it does | Runs when |
|---|---|---|---|
| 1 | `feedback_weights` | Adjusts the weight of graph elements that rated answers used | `session_ids` given; adapter supports feedback weights |
| 2 | `persist_session_qa` | Turns session Q&A into graph content (node set `user_sessions_from_cache`) | `session_ids` given. **The only fatal stage** |
| 3 | `persist_agent_traces` | Turns agent-trace feedback into graph content | `session_ids` given |
| 4 | `extract_agent_context` | Drafts agent-profile lessons from traces | `session_ids`, `CACHING` + `AUTO_FEEDBACK`, an LLM |
| 5 | `distill_sessions` | Distills session learnings into the graph | `session_ids`, an LLM |
| 6 | `update_user_preferences` | Folds rated turns into per-user preferences | `session_ids`, `PERSONALIZATION_ENABLED=true` (default false) |
| 7 | `build_truth_subspace` | Builds the truth subspace from distilled learnings | `session_ids`, `build_truth_subspace=True`, a Ladybug graph |
| 8 | `triplet_enrichment` | Triplet embeddings over the graph (memify) | `TRIPLET_EMBEDDING=true` (**default false**), or custom tasks/data passed |
| 9 | `global_context_index` | Bucket and root summaries for global questions | `build_global_context_index=True`, an LLM |

Stages 1–7 need `session_ids`; 8 and 9 work on the graph alone. The order
matters: 4 feeds 5, 5 feeds 7, and 7 runs before 8.

### Reading the result

`improve()` returns an `ImproveResult`, and so do `POST /api/v1/improve`, the
CLI, and `RememberResult.improve`.

- `result.status`: `completed`, `errored` (any stage errored), `skipped`
  (every stage skipped), or `running` (background, not finished).
- `result.stages`: one `StageResult` per stage, with `stage`, `status`
  (`completed` / `already_completed` / `skipped` / `errored`), `reason`,
  `error`, `counts`, `duration_ms`.
- `result.stage("distill_sessions")`, `result.stage_summary()`,
  `result.lock_held`, `result.rerun_requested`, `result.rerun_passes`.
- `await result.wait()` finishes a background run (no-op otherwise).

**Skip and no-op reasons:**

| Reason | Meaning / fix |
|---|---|
| `no_session_ids` | Session stage, no `session_ids` passed |
| `disabled_by_config` | Listed in `IMPROVE_STAGES_DISABLED` |
| `triplet_embedding_disabled` | Set `TRIPLET_EMBEDDING=true` to enable stage 8 |
| `opt_in_disabled` | Pass `build_truth_subspace=True` / `build_global_context_index=True` |
| `personalization_disabled` | Set `PERSONALIZATION_ENABLED=true` |
| `auto_feedback_disabled` | Stage 4 needs `CACHING=true` and `AUTO_FEEDBACK=true` |
| `no_llm_configured` | Stages 4, 5, 9 draft text with an LLM; none configured |
| `backend_unsupported` | The graph adapter lacks the feature (feedback weights, truth subspace) |
| `session_manager_unavailable` | The session cache is not reachable |
| `lock_held` | Another improve for the same dataset or session is running (below) |
| `aborted_by_fatal_stage` | Stage 2 failed, so the rest did not run |
| `no_new_entries`, `no_new_trace_steps`, `no_writes_since_last_improve` | With status `already_completed`: nothing new since the last run |

### How remember() triggers improve

- **Without a session:** `add`, then `cognify`, then a foreground `improve()`.
  Its outcome is on `result.improve` / `result.improve_error`. A failed
  improve never marks the remember as errored.
- **With `session_id`:** the text is cached, then a background
  `improve(dataset, session_ids=[session_id])` starts if the debounce allows
  it. `result.improve` fills in only after `await result` or
  `wait_for_background_tasks()`.
- `self_improvement=False` turns it off per call; `IMPROVE_AUTO_ENABLED=false`
  turns it off everywhere and overrides `self_improvement=True`.

### Settings (`IMPROVE_*`, `cognee/modules/improve/config.py`)

| Env var | Default | Meaning |
|---|---|---|
| `IMPROVE_AUTO_ENABLED` | `true` | Automatic improve after `remember()` |
| `IMPROVE_DEBOUNCE_ENTRIES` | `1` | Session auto-improve fires after this many new entries |
| `IMPROVE_DEBOUNCE_SECONDS` | `0` | ...or this many seconds since the last one. Seconds alone (entries left at 1) means time-only |
| `IMPROVE_STAGES_DISABLED` | empty | CSV of stage names to skip |
| `IMPROVE_FEEDBACK_ALPHA` | `0.1` | Feedback learning rate, in (0, 1] |

There is no debounce timer: held-back entries wait for the next
`remember()` with that session, or an explicit `improve()`.

## Pitfalls

- **A plain `improve(dataset)` often does nothing.** Without `session_ids`,
  only stages 8 and 9 can run, and both are off by default. Result: every
  stage `skipped`. That is expected, not an error.
- **Typed entries do not auto-improve.** `remember(QAEntry/TraceEntry/
  FeedbackEntry, session_id=...)` stores the entry but never starts an
  improve. Call `improve(session_ids=[...])` yourself.
- **`lock_held` does not wait.** Improves for the same dataset or session
  run one at a time: a second call returns at once with every stage
  `skipped: lock_held`. If it shares a session with the running one, it sets
  `rerun_requested=True` and the holder runs up to 3 extra passes
  (`rerun_passes`). The lock is per process only; multiple API workers do not
  share it.
- **Sessions bridge once.** Progress is tracked per user and session, not
  per dataset. Bridging a session into dataset A and then into dataset B
  adds nothing new to B.
- **`IMPROVE_STAGES_DISABLED` is validated.** An unknown stage name, or
  `persist_session_qa` (fatal, cannot be disabled), raises `ValueError`, and
  the API server refuses to start.
- **Session writes with the cache off.** `remember(text, session_id=...)`
  with `CACHING=false` only logs a warning and stores nothing; typed entries
  raise `RuntimeError`. An explicit `improve(session_ids=...)` then fails in
  the fatal stage.
- **Fatal stage failure.** If stage 2 errors, `improve()` raises (HTTP 409)
  and the error carries `.improve_result`. In background mode it is recorded
  on `result.error` instead. Other stages failing only mark themselves
  `errored`; HTTP still returns 200, so check `status`.
- **Remote mode.** After `cognee.serve(url)`, a background improve is
  fire-and-forget: `status` stays `running` and there is no polling.
- `cognee/modules/session_bridge/` is gone (only stale `__pycache__` may be
  left). The bridging now lives in the improve stages; some test names still
  say "session_bridge".

## How it works

`improve()` resolves the dataset once (creating it if the name is new),
claims the improve lock for `dataset:<id>` plus every
`session:<user>:<session_id>`, probes the graph adapter's capabilities, and
runs the stages with one frozen `ImproveRunInputs`. Each stage is a gate
plus a call into existing pipelines plus a result mapping. Stages never own
retries or ordering.

Watermarks keep repeat runs cheap:

- Session stages track how many entries were already persisted, per user
  and session.
- Stage 8 compares the last completed enrichment against later write
  pipelines for the dataset. A `node_name` or custom-task run bypasses that
  check.

The improve operation row is written when the run finishes: `failed` if any
stage errored (a retry is never gated off), `noop` if nothing ran.

- Orchestrator: `cognee/api/v1/improve/improve.py` (HTTP:
  `routers/get_improve_router.py`; CLI: `cognee/cli/commands/improve_command.py`)
- Stages, registry, results: `cognee/modules/improve/` (`stages.py`,
  `registry.py`, `stage.py`, `result.py`, `inputs.py`, `capabilities.py`,
  `config.py`, `graph_changes.py`, `constants.py`)
- Lock: `cognee/infrastructure/locks/session_lock.py`
- Session store and watermarks: `cognee/infrastructure/session/`
  (`session_manager.py`, `session_persist_watermark.py`,
  `feedback_detection.py`); cache backends in
  `cognee/infrastructure/databases/cache/`
- Auto-improve from remember: `cognee/api/v1/remember/remember.py`,
  `cognee/api/v1/remember/auto_improve_debounce.py`
- Entry types: `cognee/memory/entries.py`
- Background tasks: `cognee/infrastructure/background_tasks.py`

Examples: `examples/guides/improve_quickstart.py`, `sessions.py`,
`session_distillation.py`, `global_context_index.py`,
`agent_memory_quickstart.py`, and
`examples/advanced_guides/remember_recall_improve_example.py`.

## Extending it

Adding a stage:

1. Subclass `BaseStage` in `cognee/modules/improve/stages.py`. Set `name`,
   `needs_sessions`, and `fatal` (leave it `False`; exactly one fatal stage
   is enforced at import). Implement `gate()` (return a skip reason
   constant, or `None`, before any LLM/embedding cost) and `run()` (call
   existing pipeline code, return a `StageResult`).
2. Insert it into `DEFAULT_STAGES` in `registry.py` at the right position.
   The order is pinned by `cognee/tests/unit/modules/improve/test_registry_order.py`;
   update it deliberately.
3. If it can re-run cheaply, give it a watermark so a repeat run reports
   `already_completed`.
4. If it writes graph data under a new pipeline name, add that name to
   `WRITE_PIPELINE_NAMES` in `graph_changes.py`, or stage 8 will not notice
   the change.
5. Tests: `cognee/tests/unit/modules/improve/` (gates, results, config) and
   `cognee/tests/unit/api/v1/improve/` (orchestration, rerun, router).
