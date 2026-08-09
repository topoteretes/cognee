# Cognee Self-Improvement Loop — Consolidated Improvement Plan

This document consolidates the verified findings of six deep-analysis passes over the self-improvement loop (feedback capture → `improve()` → graph weights / session distillation / truth subspace → retrieval) and merges three candidate roadmaps into one phased plan. It is self-contained: all claims below were verified against the code at the cited file:line locations.

---

## 1. Current state

### 1.1 How the loop is wired today

1. **Capture.** Session turns are recorded as QA entries. Answer-generating retrievers record `used_graph_element_ids` (graph_completion_retriever.py:354, hybrid_retriever.py:265, completion_retriever.py:148, triplet_retriever.py:158, agentic_retriever.py:450, temporal_retriever.py:73-77) and `used_session_context_ids`. `served_ids` round-trip correctly through all four cache adapters (verified: session_turn.py:150-225, cache models.py:45,66-71). Explicit ratings (`feedback_score` 1-5) come only from `add_feedback` (api/v1/session/session.py:113-131), the CLI (cli/commands/feedback_command.py:61-67), or explicit `remember()` entries (remember.py:356-391). Implicit per-turn LLM analysis (`analyze_turn_for_session_context`) rates *served context entries* (helpful/harmful counters) but never produces a `feedback_score`.
2. **Improve.** `improve()` (api/v1/improve/improve.py) runs a stage chain: apply feedback weights → persist Q&A to graph → persist agent traces → agent-context extraction → per-session distillation (curator/writer LLM gates → publish via `add()`+`cognify()`) → optional truth-subspace build → memify enrichment → optional global context index. It is fired automatically as a fire-and-forget task by `remember(session_id, self_improvement=True)` (remember.py:1228-1239).
3. **Consume.** `feedback_weight` is blended into triplet distance only when `feedback_influence > 0` (CogneeGraph.py:481-536); the hybrid chunk lane uses a static `importance_weight` and an optional truth factor (hybrid/ranking.py:36-59). The truth subspace scores DocumentChunks against centroids built from accepted lessons and applies a [0.75, 1.25] factor in the hybrid chunk lane only, when `use_truth_weight=True`.

### 1.2 The headline problem

**The loop is open at both ends by default, and worse than inert where it does run:**

- No implicit signal ever becomes a `feedback_score`, so stage 1 processes nothing for users who never explicitly rate (extract_feedback_qas.py:17-19 rejects unscored entries).
- `DEFAULT_FEEDBACK_INFLUENCE` defaults to `0.0` (base_config.py:20), so even written weights are read by nothing at query time (brute_force_triplet_search.py:70-74, 229).
- Partial-failure retry semantics **compound the same feedback across every `improve()` run** (see B1 below) — the loop actively corrupts weights.
- Any re-cognify of a document **resets all learned weights and truth coords to defaults** (see B5).
- There is **zero efficacy measurement anywhere** — no ablation, no A/B, no before/after metric (eval_framework grep is empty for feedback_weight/truth_subspace; beam/REPORT.md:120 admits it).

### 1.3 Corrections to the record (refuted / narrowed claims)

- **REFUTED: "learned weights survive re-cognify of the same document."** They do not. Ladybug `ON MATCH SET n.properties = node.properties` (ladybug/adapter.py:1147-1161) and Neo4j `SET n += node.properties` (neo4j_driver/adapter.py:379-392) overwrite `feedback_weight` back to 0.5; `prepare_edges_for_storage.py:117-118` re-injects 0.5 on edges; `DocumentChunk` truth fields reset to `None`. This is treated below as a fix, not an assumption.
- **REFUTED (suspicion): truth-centroid dataset isolation is broken.** It is correct — centroid ids are uuid5-namespaced per dataset+slot with a payload double-check (centroids.py:20-21, 173-175; build.py:126).
- **NARROWED: "all improve() stages fail open."** Two stages are fail-*closed*: persist Q&A (improve.py:348-354, no try/except) and stage-3 memify (improve.py:254-260) abort all later stages. The rest warn-and-continue.
- **NARROWED: "many retrievers don't record used_graph_element_ids."** All session-answering *completion* retrievers do; only raw-result search types (chunks, summaries, cypher, bm25, etc.) don't, and code_retriever opts out explicitly (code_retriever.py:743).
- **NARROWED: "the improve lock covers idle-watcher/SessionEnd dedup."** The lock is an in-process asyncio set (session_lock.py:72-91); no idle-watcher/SessionEnd trigger exists in this repo (they live in the external plugin), and CLI-vs-server improves are not deduped.

### 1.4 Confirmed gaps by subsystem

Severity: **H** high / **M** medium / **L** low. All confirmed or partial-confirmed unless noted.

#### A. Feedback capture & attribution (session layer)

| # | Gap | Evidence | Sev |
|---|-----|----------|-----|
| A1 | No implicit `feedback_score` path: `SessionTurnAnalysis` has no rating field; only explicit paths write scores, so stage 1 is a no-op for non-rating users | feedback_models.py:10-33; extract_feedback_qas.py:17-19; session_turn.py:228-276 | H |
| A2 | The two loops never bridge: explicit scores never demote the session entries/lessons that produced the answer; implicit ratings never touch graph weights, despite both ids sitting on the same QA row | apply_feedback_weights.py; cache/models.py:41-45; distill.py:143 | H |
| A3 | Turn analysis reads only `last_n=1`; feedback-only turns are recorded as "Thanks for your feedback." QA entries, so consecutive/delayed feedback is misattributed | session_turn.py:300, 358-372; session_manager.py:439-455 | M |
| A4 | helpful/harmful counters only increment; `harmful_count==0` distillation gate means one LLM-misread rating permanently disqualifies an entry (serving-time ranker has clamped recovery; distillation does not) | session_turn.py:253-272; distill.py:143; session_context_builder.py:136 | M |
| A5 | Inline LLM calls on hot paths: turn analysis awaited before every answer; batch trace extraction awaited inline every 10 traces; and by default **one LLM summary call per traced tool call** (`generate_feedback_with_llm=True`) | session_turn.py:383-388; session_manager.py:206, 226-232, 295-299; session_agent_trace.py:55-73 | H |
| A6 | `feedback_text` is persisted everywhere, consumed nowhere downstream (not in extraction, weighting, or distillation prompts) | extract_feedback_qas.py:88-94; distill.py:154-160 | M |
| A7 | Read-modify-write races on counters and the trace watermark lose increments / duplicate LLM extraction windows | session_turn.py:238-272; agent_context_extraction.py:163-222, 364-391 | M |

#### B. Weight application & retrieval consumption

| # | Gap | Evidence | Sev |
|---|-----|----------|-----|
| B1 | Partial-failure retry **compounds feedback**: `_update_element_weights` writes the found subset, returns `False`, entry stays eligible, and every subsequent `improve()` re-applies the alpha step. One stale node id → permanent weight drift | apply_feedback_weights.py:96-112, 165-174, 191-199; extract_feedback_qas.py:21-26 | H |
| B2 | `DEFAULT_FEEDBACK_INFLUENCE=0.0`: stage 1 writes weights nothing reads by default; `improve()` runs it unconditionally | base_config.py:18-20; CogneeGraph.py:482-483; improve.py:331-341 | H |
| B3 | The hybrid retriever — the lane that records `used_graph_element_ids` — never consumes feedback weights in any lane; only graph-completion does, and only with non-default influence | hybrid/ranking.py:36-59; grep empty for "feedback" under hybrid/ | H |
| B4 | Importance scaling applied *before* the [0,2] feedback-blend gate: scaled distances lie in [0,4], so elements can silently escape blending | CogneeGraph.py:534-536, 488, 499 | M |
| B5 | Re-cognify wipes learned weights (see refutation above) | ladybug/adapter.py:1147-1161; neo4j_driver/adapter.py:379-392; prepare_edges_for_storage.py:117-118 | H |
| B6 | `apply_frequency_weights` is dead machinery: no adapter implements the getters/setters; no ranker reads `frequency_weight`; invocation raises | graph_db_interface.py:727-757; apply_frequency_weights.py:151 | M |
| B7 | Weight application silently no-ops on Postgres-graph/Neptune (NotImplementedError swallowed as non-fatal warning) | graph_db_interface.py:643-711; improve.py:340-341 | M |
| B8 | Non-atomic weight+flag writes: overlapping improves double-apply; failure after write but before mark re-applies | apply_feedback_weights.py:96-112, 131-135, 192-199 | M |
| B9 | Uniform credit assignment, weak signal: identical alpha=0.1 update to every used element; ~7 unanimous 5-star ratings to move 0.5→0.76 | apply_feedback_weights.py:57; improve.py:153; hybrid/context.py:38-56 | L |
| B10 | Edge feedback path works only for graph-completion sessions with non-default influence (hybrid excludes facts; triplet returns None) | hybrid/context.py:34; triplet_retriever.py:93-95; brute_force_triplet_search.py:73-74 | M |

#### C. Session distillation & agent-context extraction

| # | Gap | Evidence | Sev |
|---|-----|----------|-----|
| C1 | No distillation watermark: every `improve()` re-runs the full curator+writer LLM fan-out per session; date-in-title makes cross-day re-accepted lessons distinct duplicate documents | distill.py:361, 384-405 (contrast agent_context_extraction.py:163-198) | H |
| C2 | Intra-run duplicate lessons: 6-block batches evaluated in parallel; novelty search sees only previously persisted chunks, so one theme restated across batches → N near-identical accepted lessons | models.py:24; distill.py:302-309, 336-344, 404 | H |
| C3 | Lesson store is append-only: no merge/supersede/retire; no outcome tracking (helpful counters exist only on session rows, never on published lessons); truth subspace replays all lessons forever | distill.py:302-309, 378-380; session_context_models.py:358-362; truth_subspace/build.py:3 | H |
| C4 | Pending-trace extraction drops history beyond the 40-trace window (watermark jumps to total); store errors swallowed while watermark advances | agent_context_extraction.py:146-148, 219-222, 386-391; session_context_builder.py:453-458 | M |
| C5 | `MIN_GATE_CONFIDENCE=0.75` is a no-op (equals the creation gate; confidence never mutated); gate reduces to `harmful_count==0`; `helpful_count` ignored | distill.py:143; models.py:19; session_context_models.py:8 | M |
| C6 | `improve()` violates its own dataset-resolution invariant: `_distill_sessions` and `build_truth_subspace` get the raw `dataset`, not `write_dataset_ref` — the shared-dataset retargeting hazard its own comment warns about | improve.py:141-151, 211, 229; distill.py:106 | M |
| C7 | Sessions with QA but zero gated context entries distill nothing (`no_gated_entries` short-circuit) | distill.py:392-394 | M |
| C8 | One full `add()`+`cognify()` per session; sessions processed serially | distill.py:379-380; improve.py:421-437 | M |
| C9 | Distilled lessons never feed future sessions' guidance: `build_active_context_block` reads only the same session's rows | session_context_builder.py:264-267 | M |
| C10 | Hygiene: `extract_batch_agent_context` is dead code; curator prompt/candidate format drift; `WrittenLesson.entities` and `member_entry_ids` provenance dropped; lexical timeline sort | agent_context_extraction.py:313-339; distill.py:164, 167, 347-381; models.py:74-77 | L |

#### D. Truth subspace

| # | Gap | Evidence | Sev |
|---|-----|----------|-----|
| D1 | "NEUTRAL" padding for failed embedding batches is a durable **25% penalty**: `[0.0]*k` coords at current epoch → truth_score 0 → factor 0.75 | build.py:96-98, 256-260; align.py:14-17, 65-83 | H |
| D2 | Epoch commits **before** scoring: any mid-build failure (or NotImplementedError on non-Ladybug backends) leaves centroids at N+1 with all chunks ≤ N — entire corpus silently neutralized | build.py:192-204, 274-283; hybrid/ranking.py:42 | H |
| D3 | Every build re-embeds the whole DocumentChunk corpus (embeddings already exist in DocumentChunk_text) and loads the full graph via `get_graph_data()` | build.py:219-228, 253 | H |
| D4 | Doubly opt-in and never auto-triggered; new chunks after cognify have no coords until a manual rebuild; router doesn't even expose `build_truth_subspace` | improve.py:45, 222; remember.py:1230-1234, 1278-1281; get_improve_router.py:23-37; DocumentChunk.py:43-44 | M |
| D5 | Only Ladybug implements `get/set_node_truth_state`; other backends pay full embedding cost then no-op | ladybug/adapter.py:2320, 2345; graph_db_interface.py:659-673 | M |
| D6 | Asymmetric lane coverage: BM25-only and summary-sourced chunks are never in the truth map (implicit 1.0) while vector-lane chunks can be demoted to 0.75× | hybrid_retriever.py:145-165; hybrid/chunks.py:43-81; ranking.py:41-43 | M |
| D7 | With `use_truth_weight` on, every query pays 3 extra serial round trips including a duplicate DocumentChunk_text search | hybrid_retriever.py:87-89, 145-159 | M |
| D8 | Positive-only axis: truth_score is a lesson-similarity prior, not a truth signal (no anti-centroids/contradiction axis); slot seeding is first-k-by-content-hash (effectively random); learnings are never removed or decayed | align.py:52-74; centroids.py:24-26, 84-124; build.py:143-179 | M |
| D9 | Re-cognify wipes chunk truth coords (same mechanism as B5) | ladybug/adapter.py:1156-1160; DocumentChunk.py:43-44 | M |
| D10 | Dead surface: `Entity.truth_alignment/truth_subspace_signature/truth_epoch` never scored or read; `unique_learning_vectors` uncalled; build signature computed but never persisted; retriever hardcodes `DEFAULT_K` while build accepts arbitrary k | Entity.py:14-16; centroids.py:55, 165-166; build.py:108-112, 153, 294-299; hybrid_retriever.py:125, 130 | L |

#### E. Orchestration, lifecycle, observability

| # | Gap | Evidence | Sev |
|---|-----|----------|-----|
| E1 | Callers cannot see which stages ran/failed: return is raw memify run info or bare `{}` on lock contention; `stages_run` only a span attribute, lost entirely if memify raises; auto-improve failures swallowed with a bare warning (`result.error` never set) | improve.py:167-172, 277-284; remember.py:1236-1237, 539-540 | H |
| E2 | Fire-and-forget improve task: only strong ref is `result._task` (GC hazard if result discarded); no drain/atexit hook → orphaned at process exit | remember.py:467, 1228-1239 | H |
| E3 | Agent-trace persistence has no watermark: `last_n_steps=None` re-extracts/re-cognifies the full growing trace blob per improve — O(n²), defeats dedup, leaves stale snapshots | improve.py:469; extract_agent_trace_feedbacks.py:88-113; session_persist_watermark.py:3-15 | H |
| E4 | Auto-improve fires the full multi-stage pipeline on **every** `remember()` with no debounce (lock prevents overlap, not repeated cost) | remember.py:1225-1239; improve.py:167-172, 238-261 | H |
| E5 | `run_in_background=True` breaks stage ordering and lock coverage (distillation reads while persist pipelines still write; lock released with writes in flight); only GCI is guarded | improve.py:183, 195, 258, 263-268, 280-284 | M |
| E6 | Documented graph→session sync does not exist (remember docstring + improve router promise it; no stage implements it) | remember.py:692-695; get_improve_router.py:33-35 | M |
| E7 | Post-cognify improve failure flips a completed `remember()` to 'errored' | remember.py:1272-1305 | M |
| E8 | Raw caller-chosen `session_ids` sent to PostHog product telemetry | improve.py:116; remember.py:856; shared/utils.py:239-253 | M |
| E9 | Lock-contention `{}` indistinguishable from a successful empty run (SDK and HTTP) | improve.py:167-172; get_improve_router.py:42, 97-99 | L |

#### F. Test & eval coverage

| # | Gap | Evidence | Sev |
|---|-----|----------|-----|
| F1 | No integration test covers the full loop (feedback → improve → changed ranking); at default influence 0.0 a silently dead loop is undetectable | test_feedback_weights_memify_integration.py:113 (weights only); cognee_graph_test.py:735 (ranking only, hand-built) | H |
| F2 | eval_framework has zero hooks for feedback weights or truth subspace; BEAM measures lesson *counts*, not utility; REPORT.md:120 admits no ablations | grep empty across eval_framework/ | H |
| F3 | improve() orchestration untested: stage order, fail-open vs the fatal persist stage, lock semantics, `feedback_alpha` plumbing | improve.py:155-354; only three narrow slices tested | H |
| F4 | LLM acceptance gates of distillation (curate/write_or_reject) and the truth extend path (pre-existing centroids) untested; no per-backend degradation tests; no lock concurrency test | distill.py:176, 270-345; test_build_centroid_slots.py mocks retrieve→[] | M |

---

## 2. Phased roadmap

Sequencing logic: **Phase 1** stops confirmed active damage with small PRs and lands the observability/measurement substrate (you cannot tune what you cannot see — and you must not turn on a loop that double-applies signal). **Phase 2** closes the loop end-to-end under defaults, guarded by the Phase-1 harness. **Phase 3** makes the now-working loop cheap enough to leave on. **Phase 4** holds semantic redesigns and architecture unification, each gated on Phase 1–3 data.

Effort: **S** ≤ 1 day, **M** ≤ 1 week, **L** > 1 week.

### Phase 1 — Stop active damage + make the loop visible (all independently shippable)

**1.1 Fix compounding re-application in `apply_feedback_weights`**
Record per-QA applied element ids and an attempt counter in `memify_metadata`; treat "some nodes deleted" as success-with-warnings after pruning missing ids; mark entries with no/empty `used_graph_element_ids` as processed so they are never rescanned.
- Files: `cognee/tasks/memify/apply_feedback_weights.py:96-210`, `cognee/tasks/memify/extract_feedback_qas.py:21-35`
- Effort: **S** · Impact: **High** (fixes B1, half of B8, part of A gap on skipped-forever entries) · Risk: Low — metadata-only; update existing unit/integration tests deliberately.
- **Acceptance:** a QA whose `used_graph_element_ids` include one deleted node id, run through `improve()` three times, moves each surviving element's weight exactly once; a no-id entry is fetched at most once across runs.

**1.2 Truth build: stop persisting `[0.0]*k` for failed embedding batches**
Skip those node_ids entirely (genuinely neutral), matching the code's own "NEUTRAL" comment.
- Files: `cognee/modules/truth_subspace/build.py:96-98, 253-260`; regression test under `cognee/tests/unit/truth_subspace/`
- Effort: **S** · Impact: **High** (D1) · Risk: Very low.
- **Acceptance:** a build with one failing embedding batch leaves those chunks with no truth state; their ranking factor is 1.0, not 0.75.

**1.3 Truth build: make the epoch bump the commit**
Score and persist chunk truth states tagged epoch N+1 first; upsert centroids at N+1 last. A mid-build failure leaves the old epoch live.
- Files: `cognee/modules/truth_subspace/build.py:192-283`
- Effort: **S** · Impact: **High** (D2) · Risk: Low-medium; needs a mid-build-failure test.
- **Acceptance:** killing the build after scoring 50% of chunks leaves retrieval using epoch-N factors unchanged; no chunk carries an epoch newer than the live centroids.

**1.4 Capability-gate the truth build**
Probe `set_node_truth_state` support before paying any embedding cost; surface a clear per-backend "skipped: backend unsupported" instead of embed-everything-then-no-op.
- Files: `cognee/modules/truth_subspace/build.py` (early check), `cognee/infrastructure/databases/graph/graph_db_interface.py:659-673`
- Effort: **S** · Impact: **Medium** (D5 cost half) · Risk: Low.
- **Acceptance:** on a Neo4j/Neptune backend, `build_truth_subspace` returns a skipped status with zero embedding calls.

**1.5 Stop paying for writes nothing reads**
`improve()` skips (or loudly warns on) stage 1 when `default_feedback_influence == 0`; document `DEFAULT_FEEDBACK_INFLUENCE` in `.env.template`/CLAUDE.md; add a one-line test pinning the default so it can only change consciously.
- Files: `cognee/api/v1/improve/improve.py:331`, `cognee/base_config.py:20`, `.env.template`, new unit test
- Effort: **S** · Impact: **Medium** (B2 visibility; the flip itself is Phase 2) · Risk: Very low.
- **Acceptance:** with influence 0, improve's result marks stage 1 `skipped` with a reason; the pin test fails if the default changes without editing the test.

**1.6 Structured `ImproveResult` + surface auto-improve errors**
Return `{stage: ok|failed|skipped, error, counts, skipped_reason}` instead of raw memify run info (keep run info nested for compatibility); make the lock-contention `{}` an explicit `skipped: lock_held`; `_session_improve` sets `result.improve_error` instead of a bare `logger.warning`; a post-cognify improve failure annotates rather than flips a completed `remember()` to errored.
- Files: `cognee/api/v1/improve/improve.py:108-284`, `cognee/api/v1/improve/routers/get_improve_router.py:42-99`, `cognee/api/v1/remember/remember.py:1225-1305`
- Effort: **M** · Impact: **High** (E1, E7, E9) · Risk: Medium — return-shape change; version/nest for one release.
- **Acceptance:** a forced distillation failure is visible in the returned result and on `RememberResult.improve_error`, while `remember()` status stays `completed`; two concurrent improves show one `ok` chain and one `skipped: lock_held`.

**1.7 improve() orchestration test suite + closed-loop integration test**
(a) Unit suite with all pipelines mocked: stage order, `stages_run`, fail-open vs the fatal persist stage at improve.py:348 (pin deliberately, marked as a decision test), lock-held early return, lock release on exception, lock concurrency (exactly one of two concurrent improves runs). (b) Integration test: seed session QA with `used_graph_element_ids` + `feedback_score`, run `improve()`, assert `brute_force_triplet_search` with `feedback_influence>0` re-ranks (boosted up, penalized down).
- Files: new `cognee/tests/unit/api/v1/improve/test_improve_orchestration.py` (pattern: test_improve_agent_context.py); extend `cognee/tests/integration/infrastructure/session/test_feedback_weights_memify_integration.py`
- Effort: **M** · Impact: **High** (F1, F3; safety net for every later phase) · Risk: None to prod.
- **Acceptance:** both suites green in CI; deleting the feedback blend in `CogneeGraph._effective_distance` fails the integration test.

**1.8 Telemetry hygiene**
Hash or drop raw `session_ids` from PostHog payloads (keep `session_count`); span attributes keep raw ids.
- Files: `cognee/api/v1/improve/improve.py:116`, `cognee/api/v1/remember/remember.py:856`
- Effort: **S** · Impact: **Medium** (E8 privacy) · Risk: Very low.
- **Acceptance:** no PostHog payload contains a raw session id string.

### Phase 2 — Close the loop end-to-end under defaults (guarded by Phase-1 tests, measured by the new harness)

**2.1 Eval ablation harness + lesson-utility metric** *(land first in this phase; gates 2.5)*
Sweep axis running the same question set with `feedback_influence ∈ {0, x}` and with/without `build_truth_subspace`, reporting delta mean score; add lesson retrieval-hit-rate (lesson-chunk ids appearing in `used_graph_element_ids` per answer) to BEAM reporting, replacing count-only metrics.
- Files: `cognee/eval_framework/sweeps/retriever_sweep_runner.py:180` (+ configs), `cognee/eval_framework/beam/local_ingest.py`, `cognee/eval_framework/beam/session_io.py:70-72`, new comparison report
- Effort: **L** · Impact: **High** (F2 — the loop's efficacy has literally never been measured) · Risk: Low to prod; LLM spend — keep the ablation set small and cached.
- **Acceptance:** one command produces a report with per-axis delta scores on the BEAM corpus; lesson hit-rate appears per lesson.

**2.2 Make learned state survive re-cognify**
Preserve `feedback_weight` (and `truth_alignment`/`truth_epoch`) on node/edge upserts: Neo4j coalesce pattern (precedent: `belongs_to_set` at adapter.py:382-385); Ladybug merge the existing properties blob's learned fields before replacement; stop re-injecting 0.5 on edge upserts.
- Files: `cognee/infrastructure/databases/graph/neo4j_driver/adapter.py:379-392, 1119-1121`, `.../ladybug/adapter.py:1147-1161`, `cognee/tasks/storage/prepare_edges_for_storage.py:117-118`, optionally `DataPoint.py:77`
- Effort: **M** · Impact: **High** (B5, D9 — prerequisite for any default flip) · Risk: Medium — hot ingestion path; per-adapter regression tests for both preserved fields and normal property updates.
- **Acceptance:** set a node's weight to 0.9, re-add + re-cognify the same document, weight remains 0.9 on both adapters; a fresh node still starts at 0.5.

**2.3 Bridge the two loops: implicit `feedback_score` from turn analysis**
Add optional `overall_answer_rating` to `SessionTurnAnalysis`; when turn analysis detects clear sentiment about the previous answer, write a conservative implicit score (helpful→4, harmful→2, never 1/5) to `previous_qa_id` via the existing `add_feedback` path (which already resets the memify flag). Explicit ratings stay authoritative (skip if score already set). Symmetrically, an explicit low score increments `harmful_count` on the QA's `used_session_context_ids` entries.
- Files: `cognee/infrastructure/session/feedback_models.py:10-33`, `cognee/infrastructure/session/session_turn.py:279-329`, `cognee/api/v1/session/session.py:113-131`
- Effort: **M** · Impact: **High** (A1, A2 — makes stage 1 work for users who never rate, reusing existing machinery end-to-end) · Risk: Medium — LLM-inferred scores; mitigate with conservative range, explicit-over-implicit precedence, and the auto-feedback flag gate.
- **Acceptance:** a session where the user says "that was wrong" (no explicit rating) produces a `feedback_score` on the previous QA and, after `improve()`, moved graph weights; an explicit rating is never overwritten by an implicit one.

**2.4 Consume weights where feedback is generated: hybrid-lane feedback factor + triplet ordering fix**
(a) Add a feedback factor to `rank_chunk_summary_pairs` mirroring `_importance_factor` (0.75 + 0.5·w), batch-fetching `get_node_feedback_weights` for the lane's candidate chunk ids. (b) Fix the importance/feedback ordering bug: apply `_effective_distance` to the raw cosine distance *before* the importance multiplier (or widen the gate to the scaled domain), with a regression test (distance ~1.5, importance 0.5).
- Files: `cognee/modules/retrieval/hybrid/ranking.py:36-59`, `cognee/modules/retrieval/hybrid_retriever.py`, `cognee/modules/graph/cognee_graph/CogneeGraph.py:481-536`
- Effort: **M** · Impact: **High** (B3, B4) · Risk: Low-medium — one extra batched graph call per hybrid query; gate behind the same influence config.
- **Acceptance:** two equal-similarity chunks with weights 0.9 vs 0.1 rank in that order in the hybrid lane when influence > 0; the triplet regression test passes; with influence 0, behavior is byte-identical to today.

**2.5 Flip `DEFAULT_FEEDBACK_INFLUENCE` off 0.0 (e.g. 0.2) — gated on 2.1's ablation result**
Only after 1.1 (no double-apply), 2.2 (persistence), and 2.4 (consumption in both lanes), and only if the ablation shows non-negative delta. Ship with an env escape hatch and a changelog note; update the Phase-1 pin test.
- Files: `cognee/base_config.py:20`, `.env.template`, pin test
- Effort: **S** · Impact: **High** (closes B2) · Risk: Medium — changes default ranking for all users; the gate is the mitigation.
- **Acceptance:** ablation report attached to the PR shows ≥ 0 delta; setting `DEFAULT_FEEDBACK_INFLUENCE=0` restores exact previous behavior.

**2.6 Attribution + text-signal plumbing**
(a) Widen turn analysis to the last 3–5 QA turns with model-selected `referenced_qa_ids`; skip "Thanks for your feedback." acknowledgement entries when picking the previous answer. (b) Yield `feedback_text` from `extract_feedback_qas` and include "User feedback on this answer: …" in distillation curator timeline blocks.
- Files: `cognee/infrastructure/session/session_turn.py:300, 358-372`, `cognee/infrastructure/session/feedback_detection.py`, `cognee/tasks/memify/extract_feedback_qas.py:88-94`, `cognee/modules/session_distillation/distill.py:154-167`
- Effort: **S** · Impact: **Medium** (A3, A6) · Risk: Low — additive prompt/plumbing.
- **Acceptance:** in a fixture with feedback delivered two turns late, the correct QA id receives the rating; curator prompts contain the feedback text.

### Phase 3 — Cut hidden cost, latency, and duplicate work (make the loop cheap enough to leave on)

**3.1 Move inline LLM calls off hot paths**
Flip `generate_feedback_with_llm` default to **False** (currently one LLM call per traced tool call; the batch pass already reads `method_return_value`); run `analyze_turn_for_session_context` concurrently with retrieval via `asyncio.gather` (context updates apply post-answer — a deliberate one-turn staleness); make the every-10-traces batch extraction fire-and-forget behind a per-session lock.
- Files: `cognee/infrastructure/session/session_manager.py:206, 226-232, 286-299`, `session_turn.py:383-388`, `session_agent_trace.py:55-73`
- Effort: **M** · Impact: **High** (A5 — the single biggest hidden latency) · Risk: Medium — behavior change for trace consumers; release-note it; verify answer quality via the 2.1 harness.
- **Acceptance:** a session turn issues zero serial LLM calls before retrieval starts; a traced tool call with default settings issues zero inline LLM calls.

**3.2 Distillation watermark + dedup**
Per-(session, dataset) watermark (processed entry ids / max `created_at`, mirroring agent_context_extraction.py:163-198); skip `distill_session` when nothing new is gated; drop the run date from the lesson document title (or key it to content) so re-accepted lessons dedup at ingestion; deterministic intra-run dedup — cluster accepted statements by normalized text before publish.
- Files: `cognee/modules/session_distillation/distill.py:336-405` (esp. :361)
- Effort: **S** · Impact: **High** (C1, C2) · Risk: Low — no acceptance-semantics change.
- **Acceptance:** running `improve()` twice on an unchanged session performs zero curator/writer LLM calls the second time; a 10-batch session restating one theme publishes exactly one lesson.

**3.3 Agent-trace persist watermark + lossless extraction drain**
Replace `last_n_steps=None` with the `session_persist_watermark` count-window pattern (advance only after successful cognify); drain pending extraction in `BATCH_TRACE_LIMIT` windows advancing the watermark per window (never jump to total), with a catch-up bound; only advance when candidate application succeeded.
- Files: `cognee/api/v1/improve/improve.py:469`, `cognee/tasks/memify/extract_agent_trace_feedbacks.py:88-113`, `cognee/infrastructure/session/agent_context_extraction.py:146-148, 342-395`
- Effort: **M** · Impact: **High** (E3, C4) · Risk: Low-medium — established pattern applied to a third pipeline; watermark tests mirror the existing suite.
- **Acceptance:** improve() on a session with 300 traces and 290 already processed re-cognifies only the new 10; a 300-trace backlog is fully drained across windows, none skipped.

**3.4 Debounce auto-improve + background-task lifecycle**
Fire the background improve only when ≥ N new entries or ≥ T seconds since last run (per-session dirty counter in existing state rows; N/T env-configurable); anchor background tasks in a module-level registry with a drain hook (atexit / FastAPI lifespan / `cognee.wait_for_background_tasks()`); run the whole improve chain inside one background task holding the lock for its lifetime instead of forwarding `run_in_background` into each pipeline (delete the GCI skip workaround).
- Files: `cognee/api/v1/remember/remember.py:1225-1239, 467`, `cognee/api/v1/improve/improve.py:160-284`, new registry module
- Effort: **M** · Impact: **High** (E2, E4, E5) · Risk: Medium — bounded staleness trade-off; lifespan/shutdown ordering needs an app-level test.
- **Acceptance:** N rapid `remember()` calls trigger at most ⌈N/debounce⌉ improve runs; a script that discards the `RememberResult` and exits still completes (or cleanly drains) the pending improve; no "Task was destroyed but it is pending" warnings.

**3.5 Fold truth-state fetching into the chunk lane**
Collect candidate ids from the lane's own BM25+vector+summary hits, batch `get_node_truth_state` once before ranking, delete `_candidate_chunk_ids` — removes the duplicate DocumentChunk_text search, the serial pre-gather round trips, and the lane-coverage bias in one move. Add the end-to-end HybridRetriever truth-ordering test alongside.
- Files: `cognee/modules/retrieval/hybrid_retriever.py:87-165`, `cognee/modules/retrieval/hybrid/chunks.py`, `hybrid/ranking.py:40-43`
- Effort: **S** · Impact: **Medium** (D6, D7) · Risk: Low — ranking already takes a plain dict.
- **Acceptance:** with truth weighting on, exactly one DocumentChunk_text search per query; a BM25-only chunk with poor alignment is demoted the same as a vector-lane one.

**3.6 Incremental truth build**
Reuse existing DocumentChunk_text embeddings (read vectors back) or add an incremental mode scoring only `truth_epoch < current`; replace `get_graph_data()` with a type-filtered node query; implement `get/set_node_truth_state` for Neo4j (mirrors the feedback-weight methods). Consider a lightweight cognify-time projection task (no-op when no centroids exist) that also fixes "new chunks have no coords."
- Files: `cognee/modules/truth_subspace/build.py:39, 219-283`, `neo4j_driver/adapter.py` (new methods)
- Effort: **M** · Impact: **High** (D3, D4 partially, D5) · Risk: Medium — vector read-back differs per backend; keep re-embedding as fallback.
- **Acceptance:** a second build on an unchanged corpus performs zero embedding calls; newly cognified chunks get coords without a manual full rebuild; the Ladybug truth tests pass on Neo4j.

**3.7 Race hardening: atomic increments**
Add `increment_context_entry_counter(entry_id, field, delta)` (and CAS for the trace watermark) to the cache adapter interface; use for helpful/harmful counters and extraction watermarks. SQL/Redis native; fs/tapes via their existing per-session write locks.
- Files: `cognee/infrastructure/databases/cache/cache_db_interface.py` + 4 adapters, `session_turn.py:238-272`, `agent_context_extraction.py:151-222`
- Effort: **M** · Impact: **Medium** (A7, C4 race half, B8 residual) · Risk: Medium — interface change across 4 adapters; cover via adapter contract tests.
- **Acceptance:** 10 concurrent rating applications yield exactly 10 counter increments on every adapter; concurrent trace writes never run overlapping extraction windows.

**3.8 Hygiene batch (small PRs, bundle freely)**
- Pass resolved `write_dataset_ref` to `_distill_sessions` and `build_truth_subspace` (C6) — closes a confirmed shared-dataset retargeting hazard.
- Replace the `harmful_count==0` distillation gate with clamped net-helpfulness reusing the ranker's `net_help` (A4, C5); use `helpful_count` as a positive signal.
- Delete dead code: `extract_batch_agent_context`, `unique_learning_vectors`, `apply_frequency_weights` task/pipeline/stubs (B6 — deprecate the export), `Entity.truth_subspace_signature` unless 4.4 wires it.
- Fix curator prompt candidate-format drift and the lexical timeline sort; persist `WrittenLesson.entities` + `member_entry_ids` as provenance on lesson documents (C10 — prerequisite for 4.2).
- Delete the false graph→session-sync promises from remember() and the improve router (E6) — implement only if 4.3 proves cross-session value.
- Capability-gate weight application per backend with a visible `skipped: backend unsupported` stage status (B7).
- Files: improve.py:211, 229; distill.py:140-167, 347-381; agent_context_extraction.py:313-339; centroids.py:55; Entity.py:15; remember.py:692-695; get_improve_router.py:33-35; graph_db_interface.py:727-757
- Effort: **M** total (each item S) · Impact: **Medium** · Risk: Low.
- **Acceptance:** an entry with harmful 1 / helpful 3 is distillable; a shared-dataset distillation writes to the resolved UUID's dataset; grep for the deleted symbols returns nothing; lesson documents carry member/entity provenance.

**3.9 Remaining test gaps**
distill_session orchestration with mocked LLM gates (accepted published / rejected dropped / `no_gated_entries` / one failing batch isolated); truth extend path with pre-existing centroids; per-backend truth degradation tests.
- Files: new tests beside `test_distillation_helpers.py`, `test_build_centroid_slots.py`
- Effort: **S-M** · Impact: **Medium** (F4) · Risk: None to prod.
- **Acceptance:** suites green; the extend branch of `build_for_epoch` is exercised with non-empty `vector_engine.retrieve`.

### Phase 4 — Semantic redesign and architecture unification (each item gated on Phase 1–3 data)

**4.1 Unified `FeedbackEvent` model** *(gate: Phases 1–3 shipped; equivalence tests green)*
New `cognee/modules/feedback/{models.py,dispatch.py}`: every signal — explicit score, implicit turn rating, served-context judgment, trace-derived candidate — becomes one normalized event `{source, session_id, referenced_qa_ids, target_context_entry_ids, target_graph_element_ids, score, text, confidence, timestamp}`. Existing writers emit events; consumers (counters, weight extraction, distillation gating) read the stream. Keep legacy fields as projections during transition. Collapse `apply_feedback_weights` (and the frequency remnant, if kept) into one parametrized weight-update task consuming events.
- Effort: **L** · Impact: **High** (collapses A2's structural cause; halves the weight-task surface) · Risk: High — four write paths, three read paths; the Phase-1 test net plus projection-equivalence tests are the mitigation.
- **Acceptance:** all existing feedback tests pass against the projections; a new event source (e.g. thumbs-up API) reaches graph weights, counters, and distillation gating with zero new plumbing.

**4.2 Lesson lifecycle** *(gate: 2.1 lesson hit-rate metric live; 3.8 provenance shipped)*
Writer emits supersede/merge decisions against the top-5 similar lessons (soft-retire via node_set flag before any hard delete); track usage via lesson-chunk appearances in `used_graph_element_ids` / FeedbackEvents; prune or downweight never-retrieved and superseded lessons; remove retired `learning_ids` from truth-centroid slots.
- Files: `distill.py:270-381`, `models.py:66-77`, `truth_subspace/centroids.py:84-124`, `build.py:143-179`, writer prompt
- Effort: **L** · Impact: **High** (C3 — today lesson quality can only degrade) · Risk: High — first destructive operation in the loop; soft-retire + hit-rate evidence required before hard delete.
- **Acceptance:** a superseding lesson retires its predecessor (excluded from serving, distillation novelty, and centroid slots); a lesson with zero hits over the pruning window is downweighted in the next truth build.

**4.3 Seed new sessions from distilled lessons** *(gate: 4.2 shipped, so bad lessons can be demoted)*
At first `build_active_context_block`, vector-search `session_learnings` for the current query and inject top hits as low-priority `lessons_learned` entries, behind `is_auto_feedback_enabled`.
- Files: `cognee/infrastructure/session/session_context_builder.py:264-267`
- Effort: **M** · Impact: **High** (C9 — closes the cross-session half of the loop) · Risk: Medium — adds a retrieval call at session start; can amplify a bad lesson without 4.2.
- **Acceptance:** a lesson distilled from session A appears in session B's guidance block when relevant; 2.1 harness shows non-negative answer-quality delta with seeding on.

**4.4 Truth subspace: anti-centroids or trim** *(gate: 2.1 ablation of the positive-only prior)*
If the ablation shows lift: add a negative axis from rejected distillation lessons and `contradicts`-edge facts, subtracting anti-alignment in `truth_score`; replace first-k-by-hash seeding with farthest-point seeding + periodic re-cluster; either wire Entity truth fields into the entity/facts lanes or delete them. If no lift: trim the subspace to the minimal maintained surface (or deprecate).
- Files: `align.py:52-83`, `centroids.py`, `build.py`, `Entity.py:14-16`
- Effort: **L** · Impact: **High if kept / Medium if trimmed** (D8, D10) · Risk: High — new scoring semantics; ship behind a flag, validate via ablation before any default.
- **Acceptance (keep path):** a chunk contradicting an accepted lesson receives factor < 1.0 in the harness; ablation delta ≥ 0 on the BEAM set. **(trim path):** dead fields and unused plumbing removed; unit tests updated.

**4.5 Auto-trigger truth build** *(gate: 3.6 incremental build shipped)*
Pass `build_truth_subspace` through `remember(self_improvement=True)` behind an env flag; expose it on the improve router.
- Files: `remember.py:1230-1234`, `improve.py:222-236`, `get_improve_router.py`
- Effort: **S** · Impact: **Medium** (D4) · Risk: Medium — only sensible post-3.6 (today it would re-embed the corpus per session end).
- **Acceptance:** with the flag on, coords stay current after each debounced improve with no full-corpus embedding pass.

**4.6 Attributed credit assignment from `feedback_text`** *(gate: 2.1 harness; optional)*
On negative feedback, LLM-classify which retrieved elements the complaint targets and weight alpha per element instead of uniformly; positive feedback stays uniform.
- Files: `apply_feedback_weights.py`, `extract_feedback_qas.py`
- Effort: **L** · Impact: **Medium** (B9) · Risk: Medium; stays within the existing EWMA.
- **Acceptance:** in a fixture where feedback text names one wrong fact, only that element's weight drops materially; harness shows no regression.

---

## 3. Non-goals

- **A durable cross-process improve lock.** The in-process lock's single-worker scope stays a documented limitation; CLI-vs-server dedup relies on the Phase-3 debounce, not a distributed lock.
- **Implementing graph→session sync.** The false docstrings are deleted in 3.8; a real sync stage is only reconsidered if 4.3 demonstrates cross-session value.
- **Frequency weights as a feature.** Default decision is deletion (3.8). Resurrect only with a concrete ranking consumer and an ablation plan.
- **Per-user personalization of weights, multi-armed-bandit exploration, or online learning.** Out of scope until the basic loop is measurably positive.
- **Postgres-graph / Neptune weight+truth support.** These backends get honest `skipped` statuses (1.4, 3.8), not implementations, in this plan.
- **Changing the EWMA weight formula or the [0.75, 1.25] truth-factor bounds** beyond the ordering fix (2.4) — tuning waits for the harness.

## 4. Open questions

1. **What ablation delta justifies flipping `DEFAULT_FEEDBACK_INFLUENCE`?** Proposal: non-negative mean delta and no per-question regression > X% on the BEAM set — needs sign-off before 2.5.
2. **Implicit-score calibration (2.3):** are helpful→4 / harmful→2 the right conservative mappings, and should implicit scores use a smaller alpha than explicit ones?
3. **Debounce defaults (3.4):** what N entries / T seconds balances learning freshness against per-message cost for typical agent sessions? Needs telemetry from the Phase-1 ImproveResult data.
4. **Lesson retirement policy (4.2):** soft-retire window length and whether hard deletion is ever safe given lessons may be cited by external consumers of the graph.
5. **Truth subspace keep-or-trim:** if the positive-only prior shows no ablation lift and anti-centroids are expensive, is the feature worth its maintenance surface at all?
6. **`k` mismatch (D10):** standardize the centroid slot count on `DEFAULT_K` end-to-end, or plumb `k` through retrieval? (Cheap either way; decide when touching 3.6.)
7. **Backward compatibility window for the `ImproveResult` shape (1.6)** — one release with nested legacy run info, or a versioned router response model?
8. **Should the fatal persist_sessions stage (improve.py:348) become fail-open** like its siblings, or is aborting later stages on Q&A-persist failure intentional? Phase 1.7 pins current behavior; the decision itself is open.