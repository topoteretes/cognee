# LoCoMo on Cognee with GPT-5.1 Mini — Plan

Branch: `Vasilije1990/locomoco` (clean, identical to `dev`; no prior LoCoMo work anywhere in git history).

## Status (2026-09-05)

Implemented under `cognee/eval_framework/locomo/` (see its README) — adapter, preprocessing,
sessions → `improve()` ingestion, retriever sweep, F1 + LLM-judge evaluator, aggregation, and
the `run_locomo_eval.py` driver. Unit tests: `cognee/tests/unit/eval_framework/locomo_*.py`.

Decisions taken vs. the plan above:

- **Model**: `gpt-5.1-mini` does not exist on the OpenAI API (probe returned NotFoundError, and
  litellm 1.96.2 lists `gpt-5.1`, `gpt-5.1-codex-mini`, `gpt-5.2` but no `gpt-5.1-mini`). The
  answerer defaults to `openai/gpt-5-mini`; the judge is `openai/gpt-5.1` as requested. Pass
  `--answer-model` to change; the driver probes the id before spending anything.
- **Ingestion path**: sessions first. Dialogue windows go to the session cache via
  `remember(session_id=...)` and reach the graph only through `improve(session_ids=...)`
  (persist + distill + preference weights + enrichment + global context index). The BEAM-style
  `add`/`cognify` JSON-list path is kept only as an optional artifact (`preprocessed/`).
- **Isolation**: one pair of DATA/SYSTEM roots per conversation, `ENABLE_BACKEND_ACCESS_CONTROL`
  off inside each child process, so retrievers see exactly one conversation's memory.
- **Adversarial**: kept by default; the aggregate reports both "overall" and a mem0-comparable
  slice without it.

## 0. What already exists and what is missing

The BEAM work (`cognee/eval_framework/beam/`) is the template. Everything below is reusable as-is:

| Piece | Where | Reuse for LoCoMo |
| --- | --- | --- |
| Benchmark adapter registry + `cognee eval` CLI | `benchmark_adapters/benchmark_adapters.py`, `cli/commands/eval_command.py` | Register a `LoCoMo` adapter; the flattened one-document path then works with zero further wiring |
| Turn-preserving ingestion (one JSON-list file per session, one list item per chunk, no overlap) | `beam/local_ingest.py`, `modules/chunking/JsonListChunker.py` | Same representation; only the preprocessing step differs |
| Benchmark-agnostic QA sweep (retriever configs, per-question-type prompt files, answer cache, repeats, concurrency) | `sweeps/retriever_sweep_runner.py`, `beam/eval/registry.py` | Call `run_retriever_sweep_for_questions` with LoCoMo questions; `question_type` = LoCoMo category |
| Async evaluator pattern (fresh metric per task, semaphore, no deepeval dependency) | `beam/eval/beam_eval_adapter.py` | Copy the shape; swap metrics for token-F1 + LLM judge |
| Cross-run aggregation by question type + bootstrap CIs | `beam/eval/aggregate_cross_run.py`, `analysis/metrics_calculator.py` | Extend to aggregate across 10 conversations |
| Reported BEAM retrieval config (hybrid, chunks_top_k=20, entities_top_k=20) | `beam/report_artifacts/100k_fixed/*.json` | Starting point for the LoCoMo sweep |
| GPT-5 family handling (no temperature sent unless set, `max_completion_tokens` capped, reasoning_effort plumbing) | `infrastructure/llm/config.py`, `LLMGateway.py` | Nothing to change |

Missing, must be built:

1. LoCoMo dataset adapter (download, parse, question typing, golden evidence).
2. LoCoMo preprocessing into per-session JSON-list files (two human speakers, session date headers, image captions).
3. Per-conversation ingestion driver (10 independent memories; LoCoMo questions never cross conversations).
4. LoCoMo evaluator: token-F1 (paper metric) + binary LLM judge (mem0/Zep-style), per-category reporting.
5. Answer prompts per LoCoMo category (short-answer style; F1 punishes verbosity).
6. Registration of the `gpt-5.1-mini` model id in litellm's capability table (see §1.3).

## 1. Environment (Phase 0, ~1 h)

### 1.1 Worktree venv
This worktree has no `.venv` and no `.env`. The main checkout's venv
(`~/Projects/cognee_experimenal/cognee/.venv`, Python 3.11, litellm 1.96.2) imports fine but
is an editable install of a different checkout.

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,evals]"
uv pip install datasets          # only if we also want BEAM loaders; LoCoMo is a plain JSON download
```

`deepeval` is NOT needed: the LoCoMo evaluator will be async and deepeval-free like `BeamEvalAdapter`.
(`evaluation/metrics/f1.py` imports deepeval at module top, so it cannot be reused directly.)

### 1.2 .env
Copy `LLM_API_KEY` from the main checkout's `.env` (it is the only key set there). Everything else
stays default: `litellm_native` structured output, LanceDB + Ladybug + SQLite, `text-embedding-3-large`.

```bash
LLM_API_KEY=...
LLM_MODEL=openai/gpt-5.1-mini
LLM_PROVIDER=openai
# keep memory layer on, but drop the per-turn LLM call during QA
CACHING=true
AUTO_FEEDBACK=false
```

### 1.3 Verify the model id — BLOCKER TO RESOLVE FIRST
litellm 1.96.2 (pinned `<1.97.0` for Python 3.10 compat) knows `gpt-5.1`, `gpt-5.1-chat-latest`,
`gpt-5.1-codex-mini`, `gpt-5.2`, but **not** `gpt-5.1-mini`. Consequences if we run without fixing:

- `litellm.supports_response_schema("openai/gpt-5.1-mini")` returns `False`, so every structured
  extraction call silently takes the prompted-JSON fallback path instead of schema-native
  `response_format`. Worse extraction, more parse retries.
- `get_model_max_completion_tokens` returns `None`, so the user cap (16384) is used. Fine.
- Cost tracking returns nothing for the model. Cosmetic.

Steps:
1. One-shot `litellm.acompletion(model="openai/gpt-5.1-mini", ...)` against the real API to
   confirm the id exists. If OpenAI does not serve it, decide between `openai/gpt-5-mini`
   (cognee default, fully known to litellm) and another GPT-5.x id.
2. If it exists, register it before any cognee call in the LoCoMo driver:
   ```python
   litellm.register_model({"openai/gpt-5.1-mini": {
       "litellm_provider": "openai", "mode": "chat",
       "max_tokens": 128000, "max_input_tokens": 400000, "max_output_tokens": 128000,
       "supports_response_schema": True, "supports_function_calling": True,
       "supports_reasoning": True, "supports_system_messages": True,
       "input_cost_per_token": ..., "output_cost_per_token": ...}})
   ```
   Also register the bare `gpt-5.1-mini` key (litellm strips the `openai/` prefix in some lookups).
3. Confirm with a 3-sentence `cognee.add` + `cognify` + `search` smoke that the native schema path
   is taken (log line from `native_adapter`).

## 2. Data (Phase 1, ~half day)

LoCoMo source: `snap-research/locomo`, file `data/locomo10.json` (single JSON, ~10 conversations).
Structure (verify on download; this is from memory):

- `conversation`: `speaker_a`, `speaker_b`, `session_1 … session_N` (lists of turns:
  `speaker`, `dia_id` like `D1:3`, `text`, optional `img_url` + `blip_caption`, optional `query`),
  and `session_N_date_time` strings such as `"1:56 pm on 8 May, 2023"`.
- `qa`: list of `{question, answer | adversarial_answer, evidence: [dia_id…], category}`.
  Categories: 1 multi-hop, 2 temporal, 3 open-domain / commonsense, 4 single-hop, 5 adversarial
  (unanswerable). Roughly 1,986 questions total; the mem0 paper excludes category 5 (~1,540 left).
- Also `event_summary`, `observation`, `session_summary` (not used for ingestion; they are oracle
  artefacts).

### 2.1 `benchmark_adapters/locomo_adapter.py` — `LocomoAdapter(BaseBenchmarkAdapter)`
- `dataset_info` with filename + raw GitHub URL; download-once pattern copied from `HotpotQAAdapter`
  (uses `requests`, already available transitively).
- ctor: `conversation_index: int | None` (None = all 10), `max_sessions`, `include_adversarial=True`.
- `load_corpus()` returns
  - corpus: one flattened transcript string per conversation (for the quick `cognee eval` path), and
  - questions: `question`, `answer` (or `adversarial_answer`), `question_type` in
    `{single_hop, multi_hop, temporal, open_domain, adversarial}`, `category` (int),
    `evidence` (dia_ids), `conversation_id`, `conversation_index`, `question_idx`,
    `golden_context` (evidence dia_ids resolved to `"[date] speaker: text"` when `load_golden_context`).
- Register `LOCOMO = ("LoCoMo", LocomoAdapter)` in `BenchmarkAdapter`.
- Unit tests in `cognee/tests/unit/eval_framework/` with a tiny synthetic `locomo` fixture:
  category mapping, adversarial answer field, evidence resolution, `max_sessions` truncation,
  `_filter_instances` interplay.

Deliverable check: `cognee eval --benchmark LoCoMo --engine direct_llm --limit 20 --no-dashboard`
runs end to end on the flattened single-document representation. This is the first real smoke
test of GPT-5.1-mini through the whole pipeline and needs no new ingestion code.

### 2.2 `eval_framework/locomo/preprocess.py` — session JSON-list files
Output: `temp/locomo_preprocessed/conv_<idx>/session_<NN>.json`, each a JSON list of strings,
plus a `manifest.json` (speakers, session dates, turn counts, chunk-size audit).

Representation decisions (mirroring BEAM's `format_turn_pair` but for two humans):
- Item = a window of consecutive turns within one session (default 6 turns, ~150–250 words),
  never spanning sessions. One turn per item would give ~600 chunks per conversation and ~6k
  extraction calls total, too fine for graph extraction; whole-session items blow past useful
  chunk sizes for entity extraction. Window size is a CLI flag so it can be swept.
- Every item starts with a header: `Session 3 — 1:56 pm on 8 May, 2023 — Caroline and Melanie`.
  Temporal questions (category 2) are answerable only if the session date is inside the chunk.
- Each turn: `Caroline: <text>`; image turns append `[shared a photo: <blip_caption>]`.
  `dia_id` is kept in the manifest, not in the text, so it does not pollute extraction.
- No LLM compression step: LoCoMo turns are short and clean.

## 3. Ingestion (Phase 2, ~1 day)

`eval_framework/locomo/ingest.py`, structured like `beam/local_ingest.py`:

- One subprocess per conversation with `DATA_ROOT_DIRECTORY` / `SYSTEM_ROOT_DIRECTORY` pointed at
  `temp/locomo_runs/<run_id>/conv_<idx>/{data,system}`. This gives 10 fully isolated on-disk
  memories that stay re-queryable for later sweeps (BEAM pruned between conversations; we should
  not, because we will want to re-run QA many times without re-paying for ingestion).
- Per conversation: `cognee.add(session_files, dataset_name="locomo_conv_<idx>")`, then
  `cognee.cognify(datasets=[...], chunker=JsonListChunker)`, then optionally
  `global_context_index_pipeline` (flag, default on: cheap and BEAM found it useful for temporal
  links).
- Session distillation OFF in v1: `session_io.parse_turns` expects `User:`/`Assistant:` blocks and
  the distillation logic is about a user's preferences toward an assistant; LoCoMo is two humans.
  Phase 6 can test mapping `speaker_a → user`, `speaker_b → assistant`.
- Writes `ingestion_report.json` per conversation: chunk counts, wall time, LLM token usage from
  the usage recorder, so we can report ingestion cost.

Sizing (rough, verify on the smoke run): ~24k tokens of dialogue per conversation, ~100 windowed
chunks per conversation, ~1k chunks overall → ~1k extraction + ~1k summarization calls. At mini
pricing this is single-digit to low-double-digit dollars for the whole benchmark, and roughly an
hour wall-clock with default concurrency.

## 4. Question answering (Phase 3, ~1 day)

`eval_framework/locomo/run_eval.py`:
- Loads questions via `LocomoAdapter`, groups them by conversation, and for each conversation
  points the env at that conversation's roots (subprocess again) and calls
  `run_retriever_sweep_for_questions(...)` from `sweeps/retriever_sweep_runner.py` with
  `evaluation_engine="LocomoEval"`.
- Sweep config JSON (`eval_framework/locomo/configs/locomo_v1.json`) with three variants, so the
  first full run already yields an ablation:
  - `hybrid_completion_20_20` (BEAM's reported config),
  - `cognee_completion_20` (chunk-RAG baseline — the fair comparison point to mem0/Zep's "RAG"),
  - `cognee_graph_completion_20`.
- `qa_prompt_paths` per category, files under `eval_framework/locomo/qa_prompts/`:
  `DEFAULT`, `single_hop`, `multi_hop`, `temporal`, `open_domain`, `adversarial`. All demand a
  short answer (a phrase, a date, a name), no preamble, because token-F1 against 2–6 word gold
  answers is the paper metric. `temporal` tells the model to resolve relative expressions
  ("last week") against the session date in the context. `adversarial` tells it to answer
  "Not mentioned in the conversation" when evidence is absent.
- `AUTO_FEEDBACK=false`, no `session_id`: pure graph/vector retrieval, no per-turn analysis call.
- Answer cache is already in the sweep runner, so a crash resumes.

## 5. Evaluation (Phase 3, same day)

`eval_framework/locomo/eval_adapter.py` — `LocomoEvalAdapter`, async, modeled on
`BeamEvalAdapter`, registered as
`LOCOMO = ("LocomoEval", "cognee.eval_framework.locomo.eval_adapter", "LocomoEvalAdapter", None)`
in `EvaluatorAdapter`. Metrics:

- `f1`: SQuAD-style token F1 (lowercase, strip punctuation and articles, whitespace split). This
  is the LoCoMo paper's headline metric. Pure Python, no deepeval.
- `llm_judge`: binary CORRECT/WRONG with the gold answer visible to the judge, ported from the
  mem0 evaluation prompt (that is what mem0, Zep and most 2025 reports quote as "LoCoMo accuracy").
  For adversarial questions the judge credits an explicit abstention.
- Judge model: run the evaluation step under a different `LLM_MODEL` (recommend `openai/gpt-5.1`)
  by giving `run_eval.py` a `--judge-env` override, so the answerer stays mini while the judge is
  strong. Judging is cheap (~2k questions × a few hundred tokens).

Aggregation (`eval_framework/locomo/aggregate.py`): per category and overall, per conversation
and pooled, mean F1 and judge accuracy with bootstrap CIs (reuse `metrics_calculator.bootstrap_ci`),
plus the "mem0-comparable" slice (categories 1–4 only). Output JSON + a markdown table.

## 6. Run protocol (Phase 4–5, ~2 days incl. waiting)

1. **Smoke**: conversation 0, `--max-sessions 3`, 30 questions, `hybrid_completion_20_20`. Check:
   schema-native path taken, chunks carry session dates, answers are short, adversarial answers
   abstain, F1 and judge agree on obvious cases.
2. **Single full conversation**: conversation 0, all sessions, all questions, all three variants.
   Read 20 wrong answers by hand; fix prompts/windowing once. This is the only tuning loop.
3. **Freeze** prompts + config. Ingest conversations 1–9.
4. **Full run**: all 10 conversations × 3 variants × 3 QA+judge repeats on the frozen ingestion.
   Report mean and run std, like the BEAM report.
5. **Report**: `cognee/eval_framework/locomo/REPORT.md` with the same sections as BEAM's, plus the
   published numbers we compare against (mem0 paper's table: Mem0, Mem0-graph, Zep, LangMem,
   OpenAI memory, full-context, RAG — pull the exact figures from the paper when writing, do not
   quote from memory), and a clear statement of which categories and which metric each number uses.

Ballpark budget for the whole thing (ingestion once + 3 variants × 3 repeats + judging): well
under $100 in API spend; a few hours of wall clock per full QA pass at concurrency 8–10.

## 7. Optional tuning (Phase 6, only if the headline is disappointing)

- Sweep `chunks_top_k`/`entities_top_k` (10/20/40) and window size (4/6/10 turns).
- `include_global_context_index=True` in the hybrid retriever.
- Session distillation with the speaker→user/assistant mapping.
- Per-category routing (temporal → `TEMPORAL`-aware retriever; multi_hop → CoT variant), the way
  BEAM's 10M run routed by type. Note this uses gold category labels and must be disclosed.
- Contradiction detection on ingestion (`CONTRADICTION_DETECTION=true`), relevant to knowledge
  updates inside LoCoMo conversations.

## 8. Decisions needed from you

1. **Model id**: if `openai/gpt-5.1-mini` does not exist on the API, fall back to `openai/gpt-5-mini`
   or pick another? (Verified in Phase 0 before any spend.)
2. **Judge model**: mini as well (cheapest, but self-judging) or `gpt-5.1` (recommended)?
3. **Adversarial category**: run all five categories and report both slices (recommended), or drop
   category 5 like mem0?
4. **Headline metric**: token F1 (paper) vs LLM-judge accuracy (what competitors quote). Plan reports
   both; pick one for the title line.

## 9. Files to add

```
cognee/eval_framework/benchmark_adapters/locomo_adapter.py
cognee/eval_framework/benchmark_adapters/benchmark_adapters.py        (register)
cognee/eval_framework/evaluation/evaluator_adapters.py                (register LocomoEval)
cognee/eval_framework/locomo/__init__.py
cognee/eval_framework/locomo/preprocess.py
cognee/eval_framework/locomo/ingest.py
cognee/eval_framework/locomo/run_eval.py
cognee/eval_framework/locomo/eval_adapter.py
cognee/eval_framework/locomo/metrics/{f1.py,llm_judge.py}
cognee/eval_framework/locomo/aggregate.py
cognee/eval_framework/locomo/model_registry.py                        (litellm.register_model for gpt-5.1-mini)
cognee/eval_framework/locomo/configs/locomo_v1.json
cognee/eval_framework/locomo/qa_prompts/{default,single_hop,multi_hop,temporal,open_domain,adversarial}.txt
cognee/eval_framework/locomo/prompts/locomo_judge_prompt.txt
cognee/tests/unit/eval_framework/locomo_adapter_test.py
cognee/tests/unit/eval_framework/locomo_preprocess_test.py
cognee/tests/unit/eval_framework/locomo_metrics_test.py
cognee/eval_framework/README.md, evals/README.md                      (mention LoCoMo)
```
