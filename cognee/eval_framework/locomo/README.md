# LoCoMo on cognee

[LoCoMo](https://github.com/snap-research/locomo) is a long-term conversational memory
benchmark: 10 multi-session conversations between two people (19–32 dated sessions, 370–690
turns each) and 1,986 questions in five categories — single-hop, multi-hop, temporal,
open-domain, and adversarial (unanswerable).

This package runs it through cognee's **memory API**, sessions first:

1. `remember(overview, dataset_name=...)` — a short permanent document with the speakers and the
   session timeline (creates the dataset).
2. For every LoCoMo session the dialogue is cut into dated windows (default 6 turns, each
   prefixed with the session date) and written to the session cache with
   `remember(text, session_id=<conversation>_s<nn>)`. Every window also goes through the
   session-context analyzer so durable facts/preferences become gated session guidance.
3. `improve(dataset, session_ids=[all sessions], build_global_context_index=True)` — the only
   path by which dialogue reaches the graph: persists the session windows (add + cognify under
   the `user_sessions_from_cache` node set), distills the gated guidance into lessons, updates
   preference weights, runs the default enrichment (triplet embeddings) and builds the global
   context index.

Questions are then answered by the retrievers listed in a sweep config (`configs/locomo_v1.json`:
hybrid 20/20, hybrid + global context index, graph completion, chunk RAG) with a short-answer
prompt per category (`qa_prompts/`), and scored two ways:

- **token F1** (`metrics/f1.py`) — the LoCoMo paper's metric, SQuAD normalization;
- **LLM judge** (`metrics/llm_judge.py`) — binary CORRECT/WRONG with the gold answer visible,
  the number mem0/Zep-style reports quote. Adversarial questions are graded on abstention. The
  judge runs on its own model (`LOCOMO_JUDGE_MODEL`, default `openai/gpt-5.1`) so the answering
  model never grades itself.

Aggregation (`aggregate.py`) reports overall, per-category and a *mem0-comparable* slice
(adversarial excluded), pooled over conversations, with run-to-run std across repeats.

## Run it

```bash
uv pip install -e ".[dev,evals]"      # once; deepeval is NOT needed
export LLM_API_KEY=...                # or put it in .env

# smoke: one conversation, three sessions, ~20 questions, one retriever
uv run python -m cognee.eval_framework.locomo.run_locomo_eval \
    --conversations 0 --max-sessions 3 --max-questions 20 --retrievers hybrid_completion_20_20

# full: all ten conversations, every retriever in the config, three QA+judge repeats
uv run python -m cognee.eval_framework.locomo.run_locomo_eval --conversations 0-9 --num-runs 3

# re-score an existing ingestion with another retriever set (ingestion is skipped when its
# report exists; use --force-ingest to redo it)
uv run python -m cognee.eval_framework.locomo.run_locomo_eval \
    --run-dir temp/locomo_runs/<run> --stage answer --retrievers rag_completion_20
```

The driver launches two child processes per conversation (`ingest`, then `answer`) with their
own `DATA_ROOT_DIRECTORY` / `SYSTEM_ROOT_DIRECTORY` under `<run-dir>/conv_NN/`, so the ten
memories are isolated and stay on disk. Before anything expensive it probes both models with a
one-token completion and registers unknown OpenAI ids in litellm's capability table so
structured output keeps the schema-native path.

Models: `--answer-model` (default `openai/gpt-5-mini`; env `LOCOMO_ANSWER_MODEL`) is used for
extraction, distillation and answering; `--judge-model` (default `openai/gpt-5.1`) grades.
`gpt-5.1-mini` does not exist on the OpenAI API at the time of writing — the probe fails fast if
you pass it.

Artifacts per conversation: `ingestion_report.json`, `ingest.log`, `answer.log`,
`preprocessed/` (windows as JSON-list files + manifest), `qa/locomo_{questions,answers,metrics,
aggregate_metrics}_conv<i>[_<retriever>_run<r>].json`. Run-level: `locomo_summary.json`,
`locomo_summary.md`, `run_manifest_*.json`.

The dataset is downloaded once to `temp/locomo_data/locomo10.json` (override with
`--data-path` / `LOCOMO_DATA_PATH`). `cognee eval --benchmark LoCoMo` also works for the generic
flattened-transcript corpus path.

## Knobs worth sweeping

`--window-turns` (chunk granularity), `--skip-turn-analysis` (saves one LLM call per window),
`--skip-global-context-index`, retriever `kwargs` in the sweep config (`chunks_top_k`,
`entities_top_k`, `top_k`, `include_global_context_index`), and the per-category prompts.
