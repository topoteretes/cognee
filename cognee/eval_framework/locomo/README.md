# LoCoMo on cognee

[LoCoMo](https://github.com/snap-research/locomo) is a long-term conversational memory
benchmark: 10 multi-session conversations between two people (19–32 dated sessions, 370–690
turns each) and 1,986 questions in five categories — single-hop, multi-hop, temporal,
open-domain, and adversarial (unanswerable).

This package ingests each conversation through cognee's **memory API** in one of two modes
(`--ingest-mode`):

- **`remember`** (default) — one document per LoCoMo session (a dated header plus every
  turn) and a short overview document with the speakers and the session timeline, all through a
  single `remember(documents, dataset_name=...)`, i.e. `add` + `cognify` + `improve()` (triplet
  embeddings, no global context index). With the default chunk budget a session fits in one
  chunk, so the graph is built from ~20 dated chunks per conversation. This measures cognee's
  graph built the way any user builds it.
- **`sessions`** — the session-memory path:
  1. `remember(overview, dataset_name=...)` creates the dataset.
  2. Every session is cut into dated windows (default 6 turns, each prefixed with the session
     date) and written to the session cache with `remember(text, session_id=<conversation>_s<nn>)`;
     every window also goes through the session-context analyzer so durable facts/preferences
     become gated session guidance.
  3. `improve(dataset, session_ids=[all sessions], build_global_context_index=True)` is the
     only path by which dialogue reaches the graph: it persists the session windows (add +
     cognify per session, under the `user_sessions_from_cache` node set), distills the gated
     guidance into lessons, updates preference weights and runs the default enrichment.

Questions are then answered by the retrievers listed in a sweep config (`configs/locomo_v1.json`:
hybrid 20/20, hybrid + global context index, graph completion, chunk RAG). The answer prompt
follows one of two protocols (`--prompt-style`):

- **`locomo`** (default) — the official LoCoMo QA protocol (`task_eval/gpt_utils.py`): one
  short-answer system prompt for every question (`qa_prompts/locomo.txt`; adversarial questions
  get the official category-5 variant), temporal questions carry the *"Use DATE of CONVERSATION
  to answer with an approximate date."* hint, and adversarial questions become a two-way choice
  between *"Not mentioned in the conversation"* and the dataset's distractor (option order
  alternates deterministically). The plain question still drives retrieval; the augmented one
  only reaches the LLM. A chosen option letter is mapped back to its text before scoring.
- **`category`** — the per-category prompts in the sweep config (`qa_prompts/*.txt`), plain
  questions.

Both styles use the category label the dataset provides, exactly like the official script does;
say so when reporting numbers.

Answers are scored three ways:

- **`f1_locomo`** (`metrics/f1_locomo.py`) — the **official LoCoMo scorer**
  (`task_eval/evaluation.py`): Porter-stemmed tokens, `and`/commas dropped, comma-split
  sub-answers for multi-hop, first `;`-part of the gold for open-domain, and for adversarial
  questions 1/0 on whether the answer contains "not mentioned" / "no information available".
  Comparable with the paper. Needs `nltk` (in the `evals` extra).
- **`f1`** (`metrics/f1.py`) — plain SQuAD-style token F1, kept for continuity with the harness.
- **`llm_judge`** (`metrics/llm_judge.py`) — binary CORRECT/WRONG with the gold answer visible,
  the number mem0/Zep-style reports quote. Adversarial questions are graded on abstention. The
  judge runs on its own model (`LOCOMO_JUDGE_MODEL`, default `openai/gpt-5.1`) so the answering
  model never grades itself; empty verdicts are retried.

Aggregation (`aggregate.py`) reports overall, per-category and a *mem0-comparable* slice
(adversarial excluded), pooled over conversations, with run-to-run std across repeats.

## Run it

```bash
uv pip install -e ".[dev,evals]"      # once; deepeval is NOT needed
export LLM_API_KEY=...                # or put it in .env

# smoke: the smallest conversation (conv-30 = index 1), three sessions, ~20 questions,
# one retriever, remember() ingestion and the official prompt protocol (both defaults)
uv run python -m cognee.eval_framework.locomo.run_locomo_eval \
    --conversations 1 --max-sessions 3 --max-questions 20 --retrievers hybrid_completion_20_20

# one full conversation, one retriever
uv run python -m cognee.eval_framework.locomo.run_locomo_eval \
    --conversations 1 --retrievers hybrid_completion_20_20

# the session-memory path and the per-category prompts instead
uv run python -m cognee.eval_framework.locomo.run_locomo_eval \
    --conversations 1 --ingest-mode sessions --prompt-style category

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
`preprocessed/` (`remember` mode: the per-session documents; `sessions` mode: windows as
JSON-list files + manifest), `qa/locomo_{questions,answers,metrics,
aggregate_metrics}_conv<i>[_<retriever>_run<r>].json`. Run-level: `locomo_summary.json`,
`locomo_summary.md`, `run_manifest_*.json`.

The dataset is downloaded once to `temp/locomo_data/locomo10.json` (override with
`--data-path` / `LOCOMO_DATA_PATH`). `cognee eval --benchmark LoCoMo` also works for the generic
flattened-transcript corpus path.

## Knobs worth sweeping

`--ingest-mode` and `--prompt-style` (see above), retriever `kwargs` in the sweep config
(`chunks_top_k`, `entities_top_k`, `top_k`, `include_global_context_index`), and — in
`sessions` mode only — `--window-turns` (window granularity), `--skip-turn-analysis` (saves one
LLM call per window) and `--skip-global-context-index`.
