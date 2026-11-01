# LoCoMo on cognee

Runs the [LoCoMo](https://github.com/snap-research/locomo) long-term conversational memory benchmark on cognee: dated 6-turn windows through `remember()`, hybrid retrieval, one answer prompt for every question, and an LLM judge.

Cognee answered about **91%** of 1,762 evaluated questions correctly according to `gpt-5.1`, with answers by `gpt-5-mini` from hybrid retrieval 20/20/20 over memories built with Cognee's standard ingestion prompts. For transparency, we report two runs on the same memories, which scored 90.7% and 91.3%. The evaluated questions leave out open-domain questions and 143 questions whose gold answer an audit flags; [`report_artifacts/README.md`](report_artifacts/README.md#evaluation-scope) gives the rule and the counts.

The [technical report](REPORT.md) describes the method, the scope and the grading rules, and an additional configuration: the hybrid retriever's default depth (5/5/5) with adapted ingestion and the global context index, 89.7% in one run. Answers, verdicts, summaries and run configurations of the 91.3% run and of the additional configuration are in [`report_artifacts/`](report_artifacts/).

## Running

```bash
# main configuration: hybrid retrieval 20/20/20, standard ingestion
uv run python -m cognee.eval_framework.locomo.run_locomo_eval --conversations 0-9

# additional configuration: default retrieval depth 5/5/5 with the global context index, adapted ingestion
uv run python -m cognee.eval_framework.locomo.run_locomo_eval --conversations 0-9 \
    --resolve-dates \
    --extraction-prompt cognee/eval_framework/locomo/prompts/graph_extraction_conversations.txt \
    --summary-prompt cognee/eval_framework/locomo/prompts/summarize_content_questions.txt \
    --global-context-index \
    --config-json-path cognee/eval_framework/locomo/configs/locomo_hybrid_5_5_5_gci.json
```

Each command writes a fresh run directory (`temp/locomo_runs/<timestamp>` unless `--run-dir` is given) and summarises it into `locomo_summary.json` and `locomo_summary.md` there. The dataset is downloaded to `temp/locomo_data/locomo10.json` when missing; compare its SHA-256 with `dataset_sha256` in the run configurations of `report_artifacts/`. Values in a `.env` file take precedence over the settings the driver passes to its child processes, such as `LLM_MODEL` and the database locations, so keep the file consistent with the driver's options.

Only evaluated questions are answered and judged: `answer.py` drops the excluded ones before retrieval, following [`scope.py`](scope.py) and `--gold-exclusions` (default `report_artifacts/gold_exclusions.json`). The driver refuses to aggregate a run directory that holds answers to excluded questions; [`report_artifacts/README.md`](report_artifacts/README.md#filtering-existing-run-records) shows how to filter such records.

## Layout

- `run_locomo_eval.py` — the driver: one command for the whole benchmark
- `ingest.py`, `answer.py` — one conversation each, started by the driver: ingestion, then answers and judge; the four ingestion options are documented in `ingest.py`
- `scope.py` — which questions are evaluated
- `preprocess.py` — the dated 6-turn windows and the overview document
- `dates.py` — relative dates resolved before ingestion (`--resolve-dates`)
- `aggregate.py` — the summary tables of a run
- `metrics/llm_judge.py`, `eval_adapter.py` — the LLM judge and its hook into the harness
- `prompts/` — the judge prompt, the date-resolution prompt and the two adapted ingestion prompts
- `qa_prompts/unified.txt` — the answer prompt
- `configs/` — retrieval settings: 20/20/20, and 5/5/5 with the global context index
- `report_artifacts/` — per-question answers and verdicts of one run of each configuration (evaluated questions only), their summaries, ingestion reports and run configurations, and the gold exclusions
- `REPORT.md` — the technical report
