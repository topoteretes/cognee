# LoCoMo on cognee

Runs the [LoCoMo](https://github.com/snap-research/locomo) long-term conversational memory
benchmark on cognee: dated 6-turn windows through `remember()`, hybrid retrieval, one answer
prompt for every question, and an LLM judge.

| Configuration | Accuracy |
| --- | ---: |
| Default retrieval depth (5/5/5), adapted ingestion prompts, global context index | **89.7%** |
| Deeper retrieval (20/20/20), standard ingestion | **91.3%** |

Share of 1,762 questions answered correctly according to `gpt-5.1`, with answers by
`gpt-5-mini`; open-domain questions and 143 questions with a corrupted gold answer are left
out. The method, the two configurations and the full analysis are in the technical report,
available soon. Answers, verdicts and run configurations are in
[`report_artifacts/`](report_artifacts/).

## Running

```bash
# deeper retrieval, standard ingestion
uv run python -m cognee.eval_framework.locomo.run_locomo_eval --conversations 0-9

# default retrieval depth, adapted ingestion prompts
uv run python -m cognee.eval_framework.locomo.run_locomo_eval --conversations 0-9 \
    --resolve-dates \
    --extraction-prompt cognee/eval_framework/locomo/prompts/graph_extraction_conversations.txt \
    --summary-prompt cognee/eval_framework/locomo/prompts/summarize_content_questions.txt \
    --global-context-index \
    --config-json-path cognee/eval_framework/locomo/configs/locomo_hybrid_5_5_5_gci.json
```

## Layout

- `run_locomo_eval.py` — the driver: one command for the whole benchmark
- `ingest.py`, `answer.py` — one conversation each, started by the driver: ingestion, then
  answers and judge; the four ingestion options are documented in `ingest.py`
- `preprocess.py` — the dated 6-turn windows and the overview document
- `dates.py` — relative dates resolved before ingestion (`--resolve-dates`)
- `aggregate.py` — the summary tables of a run
- `metrics/llm_judge.py`, `eval_adapter.py` — the LLM judge and its hook into the harness
- `prompts/` — the judge prompt, the date-resolution prompt and the two adapted ingestion prompts
- `qa_prompts/unified.txt` — the answer prompt
- `configs/` — retrieval settings: 20/20/20, and 5/5/5 with the global context index
- `report_artifacts/` — per-question answers and verdicts, summaries, ingestion reports, run
  configurations and gold exclusions: the first 20/20/20 run at the top level, the 5/5/5 run in
  `topk5_adapted_prompts/`, a second 20/20/20 run in `topk20_run2/`
