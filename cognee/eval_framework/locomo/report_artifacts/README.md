# Attribution

These artifacts refer to questions of the [LoCoMo](https://github.com/snap-research/locomo) dataset by Maharana et al. (2024), licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), by id, and include error types from the LoCoMo audit by Penfield Labs ([dial481/locomo-audit](https://github.com/dial481/locomo-audit), CC BY-NC 4.0). They are packaged with Cognee answers and judge verdicts; see the [technical report](../REPORT.md) for the results they support.

# Layout

- `topk20_run2/`: a run of the main configuration, hybrid retrieval 20/20/20 over memories built with Cognee's standard ingestion prompts: 1,608 of 1,762 questions correct (91.3%). Its memories are described by `ingestion_reports.json` at this level.
- `topk5_adapted_prompts/`: the additional experiment. Hybrid retrieval 5/5/5 with the global context index, over memories built with resolved dates and adapted extraction and summary prompts: 1,580 of 1,761 scored questions correct (89.7%). It has its own `ingestion_reports.json`, and `date_annotations.json` with the date annotations inserted in each window, by offset into the window text built from `locomo10.json`.
- `ingestion_reports.json`: per conversation, the windows, chunks, graph size, ingestion time and window map (which turns are in which window) of the memories behind `topk20_run2/`.
- `gold_exclusions.json`: the questions whose gold answer an audit flags, with each audit's error type or label and reason. `gold_audit_prompt.txt` is the prompt of our audit.

Each run folder has one per-question file per conversation in `qa/` (ids, category, answer, judge verdict and reason), a `run_config.json` with models, settings, dates and prompt hashes, and a summary (`locomo_summary.json`, `locomo_summary.md`) that `aggregate.py` regenerates from its `qa/` files:

```bash
uv run python -m cognee.eval_framework.locomo.aggregate cognee/eval_framework/locomo/report_artifacts/topk20_run2
```

Question and gold text are not copied here: they come from `locomo10.json` (SHA-256 in each `run_config.json` as `dataset_sha256`), joined on `conversation_index` and `question_idx`, the position of the question in the conversation's `qa` list.

# Evaluation scope

A question is evaluated unless its category is open-domain or an audit flags its gold answer. That is the complete rule, implemented in [`scope.py`](../scope.py).

- Open-domain: 96 questions. They combine what a speaker said with knowledge from outside the conversation.
- Audits (`gold_exclusions.json`, variant `without_corrupted_golds`, which takes both sets together): 99 questions from the external audit by Penfield Labs (its score-corrupting errors; `WRONG_CITATION` entries, where the gold answer is right and only the cited evidence is wrong, stay in) and 93 from ours (final label WRONG, UNSUPPORTED, MALFORMED or AMBIGUOUS with high or medium confidence). 49 questions are flagged by both audits, so 143 are flagged in all, none of them adversarial.
- 15 of the 143 flagged questions are also open-domain, so 96 + 143 − 15 = 224 questions are excluded and 1,762 of the 1,986 remain: 796 single-hop, 236 multi-hop, 284 temporal and 446 adversarial.

Excluding a question removes nothing from ingestion: every conversation is stored in full.

# Runs

The runs in both folders answered and judged all 1,986 questions (`"questions": 1986` in `run_config.json`). The `qa/` files here keep only the 1,762 evaluated questions, each record unchanged and in its original order, and the summaries are computed from them. The `publication` block of each `run_config.json` gives the commit with the complete records (`source_commit`), the number of evaluated questions and the number with a verdict.

In `topk5_adapted_prompts/`, one verdict could not be parsed: conversation 6 (`conv-47`), question 163, where the judge replied with the label `WRNG`. Its score is `null` and its record is kept as it is, so that run has 1,762 evaluated questions and 1,761 scored ones. The unscored question is not counted as wrong.

Another run of the 20/20/20 configuration, answered on 9 October 2026 on the same memories with the same configuration, models and prompts, scored 1,598 of 1,762 evaluated questions (90.7%). Its complete records are in [`report_artifacts/qa/` at commit `012e8a0`](https://github.com/topoteretes/cognee/tree/012e8a039b5116bc30325a9c10192d9c9336e399/cognee/eval_framework/locomo/report_artifacts/qa).

# Filtering existing run records

The benchmark package applies the scope before retrieval: `answer.py` drops excluded questions, so they are never answered or judged, and the driver refuses to aggregate a run directory that holds answers to excluded questions. The scope check never deletes or rewrites files. Records that cover every question, like the runs above, need a one-time filter instead. This snippet, run from the repository root, filtered the two `qa/` folders here; it keeps every evaluated record unchanged and changes nothing when run again:

```python
import json
from pathlib import Path

artifacts = Path("cognee/eval_framework/locomo/report_artifacts")
exclusions = json.loads((artifacts / "gold_exclusions.json").read_text(encoding="utf-8"))
excluded = {
    (q["conversation_index"], q["question_idx"])
    for name in exclusions["variants"]["without_corrupted_golds"]
    for q in exclusions["sets"][name]["questions"]
}

for run in ("topk20_run2", "topk5_adapted_prompts"):
    paths = sorted((artifacts / run / "qa").glob("locomo_metrics_*.json"))
    assert len(paths) == 10, paths
    # Filter every file before writing any, so a malformed file leaves the folder untouched.
    kept = {
        path: [
            row
            for row in json.loads(path.read_text(encoding="utf-8"))
            if row["question_type"] != "open_domain"
            and (row["conversation_index"], row["question_idx"]) not in excluded
        ]
        for path in paths
    }
    for path, rows in kept.items():
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
```

A fresh run of the driver needs no filter: its `conv_NN/qa/` files hold only evaluated questions and aggregate in place.
