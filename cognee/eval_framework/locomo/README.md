# LoCoMo on cognee

Runs the [LoCoMo](https://github.com/snap-research/locomo) long-term conversational memory
benchmark on cognee: each conversation is ingested as dated 6-turn windows through `remember()`,
and questions are answered with hybrid retrieval. There is a single answer prompt
([`qa_prompts/unified.txt`](qa_prompts/unified.txt)), the same for every question. An LLM judge grades the answers.

## Results

Cognee answered **90.7%** of 1,762 questions correctly, according to an LLM judge (`gpt-5.1`).
The total leaves out the open-domain questions and the 143 questions whose gold answer is
corrupted according to an external audit or to ours (listed in
[`report_artifacts/gold_exclusions.json`](report_artifacts/gold_exclusions.json)). The numbers
come from one run on all ten conversations, with answers by `gpt-5-mini`.

The answers and judge verdicts behind these numbers are in
[`report_artifacts/`](report_artifacts/). A technical report with the method and the full
analysis will be available soon.

## Running

```bash
uv run python -m cognee.eval_framework.locomo.run_locomo_eval --conversations 0-9
```
