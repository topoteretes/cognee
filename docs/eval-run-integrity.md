# Eval run integrity

`cognee eval` / `run_eval` record what a run did and refuse to present a partly failed run as a clean one.
Nothing below changes how a run fails unless noted under "What changes by default".

## Answer status

Every record in the answers file has a `status`:

| status | meaning | scored? |
| --- | --- | --- |
| `ok` | the retriever returned an answer | yes |
| `empty` | the retriever returned a blank answer | yes (a miss), counted separately |
| `retrieval_error` | retrieval raised | no |
| `llm_error` | completion raised after retrieval | no |

A failed record has `answer: ""` and an `error` string (`ExcType: message`). It is not sent to the
evaluator but keeps its place in the metrics file with `"metrics": {}`, so the files stay aligned.

One policy decides what a failed record is (`integrity.status.answer_status`): an explicit `status`
wins; a record with no `status` is `ok`, except when its answer starts with `ERROR:`, the text the old
code stored for a failure, which is read as `llm_error`. An unknown `status` raises `RunIntegrityError`.

Who records failures:

- `BEAMRouter` and the retriever sweeps (LoCoMo, BEAM) already swallowed answer errors; they now
  store `status` and `error` instead of `"ERROR: ..."` as the answer.
- The standard runner (`AnswerGeneratorExecutor`) still raises on the first answer error, as before.
  Setting `max_answer_failure_rate` (`--max-answer-failure-rate`) opts in to recording the failure and
  continuing; the run then fails only if the failed fraction exceeds that rate.

## Metric coverage

A metric that fails after its retries has `score: null` (plus an `error` naming the exception for
DeepEval). Means and confidence intervals use scored entries only, so each metric in
`aggregate_metrics.json` now also carries
`coverage: {scored, total, unscored, unscored_fraction}` and `answers: {total, ok, empty, failed}`.
`total` is every question in the metrics file: answers that failed count as unscored, so 50 failed
answers out of 100 reports `scored 50 / total 100`. The run summary always prints the answer counts.

## Run manifest

`run_manifest.json` is written next to the run's artifacts (atomically, after every stage change):
schema version, the result-determining config, an answers fingerprint, and per stage (`corpus`,
`answers`, `evaluation`, `dashboard`) a status, timestamps, error, counts, failed question ids and
metric coverage. It also holds the findings of `validate_run`: missing or orphaned entries,
duplicates and empty golden answers, matched by a digest of question and golden answer.
A stage that a caller replaced and that wrote no file is recorded without counts; with
`--strict-integrity` that is an error.

## Resume

`--resume` (`EvalConfig.resume`) continues an interrupted run in the same `--output-dir`. Without a
manifest it raises. The manifest is the only gate for reusing earlier work:

- the run config (everything except output paths, dashboard and the integrity flags) must equal the
  manifest's, otherwise `RunIntegrityError`;
- the answers fingerprint (config, SHA-256 of the questions file, system prompt name and text, `top_k`,
  LLM model/provider/temperature) must match, otherwise `RunIntegrityError`.

Answers are journaled to `<answers_path>.journal.jsonl` as they are produced. A resumed run only
appends to the journal (a partial last line from a crash is cut off and logged), reuses every `ok`
answer and asks the rest again, failed and empty ones included. A journal is never reused without a
verified resume. The built corpus is reused when the manifest says it completed.

Only one runner may use a run directory: `run.lock` is created with `O_EXCL` and removed when the run
ends. A killed runner leaves it behind; it is never taken over automatically, delete it by hand once
no run is active.

## Strict mode

All thresholds are yours; none has a default and unset means failures are reported but never fatal.
A fraction exactly equal to the threshold is allowed.

- `--max-answer-failure-rate R`: record answer failures and fail if more than fraction `R` failed.
- `--max-unscored-rate R`: fail if any metric left more than fraction `R` of its questions unscored.
- `--strict-integrity`: fail if `validate_run` finds an error or a stage wrote no file.

Failures raise `RunIntegrityError` after the manifest has been written.

## What changes by default

- Answers and metrics gain `status`/`error` fields; aggregate metrics gain `coverage` and `answers`.
  Older readers ignore the extra keys. Answers files without `status` still load.
- `run_eval` writes `run_manifest.json` and `run.lock` in the run directory, and fails with
  `RunIntegrityError` (not `FileNotFoundError`/`ValueError`) for a missing, truncated or wrongly
  shaped answers or metrics file.
- Answers and metrics files are written atomically.
- A retriever result that is a non-list is wrapped as one answer, and an empty result list is now an
  `empty` answer instead of an `IndexError`.
- `run_evaluation` returns `[]` whenever `calculate_metrics` is off (before it returned an unbound
  variable in some paths).
- `EvalConfig.to_dict()` has four new keys (`resume`, `max_answer_failure_rate`, `max_unscored_rate`,
  `strict_integrity`); its consumers (`run_eval`, `modal_run_eval`, `run_beam_eval`, the LoCoMo and BEAM
  sweeps) pass the dict through and ignore unknown keys.

## Not covered

The LoCoMo and BEAM sweep runners share `integrity.answering` and the status field but write no
manifest and have no resume. `BeamEvalAdapter`, `DirectLLMEvalAdapter` and the LoCoMo judge do not
turn evaluator exceptions into `score: null` with an `error`. The dashboard does not show coverage.
