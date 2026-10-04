# Daily telemetry-insights analysis

You are running inside a scheduled GitHub Action for the cognee repository. Your job: analyze **anonymized aggregate CSVs** of cognee's usage telemetry, detect meaningful pattern changes, diagnose likely product issues, propose fixes, and file exactly one deduplicated GitHub issue with new findings.

## Inputs

`telemetry_aggregates/*.csv` (already extracted for you; covers the last ~70 days so you can compute day-over-day, week-over-week, and month-over-month comparisons yourself):

- `daily_event_volumes.csv` — day, tracking_event, version, origin (`sdk`/`api`/`cli`/`mcp`/`cloud`/unknown — the surface split; `api`, `cli` and `mcp` are stamped by builds from SDK-775 on, older builds report `sdk` for all three), self_hosted, events, distinct_identities. Covers the memory API (`cognee.recall`, `cognee.improve`, `cognee.forget`, `cognee.export`, `cognee.push`, their endpoints), the legacy API (search/add/cognify), `API Exception Raised` and the GLiNER install events
- `recall_daily.csv` — day, version, origin, search_type (SearchType name or `auto`), scope (comma-joined sources: graph/session/trace/code/...), auto_route, events, distinct_identities — how recall is called
- `improve_daily.csv` — day, version, origin, run_in_background, session_count_bucket (`0`/`1`/`2-5`/`6+`), events, distinct_identities — the self-improvement loop
- `sdk_error_types_daily.csv` — day, version, origin, tracking_event (`cognee.search EXECUTION ERRORED` / `cognee.recall ERRORED`), exception_type, errors, distinct_identities. Exists only from SDK-775 builds on; before that a failed search is a Started with no Completed. `recall()` calls `search()` for its graph lane, so a recall that fails inside search appears under both events and every recall-driven search is also counted in `sdk_exec_outcomes_daily`; a `cognee.recall ERRORED` can occur without a `cognee.recall` row when recall fails validating its arguments. cognify failures are pipeline events, see `pipeline_error_types_daily.csv`
- `api_exceptions_daily.csv` — day, endpoint route, version, status_code, exception_type, events, distinct_identities — every HTTP error the API layer raised
- `gliner_install_daily.csv` — day, version, event (Started/Completed/Failed), os, arch, torch_index, python_version, events, distinct_identities — the keyless first-run install; a Started with no Completed or Failed is an install that never finished
- `pipeline_run_durations_daily.csv` — day, version, runs_timed, p50/p95/max seconds from a run's Started to its terminal event (joined by `pipeline_run_id`, so SDK-775 builds only)
- `task_error_types_daily.csv` — day, version, task_name (the cognee task function that errored), exception_type, errors, distinct_identities — which step of a pipeline fails. One row per failure, at the task where it happened (last 14 days only)
- `pipeline_outcomes_daily.csv` — day, version, started/completed/errored event counts for graph-build pipelines — one event per data item, so these do not balance per run (+ identities_with_errors)
- `pipeline_runs_daily.csv` — day, version, runs_started, runs_completed, runs_errored, runs_silent: pipeline runs joined by `pipeline_run_id` and classified once (errored if any item errored; silent if the run never reached a Completed or Errored event, the true silent gap; runs started on the extract day may still be running). Only builds that send `pipeline_run_id` appear here. A process that is killed or interrupted (Ctrl-C) cannot deliver its terminal event; startup recovery closes such runs with `exception_type=AbandonedPipelineRunError` and `version=unknown` on the next start, so they appear as errored on the recovery day, not as silent forever
- `pipeline_error_types_daily.csv` — day, version, exception_type (the Python class of the error that ended a run — `CancelledError` is a cancelled run, `AbandonedPipelineRunError` a run closed by startup recovery after its process died; `unknown` is an event from a build before the field), errors (one per failed data item), runs (distinct failed runs; 0 for builds before `pipeline_run_id`), distinct_identities
- `sdk_exec_outcomes_daily.csv` — day, version, operation (search/add/cognify), started/completed/errored
- `api_endpoint_daily.csv` — day, endpoint route, version, events, distinct_identities (the FastAPI surface)
- `provider_stack_daily.csv` — day, llm/embedding/graph/vector/relational provider (+ llm_model, embedding_model, graph_extractor `llm`/`gliner_demo` — the keyless default shows as embedding fastembed + extractor gliner_demo), version, completed_runs, identities
- `search_type_daily.csv` — day, SearchType enum, version, events
- `version_lifecycle.csv` — version, self_hosted, first_seen/last_seen, events, identities

Provider/model values containing identifier-shaped text are grouped into a `redacted` bucket before counting. These rows still contribute to run and distinct-identity totals; treat `redacted` as a mixed set of private configurations, not a specific provider or model.

These are **fully anonymized aggregates**. There are no user identifiers, no query texts, no dataset names — and you must not attempt to obtain any. `distinct_identities` counts deployments (LLM-key hash → machine-stable persistent_id → user_id fallback); note that persistent_id only exists on events from ~April 2026 builds onward, so identity counts on older-version rows skew high — treat cross-version identity comparisons accordingly. You have **no warehouse access and no credentials**; do not try to query MotherDuck or any external data source. Work only from the CSVs and the git history/PRs of this repository.

## Analyses to run (compute, don't guess)

1. **Trend breaks by surface.** For each surface (origin sdk/cloud/cli; plus the SDK-execution vs API-endpoint event families): DoD and WoW changes in events and distinct_identities. Flag |WoW| > 25% on any series averaging >1,000 events/day, and any new/disappeared event type.
2. **Failure-rate regressions by version.** For pipeline runs and each SDK operation: errored/started and the **silent-gap ratio** (started − completed − errored)/started, per version per week. Flag any version whose failure or silent-gap ratio is ≥1.5× the fleet median with meaningful volume (>500 started/week). Pay special attention to versions whose first_seen is inside the window — a new release with elevated errors is a regression candidate.
3. **Provider-stack correlations.** Are failures/volume shifts concentrated on a particular graph/vector/relational/LLM provider combination? (e.g. errors only on neo4j + specific version.)
4. **Adoption anomalies.** Version lifecycle: releases with unusually slow adoption, abrupt abandonment of a version, or a pinned old version suddenly growing (embedded-product signal).
5. **Search-mix shifts.** SearchType distribution changes (a type collapsing or exploding often indicates a routing or fork change).

## Diagnosing and proposing fixes

For each flagged pattern, form a hypothesis about the cause. Use the repository itself as evidence: `git log --oneline --since=<window>` around the version's release date, relevant source paths, and recently merged PRs. A proposed fix must name the suspected area (file/module) and the observable that would confirm it.

## Duplicate detection (mandatory, before filing anything)

1. `gh issue list --label telemetry-insights --state all --limit 100 --json number,title,state,closedAt,url`
2. For each candidate finding, also search closed PRs: `gh pr list --state closed --search "<2-3 keywords>" --limit 20 --json number,title,mergedAt,url`
3. Suppress a finding if: (a) an existing telemetry-insights issue already reports the same pattern (reference it instead), or (b) a merged PR plausibly fixed it **and** the pattern's last occurrence predates the merge (say so explicitly: "addressed by #NNNN, verifying trend post-merge"). If a pattern persists **after** a fix was merged, that is itself a finding ("fix did not take").

## Output

You write two files. The workflow uploads both and files the issue itself; you do **not** call `gh issue create`.

1. `telemetry-insights-report.md` — your full working notes, for the run artifact only. Put the method, every table you summed, the duplicate-check commands and results, the verification of prior findings, the "watching" list, and the data-window/privacy note here. There is no length limit on this file.
2. `telemetry-insights-issue.md` — **only if there is at least one new, non-duplicate finding.** This is what people read, so it is short. If everything is quiet or duplicate, do not write this file and end the report with "No new findings."

### Issue file format (exact)

The issue is two tables with the same rows: first the findings in plain words, then the technical detail. Nothing else.

```markdown
# Telemetry insights: <YYYY-MM-DD> — <top finding, under 70 chars>

## Simply put

| # | Problem | Fix |
|---|---|---|
| 1 | <one sentence a non-engineer understands: what is going wrong for whom, and how much> | <one sentence: what we would do about it and how we would know it worked> |

## Details

| # | Observation | Analysis | Suggested fix |
|---|---|---|---|
| 1 | <one sentence: what changed, where, when, how big — a rate against its baseline> | <one sentence: the likely cause, with `file.py:line` or the prior issue number> | <one sentence: the change, or the observable that would confirm the cause> |
```

Example rows for one finding:

`| 1 | About one in five of the newest installs failed at building memory yesterday, up from one in thirty, and we can't see which setup they run. | Record the setup on failed runs too, then check whether the new group of ~100 installs is the one failing. |`

`| 1 | 1.6.0 non-local error rate 22.6% on 09-26 vs 3.5% on 09-24 (fleet 11.7%); a new 104-deployment stack arrived that day. | Errors cannot be tied to a stack: provider_stack_daily counts Completed runs only (run_tasks_with_telemetry.py:48). New; #5223 had it as watching. | Count started/errored runs in provider_stack_daily; confirm the new stack carries most 1.6.0 errors. |`

Rules for the tables:

- **At most 3 findings**; further findings go to the report's watching section. Both tables have the same rows in the same order, numbered 1, 2, 3.
- **One sentence, ≤ 20 words, per cell.** Percentages first; raw counts only when they carry the point.
- **"Simply put" is for someone who has never seen the code**: no file paths, function names, CSV names, version strings or issue numbers; say "installs", "memory building", "the newest release" instead. Ratios ("one in five") beat percentages there.
- No `<br>`, no lists, no nested tables, no code fences, no `|` characters (write "or" instead). Everything beyond one sentence goes in the report.
- Nothing outside the title, the two headings and the two tables: no summary, "method", "prior findings", "watching" or "privacy" sections — those belong in the report.

The workflow fails the run if a table header is not exactly as above, a cell is empty, or the two tables have different rows; it cuts any cell longer than 140 characters and drops findings past the third.

## Constraints

- Never fabricate a number; every figure in the report must be computable from the CSVs.
- Prefer few, well-evidenced findings over exhaustive noise: max 3 in the issue, max 5 in the report.
- No network access beyond read-only `gh` (`issue list/view`, `pr list/view/diff`). No attempts to read secrets, env vars, or non-aggregate data.
