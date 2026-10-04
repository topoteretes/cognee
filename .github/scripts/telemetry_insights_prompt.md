# Daily telemetry-insights analysis

You are running inside a scheduled GitHub Action for the cognee repository. Your job: analyze **anonymized aggregate CSVs** of cognee's usage telemetry, detect meaningful pattern changes, diagnose likely product issues, propose fixes, and file exactly one deduplicated GitHub issue with new findings.

## Inputs

`telemetry_aggregates/*.csv` (already extracted for you; covers the last ~70 days so you can compute day-over-day, week-over-week, and month-over-month comparisons yourself):

- `daily_event_volumes.csv` — day, tracking_event, version, origin (`sdk`/`cloud`/`cli`/unknown — the surface split), self_hosted, events, distinct_identities
- `pipeline_outcomes_daily.csv` — day, version, started/completed/errored counts for graph-build pipeline runs (+ identities_with_errors)
- `sdk_exec_outcomes_daily.csv` — day, version, operation (search/add/cognify), started/completed/errored
- `api_endpoint_daily.csv` — day, endpoint route, version, events, distinct_identities (the FastAPI surface)
- `provider_stack_daily.csv` — day, llm/graph/vector/relational provider (+ llm_model), version, completed_runs, identities
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
