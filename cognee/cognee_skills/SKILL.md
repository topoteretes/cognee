---
name: skill self-improvement
description: >
  Execute skills, record outcomes, and automatically fix failing skills.
  Triggers: "run this skill", "this skill keeps failing", "fix this skill",
  "why does X fail", "improve this skill's instructions",
  "the skill isn't working", "apply an amendment", "rollback the fix".
---

# Skill Self-Improvement

Skills that fix themselves. When a skill produces bad results repeatedly,
the system diagnoses why, generates improved instructions, applies the fix,
and verifies it helped — automatically or step-by-step.

## When to Activate

- User wants to run a skill against a task
- A skill has failed or produced poor results multiple times
- You want to understand why a skill is underperforming
- You want to review a proposed fix before applying it
- You want to revert a fix that didn't improve things

## Available Tools

### Execute

| Tool | Description |
|------|-------------|
| `run_skill` | **Start here.** Find the best skill and execute it. One call does everything: route → execute → observe → self-repair on failure |
| `execute_skill` | Execute a specific skill by ID (use when you already know which skill) |
| `observe_skill_run` | Record an execution outcome — feeds the self-improvement loop and routing preferences |

### Self-improvement

| Tool | Description |
|------|-------------|
| `inspect_skill` | LLM analyzes failed runs → root cause, severity, improvement hypothesis |
| `preview_amendify_skill` | LLM generates improved instructions (not applied yet) |
| `amendify_skill` | Apply a proposed amendment to the skill in the graph |
| `rollback_amendify_skill` | Revert to original instructions |
| `evaluate_amendify_skill` | Compare pre/post amendment success scores |
| `auto_amendify_skill` | Full pipeline: inspect → preview → apply in one call |

### Management

| Tool | Description |
|------|-------------|
| `ingest_skills` | Parse SKILL.md files and store in graph + vector |
| `upsert_skills` | Re-ingest: skip unchanged, update changed, remove deleted |
| `remove_skill` | Delete a skill from graph and vector |
| `list_skills` | List all ingested skills with summaries |
| `load_skill` | Load full details for a skill by ID |
| `get_skill_context` | Ranked skill recommendations for a task |

## Process

### One-call (recommended)

```
run_skill(task_text="compress this conversation")
```

Finds the best skill, executes it, records the outcome, and repairs
the skill if it fails. This is the default for most use cases.

### Manual self-improvement (review before applying)

Use when the skill is important enough to review proposed changes.

1. Call `inspect_skill(skill_id)` — returns `failure_category`, `root_cause`, `severity`, `improvement_hypothesis`. Returns `None` if not enough failures yet.
2. Review the diagnosis — does the root cause match what you've observed?
3. Call `preview_amendify_skill(skill_id)` — returns `amendment_id`, `change_explanation`, `amended_instructions`, `amendment_confidence`. Nothing is changed yet.
4. Read the `change_explanation` and `amended_instructions`
5. Call `amendify_skill(amendment_id)` to apply
6. After more runs accumulate, call `evaluate_amendify_skill(amendment_id)`
7. If scores didn't improve, call `rollback_amendify_skill(amendment_id)`

### Automatic self-improvement

```
auto_amendify_skill(skill_id, min_runs=3)
```

Runs inspect → preview → apply in one step without manual review.

## Decision Guide

| Situation | Action |
|-----------|--------|
| Run a skill against a task | `run_skill` |
| Run a specific skill you already know | `execute_skill` |
| Skill fails, want to understand why | `inspect_skill` |
| Want to see the fix before applying | `preview_amendify_skill` → review → `amendify_skill` |
| Trust the LLM, just fix it | `auto_amendify_skill` |
| Fix applied, want to check if it helped | `evaluate_amendify_skill` |
| Fix didn't help | `rollback_amendify_skill` |

## Guidelines

1. **Start with `run_skill`** — it handles routing, execution, observation, and self-repair in one call
2. `inspect_skill` needs at least 1 failed run (default) to produce a diagnosis — more runs produce more accurate root causes
3. `amendment_confidence` below 0.6 is a signal to use the manual flow and review proposed instructions before applying
4. Run `evaluate_amendify_skill` after a few post-amendment executions, not immediately — give the skill time to accumulate new results
5. `rollback_amendify_skill` restores the original instructions but keeps the amendment and inspection history in the graph
6. Each amendment stores the original instructions — rollback is always available regardless of how many amendments have been applied
7. `observe_skill_run` is called automatically by `run_skill` and `execute_skill` — you only need to call it manually for custom execution flows
