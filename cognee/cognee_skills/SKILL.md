---
name: skill self-improvement
description: >
  Inspect why a skill keeps failing, generate improved instructions,
  apply the fix, and evaluate whether it helped.
  Triggers: "this skill keeps failing", "fix this skill",
  "why does X fail", "improve this skill's instructions",
  "the skill isn't working", "apply an amendment", "rollback the fix".
---

# Skill Self-Improvement

This skill diagnoses failing skills and fixes them. When a skill
produces bad results repeatedly, use this to identify the root cause,
generate improved instructions, apply the fix, and verify it helped.

## When to Activate

- A skill has failed or produced poor results multiple times
- You want to understand why a skill is underperforming
- You want to review a proposed fix before applying it
- You want to revert a fix that didn't improve things
- You want to compare success scores before and after a fix

## Available Tools

| Tool | Description |
|------|-------------|
| `inspect_skill` | LLM analyzes failed runs → root cause, severity, improvement hypothesis |
| `preview_amendify_skill` | LLM generates improved instructions (not applied yet) |
| `amendify_skill` | Apply a proposed amendment to the skill in the graph |
| `rollback_amendify_skill` | Revert to original instructions |
| `evaluate_amendify_skill` | Compare pre/post amendment success scores |
| `auto_amendify_skill` | Full pipeline: inspect → preview → apply in one call |

## Process

### Manual (review before applying)

Use this when the skill is important enough that you want to read the
proposed changes before they go live.

1. Call `inspect_skill(skill_id)`
   - Returns `failure_category`, `root_cause`, `severity`, `improvement_hypothesis`
   - Returns `None` if there aren't enough failed runs yet
2. Review the diagnosis — does the root cause match what you've observed?
3. Call `preview_amendify_skill(skill_id)` to generate a fix
   - Returns `amendment_id`, `change_explanation`, `amended_instructions`, `amendment_confidence`
   - Nothing is changed yet — the original is still active
4. Read the `change_explanation` and `amended_instructions`
5. Call `amendify_skill(amendment_id)` to apply
6. After more runs accumulate, call `evaluate_amendify_skill(amendment_id)`
7. If scores didn't improve, call `rollback_amendify_skill(amendment_id)`

### Automatic (trust the LLM to fix it)

Use when you want the full pipeline without manual review.

```
auto_amendify_skill(skill_id, min_runs=3)
```

Runs inspect → preview → apply in one step. Returns the full inspection,
amendment proposal, and apply result.

## Decision Guide

| Situation | Action |
|-----------|--------|
| Skill fails, want to understand why | `inspect_skill` |
| Want to see the fix before applying | `preview_amendify_skill` → review → `amendify_skill` |
| Trust the LLM, just fix it | `auto_amendify_skill` |
| Fix applied, want to check if it helped | `evaluate_amendify_skill` |
| Fix didn't help | `rollback_amendify_skill` |

## Guidelines

1. `inspect_skill` needs at least 1 failed run (default) to produce a
   diagnosis — more runs produce more accurate root causes
2. `amendment_confidence` below 0.6 is a signal to use the manual
   flow and review the proposed instructions before applying
3. Run `evaluate_amendify_skill` after a few post-amendment executions,
   not immediately — give the skill time to accumulate new results
4. `rollback_amendify_skill` restores the original instructions but
   keeps the amendment and inspection history in the graph
5. Each amendment stores the original instructions — rollback is always
   available regardless of how many amendments have been applied
