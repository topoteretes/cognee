---
name: skill-routing
description: >
  Route tasks to the best skill using cognee's skill memory.
  Use before starting any complex task to find which skill fits best,
  and after completing a task to record whether the skill worked.
  When a skill keeps failing, inspect it and apply an amendment.
  Triggers: "which tool should I use", "find the right skill",
  "what's the best approach for this", or at the start of any
  multi-step task where tool selection matters.
---

# Skill Routing

This skill selects the best tool/skill for a task using semantic
matching and historical performance, records outcomes to improve
future routing, and self-fixes skills that keep failing.

## When to Activate

- Before starting a task that could be handled by multiple skills
- When unsure which tool or approach to use
- After completing a task, to record whether the chosen skill worked
- When a skill has failed multiple times and needs inspection or amendment

## Available Tools

| Tool | Description |
|------|-------------|
| `get_skill_context` | Find best skills for a task (semantic search + learned preferences) |
| `load_skill` | Load full skill details by skill_id |
| `list_skills` | List all skills currently ingested in the knowledge graph |
| `execute_skill` | Load a skill and execute it against a task via LLM |
| `observe_skill_run` | Record an execution outcome (persists to graph, updates preferences immediately) |
| `ingest_skills` | Parse SKILL.md files from a folder and store in the knowledge graph |
| `upsert_skills` | Re-ingest a folder: skip unchanged, update changed, remove deleted |
| `remove_skill` | Remove a single skill by skill_id |
| `inspect_skill` | Analyze why a skill keeps failing (root cause, severity, hypothesis) |
| `preview_amendify_skill` | Generate improved instructions based on an inspection |
| `amendify_skill` | Apply a proposed amendment to fix a failing skill |
| `rollback_amendify_skill` | Revert an applied amendment |
| `evaluate_amendify_skill` | Compare pre/post amendment success scores |
| `auto_amendify_skill` | Full pipeline: inspect, preview, apply in one call |

## Process

### First-time setup

1. Call `ingest_skills(skills_folder)` with the path to the project's
   skills directory to register all skills in the knowledge graph
2. On subsequent runs, call `upsert_skills(skills_folder)` instead
   to sync changes without re-processing unchanged skills

### Task routing

1. Call `get_skill_context(task_text)` with the user's request
2. Review the ranked results (score, prefers_score, instruction_summary)
3. If needed, call `load_skill(skill_id)` for the top candidate
   to read its full instructions
4. Execute the chosen skill
5. Call `observe_skill_run` with the outcome:
   - task_text: what was asked
   - selected_skill_id: which skill was used
   - success_score: 0.0 (failed) to 1.0 (perfect)
   - result_summary: brief description of what happened

### Self-amendifying (when a skill keeps failing)

1. Call `inspect_skill(skill_id)` to analyze failures
2. Call `preview_amendify_skill(skill_id)` to generate a fix
3. Review the proposed changes
4. Call `amendify_skill(amendment_id)` to apply
5. After more runs, call `evaluate_amendify_skill(amendment_id)` to check improvement
6. If it didn't help, call `rollback_amendify_skill(amendment_id)` to revert

Or call `auto_amendify_skill(skill_id)` for the full pipeline in one step.

### Skill management

- Call `remove_skill(skill_id)` to delete a skill that is no longer needed
- Call `upsert_skills(skills_folder)` after adding, editing, or removing
  SKILL.md files to keep the graph in sync

## Output Format

No direct output to the user -- this skill operates behind the
scenes to pick the right tool. The user sees the output of
whichever skill was selected.

## Guidelines

1. Always check skill context before defaulting to a generic approach
2. Record both successes AND failures -- failures improve future routing
   and enable the self-amendifying loop
3. Trust prefers_score > 0 as a signal that a skill has worked before
   for similar tasks
4. If vector_score is high but prefers_score is 0, the skill is
   semantically relevant but untested -- proceed but observe carefully
5. If a skill keeps failing (success_score < 0.5), use inspect_skill
   to understand why before trying to fix it manually
6. Use `upsert_skills` rather than `ingest_skills` when skills may
   already be ingested -- it avoids duplicates and only re-processes
   what changed
