---
name: skill-routing
description: >
  Route tasks to the best skill using cognee's skill memory.
  Use before starting any complex task to find which skill fits best,
  and after completing a task to record whether the skill worked.
  Triggers: "which tool should I use", "find the right skill",
  "what's the best approach for this", or at the start of any
  multi-step task where tool selection matters.
---

# Skill Routing

This skill selects the best tool/skill for a task using semantic
matching and historical performance, then records outcomes to
improve future routing.

## When to Activate

- Before starting a task that could be handled by multiple skills
- When unsure which tool or approach to use
- After completing a task, to record whether the chosen skill worked

## Available Tools

| Tool | Description |
|------|-------------|
| `get_skill_context` | Find best skills for a task (semantic search + learned preferences) |
| `load_skill` | Load full skill details by skill_id |
| `list_skills` | List all skills currently ingested in the knowledge graph |
| `observe_skill_run` | Record an execution outcome (success/failure) |
| `promote_skill_runs` | Promote cached runs to the graph (updates preference weights) |
| `ingest_skills` | Parse SKILL.md files from a folder and store in the knowledge graph |
| `upsert_skills` | Re-ingest a folder: skip unchanged, update changed, remove deleted |
| `remove_skill` | Remove a single skill by skill_id |

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
3. Trust prefers_score > 0 as a signal that a skill has worked before
   for similar tasks
4. If vector_score is high but prefers_score is 0, the skill is
   semantically relevant but untested -- proceed but observe carefully
5. Call `promote_skill_runs` periodically (e.g. end of session) to
   bake learning into the graph
6. Use `upsert_skills` rather than `ingest_skills` when skills may
   already be ingested -- it avoids duplicates and only re-processes
   what changed
