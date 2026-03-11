# Cognee Skill Routing

You have access to cognee skill routing via MCP. It finds the best skill for a task using semantic search and learned preferences, improves over time as you record outcomes, and can self-fix skills that keep failing.

## Setup (first session only)

If skills have not been ingested yet, call `ingest_skills` with the project's skills folder:

```
ingest_skills(skills_folder="./my_skills")
```

On later sessions, call `upsert_skills` instead to sync any changes without reprocessing unchanged skills:

```
upsert_skills(skills_folder="./my_skills")
```

Replace `./my_skills` with the actual path to the project's SKILL.md folder.

## Task routing workflow

1. Before starting a complex task, call `get_skill_context(task_text)` with a description of what needs to be done
2. Review the ranked results -- higher `score` means better match, `prefers_score > 0` means the skill has worked before for similar tasks
3. If needed, call `load_skill(skill_id)` on the top result to read its full instructions
4. Execute the chosen skill
5. After completing the task, call `observe_skill_run` with the outcome:
   - `task_text`: what was asked
   - `selected_skill_id`: which skill was used
   - `success_score`: 0.0 (failed) to 1.0 (perfect)
   - `result_summary`: brief description of what happened

Runs are persisted to the graph immediately and preference weights update in real-time.

## Self-amendifying workflow

When a skill keeps failing, use the amendify tools to diagnose and fix it:

1. Call `inspect_skill(skill_id)` to analyze why it fails (root cause, severity, improvement hypothesis)
2. Call `preview_amendify_skill(skill_id)` to generate improved instructions
3. Review the proposed changes, then call `amendify_skill(amendment_id)` to apply
4. After more runs, call `evaluate_amendify_skill(amendment_id)` to compare before/after scores
5. If the fix didn't help, call `rollback_amendify_skill(amendment_id)` to revert

Or call `auto_amendify_skill(skill_id)` to run the full inspect-preview-apply pipeline in one step.

## When to activate

- Before starting a task that could be handled by multiple skills
- When unsure which tool or approach to use
- After completing a task, to record the outcome
- When a skill has failed multiple times, to inspect and fix it

## Guidelines

- Record both successes AND failures -- failures improve future routing and enable self-amendifying
- Use `upsert_skills` rather than `ingest_skills` when skills may already be ingested
- Call `list_skills` to see what skills are currently available
- Trust `prefers_score > 0` as a signal that a skill has worked before for similar tasks
- If a skill keeps failing (`success_score < 0.5`), use `inspect_skill` to understand why before trying to fix it manually
