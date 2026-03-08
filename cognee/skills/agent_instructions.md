# Cognee Skill Routing

You have access to cognee skill routing via MCP. It finds the best skill for a task using semantic search and learned preferences, and improves over time as you record outcomes.

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
6. Call `promote_skill_runs` at the end of the session to update preference weights

## When to activate

- Before starting a task that could be handled by multiple skills
- When unsure which tool or approach to use
- After completing a task, to record the outcome

## Guidelines

- Record both successes AND failures -- failures improve future routing
- Use `upsert_skills` rather than `ingest_skills` when skills may already be ingested
- Call `list_skills` to see what skills are currently available
