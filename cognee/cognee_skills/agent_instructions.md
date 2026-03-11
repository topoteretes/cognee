# Cognee Skills

You have cognee skills via MCP. Skills are instructions that get better with every run — they self-fix when they fail.

## Quick start

First time only — ingest your skills:

```
ingest_skills(skills_folder="./my_skills")
```

Then just use `run_skill` for everything:

```
run_skill(task_text="compress this conversation")
```

That's it. `run_skill` finds the best skill, executes it, records the outcome, and repairs the skill if it fails. Everything is automatic.

## When to use what

| What you want | Tool |
|---------------|------|
| Run a task | `run_skill(task_text)` |
| Run a specific skill you already know | `execute_skill(skill_id, task_text)` |
| See what skills exist | `list_skills()` |
| Sync skill changes from disk | `upsert_skills(skills_folder="./my_skills")` |
| Fix a failing skill manually | `inspect_skill(skill_id)` → `preview_amendify_skill(skill_id)` → `amendify_skill(amendment_id)` |
| Fix a failing skill automatically | `auto_amendify_skill(skill_id)` |
| Undo a fix that didn't help | `rollback_amendify_skill(amendment_id)` |
