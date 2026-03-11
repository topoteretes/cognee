# cognee-skills

> Turn `SKILL.md` folders into a self-improving skill router for AI agents.

AI agents usually scan all `SKILL.md` files or rely on keyword matching to choose which skill to run. That works for a few skills, but breaks down quickly once you have dozens.

**cognee-skills** turns skill folders into a structured, self-improving skill router:

- **Semantic routing** — finds the right skill by meaning, not keywords
- **Learns from outcomes** — skills that work rank higher, skills that fail rank lower
- **Self-amendifying** — skills that keep failing get automatically inspected and fixed
- **Works everywhere** — Python, CLI, MCP (Cursor, Claude Code, Windsurf, Cline, etc.)

Instead of repeatedly scanning all skills or planning from scratch, agents can retrieve the best skill instantly and improve their routing over time.

---

## Quickstart

### 1. Install

```bash
pip install cognee
```

Set `LLM_API_KEY` in your `.env` (defaults to OpenAI).

### 2. Write your skills

One folder per skill, each with a `SKILL.md`:

```text
my_skills/
  summarize/
    SKILL.md
  code-review/
    SKILL.md
```

A `SKILL.md` needs two frontmatter fields (`name`, `description`) and a free-form markdown body:

```markdown
---
name: summarize
description: >
  Summarize documents, articles, or text into concise key points.
---

## When to Activate

- User asks to summarize, condense, or compress text

## Process

1. Identify key points
2. Produce a concise summary

## Guidelines

- Preserve the original meaning
- Keep it under 20% of the original length
```

### 3. Ingest, route, learn

```python
from cognee import skills

# Ingest skills into the knowledge graph
await skills.ingest("./my_skills")

# Find the best skill for a task
recs = await skills.get_context("compress my conversation to 8k tokens")
# [{"skill_id": "summarize", "score": 0.98, ...}]

# Record what happened (persists to graph and updates preferences immediately)
await skills.observe({
    "task_text": "compress my conversation to 8k tokens",
    "selected_skill_id": "summarize",
    "success_score": 0.92,
})

# Next time, summarize ranks even higher for compression tasks
recs = await skills.get_context("compress my conversation to 8k tokens")
```

`skills.observe()` persists the run to the graph and updates routing preferences immediately — no extra step needed.

---

## Why this helps

Without structured routing, agents often:

- scan too many skills before acting
- re-plan the same workflow over and over
- fail to learn which skills actually work

With `cognee-skills`, the routing loop becomes:

```text
retrieve best skill → execute → observe outcome → route better next time
```

When a skill keeps failing, the self-amendifying loop kicks in:

```text
inspect why it fails → preview an amendment → apply the fix → evaluate improvement
```

---

## Features

| What | How |
|------|-----|
| **Ingest skills** | `skills.ingest("./my_skills")` — parses `SKILL.md`, enriches via LLM, stores in graph + vector |
| **Route tasks** | `skills.get_context("task description")` — semantic search + learned preferences |
| **Execute a skill** | `skills.execute("skill-id", "task text")` — load and run via LLM |
| **Load full details** | `skills.load("skill-id")` — returns instructions, patterns, metadata |
| **List all skills** | `skills.list()` — everything currently ingested |
| **Record outcomes** | `skills.observe({...})` — persist run and update preferences immediately |
| **Sync changes** | `skills.upsert("./my_skills")` — skip unchanged, update changed, remove deleted |
| **Remove a skill** | `skills.remove("skill-id")` — delete from graph and vector |
| **Inspect failures** | `skills.inspect("skill-id")` — LLM analyzes why a skill keeps failing |
| **Preview amendment** | `skills.preview_amendify("skill-id")` — generate improved instructions |
| **Apply amendment** | `skills.amendify(amendment_id)` — apply the fix to the graph |
| **Rollback amendment** | `skills.rollback_amendify(amendment_id)` — revert if the fix didn't help |
| **Evaluate amendment** | `skills.evaluate_amendify(amendment_id)` — compare pre/post success scores |
| **Auto-amendify** | `skills.auto_amendify("skill-id")` — full pipeline in one call |
| **Visualize the graph** | `await cognee.visualize_graph("graph.html")` — interactive HTML |

All of these are also available as **MCP tools**.

---

## MCP tools

When running the cognee MCP server, all skill operations are available as tools:

```text
get_skill_context
load_skill
list_skills
execute_skill
observe_skill_run
ingest_skills
upsert_skills
remove_skill
inspect_skill
preview_amendify_skill
amendify_skill
rollback_amendify_skill
evaluate_amendify_skill
auto_amendify_skill
```

---

## Using with AI coding agents

Works with any **MCP-capable IDE**.

### 1. Start the MCP server

```bash
cd cognee-mcp && python src/server.py --transport sse
```

### 2. Connect your IDE

```json
{ "mcpServers": { "cognee": { "type": "sse", "url": "http://localhost:8000/sse" } } }
```

| IDE | Config location |
|-----|-----------------|
| Cursor | `.cursor/mcp.json` |
| Claude Code | `~/.claude.json` or `claude mcp add cognee -t sse http://localhost:8000/sse` |
| Windsurf | MCP settings panel |
| Cline / Roo | VS Code MCP settings |

### 3. Ingest your skills

```bash
cognee-cli skills ingest ./my_skills
```

### 4. Teach your agent

Copy [`agent_instructions.md`](agent_instructions.md) into your IDE's agent rules (`.cursor/rules/`, `CLAUDE.md`, etc.) and update the skills folder path.

Your agent will now **semantically route tasks to skills, improve over time, and self-fix failing skills**.

---

## How it works

```text
SKILL.md files
    → skills.ingest()        parse + LLM enrich + store in graph/vector
    → skills.get_context()   semantic search + preference weights → ranked skills
    → agent executes skill
    → skills.observe()       persist run + update preference edges immediately
    → skills.get_context()   preferences now reflect historical performance

When a skill keeps failing:
    → skills.inspect()           analyze failed runs, identify root cause
    → skills.preview_amendify()  generate improved instructions
    → skills.amendify()          apply the fix to the graph
    → skills.evaluate_amendify() compare before/after success scores
```

---

## Running the example

```bash
python -m cognee.skills.example
```

Runs the full loop:

```text
ingest → route → execute → record → re-route
```

You'll see the **preference boost appear in the routing scores** after recording outcomes.

An interactive graph visualization is also generated at the end.

Bundled examples (`summarize`, `code-review`, `data-extraction`) are taken from [Agent-Skills-for-Context-Engineering](https://github.com/muratcankoylan/Agent-Skills-for-Context-Engineering/tree/main).

> **Warning:** This calls `cognee.prune.prune_system()` at startup, which deletes all existing cognee data.

---

## Repository structure

```text
my_skills/
  skill-a/
    SKILL.md
    references/
    scripts/
  skill-b/
    SKILL.md
```

`cognee-skills` treats each skill folder as a structured unit: metadata, instructions, resources, execution history, and learned routing preferences.

---

## Summary

`cognee-skills` makes `SKILL.md` repos:

- **searchable**
- **routable**
- **learnable**
- **self-improving**

So agents stop treating skills as loose markdown files and start using them as structured workflows that improve with use.
