# cognee-skills

AI agents pick skills by matching keywords or following static rules. When you have more than a handful of skills, this breaks down -- the agent picks the wrong one, or defaults to a generic approach. cognee-skills solves this by storing skills in a knowledge graph, matching them semantically, and learning from outcomes. Skills that work get ranked higher next time. Skills that fail get ranked lower.

It works with any MCP-capable IDE (Cursor, Claude Code, Windsurf, Cline, etc.) and as a Python library.

## Quickstart

### Prerequisites

- Python 3.10+
- cognee installed (`pip install -e .` from the repo root, or `pip install cognee`)
- `LLM_API_KEY` set in your `.env` (defaults to OpenAI)

### 1. Write your skills

Create a folder with one subdirectory per skill, each containing a `SKILL.md`:

```
my_skills/
  summarize/
    SKILL.md
  code-review/
    SKILL.md
```

Flat files are also supported -- you can place `SKILL.md` files directly in the root:

```
my_skills/
  summarize.md
  code-review.md
```

### 2. Ingest and use

```python
from cognee import skills

# Ingest skills from a folder
await skills.ingest("./my_skills")

# Find the best skill for a task
recs = await skills.get_context("compress my conversation to 8k tokens")
# [{"skill_id": "summarize", "score": 0.98, "instruction_summary": "...", ...}]

# Load full details
skill = await skills.load("summarize")

# Record an execution outcome
await skills.observe({
    "task_text": "compress my conversation to 8k tokens",
    "selected_skill_id": "summarize",
    "success_score": 0.92,
    "result_summary": "Compressed 32k tokens to 7.5k",
})

# Promote cached runs to the graph (updates preference weights)
await skills.promote()
```

### 3. Query again -- preferences kick in

```python
recs = await skills.get_context("compress my conversation to 8k tokens")
# prefers_score > 0 now, final score is higher for skills that worked before
```

## SKILL.md format

Each skill is a markdown file with YAML frontmatter:

```markdown
---
name: my-skill
description: >
  One-paragraph description of what this skill does and when to use it.
  Include trigger phrases the user might say.
---

# Skill Title

Brief explanation.

## When to Activate

- Trigger condition 1
- Trigger condition 2

## Process

1. Step one
2. Step two
3. Step three

## Output Format

Describe the expected output structure.

## Guidelines

1. Rule one
2. Rule two
```

**Required frontmatter fields:** `name`, `description`

The markdown body is free-form. The LLM enrichment step reads the full file and generates: `instruction_summary`, `task_pattern_candidates`, `tags`, and `complexity`.

## Example skills

The repo ships with four example skills in `cognee/skills/example_skills/`, the first three adapted from [Agent-Skills-for-Context-Engineering](https://github.com/muratcankoylan/Agent-Skills-for-Context-Engineering/tree/main):

| Skill | Description |
|-------|-------------|
| `summarize` | Summarize documents, articles, or text into concise key points |
| `code-review` | Review code for bugs, style issues, and improvements |
| `data-extraction` | Extract structured data from unstructured text, PDFs, or web pages |
| `skill-routing` | Meta-skill that teaches agents to use all 8 cognee-skills MCP tools for autonomous skill discovery, routing, and learning |

### Try them out

```python
import asyncio
from cognee import skills

async def main():
    # Ingest the bundled example skills
    await skills.ingest("cognee/skills/example_skills")

    # Query for the best skill
    recs = await skills.get_context("summarize this article for me")
    for r in recs:
        print(f'{r["name"]:20s}  score={r["score"]:.3f}  prefers={r["prefers_score"]:.3f}')

asyncio.run(main())
```

Or run the full closed-loop demo (ingestion, retrieval, observe, promote, re-retrieval) -- see [Running the example](#running-the-example) below.

## API reference

### Client API (recommended)

```python
from cognee import skills
```

| Method | Description |
|--------|-------------|
| `skills.ingest(skills_folder, dataset_name="skills")` | Parse and ingest SKILL.md files |
| `skills.upsert(skills_folder, dataset_name="skills")` | Re-ingest: skip unchanged, update changed, delete removed |
| `skills.remove(skill_id)` | Remove a single skill from graph and vector stores |
| `skills.get_context(task_text, top_k=5)` | Ranked skill recommendations with scores |
| `skills.load(skill_id)` | Full skill details including instructions from the graph |
| `skills.list()` | List all ingested skills with summaries |
| `skills.observe(run_dict)` | Record a skill execution to short-term cache |
| `skills.promote(session_id=None)` | Promote cached runs and update preference edges |

#### `skills.observe()` payload

| Key | Required | Description |
|-----|----------|-------------|
| `task_text` | yes | What the user asked |
| `selected_skill_id` | yes | Which skill was used |
| `success_score` | yes | 0.0 (failed) to 1.0 (perfect) |
| `session_id` | no | Groups runs for batch promotion (default: "default") |
| `task_pattern_id` | no | Resolved automatically by `get_context()` |
| `result_summary` | no | Brief description of the outcome |
| `candidate_skills` | no | Full candidate list from `get_context()` |
| `feedback` | no | -1.0 to 1.0 user feedback signal |
| `error_type` | no | Error class name if the skill failed |
| `error_message` | no | Error details |
| `latency_ms` | no | Execution time |

### Lower-level API

```python
from cognee.skills import ingest_skills, upsert_skills, remove_skill, recommend_skills, record_skill_run, promote_skill_runs
```

| Function | Description |
|----------|-------------|
| `ingest_skills(skills_folder, dataset_name)` | Parse and ingest SKILL.md files |
| `upsert_skills(skills_folder, dataset_name)` | Diff-based re-ingestion of a skills folder |
| `remove_skill(skill_id)` | Remove a single skill from graph + vector |
| `recommend_skills(task_text, top_k)` | Raw retrieval with full metadata |
| `record_skill_run(session_id, task_text, ...)` | Record with explicit parameters |
| `promote_skill_runs(session_id)` | Promote with explicit session |

## CLI

```bash
cognee-cli skills ingest ./my_skills          # Ingest SKILL.md files from a folder
cognee-cli skills recommend "summarize this"  # Find best skills for a task
cognee-cli skills list                        # List all ingested skills
cognee-cli skills list -f json                # List as JSON
cognee-cli skills observe '{"task_text":"...", "selected_skill_id":"summarize", "success_score":0.9}'
cognee-cli skills promote                     # Promote cached runs to the graph
```

## MCP tools

When running the cognee MCP server, eight tools are available:

| Tool | Description |
|------|-------------|
| `get_skill_context` | Find best skills for a task (semantic search + learned preferences) |
| `load_skill` | Load full skill details by skill_id |
| `list_skills` | List all skills currently ingested in the knowledge graph |
| `observe_skill_run` | Record an execution outcome (success/failure) |
| `promote_skill_runs` | Promote cached runs to the graph and update preference weights |
| `ingest_skills` | Parse SKILL.md files from a folder and store in the knowledge graph |
| `upsert_skills` | Re-ingest a folder: skip unchanged, update changed, remove deleted |
| `remove_skill` | Remove a single skill by skill_id |

## Using with AI coding agents

Works with any MCP-capable IDE: Cursor, Claude Code, Windsurf, Cline, Roo, etc.

### Step 1: Start the MCP server

```bash
cd cognee-mcp
python src/server.py --transport sse
```

Or with Docker:

```bash
docker run -e TRANSPORT_MODE=sse --env-file ./.env -p 8000:8000 --rm -it cognee/cognee-mcp:main
```

See the [cognee-mcp README](https://github.com/topoteretes/cognee/tree/dev/cognee-mcp) for full setup options.

### Step 2: Connect your IDE

Add the MCP server to your IDE's config. The JSON payload is the same everywhere -- only the config file location differs:

```json
{
  "mcpServers": {
    "cognee": {
      "type": "sse",
      "url": "http://localhost:8000/sse"
    }
  }
}
```

| IDE | Where to put it |
|-----|-----------------|
| Cursor | `.cursor/mcp.json` in your project |
| Claude Code | `~/.claude.json`, or run `claude mcp add cognee -t sse http://localhost:8000/sse` |
| Windsurf | MCP settings panel |
| Cline / Roo | VS Code MCP settings |

### Step 3: Ingest your skills

```bash
cognee-cli skills ingest ./my_skills
```

Or from Python:

```python
from cognee import skills
await skills.ingest("./my_skills")
```

### Step 4: Teach your agent about skill routing

Copy the contents of [`agent_instructions.md`](https://github.com/topoteretes/cognee/blob/dev/cognee/skills/agent_instructions.md) into your IDE's agent instructions. Where you paste it depends on your IDE:

| IDE | Where to paste |
|-----|----------------|
| Cursor | `.cursor/rules/cognee-skills.md` in your project |
| Claude Code | `CLAUDE.md` in your project root |
| Windsurf | Rules / AI instructions in settings |
| Cline / Roo | Custom instructions in extension settings |

Edit the `./my_skills` path in the instructions to match your actual skills folder. No other changes needed.

The agent will now route tasks to the best skill via semantic search, learn from outcomes, and improve over time.

> **Optional:** The `skill-routing/` example skill in `cognee/skills/example_skills/` can be ingested into cognee itself so that skill routing shows up as a result in `get_skill_context`. This is useful if you want the agent to self-discover the routing workflow rather than having it in static instructions.

## How it works

```
SKILL.md files
    |
    v
skills.ingest() -- parse + LLM enrich + store in graph/vector
    |
    v
skills.get_context(task) -- vector search + prefers weights --> ranked skills
    |
    v
Agent executes the top skill
    |
    v
skills.observe(outcome) -- cached in short-term memory
    |
    v
skills.promote() -- moves runs to graph, updates TaskPattern->Skill "prefers" edges
    |
    v
skills.get_context(task) -- prefers_score now reflects historical performance
```

Skills that work well for a task pattern get higher `prefers_score` over time. Skills that fail get lower scores. The system learns from every execution.

## Running the example

> **Warning:** `example.py` calls `cognee.prune.prune_system()` at startup, which **deletes all existing cognee data and system state**. Run it in a clean environment or with a dedicated `DATA_ROOT_DIRECTORY`.

```bash
cd /path/to/cognee
python -m cognee.skills.example
```

This runs the full closed-loop demo: ingest -> recommend -> pick top skill -> simulate execution -> record -> promote -> recommend again (with prefers boost visible). At the end it generates an interactive graph visualization you can open in your browser.

## Visualizing the graph

After ingesting skills, you can generate an interactive HTML visualization of the knowledge graph:

```bash
python -m cognee.skills.inspect_graph
# Open cognee/skills/graph.html in a browser
```

Or from Python:

```python
import cognee
await cognee.visualize_graph("graph.html")
# Open graph.html in a browser
```

The graph shows Skills, TaskPatterns, their `solves` edges, and any `prefers` edges that have been learned from recorded runs.
