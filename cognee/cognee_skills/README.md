# cognee-skills

> Skills that fix themselves. Agents that get better with every run.

AI agents accumulate failing skills. A skill that worked last month stops working when the codebase changes, the model changes, or the task scope shifts. Usually nobody notices until a user complains.

**cognee-skills** gives every skill a self-improvement loop: every run is scored, failures are diagnosed, and fixes are proposed and applied — automatically or step-by-step.

---

## Quickstart

### 1. Install

```bash
pip install cognee
```

Set `LLM_API_KEY` in your `.env` (defaults to OpenAI).

### 2. Write a skill

```text
my_skills/
  summarize/
    SKILL.md
```

```markdown
---
name: summarize
description: Summarize documents, articles, or text into concise key points.
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

### 3. Run it

```python
from cognee import skills

await skills.ingest("./my_skills")
result = await skills.run("Compress this conversation")
```

`skills.run()` does everything: finds the best skill, executes it, scores the output, records the outcome, and self-repairs on failure. That's the full loop.

---

## Integrations

There are four ways to use cognee-skills. Pick the one that fits your setup:

| | Who it's for |
|---|---|
| [Claude Code / MCP IDEs](#claude-code--mcp-ides-no-code-required) | Vibe-coders, anyone using Claude Code or a MCP-enabled IDE |
| [Python SDK](#python-sdk) | Developers building custom workflows or agents |
| [CLI](#cli) | Terminal users, shell scripts, CI pipelines |
| [MCP programmatically](#mcp-programmatically) | Custom agents or services that speak MCP |

---

### Claude Code / MCP IDEs (no code required)

```bash
pip install cognee-mcp
export LLM_API_KEY="your-openai-api-key"
```

Add `.mcp.json` to your project root and check it into git — everyone on the team gets it:

```json
{
  "mcpServers": {
    "cognee": {
      "command": "cognee-mcp"
    }
  }
}
```

Add to your `CLAUDE.md` (replace `./my_skills` with your skills folder):

```
Skills are in ./my_skills. Call ingest_skills(skills_folder="./my_skills") and ingest_meta_skill() if skills haven't been loaded yet, then use run_skill for tasks — skills learn from their mistakes and get better over time.
```

Your IDE launches cognee locally. Claude loads your skills on first use, routes every task to the best one, evaluates output quality, and self-repairs failing skills over time — no code required.

`ingest_meta_skill()` loads the cognee-skills self-improvement guide as a skill. Once loaded, Claude knows how to inspect failing skills, review proposed fixes, apply amendments, and roll back changes on its own.

---

### Python SDK

```bash
pip install cognee
```

```python
from cognee import skills

await skills.ingest("./my_skills")
result = await skills.run("Compress this conversation")
```

For step-by-step control over the repair loop:

```python
# Inspect why a skill keeps failing
inspection = await skills.inspect("summarize")

# Preview the proposed fix before applying
amendment = await skills.preview_amendify("summarize")

# Apply it
await skills.amendify(amendment["amendment_id"])

# Roll back if it didn't help
await skills.rollback_amendify(amendment["amendment_id"])
```

See the [full API](#full-api) for all available methods.

---

### CLI

```bash
pip install cognee
```

```bash
cognee-cli skills ingest ./my_skills
cognee-cli skills run "Compress this conversation"

# Inspect and fix a failing skill
cognee-cli skills inspect summarize
cognee-cli skills preview summarize
cognee-cli skills amendify <amendment_id>

# JSON output for scripting
cognee-cli skills run "Compress this conversation" -f json
```

All commands support `-f json` for machine-readable output. Good for shell scripts and CI pipelines.

---

### MCP programmatically

Any agent or service that speaks MCP over stdio can connect directly — not just Claude Code:

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(command="cognee-mcp")

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        await session.call_tool("ingest_skills", {"skills_folder": "./my_skills"})
        result = await session.call_tool("run_skill", {"task_text": "Compress this conversation"})
```

See [`example/test_mcp.py`](example/test_mcp.py) for the full self-improvement loop over MCP.

---

## The self-improvement loop

Every run is scored 0.0–1.0 by a second LLM call. Failures accumulate. Once the threshold is hit, the loop kicks in:

```text
skill fails
  → inspect: LLM diagnoses why (root cause, severity, hypothesis)
  → preview: LLM generates improved instructions
  → amendify: fix applied to the graph, original preserved
  → evaluate: before/after scores compared
  → rollback: one call to revert if the fix didn't help
```

### Automatic (one call)

```python
# Full pipeline in one call
result = await skills.auto_amendify("summarize")

# {
#   "inspection": {"failure_category": "instruction_gap", "root_cause": "...", ...},
#   "amendment":  {"change_explanation": "...", "amendment_confidence": 0.82, ...},
#   "applied":    {"success": True, "status": "applied", ...}
# }
```

Or trigger repair automatically on execution failure:

```python
result = await skills.execute(
    "summarize",
    "Compress this conversation",
    auto_amendify=True,      # trigger repair if it fails
    amendify_min_runs=3,     # only after 3+ failures
)
```

### Manual (step-by-step)

Use the manual flow when you want to review the proposed fix before applying it.

**Step 1 — Inspect: understand why it fails**

```python
inspection = await skills.inspect("summarize")
# {
#   "failure_category": "instruction_gap",
#   "root_cause": "Instructions don't handle empty inputs",
#   "severity": "high",
#   "improvement_hypothesis": "Add guard clause for empty/short inputs",
#   "analyzed_run_count": 5,
#   "avg_success_score": 0.18,
#   "inspection_confidence": 0.88,
# }
```

Returns `None` if the skill doesn't have enough failures yet (default threshold: 1 run with score < 0.5).

**Step 2 — Preview: see the proposed fix before applying it**

```python
amendment = await skills.preview_amendify("summarize")
# {
#   "amendment_id": "amend-abc123",
#   "change_explanation": "Added guard clause for empty/short inputs",
#   "expected_improvement": "Skill will no longer fail on empty inputs",
#   "amendment_confidence": 0.82,
#   "pre_amendment_avg_score": 0.18,
# }
```

The original instructions are preserved in the graph — amendments are always reversible.

**Step 3 — Apply: update the skill**

```python
result = await skills.amendify("amend-abc123")
# {"success": True, "status": "applied", "skill_id": "summarize", ...}
```

**Step 4 — Evaluate: check whether the fix helped**

```python
eval_result = await skills.evaluate_amendify("amend-abc123")
# {
#   "pre_avg": 0.18,
#   "post_avg": 0.91,
#   "improvement": 0.73,
#   "run_count": 8,
#   "recommendation": "keep",
# }
```

**Step 5 — Rollback: revert if it didn't help**

```python
await skills.rollback_amendify("amend-abc123")
# Returns True if rolled back, False if not applicable
```

The original instructions are restored. The amendment history is kept in the graph for reference.

---

## Recording outcomes

Every observation drives both routing preferences and the repair loop. Without observations, `inspect` has nothing to analyze.

```python
await skills.observe({
    "task_text": "Compress this conversation",
    "selected_skill_id": "summarize",
    "success_score": 0.0,           # 0.0 = failed, 1.0 = perfect
    "error_type": "instruction_gap",
    "error_message": "Output was empty",
    "result_summary": "Skill returned nothing",
    "session_id": "sess-abc",
    "latency_ms": 1200,
})
```

When `skills.run()` or `skills.execute()` is used, this is handled automatically.

---

## Full API

### Self-improvement

| Call | What it does |
|------|-------------|
| `skills.inspect(skill_id)` | LLM analyzes failed runs → root cause, severity, hypothesis |
| `skills.preview_amendify(skill_id)` | LLM generates improved instructions (not applied yet) |
| `skills.amendify(amendment_id)` | Applies the fix; original preserved for rollback |
| `skills.rollback_amendify(amendment_id)` | Reverts to original instructions |
| `skills.evaluate_amendify(amendment_id)` | Before/after score comparison |
| `skills.auto_amendify(skill_id)` | Full pipeline in one call |
| `skills.execute(..., auto_amendify=True)` | Execute + repair on failure, in one call |

### Routing and observation

| Call | What it does |
|------|-------------|
| `skills.run(task_text)` | Find the best skill and execute it — one call does everything |
| `skills.execute(skill_id, task_text)` | Execute a specific skill by ID |
| `skills.observe({...})` | Record outcome; updates preferences immediately |
| `skills.get_context(task_text)` | Semantic search + learned preferences → ranked skills |
| `skills.load(skill_id)` | Full skill details: instructions, patterns, metadata |
| `skills.list()` | All ingested skills (summaries only) |

### Ingestion and management

| Call | What it does |
|------|-------------|
| `skills.ingest(folder)` | Parse SKILL.md files, enrich via LLM, store in graph + vector |
| `skills.ingest_meta_skill()` | Ingest the cognee-skills guide as a skill — agents learn the self-improvement loop |
| `skills.upsert(folder)` | Sync: skip unchanged, update changed, remove deleted |
| `skills.remove(skill_id)` | Delete from graph and vector |

All of these are also available as **MCP tools**.

---

## Visualize the graph

Every skill, run, inspection, and amendment is a node in the cognee knowledge graph. You can visualize it:

```python
from cognee.api.v1.visualize import start_visualization_server
await start_visualization_server(port=8080)
```

```bash
# Or via CLI
cognee-cli -ui  # launches UI at http://localhost:3000
```

See a live example at [graphskills.vercel.app](https://graphskills.vercel.app), built from the [Agent Skills for Context Engineering](https://github.com/muratcankoylan/Agent-Skills-for-Context-Engineering) repo.

---

## How the graph stores it

Every piece of data is a node in the cognee knowledge graph:

```text
Skill              — instructions, metadata, content hash
  └─ solves ──→  TaskPattern   — routing patterns with prefers weights
  └─ has ────→  SkillRun       — every execution (success or failure)
  └─ has ────→  SkillInspection — LLM diagnosis of failure patterns
  └─ has ────→  SkillAmendment  — proposed + applied fixes, with history
  └─ has ────→  SkillChangeEvent — temporal log of every change
```

Amendments update the `Skill` node in-place. The `SkillAmendment` node keeps the original instructions so rollback is always possible.

---

## Preference weights — how routing learns

Skills are linked to `TaskPattern` nodes via `solves` edges. When a run is observed, the `prefers` edge between that pattern and the skill is updated with an incremental mean of all `success_score` values seen so far:

```text
new_weight = (prior_weight_sum + success_score) / (prior_run_count + 1)
```

`skills.get_context()` ranks results by `vector_score + prefers_score`. A new skill starts at `prefers_score = 0` and wins only on semantic similarity. A skill with a proven track record on the same pattern will outrank a semantically closer but untested one.

For the weight to update, pass `task_pattern_id` to `observe()` — it comes back in every `get_context()` response:

```python
recs = await skills.get_context("compress this conversation")

await skills.observe({
    "task_text":          "compress this conversation",
    "selected_skill_id":  "summarize",
    "task_pattern_id":    recs[0]["task_pattern_id"],  # links the score to the right pattern
    "success_score":      0.9,
})
```

Without `task_pattern_id`, the run is still recorded as a `SkillRun` and feeds `inspect_skill`, but the preference weight is not updated.

---

## SkillChangeEvent — audit trail

A `SkillChangeEvent` node is written to the graph on every instruction change:

| Trigger | `change_type` |
|---------|--------------|
| New skill ingested | `"added"` |
| Skill content changed on upsert | `"updated"` |
| Skill deleted | `"removed"` |
| Amendment applied | `"amended"` |
| Amendment reverted | `"rolled_back"` |

Each event stores `skill_id`, `skill_name`, `change_type`, `old_content_hash`, `new_content_hash`, and a UTC `Timestamp` node. The hash pair lets you correlate a performance shift with the exact amendment that caused it.

---

## Repository structure

```text
my_skills/
  skill-a/
    SKILL.md        ← required: name, description, instructions
    references/     ← optional: docs, context files
    scripts/        ← optional: helper scripts
  skill-b/
    SKILL.md
```

Each skill folder is a structured unit: instructions, execution history, routing preferences, inspections, and amendment history all live together in the graph.

See [`example/`](example/) for a working demo with skills and test scripts for all four integration paths.
