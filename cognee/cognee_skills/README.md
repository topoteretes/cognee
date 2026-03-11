# cognee-skills

> Skills that fix themselves. Agents that get better with every run.

AI agents accumulate failing skills. A skill that worked last month stops working when the codebase changes, the model changes, or the task scope shifts. Usually nobody notices until a user complains.

**cognee-skills** gives every skill a self-improvement loop:

```text
skill fails
  → inspect: LLM diagnoses why (root cause, severity, hypothesis)
  → preview: LLM generates improved instructions
  → amendify: fix applied to the graph, original preserved
  → evaluate: before/after scores compared
  → rollback: one call to revert if the fix didn't help
```

This runs automatically on failure (`auto_amendify=True`) or manually step-by-step. Every run, pass or fail, feeds back into routing preferences so the best skills rise and the broken ones surface for repair.

---

## The self-improvement loop

### Automatic (one call)

```python
from cognee import skills

# Execute with automatic self-repair on failure
result = await skills.execute(
    "summarize",
    "Compress this conversation",
    auto_amendify=True,      # trigger repair if it fails
    amendify_min_runs=3,     # only after 3+ failures
)

# result["success"]  — whether execution succeeded
# result["amended"]  — amendment applied if it failed (or None)
```

Or trigger repair directly:

```python
# Inspect → preview → apply in one call
result = await skills.auto_amendify("summarize")

# {
#   "inspection": {"failure_category": "instruction_gap", "root_cause": "...", ...},
#   "amendment":  {"change_explanation": "...", "amendment_confidence": 0.82, ...},
#   "applied":    {"success": True, "status": "applied", ...}
# }
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

The skill's instructions in the graph are updated immediately. Future executions use the amended version.

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

When `skills.execute()` is called with `auto_observe=True` (the default), this is handled automatically.

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

### 3. Ingest, execute, observe

```python
from cognee import skills

await skills.ingest("./my_skills")

result = await skills.execute(
    "summarize",
    "Compress this conversation",
    auto_observe=True,
    auto_amendify=True,
)
```

That's the full loop. Successes raise routing scores. Failures accumulate until the repair threshold is hit, then the LLM inspects and fixes the skill automatically.

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
| `skills.execute(skill_id, task_text)` | Load skill and run via LLM |
| `skills.observe({...})` | Record outcome; updates preferences immediately |
| `skills.get_context(task_text)` | Semantic search + learned preferences → ranked skills |
| `skills.load(skill_id)` | Full skill details: instructions, patterns, metadata |
| `skills.list()` | All ingested skills (summaries only) |

### Ingestion and management

| Call | What it does |
|------|-------------|
| `skills.ingest(folder)` | Parse SKILL.md files, enrich via LLM, store in graph + vector |
| `skills.upsert(folder)` | Sync: skip unchanged, update changed, remove deleted |
| `skills.remove(skill_id)` | Delete from graph and vector |

All of these are also available as **MCP tools** — see [MCP tools](#mcp-tools) below.

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

`SkillChangeEvent` extends cognee's `Event` DataPoint, so it's queryable via the temporal retriever alongside any other time-indexed graph data.

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

## MCP tools

When running the cognee MCP server, all skill operations are available as tools:

```text
run_skill                ← one call: route → execute → observe → self-repair
inspect_skill
preview_amendify_skill
amendify_skill
rollback_amendify_skill
evaluate_amendify_skill
auto_amendify_skill
execute_skill
observe_skill_run
get_skill_context
load_skill
list_skills
ingest_skills
upsert_skills
remove_skill
```

### Start the server

```bash
cd cognee-mcp && python src/server.py --transport sse
```

### Connect your IDE

```json
{ "mcpServers": { "cognee": { "type": "sse", "url": "http://localhost:8000/sse" } } }
```

| IDE | Config location |
|-----|-----------------|
| Cursor | `.cursor/mcp.json` |
| Claude Code | `~/.claude.json` or `claude mcp add cognee -t sse http://localhost:8000/sse` |
| Windsurf | MCP settings panel |
| Cline / Roo | VS Code MCP settings |

That's it. MCP clients discover `run_skill` and the other tools automatically from their descriptions.

**Optional:** If you want the agent to prefer skills that learn from mistakes, add one line to your `CLAUDE.md` (or agent rules):

```
Use run_skill via MCP for tasks — skills learn from their mistakes and get better over time.
```

For a full tool reference, see [`agent_instructions.md`](agent_instructions.md) (also available as the `skill://agent-guide` MCP resource).

---

## Running the example

```bash
python -m cognee.cognee_skills.example
```

Runs: ingest → route → execute → observe → re-route, and shows the preference boost in scores.

> **Warning:** Calls `cognee.prune.prune_system()` at startup — deletes all existing cognee data.

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
