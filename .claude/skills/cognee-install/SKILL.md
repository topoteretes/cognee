---
name: cognee-install
description: Use when the user wants to install cognee and run their first remember → recall flow with the Python SDK — fresh setup, virtual env, extras selection, or a minimal working example.
---

# Install and run cognee

## Install

Requires Python 3.10–3.14. Prefer uv:

```bash
uv venv && source .venv/bin/activate
uv pip install cognee            # from PyPI
# or, working inside this repo:
uv pip install -e .
```

Add extras only when needed — examples: `cognee[postgres]`, `cognee[neo4j]`,
`cognee[docling]` (office/HTML document parsing, slim), `cognee[docs]`
(unstructured), `cognee[anthropic]`, `cognee[ollama]`, `cognee[aws]`. The full
list is in `pyproject.toml` under `[project.optional-dependencies]`.

## Configure

The only required setting is an LLM API key. Create `.env` in the working
directory (or export the variable):

```bash
LLM_API_KEY="your_openai_api_key"
```

Defaults need no services: SQLite (relational), LanceDB (vector), and Ladybug
(graph), all stored locally. OpenAI is the default LLM and embedding provider —
if you configure a different LLM but not embeddings (or vice versa), the other
silently stays on OpenAI. For other providers and databases use the
cognee-integrations skill.

## First run

As of cognee 1.x the memory API — `remember`, `recall`, `forget`, `improve` —
is the primary surface. All SDK functions are async. Minimal end-to-end script:

```python
import asyncio
import cognee

async def main():
    await cognee.remember("Cognee turns documents into AI memory.")
    results = await cognee.recall("What does cognee do?")
    print(results)

asyncio.run(main())
```

`remember()` is the whole ingestion path in one call — it runs `add()` +
`cognify()`, then `improve()` to index the graph (`self_improvement=True` by
default). It accepts text, file paths, URLs, and binary streams, with an
optional `dataset_name="my_project"`; pass `datasets=["my_project"]` to
`recall()` to stay inside one dataset.

`recall()` auto-routes the query to a search strategy by default. Pass
`query_type=SearchType.CHUNKS` (etc.) to pin one, or `auto_route=False` to
fall back to `GRAPH_COMPLETION`.

Session memory is the other half of the API — `remember(..., session_id="chat_1")`
writes to a fast session cache rather than running add+cognify inline, and
`recall(..., session_id="chat_1")` reads it back (session hits short-circuit the
graph search). With the default `self_improvement=True` it still bridges that
data into the permanent graph in the background; `improve(dataset=...,
session_ids=[...])` does the same explicitly. Session memory runs on the
session cache, which is on by default (`CACHING=true`); setting
`CACHING=false` disables it entirely and makes `remember(session_id=...)`
raise.

Start with `examples/demos/remember_recall_improve_example.py`, which walks
through permanent memory, session memory, and the sync between them.

The `add()` / `cognify()` / `search()` / `memify()` primitives still exist and
are what `remember`/`recall`/`improve` call underneath — reach for them when you
need to drive a stage in isolation (e.g. custom pipeline tasks), not for
ordinary ingestion. `cognee.delete` is formally deprecated (since 0.3.9);
`forget()` is the v1 replacement, unifying the old delete/prune/empty_dataset
paths behind one call. When to use `recall()` versus the low-level `search()`
is covered in `docs/recall-vs-search.md`.

## Verify / troubleshoot

- `cognee-cli remember "hello" && cognee-cli recall "hello"` exercises the same
  flow from the shell.
- To wipe local state during experiments: `cognee-cli forget --all` (or
  `await cognee.forget(everything=True)`).
- Reads slow or spending tokens on every query → set `AUTO_FEEDBACK=false`
  (keep `CACHING=true`); by default cognee makes one structured-output LLM
  call per answered query to self-tune its memory.
- Structured LLM output errors usually mean the model/provider needs an
  explicit instructor mode: `LLM_INSTRUCTOR_MODE="json_schema_mode"`.
