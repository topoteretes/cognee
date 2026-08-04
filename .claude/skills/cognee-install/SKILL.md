---
name: cognee-install
description: Use when the user wants to install cognee and run their first add → cognify → search flow with the Python SDK — fresh setup, virtual env, extras selection, or a minimal working example.
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

All SDK functions are async. Minimal end-to-end script:

```python
import asyncio
import cognee

async def main():
    await cognee.add("Cognee turns documents into AI memory.")
    await cognee.cognify()
    results = await cognee.search("What does cognee do?")
    print(results)

asyncio.run(main())
```

`add()` accepts text, file paths, and URLs, with an optional
`dataset_name="my_project"`; pass `datasets=["my_project"]` to `cognify()` and
`search()` to stay inside one dataset. More runnable examples live in
`examples/` (start with `examples/demos/simple_cognee_example.py`).

## Verify / troubleshoot

- `cognee-cli add "hello" && cognee-cli cognify && cognee-cli search "hello"`
  exercises the same flow from the shell.
- To wipe local state during experiments: `cognee-cli delete --all`.
- Structured LLM output errors usually mean the model/provider needs an
  explicit instructor mode: `LLM_INSTRUCTOR_MODE="json_schema_mode"`.
