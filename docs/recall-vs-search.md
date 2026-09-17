# recall() vs search()

Both functions query the knowledge graph. `recall()` is the memory-API entry point and the one
to reach for by default; `search()` is the low-level operation it calls underneath. This page says
what `recall()` adds, when to drop down to `search()`, and the two places where the same argument
means different things.

Source of truth: `cognee/api/v1/recall/recall.py` and `cognee/api/v1/search/search.py`.

## What recall() adds on top of search()

1. **Query routing.** When `query_type` is omitted and `auto_route=True` (the default), a
   rule-based classifier (regex scoring, no LLM call) picks a `SearchType`. With
   `auto_route=False` and no `query_type`, recall uses `HYBRID_COMPLETION`. Passing `query_type`
   bypasses the router entirely.
2. **Session memory as a source.** `scope` selects where results come from: `"graph"` (the
   permanent graph via `search()`), `"session"` (Q&A entries in the session cache),
   `"trace"` (agent trace entries), `"session_context"` (the distilled guidance block), plus
   opt-in `"tools"` (authorized external databases) and `"code"` (the code graph). `"auto"` (the
   default) and `"all"` never imply `"tools"` or `"code"`. With a bare `session_id` and no
   `datasets`/`query_type`, a session hit short-circuits the graph search.
3. **Normalized results.** Every returned entry is a dict tagged with a `_source` key
   (`"graph"`, `"session"`, `"trace"`, `"session_context"`, `"tools"`, `"code"`), so callers
   can tell where it came from. `search()` returns raw `SearchResult` objects.
4. **Skill gate.** Procedural-sounding queries trigger a concurrent `SKILLS` lookup whose hits
   are appended tagged `source="skills"` (only when exactly one dataset is targeted; disable
   with `SKILL_GATE_ENABLED=false`).
5. **Structured output shorthand.** `response_model=` validates the LLM answer against a
   Pydantic model and returns it in the result's `structured` field.

## When to call search() directly

- You need the agentic extras as first-class parameters: `skills`, `tools`, `max_iter`,
  `code_query`, `node_type`. (`recall()` reaches most of these through
  `retriever_specific_config={...}`, which you assemble yourself.)
- You want raw `SearchResult` objects rather than `_source`-tagged dicts.
- You want a pinned `query_type` with no router and no session layer in the path, for
  example when benchmarking one retriever.
- You are writing a custom pipeline task or debugging a single retrieval stage.

## Same argument, different meaning

| Argument | `recall()` | `search()` |
|---|---|---|
| `session_id` | Makes the session cache a *source* (and may short-circuit the graph) | Only adds session history to the retrieval *context*; never searches the cache as a source |
| omitted `query_type` | Router picks one; `HYBRID_COMPLETION` if routing is off | Always `HYBRID_COMPLETION` |
| `top_k` | default 15 | default 15 (the CLI's `recall --top-k` defaults to 10) |
| `only_context=True` | Same as `search()`; pin `query_type` so the hybrid retriever cannot defer to `GRAPH_COMPLETION` behind your back | Returns what the LLM would have received instead of its answer: for completion types the user prompt (history, rendered question and context, session guidance) with the system prompt (the task template) alongside; retrieval-only types return their context |

## Quick reference

```python
import cognee
from cognee import SearchType

# Ordinary retrieval: let recall route the query and tag the sources.
results = await cognee.recall("What did Alice work on?", datasets=["project"])

# Session-first: answers the current conversation from the cache before touching the graph.
results = await cognee.recall("what did I just say about deadlines?", session_id="chat_1")

# Pinned strategy, no router.
results = await cognee.recall("timeline of the migration", query_type=SearchType.TEMPORAL)

# Low level: raw SearchResult objects, agentic parameters as keywords.
raw = await cognee.search(
    "Which functions call UserService?",
    query_type=SearchType.CODE,
    code_query={"operation": "impact_analysis", "seeds": ["UserService"]},
)
```

Related: the search-type list in `CLAUDE.md` ("SEARCH: Retrieval"), `examples/guides/recall_core.py`,
`examples/guides/hybrid_retrieval_recall.py`.
