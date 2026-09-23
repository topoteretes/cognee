---
name: cognee-recall
description: Use when querying cognee memory with recall() (or search()) — picking a search type, understanding auto-routing, scoping to datasets, node sets or sessions, getting context or citations instead of an answer, reading the results, or debugging empty or unexpected results.
---

# Query memory with recall()

`recall()` is cognee's query API. It picks a search strategy, searches the
graph and (with a session) the session cache, and returns a list of tagged
results.

```python
import cognee

results = await cognee.recall("Where was Einstein born?")
for r in results:
    print(r.source, r.text)          # e.g. "graph", "Einstein was born in Ulm."
```

Without `datasets` it searches every dataset the user can read. Pass
`datasets=["research"]` (names) or `dataset_ids=[...]` (UUIDs, which win) to
narrow it; that is also faster.

## Use it

### How the search type is picked

1. An explicit `query_type=SearchType.X` always wins.
2. Otherwise, with no usable LLM key, `CHUNKS` (plain vector search).
3. Otherwise the router (`auto_route=True`, the default). It is two regex
   rules, first match wins, no LLM call:
   - the whole query is one `"quoted phrase"` → `CHUNKS_LEXICAL`
   - mentions coding rules/standards/conventions or code-review guidelines
     → `CODING_RULES`
4. Everything else → `HYBRID_COMPLETION`.

A routed type (never a pinned one) that the backend rejects, or a routed
`CHUNKS_LEXICAL` / `CODING_RULES` that returns nothing, is retried once as
`HYBRID_COMPLETION`. The router never picks `CYPHER`.

```python
from cognee import SearchType
await cognee.recall("What changed in v2?", query_type=SearchType.GRAPH_COMPLETION)
```

### Search types

The full list is `cognee/modules/search/types/SearchType.py`; the
type-to-retriever table is `cognee/modules/retrieval/README.md`.

| Type | LLM? | Use for |
|---|---|---|
| `HYBRID_COMPLETION` (default) | yes | General questions: document passages plus entity neighbourhoods, then an answer |
| `GRAPH_COMPLETION` | yes | Answers from graph relationships |
| `GRAPH_COMPLETION_COT`, `_CONTEXT_EXTENSION`, `_DECOMPOSITION` | yes | Harder multi-hop questions (more LLM calls) |
| `GRAPH_SUMMARY_COMPLETION` | yes | Answers over pre-computed summaries |
| `RAG_COMPLETION` | yes | Classic chunk RAG |
| `TRIPLET_COMPLETION` | yes | Subject-predicate-object facts (needs triplet embedding) |
| `TEMPORAL` | yes | Time questions; needs data remembered with `temporal_cognify=True` |
| `CHUNKS` | no | Raw passages by vector similarity |
| `CHUNKS_LEXICAL` | no | Keyword / exact-phrase match |
| `SUMMARIES` | no | Document summaries |
| `CODE` | no | Code-graph operations via `code_query={...}`; needs `scope="code"` in recall |
| `SKILLS` | no | Discover skill playbooks; exactly one dataset |
| `CYPHER` | no | Raw Cypher. On by default; `ALLOW_CYPHER_QUERY=false` disables it. It can write, so only pass user-authored queries deliberately |
| `NATURAL_LANGUAGE` | yes | LLM writes Cypher, then runs it (same flag) |
| `GRAPH_REPORT` | partly | Graph insight report: hubs, cross-set links, suggested questions |
| `FEELING_LUCKY` | yes | An LLM picks the type |
| `AGENTIC_COMPLETION` | yes | Multi-step loop with skills/tools; exactly one dataset. Use `search()` for its parameters |

### Scope: which sources are searched

`scope` is one of, or a list of: `graph`, `session`, `session_first`,
`trace`, `session_context`, `all`, `tools`, `code`. `all` means graph +
session + trace + session_context; `tools` and `code` are never included
implicitly.

When `scope` is omitted:

| You pass | Sources |
|---|---|
| `session_id` only | Session first; a session hit skips the graph |
| `session_id` + `datasets` | Session and graph both contribute |
| `session_id` + `query_type` | **Graph only** — pinning a type drops the session |
| no `session_id` | Graph only |

Session and trace search is keyword overlap, not embeddings.

### Filters and knobs

- `top_k=15`: results per dataset, not in total.
- `node_name=["AI"]` (+ `node_name_filter_operator="OR"|"AND"`): restrict to
  data remembered with that `node_set`.
- `system_prompt` / `system_prompt_path`: change the answering prompt.
- `response_model=MyPydanticModel`: structured answer, on `r.structured`.
- `include_references=True`: attach the document chunks that support each
  graph edge used (needs `EDGE_EVIDENCE_ENABLED=true`, the default).
- `only_context=True`: return what the LLM would have received instead of
  an answer. `r.text` is the rendered user prompt, `r.system_prompt` the
  system prompt. Pin `query_type` when you use it.
- `retriever_specific_config={...}`: retriever-only options. For the
  agentic extras (`skills`, `tools`, `max_iter`) and `node_type`, call
  `cognee.search()` instead, which takes them as parameters.

### Reading the results

Each item is a Pydantic model with a `source` discriminator: `graph`,
`session`, `trace`, `session_context`, `code`, `tools`, `skills`, or
`system`. Graph items carry `text` (always renderable), `search_type`,
`kind`, `score`, `dataset_id` / `dataset_name`, `metadata`, `raw`, and
`structured`. A `system` item is a status
marker, not data (see "memory warming up" below).

When a query sounds procedural ("how do I…", "runbook", "steps to…") and
exactly one dataset is targeted, recall also runs a small `SKILLS` lookup
and appends hits with `source="skills"`. Disable with
`SKILL_GATE_ENABLED=false`.

## Pitfalls

- **Empty results usually mean permissions.** A dataset the user cannot
  read returns `[]`, not an error, so it does not leak which datasets
  exist. Check grants (the `cognee-permissions` skill) before debugging the
  graph. An unknown dataset *name* does raise `DatasetNotFoundError`.
- **"Memory warming up".** On an empty graph recall returns one
  `source="system"` item with `status="memory_warming_up"` (or
  `"build_failed"` plus `error_message`) instead of results. Wait for the
  remember to finish, or check why it failed.
- **Hybrid silently becomes graph completion** when you pass a custom
  `node_type`, `neighborhood_depth`, `feedback_influence > 0`, or the chunk
  collection is missing. `wide_search_top_k` and `triplet_distance_penalty`
  with hybrid raise `InvalidHybridSearchConfig`; pin
  `GRAPH_COMPLETION` to use them.
- **`SKILLS` and `AGENTIC_COMPLETION` need exactly one dataset**, or they
  raise.
- **`code_query` without `scope="code"` raises**, and `scope="tools"` also
  needs `TOOL_CALLS_ENABLED=true`.
- **Latency.** Completion types make one LLM call; with a session and
  `AUTO_FEEDBACK=true` (default) each answered turn adds one more. Set
  `AUTO_FEEDBACK=false` for low-latency reads (see the `cognee-performance`
  skill).

## recall() or search()?

Use `recall()`. Drop to `cognee.search()` only for agentic parameters
(`skills`, `tools`, `max_iter`, `node_type`) as first-class arguments, raw
`SearchResult` objects, or a pinned type with no router. `search()` never
searches the session cache; its `session_id` only adds conversation history
to the prompt. Full guide: `docs/recall-vs-search.md`.

## How it works

`recall()` resolves scope and search type, then calls the same authorized
search `search()` uses: datasets resolve through the permission layer with
`read`, one search per dataset runs concurrently, and results are
normalized and tagged.

- Entry point, scope and type resolution: `cognee/api/v1/recall/recall.py`
- Router: `cognee/api/v1/recall/query_router.py`
- Skill gate: `cognee/api/v1/recall/skill_gate.py`
- Result types: `cognee/modules/recall/types/RecallResponse.py`,
  `SearchResultItem.py`
- Scope names: `cognee/memory/entries.py:normalize_scope`
- Warm-up config (`RECALL_WARMUP_*`): `cognee/modules/recall/config.py`
- Core search and fan-out: `cognee/modules/search/methods/search.py`
- Hybrid fallback rules: `cognee/modules/search/methods/hybrid_deferral.py`
- Registry: `cognee/modules/search/methods/get_search_type_retriever_instance.py`

Examples in `examples/guides/`: `recall_core.py`,
`hybrid_retrieval_recall.py`, `references_example.py`, `temporal_recall.py`,
`sessions.py`.

## Extending it

Adding a search type, per `cognee/modules/retrieval/README.md`:

1. Write the retriever in `cognee/modules/retrieval/` (subclass
   `BaseRetriever` or a completion base).
2. Add the `SearchType` member and its `search_core_registry` entry.
3. Add a row to the README table. A unit test
   (`cognee/tests/unit/modules/retrieval/retriever_readme_index_test.py`)
   fails if the table and registry disagree.
4. Optional: add it to `SEARCH_TYPE_CHOICES` in `cognee/cli/config.py` for
   the CLI, or a regex rule to `query_router.py` for auto-routing. Never
   route a type that can write.
