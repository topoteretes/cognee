# recall() vs search()

Both functions query the knowledge graph. `recall()` is the memory-API entry point and the one
to reach for by default; `search()` is the low-level operation it calls underneath. This page says
what `recall()` adds, when to drop down to `search()`, the two places where the same argument
means different things, and how the query router picks a strategy.

Source of truth: `cognee/api/v1/recall/recall.py` and `cognee/api/v1/search/search.py`.

## What recall() adds on top of search()

1. **Query routing.** When `query_type` is omitted and `auto_route=True` (the default), an
   ordered table of regex rules picks a `SearchType` — first match wins, no LLM call, so
   routing is free. With `auto_route=False` and no `query_type`, recall uses
   `HYBRID_COMPLETION`. Passing `query_type` bypasses the router entirely. See
   [The router](#the-router) below.
2. **Session memory as a source.** `scope` selects where results come from: `"graph"` (the
   permanent graph via `search()`), `"session"` (Q&A entries in the session cache),
   `"trace"` (agent trace entries), `"session_context"` (the distilled guidance block), plus
   opt-in `"tools"` (authorized external databases) and `"code"` (the code graph). `"auto"` (the
   default) and `"all"` never imply `"tools"` or `"code"`. With a bare `session_id` and no
   `datasets`/`query_type`, a session hit short-circuits the graph search; `scope="session_first"`
   asks for that short-circuit explicitly, with a pinned type and datasets in play.
3. **Normalized results.** Every returned entry is tagged with a `source` key (`"graph"`,
   `"session"`, `"trace"`, `"session_context"`, `"tools"`, `"code"`, `"skills"`), so callers
   can tell where it came from, and graph entries also carry the `search_type` that actually
   ran. `search()` returns raw `SearchResult` objects.
4. **Skill gate.** Procedural-sounding queries trigger a concurrent `SKILLS` lookup whose hits
   are appended tagged `source="skills"` (only when exactly one dataset is targeted; disable
   with `SKILL_GATE_ENABLED=false`).
5. **Structured output shorthand.** `response_model=` validates the LLM answer against a
   Pydantic model and returns it in the result's `structured` field.

## When to call search() directly

- You need the agentic extras as first-class parameters: `skills`, `tools`, `max_iter`,
  `code_query`, `node_type`. (`recall()` reaches most of these through
  `retriever_specific_config={...}`, which you assemble yourself.)
- You want raw `SearchResult` objects rather than `source`-tagged entries.
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

## The router

Source: `cognee/api/v1/recall/query_router.py`.

Rules are checked in order and the first match wins. Shape rules — what the input *looks
like* — come first and win over intent rules, so a quoted string or a Cypher statement is
handled as what it is even when its text also reads as intent: `"coding rules"` is a lexical
search for that phrase, not a request for the rule list. No query in the golden table depends
on that ordering (a test enforces this). Anything unmatched goes to `HYBRID_COMPLETION`.
Matching is case-insensitive.

| # | Rule | Signal in the query | Routes to |
|---|---|---|---|
| 1 | `quoted_phrase` | The whole query is one `"quoted phrase"` | `CHUNKS_LEXICAL` |
| 2 | `coding_rules_intent` | `coding rules` / `coding standards` / `coding conventions`, or `code review guidelines` (and the `rules`, `standards`, `checklist`, `conventions` variants) | `CODING_RULES` |
| — | `default` | Anything else | `HYBRID_COMPLETION` |

`CYPHER` is **not** in the table and never will be. The Cypher retriever runs
the query text verbatim through `graph_engine.query()`, and the whole recall
path checks read permission only — so a routable `CYPHER` would let anyone who
can read a dataset destroy it by posting
`{"query": "MATCH (n) DETACH DELETE n"}`. Reaching `CYPHER` requires an
explicit `query_type` / `searchType`, which is a deliberate act by the caller
rather than whatever text arrived in a request body. Pasted Cypher therefore
routes to `HYBRID_COMPLETION` like any other text.

The rule for what belongs in the table: **auto-routing may only pick a
strategy that is at least as good as HYBRID on a default-built graph and does
not add LLM calls without an unambiguous signal.**

`HYBRID_COMPLETION` searches document chunks, pre-computed summaries, and the
entity neighbourhood in a single LLM call. Almost every alternative strategy
sees a *subset* of that context, sometimes with an extra round trip attached.
So each rule above fires on an input that is not a natural-language question at
all — pasted Cypher, a fully quoted literal, a request for the rule list — and
for which HYBRID is the wrong operation rather than a worse one. A rule that
fires on an ordinary question is a regression even when its target sounds
better suited.

That is why these are *not* auto-routed, even though they are valid
`query_type` values:

- `GRAPH_COMPLETION_COT` runs up to four reasoning iterations. "Why" and
  "explain" questions are answered by the default strategy; pin
  `query_type=SearchType.GRAPH_COMPLETION_COT` when you want the loop.
- `GRAPH_COMPLETION_CONTEXT_EXTENSION` adds traversal rounds. HYBRID already
  includes entity neighbourhoods, so "related to" questions stay on the
  default.
- `GRAPH_SUMMARY_COMPLETION` does not read pre-computed summaries: it runs
  `GRAPH_COMPLETION` and then makes a second LLM call to summarize the answer.
  Routing "summarize the report" there would drop HYBRID's document and
  summary lanes *and* add a round trip.
- `TEMPORAL` needs `Timestamp` nodes that only `temporal_cognify=True` creates.
  On a default graph it pays an interval-extraction LLM call and then degrades
  to triplet search, so no date token — a year, a range, a decade, an ISO date,
  or the word `timeline` — routes there.
- "Exact"/"verbatim" phrasing does not select `CHUNKS_LEXICAL`. BM25 tokenizes
  the raw query, so the trigger word itself becomes a rare, high-IDF search
  term and skews the ranking it was meant to sharpen. `quoted_phrase` has no
  such problem: its trigger is punctuation, which tokenization drops.
- Incidental code tokens (`def`, `import`, `class Foo(`, `.py`, `refactor`,
  `lint`) do not select `CODING_RULES`. That retriever reads only the
  `coding_agent_rules` nodeset and returns nothing on an ordinary graph.

### When a routed strategy comes up empty

A routed type is a guess, so `recall()` never lets one do worse than the
default. When the router picked a type **other than the default**, the query is
retried once as `HYBRID_COMPLETION` in two cases:

- the backend rejects the type; or
- the search returns nothing and the empty result means the lane was
  unavailable — `CHUNKS_LEXICAL` with no lexical hits, `CODING_RULES` on a
  dataset with no rules nodeset.

The search history records the type that actually answered. Two things are
never second-guessed: a type you pinned yourself returns empty or raises as
before, and a failure of the default itself is raised rather than hidden —
there is nothing left to fall back to, so the error is real.

### Bypassing the router

| Surface | Route automatically | Pin a strategy |
|---|---|---|
| SDK `recall()` | omit `query_type` (default) | pass `query_type=SearchType.X`; `auto_route=False` forces `HYBRID_COMPLETION` without routing |
| REST `POST /api/v1/recall` | omit `searchType` or pass `null` (default) | pass a value |
| CLI `cognee-cli recall` | omit `--query-type` | `--query-type X` (choices in `cognee/cli/config.py:SEARCH_TYPE_CHOICES`) |

All three surfaces auto-route by default — with one exception. When no usable
LLM key is configured, recall picks `CHUNKS` before the router is consulted,
because nothing can write a completion answer; the router never runs and an
explicit `query_type` is the only way to select a strategy. That branch keys
off LLM availability, not off the extractor that built the graph.

On every surface, omitting the type also makes the session a search source
whenever a `session_id` is given: alone it short-circuits the graph on a hit,
alongside datasets both contribute. Pinning a type leaves the graph as the only
source unless you ask for the session by name with `scope`. REST clients that relied on the old
`HYBRID_COMPLETION` default should pass `"searchType": "HYBRID_COMPLETION"`
explicitly.

### Seeing what ran

Graph results carry the resolved type as `search_type`. The CLI prints it in
the `Found N result(s) using ...` line. The recall span carries the type as
`cognee.search.type` and, when the router chose it, the rule name as
`cognee.recall.route_rule` — so which rule fires on real traffic is answerable
without reproducing the query. The router also logs the rule name at DEBUG
level, and never logs the query text.

### Adding a rule

Add a `(name, compiled pattern, SearchType)` tuple to `_RULES` in
`query_router.py`, then add cases to the golden table and the negative
invariants in `cognee/tests/unit/api/v1/recall/test_query_router.py`. Two
structural tests constrain what you can add: the new target must be in
`ROUTABLE_TYPES`, and the new pattern must not change how any query in the
golden table routes. Keep the size principle above in mind: a rule that sends
ordinary questions to a slower or narrower retriever is a regression, not an
improvement.

Related: the search-type list in `CLAUDE.md` ("SEARCH: Retrieval"), `examples/guides/recall_core.py`,
`examples/guides/hybrid_retrieval_recall.py`.
