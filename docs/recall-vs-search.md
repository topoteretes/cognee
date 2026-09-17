# recall() vs search(), and how recall() routes a query

`recall()` is the memory API's read path. It wraps `search()` and adds three
things:

1. **Query routing.** When `query_type` is omitted, a rule-based classifier
   picks the search strategy. No LLM call is involved, so routing is free.
2. **Session memory as a source.** `scope` selects `graph`, `session`,
   `trace`, `session_context`, or a list of them. With a bare `session_id`, a
   session hit short-circuits the graph search.
3. **Normalized results.** Every entry carries `source` (`"graph"`,
   `"session"`, `"trace"`, ...) and, for graph results, the `search_type` that
   actually ran.

Use `recall()` for ordinary retrieval. Drop to `search()` when you need the
agentic extras as first-class parameters (`skills`, `tools`, `max_iter`,
`code_query`, `node_type`), raw `SearchResult` objects, or a pinned
`query_type` with no router in the path. Note that `search(session_id=...)`
only adds session history to the retrieval context. It never searches the
session cache as a source; that is `recall()`-only.

## The router

Source: `cognee/api/v1/recall/query_router.py`.

Rules are checked in order and the first match wins, but no two rules may match
the same query, so the order is cosmetic (a test enforces this). Anything
unmatched goes to `HYBRID_COMPLETION`. Matching is case-insensitive except for
Cypher.

| # | Rule | Signal in the query | Routes to |
|---|---|---|---|
| 1 | `cypher_syntax` | An upper-case Cypher clause that opens a node pattern (`MATCH (n ...`, `CREATE (a:Person {...})`, `UNWIND [...]`), or any clause plus relationship syntax (`-[`, `]->`, `)-`, `-(`) | `CYPHER` |
| 2 | `quoted_phrase` | The whole query is one `"quoted phrase"` | `CHUNKS_LEXICAL` |
| 3 | `coding_rules_intent` | `coding rules` / `coding standards` / `coding conventions`, or `code review guidelines` (and the `rules`, `standards`, `checklist`, `conventions` variants) | `CODING_RULES` |
| — | `default` | Anything else | `HYBRID_COMPLETION` |

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

- the backend rejects the type, as `CYPHER` does under
  `ALLOW_CYPHER_QUERY=false`; or
- the search returns nothing and the empty result means the lane was
  unavailable — `CHUNKS_LEXICAL` with no lexical hits, `CODING_RULES` on a
  dataset with no rules nodeset.

`CYPHER` is deliberately not retried on an empty result: a valid query that
matched no rows has answered you, and re-asking an LLM to interpret the Cypher
text as a question would replace that answer with prose.

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

All three surfaces auto-route by default. On every surface, omitting the type
also makes the session a search source whenever a `session_id` is given: alone
it short-circuits the graph on a hit, alongside datasets both contribute.
Pinning a type leaves the graph as the only source. REST clients that relied on
the old `HYBRID_COMPLETION` default should pass
`"searchType": "HYBRID_COMPLETION"` explicitly.

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
`ROUTABLE_TYPES`, and the new pattern must not match any query an existing
rule already matches. Keep the size principle above in mind: a rule that sends
ordinary questions to a slower or narrower retriever is a regression, not an
improvement.
