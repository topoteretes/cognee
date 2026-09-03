# recall() vs search(), and how recall() routes a query

`recall()` is the memory API's read path. It wraps `search()` and adds three
things:

1. **Query routing.** When `query_type` is omitted, a rule-based classifier
   picks the search strategy. No LLM call is involved, so routing is free.
2. **Session memory as a source.** `scope` selects `graph`, `session`,
   `trace`, `session_context`, or a list of them. With a bare `session_id`, a
   session hit short-circuits the graph search.
3. **Normalized results.** Every entry carries `_source` (`"graph"`,
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

Rules are checked in order and the first match wins. Anything unmatched goes
to `HYBRID_COMPLETION`. Matching is case-insensitive except for Cypher.

| # | Rule | Signal in the query | Routes to |
|---|---|---|---|
| 1 | `cypher_syntax` | Starts with `MATCH`, `RETURN`, `CREATE`, or `MERGE`, or contains `--(` / `)--` | `CYPHER` |
| 2 | `quoted_phrase` | The whole query is one `"quoted phrase"` | `CHUNKS_LEXICAL` |
| 3 | `exact_match_intent` | `exact`, `verbatim`, `literal`, `word for word` | `CHUNKS_LEXICAL` |
| 4 | `summary_intent` | `summarize`, `summary`, `overview`, `outline`, `tl;dr`, `gist`, `main points`, `key takeaways` | `GRAPH_SUMMARY_COMPLETION` |
| 5 | `explicit_time_range` | A year range (`between 1910 and 1920`, `2019 to 2021`), a preposition plus a year (`in 1915`, `since 2020`), a decade (`the 1990s`), an ISO date, `timeline`, `chronology` | `TEMPORAL` |
| 6 | `coding_rules_intent` | `coding rules` / `coding standards` / `coding conventions`, or `code review guidelines` (and the `rules`, `standards`, `checklist`, `conventions` variants) | `CODING_RULES` |
| — | `default` | Anything else | `HYBRID_COMPLETION` |

The rule for what belongs in the table: **auto-routing may only pick a
strategy that is at least as good as HYBRID on a default-built graph and does
not add LLM calls without an unambiguous signal.**

That is why these are *not* auto-routed, even though they are valid
`query_type` values:

- `GRAPH_COMPLETION_COT` runs up to four reasoning iterations. "Why" and
  "explain" questions are answered by the default strategy; pin
  `query_type=SearchType.GRAPH_COMPLETION_COT` when you want the loop.
- `GRAPH_COMPLETION_CONTEXT_EXTENSION` adds traversal rounds. HYBRID already
  includes entity neighbourhoods, so "related to" questions stay on the
  default.
- Bare temporal words (`when`, `before`, `after`, `since`, `during`) do not
  select `TEMPORAL`. Default graphs are built with `temporal_cognify=False`
  and have no event nodes, so `TEMPORAL` would pay an extra LLM call for
  interval extraction and then fall back to graph-only triplets. A year,
  date, decade, or the word `timeline` is required.
- Incidental code tokens (`def`, `import`, `class Foo(`, `.py`, `refactor`,
  `lint`) do not select `CODING_RULES`. That retriever reads only the
  `coding_agent_rules` nodeset and returns nothing on an ordinary graph.

### Bypassing the router

| Surface | Route automatically | Pin a strategy |
|---|---|---|
| SDK `recall()` | omit `query_type` (default) | pass `query_type=SearchType.X`; `auto_route=False` forces `HYBRID_COMPLETION` without routing |
| REST `POST /api/v1/recall` | `"searchType": null` | omit the field (defaults to `HYBRID_COMPLETION`) or pass a value |
| CLI `cognee-cli recall` | omit `--query-type` | `--query-type X` (choices in `cognee/cli/config.py:SEARCH_TYPE_CHOICES`) |

The REST default stays `HYBRID_COMPLETION` on purpose: with `sessionId` set,
a null `searchType` also enables the session short-circuit, and changing the
default would change that behaviour for existing clients.

### Seeing what ran

Graph results carry the resolved type as `search_type`. The CLI prints it in
the `Found N result(s) using ...` line. The router itself logs the rule name
at DEBUG level and never logs the query text.

### Adding a rule

Add a `(name, compiled pattern, SearchType)` tuple to `_RULES` in
`query_router.py` at the right precedence, then add cases to the golden table
and the negative invariants in
`cognee/tests/unit/api/v1/recall/test_query_router.py`. Keep the size
principle above in mind: a rule that sends ordinary questions to a slower or
narrower retriever is a regression, not an improvement.
