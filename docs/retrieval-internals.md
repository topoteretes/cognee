# Retrieval internals

## Search retrieval
`search(query_text, query_type)` → route to retriever type → filter by permissions → route to datasets → return results

**Dataset routing.** With backend access control on, an unscoped non-`CODE` `search()` or `recall()` across multiple readable datasets first uses routing to select the datasets most likely to hold the answer (`authorized_search` in `cognee/modules/search/methods/search.py`, module `cognee/modules/search/dataset_routing/`).
Every readable dataset is scored against the query in the active deployment-model collection `DatasetRoute_text__<digest>`. It holds each dataset's global-context root and addressable document buckets, embedded with deployment settings during indexing. It lives in the [shared routing store](architecture.md#3-multi-tenant-access-control), outside dataset contexts. `ensure_default_store` creates the default pgvector database when needed; the dataset queue pins this store against idle engine eviction. Storage retains the existing store-keyed cache; routing resolves its embedder separately, without mutating a cached adapter.
The rows are addressed by id, derived from the caller's datasets and their `Data` rows (`create_root_summary_id`, `create_document_bucket_id`) and scored with `score_by_ids`, so no other tenant's row is ever read and nothing is cut off by a result limit; a dataset's score is its best row, so a specific question is carried by the document that matches and a broad one by the root. Only the best `DATASET_ROUTING_TOP_K` (default 1) are searched. The score is a cosine distance (0 is identical text, unrelated text sits near 1): a dataset farther than `DATASET_ROUTING_DISTANCE_CUTOFF` (default 0.8) is not a candidate, and when no dataset is within it every readable dataset is searched, so a question no dataset knows about is not answered from whichever is nearest; a tie at the top-k boundary is never split, every dataset within `DATASET_ROUTING_TIE_MARGIN` (default 0.001) of the last routed one is searched too (measured against that one only, so ties do not chain). Where queries land on the distance scale depends on the embedding model, so the cutoff is a knob and the routing log prints every dataset's distance to tune it against.
A dataset with no routing rows (never indexed, or a code dataset, which has no chunks) is not searched next to a routed one, so a backlog of datasets the index never saw does not turn every unscoped search into a search of all of them (SDK-1516); it is covered by the widening pass below, and when no dataset is routed at all every readable dataset is searched, unindexed ones included. A routed search that retrieves nothing (judged on the retrieval context, `SearchResultPayload.retrieved_nothing()`, never on the completion, which is written even over an empty context; on the rows for the retrievers that report rows and no context, Cypher and natural language), is widened to the datasets routing skipped, searched as the rest of the same search (a `widening` fan-out, which reports a dataset without memory per dataset instead of raising) and merged in readable order, so the routed datasets are not searched twice; only when every routed dataset fails before retrieving (no searchable memory) is the whole set searched as one.
An explicit dataset scope is never routed, and neither is a `CODE` search. Zero/one-readable-dataset and access-control-off cases also bypass routing. A missing active collection does not embed the query; missing optional scoring support leaves datasets unindexed with a logged reason. Configuration, embedding, storage and corrupt-collection errors propagate visibly. Every unscoped search logs which datasets it covers and why, with the hint to pass `datasets=[...]`. There is no routing off switch.
`sync_dataset_routes` reconciles the full addressable set against stored text. It invalidates that dataset's IDs in noncurrent collections, retrieves changed source payloads in one batch without vectors, embeds their text together and raw-upserts with an explicit deployment width. Unchanged active rows require no source retrieval or summary embeddings. Payload IDs, document references and NodeSet tags survive; dataset-local vectors are untouched. `from_entities` synchronizes root plus document buckets; `llm` synchronizes only its root.

Deletion hooks run before relational document IDs disappear and remove the relevant root/bucket IDs from every routing namespace, including legacy `DatasetRoute_text`. The existing best-effort post-deletion root rebuild remains; failure is logged for a later improve. Prune discovers and drops every routing collection, including orphaned ones, preserving unrelated tables/schemas. Discovery/raw-write/drop capability failures are strict. Cleanup and upsert are idempotent and a partial failure can be retried.

Routing never filters readable datasets by their recorded model. Selected-dataset dimension mismatches propagate through ordinary retrieval and widening, even alongside sibling results. Populated unknown-origin stores retain unknown provenance; width cannot identify their model. A model switch or layout upgrade invalidates the improve gate and repopulates routing text on the next enabled, successful index run in a valid dataset context, without dataset re-embedding. Missing active rows before then retain unindexed semantics. See [dataset routing](recall-vs-search.md#dataset-routing) for the user-facing rule. Dataset routing is distinct from query routing (`recall()`'s search-type selection).

**Connector titles in search results.** With `include_references=True` the retrievers that cite chunks (RAG, hybrid, temporal) copy a document-mode connector chunk's title and link from its row's `system_metadata` (`title`, `url`) into the chunk payload as `document_title` / `document_url`, with the name of the nearest place the row sits in (its `structure`'s first ancestor, a Slack channel's `#name`) as `document_parent`, read once per search from the searched dataset's records (`name_chunk_documents`, `cognee/modules/retrieval/utils/document_sources.py`). They are stored values, unescaped; the answer, its Evidence block and the structured evidence do not use them, other documents' payloads gain nothing, and `Data.label` is not read.

Available search types (from `cognee/modules/search/types/SearchType.py`), passed as `query_type` to `recall()` or `search()`:
- **HYBRID_COMPLETION** (default) - Document passages plus entity neighbourhoods, then LLM completion
- **HYBRID_COMPLETION_DECOMPOSITION** - Hybrid retrieval for the full question, then LLM decomposition of question plus that context into up to `max_subqueries` sub-queries (default 7, at most 20), each retrieved through the hybrid lanes and merged content-keyed; single query only; planning errors propagate; context is uncapped (up to 8x each lane's top-k by default); never defers to a graph type: a request hybrid cannot serve (custom `node_type`, `neighborhood_depth`, `feedback_influence > 0`, no chunk collection) raises `QueryValidationError` pointing at `GRAPH_COMPLETION_DECOMPOSITION`
- **GRAPH_COMPLETION** - Graph traversal + LLM completion
- **GRAPH_SUMMARY_COMPLETION** - Graph traversal plus a second LLM call that summarizes the answer (reads no pre-computed summaries)
- **GRAPH_COMPLETION_COT** - Chain-of-thought reasoning over graph
- **GRAPH_COMPLETION_CONTEXT_EXTENSION** - Extended context graph retrieval
- **RAG_COMPLETION** - Traditional RAG with chunks
- **CHUNKS** - Vector similarity search over chunks
- **CHUNKS_LEXICAL** - Lexical (keyword) search over chunks
- **SUMMARIES** - Search pre-computed document summaries
- **CYPHER** - Direct Cypher query execution (disabled by default; `ALLOW_CYPHER_QUERY=true` enables it)
- **NATURAL_LANGUAGE** - Natural language to structured query (LLM writes Cypher, then runs it; disabled by default behind the same `ALLOW_CYPHER_QUERY=true` flag)
- **TEMPORAL** - Hybrid retrieval reranked by time: the `Timestamp` nodes overlapping the window the LLM extracts from the question ("what happened in 1969?") are looked up natively (`get_timestamps_in_range` on Ladybug, Neo4j, Postgres demo; a scan elsewhere), the chunks attached to them join the hybrid candidates (similarity alone misses most rows dated inside a window), and the anchored candidates move to the front, ranked by how tightly their matched timestamp fits the window (a chunk dated to the day before one that only names the year); nothing is dropped, and a question with no time or no overlap gets plain hybrid
- **FEELING_LUCKY** - Automatic search type selection
- **CODING_RULES** - Code-specific search rules
- **SKILLS** - Semantic discovery of skill playbooks (metadata-only, no LLM; requires exactly one dataset)
- **GRAPH_COMPLETION_DECOMPOSITION** - Splits the question into focused sub-queries, then runs graph completion over the merged context
- **AGENTIC_COMPLETION** - Multi-step LLM loop that can load `skills` and call `tools`; bounded by `max_iter`
- **CODE** - Deterministic operations over the code graph via `code_query` (no LLM); see [Code Files](memory-internals.md#code-files-cognify-code-route)
- **GRAPH_REPORT** - Graph insight report: hub nodes, cross-node-set connections, edge provenance, suggested questions

`recall()` picks one of these automatically when `query_type` is omitted; so does `cognee-cli recall` when `--query-type` is omitted, and `POST /api/v1/recall` when `searchType` is omitted or `null` (the default). The CLI's explicit `--query-type` accepts only the choices in `cognee/cli/config.py:SEARCH_TYPE_CHOICES`; the rest are SDK-only. See the [recall and search guide](recall-vs-search.md) for routing rules and bypass options.

Key files:
- `cognee/api/v1/search/search.py`
- `cognee/modules/retrieval/README.md` — SearchType → retriever class table (kept in sync by a unit test)
- `cognee/modules/search/methods/get_search_type_retriever_instance.py` — the registry itself
- `cognee/modules/retrieval/context_providers/TripletSearchContextProvider.py`
- `cognee/modules/search/types/SearchType.py`

## Session Search Modes

A session search (a `search()` with an active session cache) runs in one of two modes, chosen deployment-wide by `SESSION_SEARCH_MODE`. There is no per-request override.

Both modes make the same two LLM calls per turn — one to analyze the turn for session context, one to answer. They differ in how those calls are sequenced:

- **`concurrent`** (default) — analysis runs **concurrently** with retrieval and answering, so a turn costs one answer call of wall-clock time. Retrieval compensates for not having the analysis's rewritten query by running two lanes: the raw question, and a deterministic (LLM-free) rewrite built from the last two turns. Their results are merged by the retriever before context is formatted.
- **`sequential`** — analysis runs **first**, its rewritten query drives a single retrieval, and its context updates are applied before the answer is generated.

The practical difference: in sequential mode, guidance the user states this turn can influence this turn's answer. In concurrent mode it applies from the next turn onward.

Concurrent mode applies only to `GraphCompletionRetriever`, `HybridRetriever`, and `CompletionRetriever` (`RAG_COMPLETION`), and only through `search()`. Calling a retriever's `get_completion()` directly always takes the sequential path. Subclasses, batch queries, `only_context`, `FEELING_LUCKY`, and sessionless calls fall back to sequential mode automatically. With `AUTO_FEEDBACK=false` neither mode analyzes the turn.

## Completion prompt layout

Every completion is assembled by one function, `build_completion_prompts` in `cognee/modules/retrieval/utils/completion.py`. The **system prompt** is the retriever's task template and nothing else: cognee-authored, static per retriever, so it never changes between turns. Everything derived from the user goes into the **user prompt**, in this order: the conversation history, the rendered question-and-context template, and the guidance block last (the `## Active session guidance` block with a session, the durable preference block sessionless). The session layer travels as one `SessionPrompt(history, guidance)` value, defined next to the builder; `SessionPrompt()` is the empty layer and sessionless callers pass `SessionPrompt(guidance=preference_text)`.

The placement was measured across every combination, not chosen by taste. Soft preferences ("the user prefers German") are ignored from the system prompt by the default model and followed from the user turn; history is only used when it sits next to the context the template points at (the user template says "do not use information outside the context", so history delivered as system text or as chat turns is refused); guidance placed last wins over older turns that asked for something else; and an instruction planted in a past answer is obeyed only from the system prompt. Keep those four properties when touching the templates.

## only_context

`only_context=True` returns what the LLM would have received instead of its answer. For completion search types that is the two messages a completion sends, kept apart: the **user prompt** (conversation history, then the question and the retrieval context rendered through the retriever's user template, then the session guidance block) and the **system prompt** (the retriever's task template). `search()` returns the user prompt as the result and, with `verbose=True`, carries both as `user_prompt_result` / `system_prompt_result`; a `recall()` item has the user prompt in `text` and the system prompt in `system_prompt`. Both are built by the same code the real completion uses (`build_session_prompt` in read-only mode plus `build_completion_prompts`, see `cognee/modules/retrieval/only_context_prompt.py`), so they cannot drift from what `generate_completion` sends.

```python
items = await cognee.recall(
    "why did the migration stall?",
    query_type=SearchType.GRAPH_COMPLETION,  # pin the graph lane — with a bare
    session_id="s1",  # session_id a session hit would
    only_context=True,  # short-circuit it (see recall vs search)
)
user_prompt, system_prompt = items[0].text, items[0].system_prompt
```

What it does and does not do: no LLM completion, no turn-analysis call, nothing written to the session, no QA turn recorded. It does make **one embedding call** — the conversation-history vector recall — once per search, shared across the dataset fan-out, and only when a prompt is actually built. With `CACHING=false` the system prompt still carries the durable preference block, exactly as the real sessionless completion does. Retrievers whose *retrieval* stage calls an LLM (`GRAPH_COMPLETION_COT`, `GRAPH_COMPLETION_DECOMPOSITION`, `GRAPH_COMPLETION_CONTEXT_EXTENSION`, `TEMPORAL`'s time extraction, `GRAPH_SUMMARY_COMPLETION`'s summaries) still make those calls under `only_context`, as they always have; for them the pair is the final prompts over the final context.

Where the pair is *not* built and the bare retrieval context comes back instead: the non-generative types (`CHUNKS`, `SUMMARIES`, `CODE`, `SKILLS`, …) have no prompt template; `CYPHER` and `AGENTIC_COMPLETION` opt out via `supports_prompt_preview = False` (Cypher never prompts; the agentic loop answers through other templates); and an empty retrieval returns the empty context, never a prompt wrapped around nothing, so "nothing found" stays detectable — for `recall()` it yields zero items and the `on_empty` tools fallback still fires. The bare context stays reachable for callers that only want that: `search(verbose=True)` carries it as `context_result` next to the two prompts. `@agent_memory(memory_only_context=True)` reads `context_result`, so an agent's memory block never contains cognee's own question framing or answer instructions.

Caveats. `POST /api/v1/search` accepts `session_id`; without one the session layer is the default session's. And the pair is knowingly unfaithful in one place: a real sequential turn first rewrites the question (`effective_query`), and that rewrite fills `{{ question }}`, drives history selection, and ranks the guidance block — concurrent mode also merges a second retrieval lane. Producing the rewrite is an LLM call, so the raw query is used for all of them: the pair is the prompts for the context actually retrieved, not a replay of a full turn.
