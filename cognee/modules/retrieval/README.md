# cognee/modules/retrieval — SearchType → retriever map

Every `SearchType` (`cognee/modules/search/types/SearchType.py`) is served by one
retriever class in this package. The dispatch lives in
`cognee/modules/search/methods/get_search_type_retriever_instance.py`
(`search_core_registry`); the table below mirrors it and is checked by
`cognee/tests/unit/modules/retrieval/retriever_readme_index_test.py`, so update both
together.

| SearchType | Retriever | Module | LLM answer? | Notes |
|---|---|---|---|---|
| `HYBRID_COMPLETION` | `HybridRetriever` | `hybrid_retriever.py` | yes | Default. Chunk + entity (+ global-context) lanes. Defers to `GRAPH_COMPLETION` when given a custom `node_type` or when the chunk collection is missing (`search/methods/hybrid_deferral.py`) |
| `GRAPH_COMPLETION` | `GraphCompletionRetriever` | `graph_completion_retriever.py` | yes | Triplet search + neighbourhood expansion; base for the variants below |
| `GRAPH_COMPLETION_COT` | `GraphCompletionCotRetriever` | `graph_completion_cot_retriever.py` | yes | Chain-of-thought rounds |
| `GRAPH_COMPLETION_CONTEXT_EXTENSION` | `GraphCompletionContextExtensionRetriever` | `graph_completion_context_extension_retriever.py` | yes | Iteratively widens context |
| `GRAPH_COMPLETION_DECOMPOSITION` | `GraphCompletionDecompositionRetriever` | `graph_completion_decomposition_retriever.py` | yes | Splits the query into sub-queries first |
| `GRAPH_SUMMARY_COMPLETION` | `GraphSummaryCompletionRetriever` | `graph_summary_completion_retriever.py` | yes | Graph context built from summaries |
| `TEMPORAL` | `TemporalRetriever` | `temporal_retriever.py` | yes | Time-bounded traversal over the temporal graph |
| `RAG_COMPLETION` | `CompletionRetriever` | `completion_retriever.py` | yes | Chunk vector search + completion |
| `TRIPLET_COMPLETION` | `TripletRetriever` | `triplet_retriever.py` | yes | Triplet-embedding search + completion |
| `AGENTIC_COMPLETION` | `AgenticRetriever` | `agentic_retriever.py` | yes | Special-cased in the factory (not in the dict): tool loop with `skills`, `tools`, `max_iter` |
| `CHUNKS` | `ChunksRetriever` | `chunks_retriever.py` | no | Vector search over `DocumentChunk_text` |
| `CHUNKS_LEXICAL` | `BM25ChunksRetriever` | `bm25_retriever.py` | no | BM25 over chunks (`LexicalRetriever` subclass) |
| `SUMMARIES` | `SummariesRetriever` | `summaries_retriever.py` | no | Vector search over `TextSummary_text` |
| `SKILLS` | `SkillsRetriever` | `skills_retriever.py` | no | Metadata-only skill discovery; exactly one dataset |
| `CODE` | `CodeRetriever` | `code_retriever.py` | no | Deterministic code-graph operations via `code_query` |
| `CODING_RULES` | `CodingRulesRetriever` | `coding_rules_retriever.py` | no | Returns stored coding rules |
| `CYPHER` | `CypherSearchRetriever` | `cypher_search_retriever.py` | no | Raw Cypher; needs `ALLOW_CYPHER_QUERY=true` |
| `NATURAL_LANGUAGE` | `NaturalLanguageRetriever` | `natural_language_retriever.py` | yes | LLM writes Cypher, then runs it; same `ALLOW_CYPHER_QUERY` gate |
| `GRAPH_REPORT` | `GraphReportRetriever` | `graph_report_retriever.py` | partly | Hubs, cross-node-set edges, provenance split, LLM-suggested questions |
| `FEELING_LUCKY` | — | `search/operations/select_search_type.py` | — | An LLM picks one of the types above, then that retriever runs |

Not wired to any `SearchType`: `LexicalRetriever` (`lexical_retriever.py`, the base
for BM25) and `JaccardChunksRetriever` (`jaccard_retrival.py`, unused). Community
retrievers register through `register_retriever.py` and take precedence for their type.

## The contract

All retrievers subclass `BaseRetriever` (`base_retriever.py`) and implement three
async methods, run in order by `get_completion(query)`:

1. `get_retrieved_objects(query, query_batch)` — hit the graph/vector store.
2. `get_context_from_objects(query, query_batch, retrieved_objects)` — format for the LLM.
3. `get_completion_from_context(query, query_batch, retrieved_objects, context)` — answer.
   The "no" rows above return the context unchanged here.

Optional hooks: `extract_context_object_ids` (which nodes/edges fed a session answer),
`merge_retrieved_objects` (combine the raw-query and rewritten-query lanes of a session
turn), `append_references` / `get_context_evidence` (`include_references=True`), and the
class flags `supports_session_turn_preparation` / `supports_prompt_preview`.

## Adding a retriever

1. Subclass `BaseRetriever` (or `GraphCompletionRetriever` for a graph variant) in
   a new `*_retriever.py`.
2. Add the `SearchType` member, then the `search_core_registry` entry with the
   constructor kwargs it needs.
3. Add a row to the table above and a `*_test.py` in `cognee/tests/unit/modules/retrieval/`.
4. If it should be reachable from the CLI, add it to `SEARCH_TYPE_CHOICES` in
   `cognee/cli/config.py`; if `recall()` should route to it, add a scoring rule to
   `cognee/api/v1/recall/query_router.py` (`cognee/modules/recall/` is session
   recall, not the router).

Supporting folders: `hybrid/` (the hybrid lanes), `context_providers/` (triplet
context formatting), `entity_extractors/`, `utils/` (completion, evidence, ranking
helpers), `only_context_prompt.py` (the full LLM input an `only_context` search returns), and
`session_aware_completion.py` (the concurrent/sequential session turn).
