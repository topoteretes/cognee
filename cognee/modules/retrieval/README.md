# cognee/modules/retrieval — SearchType → retriever map

Every `SearchType` (`cognee/modules/search/types/SearchType.py`) is served by one
retriever class in this package. The dispatch lives in
`cognee/modules/search/methods/get_search_type_retriever_instance.py`
(`search_core_registry`); the table below mirrors it and is checked by
`cognee/tests/unit/modules/retrieval/retriever_readme_index_test.py`, so update both
together.

| SearchType | Retriever | Module | LLM answer? | Notes |
|---|---|---|---|---|
| `HYBRID_COMPLETION` | `HybridRetriever` | `hybrid_retriever.py` | yes | Default. Chunk + entity (+ global-context) lanes. Defers to `GRAPH_COMPLETION` for a custom `node_type`, `node_name` with `node_type=None`, `neighborhood_depth`, `feedback_influence > 0`, or a missing chunk collection (`search/methods/hybrid_deferral.py`) |
| `HYBRID_COMPLETION_DECOMPOSITION` | `HybridDecompositionRetriever` | `hybrid_decomposition_retriever.py` | yes | Hybrid pass for the full question, then LLM decomposition of question + that context into up to 7 subqueries, each retrieved through the hybrid lanes and merged content-keyed. Single query only |
| `GRAPH_COMPLETION` | `GraphCompletionRetriever` | `graph_completion_retriever.py` | yes | Triplet search + neighbourhood expansion; base for the variants below |
| `GRAPH_COMPLETION_COT` | `GraphCompletionCotRetriever` | `graph_completion_cot_retriever.py` | yes | Chain-of-thought rounds |
| `GRAPH_COMPLETION_CONTEXT_EXTENSION` | `GraphCompletionContextExtensionRetriever` | `graph_completion_context_extension_retriever.py` | yes | Iteratively widens context |
| `GRAPH_COMPLETION_DECOMPOSITION` | `GraphCompletionDecompositionRetriever` | `graph_completion_decomposition_retriever.py` | yes | Splits the query into sub-queries first |
| `GRAPH_SUMMARY_COMPLETION` | `GraphSummaryCompletionRetriever` | `graph_summary_completion_retriever.py` | yes | Graph context built from summaries |
| `TEMPORAL` | `TemporalHybridRetriever` | `temporal_hybrid_retriever.py` | yes | Hybrid retrieval reranked by time: the adapter returns the `Timestamp` nodes overlapping the window the LLM extracts from the question (`GraphDBInterface.get_timestamps_in_range`), the chunks and entities anchored to them move to the front of the candidate set, nothing is dropped; plain hybrid when the question carries no time or nothing overlaps (`temporal_hybrid/matching.py`) |
| `RAG_COMPLETION` | `CompletionRetriever` | `completion_retriever.py` | yes | Chunk vector search + completion |
| `BROAD` | `BroadRetriever` | `broad_retriever.py` | yes | Questions over a whole dataset (counts, totals, lists, lookups): a small dataset is shown whole; a large one is loaded into SQLite (`broad_store.py`) and answered by one model-written query |
| `TRIPLET_COMPLETION` | `GraphCompletionRetriever` | `graph_completion_retriever.py` | yes | Deprecated, removed in the next release: triplet embeddings were removed, so `search()` / `recall()` resolve it to `GRAPH_COMPLETION` with a `FutureWarning` (`DEPRECATED_SEARCH_TYPES`) |
| `AGENTIC_COMPLETION` | `AgenticRetriever` | `agentic_retriever.py` | yes | Special-cased in the factory (not in the dict): tool loop with `skills`, `tools`, `max_iter` |
| `CHUNKS` | `ChunksRetriever` | `chunks_retriever.py` | no | Vector search over `DocumentChunk_text` |
| `CHUNKS_LEXICAL` | `BM25ChunksRetriever` | `bm25_retriever.py` | no | BM25 over chunks (`LexicalRetriever` subclass) |
| `SUMMARIES` | `SummariesRetriever` | `summaries_retriever.py` | no | Vector search over `TextSummary_text` |
| `SKILLS` | `SkillsRetriever` | `skills_retriever.py` | no | Metadata-only skill discovery; exactly one dataset |
| `CODE` | `CodeRetriever` | `code_retriever.py` | no | Deterministic code-graph operations via `code_query` |
| `CODING_RULES` | `CodingRulesRetriever` | `coding_rules_retriever.py` | no | Returns stored coding rules |
| `CYPHER` | `CypherSearchRetriever` | `cypher_search_retriever.py` | no | Raw Cypher; disabled by default, enabled by `ALLOW_CYPHER_QUERY=true` |
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
   `cognee/cli/config.py`; if `recall()` should route to it, add a regex rule to the
   ordered, first-match-wins table in `cognee/api/v1/recall/query_router.py` (`cognee/modules/recall/` is session
   recall, not the router).

Supporting folders: `hybrid/` (the hybrid lanes), `context_providers/` (triplet
context formatting), `entity_extractors/`, `utils/` (completion, evidence, ranking
helpers), `only_context_prompt.py` (the full LLM input an `only_context` search returns), and
`session_aware_completion.py` (the concurrent/sequential session turn).

## Hybrid decomposition without retrieval

`HybridDecompositionRetriever` adds orchestration to `HybridRetriever`; it does
not implement its own chunk/entity/fact search, metadata projection, reference
formatting or answer generation. The stages are independently scoped:

- `decompose_query(question, context)` plans subquestions and can be overridden
  to use another planner without editing hybrid retrieval.
- `_retrieve_subqueries` runs expansion searches concurrently using the parent
  lane implementation. The original question reuses its first-pass results.
- `hybrid.merge.merge_hybrid_results_union` deduplicates and applies per-channel
  expansion budgets. First-pass objects retain their original metadata.
- `decomposition_state` exposes the subqueries, first pass, expansion results
  and merged result for inspection after retrieval.

Use `SearchType.HYBRID_COMPLETION_DECOMPOSITION` with `search()` or `recall()`.
Configure `max_subqueries`, `decomposition_system_prompt` (inline text),
`decomposition_system_prompt_path` and `merged_chunks_limit`,
`merged_entities_limit`, `merged_facts_limit` through
`retriever_specific_config`. All standard hybrid configuration, including
`include_external_metadata` and `external_metadata_keys`, is forwarded too.
See [the runnable example](../../../examples/guides/hybrid_decomposition.py).

`only_context=True` skips the final answer, but still runs both retrieval passes
and the decomposition LLM. To inspect subquestions without any graph, vector or
embedding calls, use the decomposition-only method:

```python
from cognee.modules.retrieval.hybrid_decomposition_retriever import HybridDecompositionRetriever

retriever = HybridDecompositionRetriever(max_subqueries=12)
subquestions = await retriever.decompose_query(
    "Which guarantees cover project X, and what are their expiry rules?",
    context="Optional source passages supplied by the caller, not retrieved here.",
)
```

The default prompt follows `max_subqueries`. An inline
`decomposition_system_prompt` is passed verbatim, so callers must give it a
compatible budget. Missing/blank prompts and decomposition errors fall back to
the original question. The method does not change retrieval run state.

In a full retrieval run, the default union budget covers actual returned objects,
including the larger fact lane hybrid can use when no entities match. Explicit
`merged_*_limit` settings cap expansion, with the distinct first-pass items as a
minimum: a ceiling below the baseline cannot remove baseline evidence. Failed
extra legs leave the first pass and successful legs intact. Cancellation still
propagates. Session preparation and session completion are excluded for this
retriever, and graph-only search knobs are rejected rather than ignored.

Invalid budgets fail at construction. The first retrieval's errors, including
the parent's empty-graph `NoDataError`, propagate. Graceful degradation applies
only to the optional decomposition/expansion stage required by SDK-694.
Preserving evidence does not guarantee a better answer: expansion increases
context size and should be evaluated for each use case. No ontology planning,
question router or application-specific prompts are built into this retriever.
