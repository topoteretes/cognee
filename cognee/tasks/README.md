# cognee/tasks — pipeline task index

A task is an async function (or generator) that a pipeline runs over a stream of
items. Tasks are wrapped in `Task(...)` from `cognee.modules.pipelines` and chained
by the runner; the runner semantics (a task's `batch_size` batches the *previous*
task's output, `enriches`, `ctx` injection, `Drop`) are documented in
`cognee/modules/pipelines/__init__.py`. This file is the map of what tasks exist.

## The default pipelines

`cognify()` (`cognee/api/v1/cognify/cognify.py`, `get_default_tasks`):

| Step | Task | In → out |
|---|---|---|
| 1 | `documents.classify_documents` | `Data` rows → typed `Document`s (`TextDocument`, `PdfDocument`, …) |
| 2 | `documents.extract_chunks_from_documents` | `Document` → `DocumentChunk`s (uses `cognee.modules.chunking`) |
| 3 | `graph.extract_graph_from_data` | chunks → chunks with `Entity`/`EntityType` nodes and edges attached (LLM, or GLiNER with `extractor="gliner_demo"`) |
| 4 | `summarization.summarize_text` | chunks → `TextSummary` nodes (LLM) |
| 5 | `storage.add_data_points` | data points → written to graph + vector DB (+ edge evidence) |
| 6 | `provenance.record_provenance` | audit-ledger rows, when `PROVENANCE_TRACKING=true` |
| 7 | `graph.detect_contradictions` | `contradicts` edges, when `CONTRADICTION_DETECTION=true` |

`cognify(temporal_cognify=True)` swaps steps 3–4 for `temporal_graph.extract_events_and_timestamps`
→ `temporal_graph.extract_knowledge_graph_from_events`. The dlt route adds
`ingestion.purge_stale_dlt_source_artifacts` and `ingestion.extract_dlt_source_edges`.
`improve()` / `memify()` run the pre-assembled lists in `cognee/memify_pipelines/`.

## Subpackages

| Package | What its tasks do | Key exports |
|---|---|---|
| `chunks/` | Split text into chunks and manage chunk relations | `chunk_by_paragraph`, `chunk_by_sentence`, `chunk_by_word`, `chunk_by_row`, `remove_disconnected_chunks`, `create_chunk_associations` |
| `cleanup/` | Remove data that no dataset references any more | `cleanup_unused_data` |
| `code_graph/` | Build the deterministic code graph with enola (no LLM): `CodeSymbol`/`CodeModule` nodes, `calls`/`imports` edges; installs the pinned enola binary | `extract_code_graph`, `extract_code_files_graph`, `install_enola` |
| `codingagents/` | Distil coding-agent traces into reusable coding rules (`CODING_RULES` search) | see its `README.md` |
| `completion/` | Exceptions shared by completion-style tasks | — |
| `documents/` | Classify `Data` rows into `Document` types and chunk them | `classify_documents`, `extract_chunks_from_documents` |
| `entity_completion/` | Pluggable entity extractors (`entity_extractors/`: LLM-based and regex-based) | `LLMEntityExtractor`, `RegexEntityExtractor` |
| `graph/` | Extract entities/relations into the graph; contradiction detection; the GLiNER (`graph/gliner_demo/`) and code (`extract_graph_from_code`) variants | `extract_graph_from_data`, `extract_graph_from_code`, `detect_contradictions`, `resolve_temporal_contradictions` |
| `ingestion/` | Ingest and normalise inputs for `add()`: resolve paths/directories, save to storage, dedup, dlt sources, relational-DB migration | `ingest_data`, `resolve_data_directories`, `save_data_item_to_storage`, `resolve_dlt_sources`, `migrate_relational_database` |
| `memify/` | Enrichment tasks: session and agent-trace persistence, feedback weights, entity dedup/consolidation, triplet embeddings, global context index | `extract_subgraph`, `cognify_session`, `apply_feedback_weights`, `extract_feedback_qas`, `detect_entity_duplicates`, … |
| `presort/` | Pre-organise a folder before ingestion (`remember(dry_run="presort")`): classify, hash, dedup, version and PII detection, proposed groupings | `build_report`, `classify_files`, `detect_duplicates`, `detect_pii`, `group_files`, `apply_presort_graph` |
| `provenance/` | Write audit-ledger provenance entries for a pipeline run | `record_provenance` |
| `schema/` | Ingest a relational database schema as graph nodes | `ingest_database_schema` |
| `storage/` | Persist data points: graph + vector writes, index rebuilds, fact validity (`close_node`) | `add_data_points`, `index_data_points`, `index_graph_edges` |
| `summarization/` | LLM summaries of chunks and code | `summarize_text`, `summarize_code` |
| `temporal_graph/` | Event/timestamp extraction and the temporal knowledge graph | `extract_events_and_timestamps`, `extract_knowledge_graph_from_events` |
| `translation/` | Translate chunk content before extraction (provider-pluggable) | `TranslationConfig`, `TranslatedContent` |
| `web_scraper/` | Fetch and crawl web pages for URL ingestion | `fetch_page_content`, `DefaultUrlCrawler` |

## Writing a task

```python
from cognee.modules.pipelines import Task
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.pipelines.tasks.task import task_summary


@task_summary("Tagged {n} chunk(s)")
async def tag_chunks(chunks: list, ctx: PipelineContext = None, label: str = "x"):
    """Say what comes in, what goes out, and any side effect (DB writes, LLM calls)."""
    for chunk in chunks:
        chunk.metadata["label"] = label
    return chunks  # or `yield` per item for streaming; return `Drop` to discard


task = Task(tag_chunks, label="reviewed", task_config={"batch_size": 10}, needs_llm=False)
```

- Put the task in the subpackage matching its stage and export it from that
  package's `__init__.py`.
- `needs_llm=False` on tasks that never call the LLM lets an LLM-free pipeline skip
  the connection probe.
- Add a unit test under `cognee/tests/unit/tasks/` and, if the task joins a default
  pipeline, an integration test that runs `remember → recall`.
