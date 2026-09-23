---
name: cognee-ingestion
description: Use when putting data into cognee memory with remember() — choosing inputs (text, files, folders, URLs, repos, databases), datasets and node_sets, loaders, ontologies, the graph extractor (LLM or GLiNER), chunking, dry-run cost estimates, temporal graphs, or when remember() raises on a keyword argument.
---

# Ingest data with remember()

`remember()` is cognee's ingestion API. One call stores the data, builds the
knowledge graph, and enriches it. Use it for all ingestion; every option in
this skill is a `remember()` argument unless it says otherwise.

```python
import cognee

result = await cognee.remember("Einstein was born in Ulm.")           # text
result = await cognee.remember(["./notes.md", "./report.pdf"],        # files
                               dataset_name="research")
print(result.status, result.dataset_id)   # "completed", UUID
```

All cognee functions are async. Without `dataset_name` data goes to
`main_dataset`. Needs `LLM_API_KEY` unless you use the GLiNER extractor
(below).

## Use it

### Inputs

`data` accepts a string, a list of strings, file paths (absolute, `file://`,
`s3://`), http(s) URLs, binary streams, or a list mixing them.

- **URLs** are fetched and scraped (needs `ALLOW_HTTP_REQUESTS=true`, the
  default).
- **Folders** are ingested file by file. A folder that looks like a code
  project, or a GitHub/GitLab URL, becomes one code repository (needs `git`
  on PATH).
- **Code files** (`.py`, `.ts`, `.go`, …) go down the code-graph route: a
  deterministic graph, no LLM calls, searchable only with
  `SearchType.CODE`. To index a whole repository explicitly, pass
  `content_type="code"`.
- **Databases and dlt sources**: a SQL connection string, a dlt
  `DltResource` / `DltSource`, or a CSV. Needs the `dlt` extra
  (`pip install "cognee[dlt]"`). Options: `primary_key` (default `"id"`),
  `write_disposition` (`"replace"` default, or `"append"`), `query`,
  `max_rows_per_table`.
- **Skill playbooks** (`SKILL.md` files): `content_type="skills"` with an
  explicit `dataset_name`.

### Where the data goes

| Argument | What it does |
|---|---|
| `dataset_name` / `dataset_id` | Target dataset. `dataset_id` wins. A dataset is the unit of permissions and isolation. |
| `node_set=["AI", "FinTech"]` | Tags the data so recall can filter to it later with `recall(..., node_name=["AI"])`. |
| `session_id="chat_1"` | Writes to the fast session cache instead of the graph; `improve()` bridges it into the graph in the background. See the `cognee-improve-sessions` skill. Requires `CACHING=true`. |

### How the graph is built

| Argument | What it does |
|---|---|
| `extractor` | `"llm"` or `"gliner_demo"` (alias `"gliner"`). Default is `GRAPH_EXTRACTOR=auto`: the LLM when an API key is configured, otherwise GLiNER. |
| `graph_model=MyModel` | Extract into your own DataPoint model instead of the generic `KnowledgeGraph`. See the `cognee-custom-graph-models` skill. |
| `custom_prompt` | Replaces the entity-extraction prompt (ignored by GLiNER). |
| `config={"ontology_config": {...}}` | Ground entities in an OWL ontology (below). |
| `temporal_cognify=True` | Builds an event/timestamp graph for `SearchType.TEMPORAL`. |
| `chunk_size`, `chunker` | Max tokens per chunk (default: derived from the embedding and LLM limits) and the chunker class (default `TextChunker`). |
| `preferred_loaders` | Choose a loader per file type (below). |
| `self_improvement` | Default `True`: runs `improve()` after the graph is built. Its outcome is on `result.improve` / `result.improve_error`; a failed improve never fails the remember. |
| `run_in_background=True` | Returns immediately with `status="running"`; `await result` to wait. |

### Ontologies

```python
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver

config = {"ontology_config": {
    "ontology_resolver": RDFLibOntologyResolver(ontology_file="./my.owl"),
    # "ontology_mode": "strict",   # drop entities with no ontology match
}}
await cognee.remember(texts, config=config)
```

Or set `ONTOLOGY_FILE_PATH` (plus `ONTOLOGY_MODE`, `MATCHING_STRATEGY`) in
`.env`. `annotate` (default) only enriches; `strict` drops entities that
match no ontology class or individual. It prunes only the graph, chunk text
is still stored. Strict mode with an empty or missing ontology file is a hard
error. Over HTTP, upload the ontology to `/api/v1/ontologies` and pass its
`ontology_key` to `POST /api/v1/remember`. Example:
`examples/guides/ontology_quickstart.py`.

### Loaders

Each file is claimed by the first loader that accepts it. Default order:
code, text, pypdf, image, audio, video, dlt_csv, csv, unstructured,
advanced_pdf, docling. Names: `text_loader`, `code_loader`, `csv_loader`,
`dlt_csv_loader`, `pypdf_loader`, `image_loader`, `audio_loader`,
`video_loader`, `unstructured_loader`, `advanced_pdf_loader`,
`docling_loader`, `beautiful_soup_loader`.

```python
# Treat a code file as a plain document (chunking + LLM extraction):
await cognee.remember("./script.py", preferred_loaders={"text_loader": {}})
```

Office formats (DOCX, PPTX, …) need the `docs` (unstructured) or `docling`
extra. A preferred loader that is not installed is skipped with only an info
log, so check the extra is installed when a file comes out wrong.

### Check the cost first

`dry_run=True` returns a token and cost estimate without ingesting anything
or calling the LLM. It excludes the calls `improve()` makes. Not supported
with GLiNER, sessions, or a remote instance.

`dry_run="presort"` on a folder returns a `PresortReport` (junk, duplicates,
version candidates, possible personal data, proposed dataset groups). Apply
it with `await cognee.remember(report)`, or pass `auto_apply=True`.

### Without an LLM: GLiNER

`extractor="gliner"` builds the graph and summaries with a local GLiNER2
model, with no LLM call (embeddings still run). Install
`pip install "cognee[gliner]"`; the model (~800 MB) downloads on first use.
It cannot be combined with a custom `graph_model`, `temporal_cognify`,
`dry_run`, `session_id`, or a remote instance.

> **For production:** the open-source GLiNER extractor is a demo. cognee's
> enterprise GLiNER extraction is more accurate and covers more labels. The
> same goes for the Postgres graph adapter (`postgres_demo`). Contact
> social@cognee.ai.

## Pitfalls

- **Unknown keyword arguments raise.** `remember()` forwards kwargs through
  a fixed allow-list and raises `TypeError: Unexpected keyword arguments`
  for anything else. These real options are not on it yet:

  | Option | Workaround through remember() |
  |---|---|
  | `ontology_file_path` | `config={"ontology_config": ...}` or `ONTOLOGY_FILE_PATH` (above) |
  | `functional_relationships`, `chunk_attachment` | None yet. Only `cognee.cognify()` accepts them. |
  | `extraction_rules`, `tavily_config`, `soup_crawler_config` (web scraping) | None yet. Only `cognee.add()` accepts them. |
  | `column_value_columns` (dlt) | None yet. Only `cognee.add()` accepts it. |

  If a user needs one of these, say so plainly: the option exists on the
  lower-level `add()` / `cognify()` but not on `remember()` yet.
- **Changed files raise `DocumentUpdateRequiredError`.** Re-remembering the
  same path (or the same filename for an upload) with different content is
  an update, not a new document. Use
  `cognee.update(data_id=..., data=..., dataset_id=...)`, which re-extracts
  only the changed chunks and keeps the document's id. Identical content is
  a no-op.
- **`content_type` is strict.** Only `None`, `"skills"`, or `"code"`.
  `"code"` rejects `session_id` and needs repository paths or git URLs;
  `"skills"` needs an explicit dataset.
- **Session mode needs `CACHING=true`**, and `extractor` cannot be combined
  with `session_id`.
- **Remote mode.** After `cognee.serve(url)`, calls go to the server;
  `extractor` and `session_ids` raise there.
- **Every remember runs `improve()`** unless `self_improvement=False` or
  `IMPROVE_AUTO_ENABLED=false`. In scripts, call
  `await cognee.wait_for_background_tasks()` before exiting.

## How it works

`remember(data)` runs `add()` (store raw data and create `Data` rows), then
`cognify()` (classify documents, chunk, extract the graph and summaries,
store in graph and vector DBs), then `improve()`. `remember(data,
session_id=...)` writes to the session cache instead.

- Entry point and kwarg routing: `cognee/api/v1/remember/remember.py`
  (`RememberKwargs`, `_ADD_ONLY` / `_COGNIFY_ONLY` / `_SHARED`)
- Storage: `cognee/api/v1/add/add.py`, `cognee/tasks/ingestion/ingest_data.py`
- Graph build: `cognee/api/v1/cognify/cognify.py`,
  `cognee/tasks/graph/extract_graph_from_data.py`,
  `cognee/tasks/storage/add_data_points.py`
- Extractor choice: `cognee/modules/cognify/config.py:resolve_extractor`;
  GLiNER package: `cognee/tasks/graph/gliner_demo/`
- Ontologies: `cognee/modules/ontology/`
- Loaders: `cognee/infrastructure/loaders/` (`supported_loaders.py`,
  `LoaderEngine.py`)
- dlt: `cognee/tasks/ingestion/resolve_dlt_sources.py`

Examples in `examples/guides/`: `simple_cognee_example.py`,
`nodeset_grouping_example.py`, `ontology_quickstart.py`,
`gliner_demo_llm_free_cognify.py`, `no_llm_remember_recall.py`,
`temporal_recall.py`, `presort_downloads.py`,
`web_url_content_ingestion_example.py`, `code_graph_example.py`.

## Extending it

- **New remember() option:** add it to `RememberKwargs` and to the matching
  routing set in `remember.py`. An option on `add()`/`cognify()` that is not
  in a routing set raises `TypeError` from `remember()`.
- **New loader:** implement `LoaderInterface`
  (`cognee/infrastructure/loaders/LoaderInterface.py`), register it in
  `supported_loaders.py` (extras-gated loaders go under `external/`), and
  add it to the priority list in `LoaderEngine.py` if it should run by
  default.
- **New cognify task:** see the `cognee-custom-pipelines` skill and
  `cognee/tasks/README.md`.
