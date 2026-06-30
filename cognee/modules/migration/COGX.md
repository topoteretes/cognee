# COGX: Cognee eXchange Format

COGX is Cognee's portable memory format. Every migration flows through it:
provider importers (Mem0, Zep/Graphiti, Letta) translate their native exports
into COGX records, and `cognee.export(format="cogx")` produces a COGX archive
that can be restored on another instance. If you are building an importer for
a new memory system, COGX is the only target you need to hit.

**Current version:** `0.1`
**Source:** `cognee/modules/migration/cogx.py`

## Table of Contents

1. [Archive layout](#archive-layout)
2. [Manifest](#manifest)
3. [Versioning](#versioning)
4. [Record types](#record-types)
5. [Export flow](#export-flow)
6. [Import flow](#import-flow)
7. [Import modes](#import-modes)
8. [Deterministic IDs and idempotent re-import](#deterministic-ids-and-idempotent-re-import)
9. [Adding a new provider source](#adding-a-new-provider-source)
10. [File reference](#file-reference)

## Archive Layout

A COGX archive is a directory with a manifest file and one JSONL file per
record kind:

```
my_dataset_cogx/
├── manifest.json
├── documents.jsonl
├── episodes.jsonl
├── entities.jsonl
├── facts.jsonl
├── memories.jsonl
├── memory_blocks.jsonl
└── nodes.jsonl
```

Only files that contain at least one record are present. Each line in a JSONL
file is a standalone JSON object discriminated by its `kind` field.

For transport over HTTP (used by `cognee.push()`), the directory is packed into
a single `.cogx.tar.gz` file by `pack_archive()` in `archive.py`.
`unpack_archive()` reverses this, with built-in decompression-bomb protection:
it caps the member count, per-member size, and total extracted size, and rejects
paths containing `..` or absolute components.

## Manifest

`manifest.json` is written when the archive writer closes successfully.

| Field            | Type             | Description                                              |
|------------------|------------------|----------------------------------------------------------|
| `cogx_version`   | `str`            | Format version. Currently `"0.1"`.                       |
| `source_system`  | `str`            | System that produced the archive, e.g. `"cognee"`.       |
| `exported_at`    | `datetime\|null` | UTC timestamp of the export.                             |
| `counts`         | `dict[str, int]` | Number of records per kind.                              |
| `embedding_model`| `str\|null`      | Embedding model used by the source dataset, if known.    |
| `notes`          | `list[str]`      | Free-form notes attached during export.                  |

## Versioning

`validate_cogx_version()` in `cogx.py` compares the archive's major version
against the reader's `COGX_VERSION`. The rules:

- **Newer major version:** rejected with `ValueError`. Upgrade cognee to read it.
- **Same major, newer minor:** accepted. Minor bumps are forward-compatible.
- **Older version:** always accepted.

This means a reader at version `0.1` will accept `0.2` but reject `1.0`.

## Record Types

Every typed record extends `COGXRecordBase`, which provides these shared fields:

| Field             | Type             | Description                                     |
|-------------------|------------------|-------------------------------------------------|
| `external_system` | `str`            | Source system identifier (e.g. `"mem0"`).        |
| `external_id`     | `str`            | Unique ID within that source system.             |
| `scope`           | `COGXScope`      | Ownership: `user_id`, `agent_id`, `session_id`, `run_id` (all optional strings). |
| `created_at`      | `datetime\|null` | Creation time in the source system.              |
| `updated_at`      | `datetime\|null` | Last modification time.                          |
| `metadata`        | `dict`           | Arbitrary provider-specific key-value pairs.     |

Below are the seven record kinds. The `kind` field is what discriminates them
in the JSONL files.

### COGXDocument (kind: `"document"`)

Raw text content: a file, archival passage, or standalone text.

| Field       | Type         | Description              |
|-------------|--------------|--------------------------|
| `content`   | `str`        | The document text.       |
| `title`     | `str\|null`  | Optional title.          |
| `mime_type` | `str\|null`  | MIME type when known.    |

### COGXEpisode (kind: `"episode"`)

A conversation episode made up of ordered turns.

| Field   | Type             | Description                    |
|---------|------------------|--------------------------------|
| `turns` | `list[COGXTurn]` | Ordered conversation turns.    |
| `title` | `str\|null`      | Optional episode title.        |

Each **COGXTurn** contains:

| Field         | Type             | Description                     |
|---------------|------------------|---------------------------------|
| `role`        | `str`            | Speaker role (`"user"`, `"assistant"`, etc.). |
| `content`     | `str`            | The turn text.                  |
| `occurred_at` | `datetime\|null` | When the turn happened.         |

### COGXEntity (kind: `"entity"`)

An extracted entity.

| Field         | Type         | Description                           |
|---------------|--------------|---------------------------------------|
| `name`        | `str`        | Entity name.                          |
| `entity_type` | `str\|null`  | Type label, e.g. `"Person"`.          |
| `description` | `str\|null`  | Description text.                     |
| `aliases`     | `list[str]`  | Alternative names for this entity.    |
| `attributes`  | `dict`       | Arbitrary key-value properties.       |

### COGXFact (kind: `"fact"`)

A subject-predicate-object triplet with bi-temporal validity.

| Field         | Type             | Description                                    |
|---------------|------------------|------------------------------------------------|
| `subject_ref` | `str`            | Source entity's `external_id` or plain name.   |
| `predicate`   | `str`            | Relationship type, e.g. `"works_at"`.          |
| `object_ref`  | `str`            | Target entity's `external_id` or plain name.   |
| `fact_text`   | `str\|null`      | Human-readable sentence expressing the fact.   |
| `valid_at`    | `datetime\|null` | When the fact became true.                     |
| `invalid_at`  | `datetime\|null` | When the fact stopped being true.              |
| `confidence`  | `float\|null`    | Confidence score from the source system.       |
| `provenance`  | `list[str]`      | Episode IDs that support this fact.            |

The temporal fields (`valid_at`, `invalid_at`) are always carried, even when
the current importer does not query them. They are stored as edge properties
so the data is in place when bi-temporal search is added.

### COGXMemory (kind: `"memory"`)

An atomic derived memory, matching the Mem0 "memory" shape.

| Field        | Type        | Description          |
|--------------|-------------|----------------------|
| `content`    | `str`       | The memory text.     |
| `categories` | `list[str]` | Category tags.       |

### COGXMemoryBlock (kind: `"memory_block"`)

A named, bounded core-memory block, matching the Letta/MemGPT shape.

| Field   | Type        | Description                                    |
|---------|-------------|------------------------------------------------|
| `label` | `str`       | Block name, e.g. `"human"` or `"persona"`.     |
| `value` | `str`       | Block content.                                 |
| `limit` | `int\|null` | Character limit from the source system.        |

### COGXRawNode (kind: `"raw_node"`)

A graph node stored verbatim when no typed COGX mapping exists. The
`properties` dict contains the full node payload including `id` and `type`.

| Field        | Type   | Description                   |
|--------------|--------|-------------------------------|
| `properties` | `dict` | Raw node property dictionary. |

On import, raw nodes are rehydrated into DataPoint instances via
`rehydrate_node()` (in `snapshot.py`), so any facts that reference them by ID
stay resolvable.

Raw nodes live in `nodes.jsonl`, separate from the typed record files.
`read_archive()` yields them last, after all typed records.

## Export Flow

```python
result = await cognee.export("my_dataset", format="cogx", destination="backup/")
```

Here is what happens inside `export_dataset()` (`export.py`):

1. The dataset is resolved and authorized (requires read permission).
2. All nodes and edges are fetched from the graph engine via `get_graph_data()`.
3. Each node is mapped to a typed COGX record when possible:
   - `Entity` nodes become `COGXEntity`.
   - `DocumentChunk` nodes become `COGXDocument` *and* `COGXRawNode`. The
     document carries the text for cross-provider portability; the raw node
     preserves the full graph structure for same-system restore.
   - All other node types become `COGXRawNode`.
4. Every edge becomes a `COGXFact`, with `valid_at`/`invalid_at` preserved.
5. `COGXArchiveWriter` writes the records into JSONL files and writes
   `manifest.json` on close.
6. When the archive needs to travel over HTTP, `pack_archive()` wraps the
   directory into a `.cogx.tar.gz`.

## Import Flow

```python
from cognee.migration import COGXArchiveSource, Mem0Source, ZepSource, LettaSource

# Restore a COGX backup (zero LLM cost, preserve mode)
await cognee.remember(COGXArchiveSource("backup/"))

# Import from another provider
await cognee.remember(Mem0Source("mem0_export.json"))
await cognee.remember(ZepSource("graphiti_dump.json", mode="hybrid"))
await cognee.remember(LettaSource("agent_file.af"))
```

When `remember()` receives a `MemorySource` instance, it routes to
`import_memory_source()` in `import_source.py`. The import proceeds in
three steps:

1. The source's `records()` async generator is iterated.
2. Records are translated by `_RecordTranslator` (in `loader.py`) according
   to the chosen import mode.
3. The translated output (data items and/or graph batches) is stored.

## Import Modes

The `mode` parameter on every `MemorySource` controls how records are handled:

| Mode        | Raw content              | Source graph                  | LLM cost |
|-------------|--------------------------|-------------------------------|----------|
| `re-derive` | Run through add+cognify  | Rendered as text digests, then re-extracted by Cognee | Full |
| `preserve`  | Stored as-is (add only)  | Mapped directly into the graph, zero LLM calls | None |
| `hybrid`    | Run through add+cognify  | Mapped directly into the graph | Partial  |

- **re-derive** is the right choice when you only have raw content (memories,
  documents, chat logs) and want Cognee to build the graph from scratch.
- **preserve** is for restoring a COGX backup or migrating a provider that
  already has an extracted graph. No LLM calls are made.
- **hybrid** does both: the source graph is preserved and the raw content is
  also cognified, so Cognee can enrich the graph beyond what the source had.

### Streaming vs. Buffered

Replayable sources (where `records()` can be called more than once) in
`preserve` mode use a two-pass streaming import via `stream_graph_from_source()`
in `loader.py`. Pass 1 streams nodes; pass 2 streams edges. This keeps peak
memory bounded to one batch plus the entity registry.

Non-replayable sources and `re-derive`/`hybrid` modes buffer through the
translation layer instead.

## Deterministic IDs and Idempotent Re-import

COGX imports are designed to be safely re-runnable:

- **Data items** get a deterministic UUID5 derived from
  `cogx:{external_system}:{external_id}` (see `record_data_id()` in
  `loader.py`). Re-importing the same record produces the same ID, so
  duplicates are detected.
- **Entity nodes** use `generate_node_id(name)`, which is the same ID scheme
  that `cognify` uses. This means preserved entities merge into the existing
  graph vocabulary instead of creating a parallel set of nodes.

If an import is interrupted, re-running it will pick up where it left off
without creating duplicates.

## Adding a New Provider Source

To import from a new memory system, create a `MemorySource` subclass in
`cognee/modules/migration/sources/`. The only method you need to implement
is `records()`, which yields COGX records:

```python
from cognee.modules.migration.cogx import COGXMemory, COGXScope, parse_timestamp
from cognee.modules.migration.sources.base import MemorySource


class MyProviderSource(MemorySource):
    source_system = "my_provider"

    def __init__(self, data, mode: str = "re-derive"):
        super().__init__(mode=mode)
        self._data = data

    async def records(self):
        for item in self._load(self._data):
            yield COGXMemory(
                external_system=self.source_system,
                external_id=item["id"],
                content=item["text"],
                scope=COGXScope(user_id=item.get("user_id")),
                created_at=parse_timestamp(item.get("created_at")),
            )
```

Users can then import with:

```python
await cognee.remember(MyProviderSource("export.json"))
```

**Things to keep in mind:**

- Use `parse_timestamp()` for all date fields. It handles ISO 8601 strings,
  epoch seconds, milliseconds, microseconds, nanoseconds, and timezone-naive
  values.
- Set `replayable = True` (the default) if `records()` can be called multiple
  times with the same results. Set it to `False` for one-shot sources like
  live API cursors; this forces the buffered import path.
- Choose the right default mode for your source: `re-derive` if the source
  only has raw content, `preserve` if it has an already-extracted graph,
  `hybrid` if it has both.

**Built-in sources to use as reference:**

| Source              | File               | Default mode | What it reads                               |
|---------------------|--------------------|--------------|---------------------------------------------|
| `Mem0Source`        | `sources/mem0.py`  | `re-derive`  | Flat memory list                            |
| `ZepSource`         | `sources/zep.py`   | `hybrid`     | Episodes + entities + facts                 |
| `GraphitiSource`    | `sources/zep.py`   | `hybrid`     | Same as ZepSource (alias)                   |
| `LettaSource`       | `sources/letta.py` | `re-derive`  | Core memory blocks + messages + passages    |
| `COGXArchiveSource` | `sources/cogx_archive.py` | `preserve` | COGX archive directory                |

## File Reference

All files are relative to `cognee/modules/migration/`.

| File                     | What it does                                              |
|--------------------------|-----------------------------------------------------------|
| `cogx.py`                | Record models, `COGXArchiveWriter`, `read_archive()`, version validation |
| `archive.py`             | `.cogx.tar.gz` packing/unpacking with bomb protection     |
| `export.py`              | `export_dataset()`: graph to COGX/JSON/GraphML/Cypher     |
| `import_source.py`       | `import_memory_source()`: orchestrates the import         |
| `loader.py`              | Record translation, graph batching, streaming two-pass import |
| `snapshot.py`            | `GraphSnapshot`, `rehydrate_node()` for pydantic export   |
| `sources/base.py`        | `MemorySource` abstract base class                        |
| `sources/mem0.py`        | Mem0 provider source                                      |
| `sources/zep.py`         | Zep and Graphiti provider source                          |
| `sources/letta.py`       | Letta/MemGPT provider source                              |
| `sources/cogx_archive.py`| COGX archive restore source                               |
