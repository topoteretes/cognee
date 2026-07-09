# DLT ingestion — profiling findings (#3626, Phase 0)

**Profile first, then propose.** This is where time and memory actually go in
cognee's DLT ingestion path, measured with
`benchmark_dlt_ingestion_profile.py` (no LLM key; per-stage timers +
tracemalloc; first-ingest + re-ingest-with-deletions).

Environment: Windows 11, Python 3.13, SQLite destination, dlt 1.28.1, cognee
`main`. Numbers are wall-clock on one machine — treat proportions, not absolute
seconds, as the signal.

## Headline

Throughput is **flat at ~40–55 items/sec regardless of source size** (500→3000
rows all land in the 42–48 rps band on the coarse harness). Ingestion is fully
**O(rows) and serial** — the bottleneck is cognee's per-row work, **not dlt**.
dlt's own extract/normalize/load is ~5% of wall-clock.

Peak memory grows **linearly** with source size (14 MB @ 320 rows → 49 MB @ 3020
rows), because rows are fully materialized before processing.

## Where first-ingest time goes (n=300 → 910 DataItems, 17.1 s)

| Stage | calls | seconds | % | note |
|---|---:|---:|---:|---|
| `ingest_data` / `id_resolve` | 910 | 4.66 | 27% | **wasted** — `identify()` re-resolves the id, result discarded |
| `resolve` / `id_resolve` (Phase 1) | 910 | 4.32 | 25% | 1 DB session + 1 SELECT per row |
| `ingest_data` / `metadata_threads` | 2730 | 3.87 | 23% | `get_metadata` ~3×/row, each spawns a thread/event-loop |
| file I/O + Phase-2 build + relational writes | — | ~3.4 | 20% | per-row temp-file write, loader read-back, `session.merge` |
| `dlt_ingest_total` (extract+normalize+load) | 1 | 0.88 | 5% | the actual dlt work |
| `schema_extract` | 1 | 0.13 | <1% | |
| `row_readback` | 1 | 0.08 | <1% | |

**~52% of first-ingest time is ID resolution, and half of that is pure waste.**

## Confirmed hotspots (ranked by leverage)

### 1. Per-row ID resolution is done **twice**, second result discarded — ~27% waste
`resolve_dlt_sources` Phase 1 computes a stable `data_id` per row via
`get_unique_data_id` ([resolve_dlt_sources.py:103](../../../tasks/ingestion/resolve_dlt_sources.py#L103))
and stores it on the `DataItem`. Then `ingest_data`'s pre-loop calls
`ingestion.identify(...)` per row ([ingest_data.py:98](../../../tasks/ingestion/ingest_data.py#L98)),
which calls `get_unique_data_id` **again** ([identify.py:11](../../../modules/ingestion/identify.py#L11)) —
but the result is **immediately overwritten** by the DataItem's existing id
([ingest_data.py:100-101](../../../tasks/ingestion/ingest_data.py#L100-L101)). For DLT rows this
entire second round-trip is dead work. **Fix:** skip `identify()` when
`DataItem.data_id` is already set. Removes ~27% of first-ingest time outright.

### 2. Per-row DB sessions — O(rows) round-trips
`get_unique_data_id` opens its own `get_async_session()` and runs one `SELECT`
per call ([get_unique_data_id.py:60-68](../../../modules/data/methods/get_unique_data_id.py#L60-L68)).
The cProfile run showed **~14,000 sessions opened for ~1000 rows**. **Fix:** batch
Phase-1 resolution — compute candidate ids in memory, resolve existing ones with a
single `SELECT ... WHERE id IN (...)`, reuse one session. Collapses N round-trips
to a few.

### 3. `get_metadata` spawns a thread per call — ~23%
`BinaryData.get_metadata` calls `run_sync(...)`, which spins up a fresh event
loop in a new thread every call ([BinaryData.py:26-27](../../../modules/ingestion/data_types/BinaryData.py#L26-L27)).
It runs ~3×/row (identify + original-file classify + storage-file classify).
cProfile: **3,063 thread joins ≈ 10 s** at 1000 rows. For in-memory structured DLT
text this file-metadata machinery is largely redundant. **Fix:** a DLT-aware fast
path in `ingest_data` that builds the `Data` row directly from the DataItem
(known text, known id, known content_hash) instead of round-tripping through temp
files + classify + metadata.

### 4. Per-row file I/O for structured text
Each DLT row's enriched text is written to a temp file
(`save_data_item_to_storage` → `save_data_to_file`), read back through a loader
into cognee storage (`data_item_to_text_file`), then reopened twice more for
`classify().get_metadata()`. Tens of thousands of filesystem ops for what is
already in-memory text. Same fast-path fix as #3.

### 5. Orphan cleanup is O(dataset), not O(delta) — dominates re-ingest
On re-ingest (tail 20% dropped, n=300), deleting **180 orphans took 11.5 s** —
*more than ingesting all 910 rows*. `_delete_dlt_orphans` calls
`get_dataset_data(dataset.id)`, loading **every** row of the dataset into Python
([resolve_dlt_sources.py:344](../../../tasks/ingestion/resolve_dlt_sources.py#L344)), filters
orphans client-side, then deletes one-by-one. Cost scales with dataset size, not
with how many rows changed. **Fix:** set-based orphan *identification* (a single
query returning only orphan ids), plus batched/ledger-aware deletion.
⚠️ *Do not* collapse deletion to a raw `DELETE ... NOT IN (...)`: the per-row
graph/vector teardown (`delete_data_nodes_and_edges`) is reference-counted and
must be preserved (see #3626 thread) — only the *identification* is safely
set-based.

### 6. Linear memory — full materialization
`_read_rows_from_tables` does `result.mappings().all()` per table and builds one
`all_rows` list across all tables ([ingest_dlt_source.py:325](../../../tasks/ingestion/ingest_dlt_source.py#L325)),
and `resolve_dlt_sources` accumulates all `DataItem`s. Peak memory ∝ source size.
**Fix:** stream rows in bounded chunks (server-side cursor / `yield_per`) feeding a
fixed-size processing window.

### 7. Concurrency limitation — dlt working dir is global (discovered empirically)
`ingest_dlt_source` hardcodes `pipeline_name="ingest_dlt_source"`
([ingest_dlt_source.py:93](../../../tasks/ingestion/ingest_dlt_source.py#L93)). dlt stores that
pipeline's schema/state in a **global** dir (`~/.dlt/pipelines/ingest_dlt_source`).
Running two DLT ingests concurrently made them clobber each other →
`SchemaNotFoundError` / `no such table: ..._dlt_pipeline_state`. **Fix:** derive a
unique pipeline name (or `pipelines_dir`) per dataset/run so concurrent ingests
are isolated. (The harness works around this with a per-process `DLT_DATA_DIR`.)

### Also: multi-resource re-read
Passing N separate `DltResource`s in one `add()` runs `ingest_dlt_source` N times,
and each run reads back the **whole** schema — earlier tables are re-read O(N)
times ([resolve_dlt_sources.py:84-92](../../../tasks/ingestion/resolve_dlt_sources.py#L84-L92)). Prefer
one `@dlt.source`, or read each table only once across the batch.

## Prioritized optimization targets

1. **Skip the redundant `identify()` for DataItems with a known `data_id`** — biggest
   win-per-line (~27% of first-ingest), low risk. (chunk: batching)
2. **DLT-aware fast path in `ingest_data`** — bypass temp-file/classify/metadata for
   structured rows (~23% + file I/O). (chunk: batching)
3. **Batch Phase-1 ID resolution** into a single `IN` query. (chunk: batching)
4. **Set-based orphan identification** + ledger-safe batched deletion + indexes on
   `data(dataset_id)`. (chunk: destination/orphan_cleanup)
5. **Stream row read-back** in bounded chunks. (chunk: memory/streaming)
6. **Per-run dlt pipeline isolation** + single-source read to unlock parallelism.
   (chunk: dlt parallelism)

## Scaling table

From `dlt_ingestion_baseline.json` (`--sizes 200,500,900`, SQLite, 3 tables:
`departments` 10 + `users` N + `orders` 2N). Re-ingest drops the tail 20% so
orphan-cleanup runs.

| rows/table (N) | DataItems | **first-ingest** total | items/sec | peak MB | re-ingest total | orphan-cleanup | orphans | ms / orphan |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 610 | 13.1 s | 46.5 | 14.8 | 21.3 s | 8.2 s | 120 | 68 |
| 500 | 1510 | 31.4 s | 48.2 | 26.7 | 55.3 s | 23.9 s | 300 | 80 |
| 900 | 2710 | 57.0 s | 47.6 | 55.2 | 108.4 s | 48.5 s | 540 | 90 |

Reading it:

- **First-ingest throughput is flat** (~47 items/s at every size) → strictly O(N),
  no economy of scale. This is the top-line problem.
- **Peak memory grows with size** (15→27→55 MB) → full materialization. (Peak in
  multi-size mode carries some residual-heap noise; single-size runs give ~15/30/40
  MB — same linear trend.)
- **Orphan-cleanup cost per orphan *rises* with dataset size** (68→80→90 ms) →
  the `get_dataset_data` full-dataset scan makes cleanup ~O(orphans × dataset),
  i.e. **super-linear**. Cleanup is ~45% of re-ingest wall-clock at N=900 and
  getting worse.

## Reproduce

```bash
python -m cognee.tests.performance.benchmark_dlt_ingestion_profile --sizes 200,500,900
python -m cognee.tests.performance.benchmark_dlt_ingestion_profile 1000 --cprofile
```
