# DLT ingestion benchmark & profiling harness (#3626)

Phase-0 deliverable for *"Make DLT ingestion more scalable and faster"*: a
reproducible benchmark that measures **where time and memory actually go** in
cognee's DLT ingestion path, so optimization targets are chosen from data, not
guesses.

## What it profiles

The DLT path never needs an LLM — structured rows bypass entity extraction — so
the harness drives the real ingestion functions directly (no `cognee.add`, no
vector/graph/LLM init) and runs **with no API key**:

```
resolve_dlt_sources
  ├─ ingest_dlt_source
  │    ├─ dlt pipeline.run (extract+normalize+load)   -> stage: dlt_ingest_total
  │    ├─ _extract_dlt_schema                          -> stage: schema_extract
  │    └─ _read_rows_from_tables (materializes rows)   -> stage: row_readback
  ├─ Phase 1: get_unique_data_id per row              -> stage: resolve/id_resolve
  └─ Phase 2: build DataItems (schema text + FK)
ingest_data (add-pipeline storage task)
  ├─ identify -> get_unique_data_id AGAIN per row     -> stage: ingest_data/id_resolve
  │              (result discarded; DataItem.data_id already set)
  └─ BinaryData.get_metadata (~3x/row, thread each)   -> stage: metadata_threads
orphan_cleanup (re-ingest only)
  └─ get_dataset_data loads the WHOLE dataset          -> stage: orphan_cleanup
```

Each run does a **first ingest** and a **re-ingest with the tail 20% dropped**
(`write_disposition="replace"`), so orphan-cleanup and incremental re-sync are
measured, not skipped.

## Running

```bash
# single size: full per-stage breakdown for first-ingest + re-ingest
python -m cognee.tests.performance.benchmark_dlt_ingestion_profile 2000

# scan several sizes (sequential, in one process) and write results JSON
python -m cognee.tests.performance.benchmark_dlt_ingestion_profile --sizes 200,500,900

# add a cProfile cumulative-time attribution pass
python -m cognee.tests.performance.benchmark_dlt_ingestion_profile 1000 --cprofile
```

`--sizes` writes `results/dlt_ingestion_baseline.json` (committed baseline). Every
optimization PR should re-run the same sizes and report before/after numbers.

## Isolation notes

- **State root**: the harness points `DATA_ROOT_DIRECTORY`, `SYSTEM_ROOT_DIRECTORY`,
  and `DLT_DATA_DIR` at a per-process temp dir and uses a fresh `DB_NAME`, so runs
  are hermetic and don't pollute the repo's `.cognee_system`.
- **`--sizes` runs sequentially in one process** (each size gets a unique dataset
  name). It is deliberately *not* a subprocess fan-out: on Windows, spawning the
  benchmark as a child while dlt parallelizes normalize/load re-imports `__main__`
  and deadlocks.
- **dlt working dir is global — a real finding**: cognee hardcodes
  `pipeline_name="ingest_dlt_source"`, whose dlt state lives in a global dir
  (`~/.dlt/pipelines/ingest_dlt_source`). Two *concurrent* cognee DLT ingests share
  it and collide (`SchemaNotFoundError`). See the findings report, hotspot #7.

## Files

- `benchmark_dlt_ingestion_profile.py` — this harness (fine-grained stages + orphan cleanup + JSON).
- `benchmark_dlt_ingestion.py` — the earlier coarse harness (resolve/ingest totals only), kept for reference.
- `results/dlt_ingestion_baseline.json` — committed baseline numbers.
- `results/dlt_ingestion_findings.md` — where-time-goes analysis + prioritized optimization targets.
