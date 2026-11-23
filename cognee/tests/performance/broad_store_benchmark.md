# BROAD record-store benchmark

This benchmark measures full-coverage BROAD retrieval for a synthetic 1 GB dataset.
It calls the real corpus reader, record parser, schema description, SQL execution and
cleanup. It does not measure ingestion, embeddings, graph/vector network calls, or
LLM decision/answer latency.

## Workload

- Exactly 1,000,000,000 bytes of generated ASCII CSV across 100 documents.
- 1,000,000 records with deterministic IDs, status, amount and a repeated text payload.
- A mock graph generates each document when its batch is requested; no model creates data.
- A mock vector adapter rejects every call, proving no similarity sampling occurs.
- Dataset-size metadata is supplied by the mock and equals the generated source bytes.
- A fixed SQL decision replaces the LLM. A guard rejects any LLM gateway call.
- The query counts `open` records and sums their amounts. The required result is
  `(500000, 2000000)`, checked against counters maintained while generating records.
- Each variant starts with a new store in a separate process. The baseline explicitly
  uses `store_mb=1000`, matching the candidate's default, so both cover the same dataset.

The baseline is the file-backed `RecordStore` from dev commit
`e919461945cc8603e1084c7380c075b6449bae46`; the candidate uses `:memory:` and
`PRAGMA temp_store=MEMORY`. All other retrieval code and dependencies are the same.

## Reproduce

Run from the repository root on macOS or Linux:

```sh
git show e919461945cc8603e1084c7380c075b6449bae46:cognee/modules/retrieval/broad_store.py > /tmp/broad-disk.py
uv run python cognee/tests/performance/benchmark_broad_store.py --store-source /tmp/broad-disk.py --json-output /tmp/broad-disk.json
uv run python cognee/tests/performance/benchmark_broad_store.py --json-output /tmp/broad-ram.json
```

The defaults are `--dataset-mb 1000 --documents 100`. Each run prints JSON with timings,
peak process RSS, SQLite logical size, the count/sum result and coverage checks.

`retrieval_total` includes source generation, parsing/insertion, schema description,
fixed decision, SQL and cleanup. It excludes Python import/startup time. Peak RSS includes
Python imports and intermediate allocations. A 10 ms heartbeat measures the largest gap
between event-loop turns; synchronous work can delay it for much longer than 10 ms.

## Local results

Measured on 2026-11-23 on an Apple M3 Max, 36 GiB RAM, macOS 27.2.1 arm64,
Python 3.13.16 and SQLite 3.53.1. Both variants returned `(500000, 2000000)`, processed
all 100 documents / 1,000,000 records, and made zero model or vector-search calls.

| Metric | File-backed baseline | In-memory candidate |
| --- | ---: | ---: |
| Total retrieval | 358.8912 s | 605.3115 s |
| Source generation (included in total) | 0.6158 s | 0.6280 s |
| Parse and insert | 11.2548 s | 10.7026 s |
| Schema description / query hints | 346.6383 s | 593.6530 s |
| Final SQL | 0.2638 s | 0.1740 s |
| Close / cleanup | 0.0350 s | 0.0732 s |
| Process CPU time | 357.6218 s | 604.1030 s |
| Peak process RSS | 855,277,568 B (0.797 GiB) | 3,149,742,080 B (2.933 GiB) |
| SQLite logical database size | 1,368,645,632 B | 1,368,645,632 B |
| Largest event-loop scheduling gap | 349.8611 s | 596.8018 s |

The candidate's `PRAGMA database_list` reported an empty filename for `main`, confirming
the store was in memory. The baseline reported a temporary `records.db` path.

The RAM candidate was 68.7% slower overall in these observations and used 3.68 times the
peak process RSS. Parsing/insertion and the final SQL were faster, but schema description
dominated both runs. This is not evidence of an overall speed improvement from this PR.
A one-second native stack sample during the RAM run showed SQLite invoking Python Unicode
normalization/join code, consistent with `_probe` repeatedly applying `norm(column)` to
text values for each question word. The benchmark times the whole `describe` stage; it
does not separately attribute every helper's cost or establish why that stage differed
between the two runs.

The large event-loop gaps are a separate existing problem: parsing, schema description
and SQL execution remain synchronous on the caller's event loop. Changing storage to RAM
does not offload that work or make cancellation interrupt synchronous processing. The
store is closed when retrieval completes or receives an exception/cancellation.

This synthetic CSV workload has repeated payloads and should not be treated as a
benchmark of arbitrary documents or semantic reading. Runs are serial with no deliberate
filesystem cache flushing. Results from one run per variant are observations, not a
statistically established speedup.
