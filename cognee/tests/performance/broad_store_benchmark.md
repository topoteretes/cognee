# BROAD record-store benchmark

This benchmark measures full-coverage BROAD retrieval for synthetic data. The current
default is **512 MB**, with synchronous BROAD work executed on a private worker thread.
Earlier 1 GB measurements below describe the implementation before that worker change.
It calls the real corpus reader, record parser, schema description, SQL execution and
cleanup. It does not measure ingestion, embeddings, graph/vector network calls, or
LLM decision/answer latency.

## Current 512 MB worker validation

The current default was exercised with `--http-probe`, which starts a real localhost
HTTP health server on the same event loop as retrieval and sends requests throughout
the search. Mock graph data generation runs off the loop too, so the fixture generator
does not introduce synchronous CPU stalls into the simulated async database adapter.

```sh
PYTHONHASHSEED=0 uv run python cognee/tests/performance/benchmark_broad_store.py --http-probe --json-output /tmp/broad-512mb-worker.json
```

Measured on the same M3 Max / 36 GiB machine, Python 3.13.16 / SQLite 3.53.1:

| Metric | Current default |
| --- | ---: |
| Actual source bytes | 512,000,000 |
| Documents / records | 100 / 512,000 |
| Result `(open_count, amount_sum)` | `(256000, 1024000)`, correct |
| Coverage | Complete; zero vector searches |
| LLM / local-model calls | 0 |
| Total retrieval and HTTP-probe lifecycle | 203.2423 s |
| Parse and insert | 5.6000 s |
| Schema description / hints | 196.5958 s |
| Final SQL | 0.1892 s |
| SQLite logical database size | 700,678,144 B |
| Peak process RSS | 1,719,042,048 B (1.60 GiB) |
| Successful HTTP responses during retrieval | 2,158 |
| HTTP errors | 0 |
| HTTP response latency, 95th percentile | 0.16794 s |
| HTTP response latency, maximum | 1.36471 s |
| Largest event-loop scheduling gap | 0.5249 s |

The heartbeat/total include local HTTP-probe startup and teardown. The latency samples
are requests completed while retrieval was active. This is a responsiveness check with
one large retrieval, not a latency guarantee or a multi-request capacity benchmark.
Threads still share the Python GIL; offloading does not eliminate CPU contention.

The 512 MB setting is a source-size coverage threshold, not a 512 MB RAM ceiling.
SQLite rows, pages and per-row metadata expand the source; loaded text batches, parsed
Python objects and SQLite temporary query structures can coexist. The process-RSS
measurement includes imports, allocators and the HTTP probe. It is not a measurement
of SQLite alone or a precise allocation breakdown. Each concurrent retrieval has its
own store and worker, so aggregate memory use still depends on concurrency.

Cancellation tests verify active SQLite queries are interrupted and the store closes
on its creating thread. Other tests cover cancellation during creation, repeated
cancellation, request context propagation, independent workers and HTTP handling during
parsing, schema description and SQL. Running Python stages finish before queued cleanup.

## Historical 1 GB measurements

The following observations were collected before the worker/default changes, using
the harness and production code at `9fafab37e54b15ffe40f81554a5a7c6f15e6bbb9` or its
earlier measurement commits. They do not describe the current default or event-loop
behavior. To exercise 1 GB with the current harness, explicitly pass
`--dataset-mb 1000 --store-mb 1000`.

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

## Reproduce the historical workload with the current harness

Run from the repository root on macOS or Linux:

```sh
git show e919461945cc8603e1084c7380c075b6449bae46:cognee/modules/retrieval/broad_store.py > /tmp/broad-disk.py
uv run python cognee/tests/performance/benchmark_broad_store.py --dataset-mb 1000 --store-mb 1000 --store-source /tmp/broad-disk.py --json-output /tmp/broad-disk.json
uv run python cognee/tests/performance/benchmark_broad_store.py --dataset-mb 1000 --store-mb 1000 --json-output /tmp/broad-ram.json
```

The current defaults are `--dataset-mb 512 --documents 100`. Each run prints JSON with timings,
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

The RAM candidate was 68.7% slower overall in this initial pair and used 3.68 times the
peak process RSS. Parsing/insertion and the final SQL were faster, but schema description
dominated both runs. The slowdown did not reproduce in the follow-up below; this pair
does not establish that the storage change caused a regression or a speed improvement.
A one-second native stack sample during the RAM run showed SQLite invoking Python Unicode
normalization/join code, consistent with `_probe` repeatedly applying `norm(column)` to
text values for each question word. The benchmark times the whole `describe` stage; it
does not separately attribute every helper's cost or establish why that stage differed
between the two runs.

In the historical implementation, parsing, schema description and SQL execution ran
synchronously on the caller's event loop. RAM storage alone did not offload that work.
The current worker validation above measures the subsequent offloading change.

This synthetic CSV workload has repeated payloads and should not be treated as a
benchmark of arbitrary documents or semantic reading. Runs are serial with no deliberate
filesystem cache flushing. Results from one run per variant are observations, not a
statistically established speedup.

## Follow-up diagnosis

An unmodified 1 GB RAM repeat, with `PYTHONHASHSEED=0`, returned the same correct result
in **339.7224 s**, compared with 605.3115 s initially. No normalizer replacement or
diagnostic callback was used in this repeat.

| Metric | RAM repeat |
| --- | ---: |
| Parse and insert | 9.7491 s |
| Schema description / query hints | 329.1134 s |
| Final SQL | 0.1781 s |
| Total retrieval | 339.7224 s |
| Process CPU time | 339.2787 s |
| Peak process RSS | 3,150,970,880 B (2.935 GiB) |
| Largest event-loop scheduling gap | 331.9523 s |

Reproduce the repeat with:

```sh
PYTHONHASHSEED=0 uv run python cognee/tests/performance/benchmark_broad_store.py --dataset-mb 1000 --store-mb 1000 --documents 100 --json-output /tmp/broad-ram-repeat.json
```

The initial 68.7% gap is therefore not a reproducible storage regression from these
measurements. Its specific cause remains unestablished. The repeat is also insufficient
to claim a reliable RAM speedup over the original disk observation.

Eight smaller, separate-process runs used 10 MB / 10 documents, `PYTHONHASHSEED=0`, and
the order RAM, disk, disk with memory temporary storage, RAM with default temporary
storage, disk, RAM, RAM with default temporary storage, disk with memory temporary
storage. The two ordinary RAM runs took 3.3934–3.4330 s and ordinary disk runs took
3.4695–3.5024 s. Changing `temp_store` did not reproduce the large gap at this size;
this does not establish its effect on arbitrary larger workloads.

### Confirmed CPU bottleneck

Temporary instrumentation of the 1 GB fixture counted **33,000,000** calls to SQLite's
Python `norm()` function, processing **10,944,972,500 characters**. `_probe` runs a
separate `SELECT DISTINCT ... WHERE norm(column) LIKE ?` for each question word and
text column. For this question, 11 words scan each of three text columns across one
million rows. Normalization repeats even when a column contains identical payloads.

An ablation replaced only the SQLite `norm` callback with an ASCII-specific equivalent
for this fixture: `str(value).casefold()`, preserving `None` and asserting ASCII input.
The fixture's fields contain no accents or whitespace requiring normalization. With
call/character counting still enabled, the full-coverage runs returned the same answer:

| Metric | Disk, ASCII ablation | RAM, ASCII ablation |
| --- | ---: | ---: |
| Total retrieval | 38.8622 s | 31.8480 s |
| Schema description / query hints | 27.2489 s | 20.9471 s |
| `_probe` alone | 22.4050 s | 19.0521 s |

This is diagnostic instrumentation, not a production optimization or a general-purpose
replacement for Unicode normalization. It isolates repeated Python character processing
as the dominant cost on this fixture, without explaining the initial timing gap.

Reducing repeated normalization while preserving Unicode matching and all existing
query hints remains an optimization to evaluate. The current worker change addresses
the long event-loop stalls; it does not optimize that normalization algorithm.
