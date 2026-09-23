---
name: cognee-performance
description: Use when cognee is slow, expensive, or hitting rate limits — speeding up or throttling ingestion (remember/cognify batching, chunk size, LLM and embedding rate limits, per-stage models), cutting read latency, estimating cost before ingesting, running work in the background, or planning how to scale a deployment.
---

# Tune cognee's performance

**The LLM is the bottleneck.** Building the graph costs two LLM calls per
chunk (graph extraction and summarization), and almost every other tuning
question comes down to how many of those calls run, how fast, and on which
model. Database and embedding work is small next to that.

## Use it

### 1. Estimate before you ingest

```python
estimate = await cognee.remember("./docs", dry_run=True)
print(estimate)   # per-stage token counts and approximate cost; no LLM calls
```

The estimate covers graph extraction and summarization only, not
`improve()`, embeddings, or contradiction detection. Not available with
GLiNER, `temporal_cognify`, or a remote instance.

### 2. Ingestion knobs

All are `remember()` arguments (and `cognify()` ones):

| Knob | Default | What it controls |
|---|---|---|
| `data_per_batch` | 20 | How many documents are processed **at the same time** (a concurrency limit, not a batch). Lower it to reduce load; raise it for many small files. |
| `chunks_per_batch` | 2000 (env `CHUNKS_PER_BATCH`) | How many chunks each extraction/storage call receives. Lower it to spread load and fail smaller. The DLT pipeline defaults to 100, temporal to 10. |
| `chunk_size` | min(embedding max tokens, LLM max tokens / 2); about 8191 tokens with default OpenAI models | Max tokens per chunk. Fewer, bigger chunks mean fewer LLM calls; smaller chunks mean finer-grained graphs. |
| `run_in_background` | `False` | Return immediately (`status="running"`); `await result` later. |
| `self_improvement` | `True` | `improve()` after the graph is built. `False` (or `IMPROVE_AUTO_ENABLED=false`) skips that extra work. |

Worst-case concurrent LLM calls are about
`data_per_batch × min(chunks per document, chunks_per_batch) × 2`, and there
is no separate concurrency cap. The rate limiter below is the brake.

Several datasets in one `cognify(datasets=[...])` call run one after
another, not in parallel.

### 3. LLM throttling and models

| Env var | Default | Notes |
|---|---|---|
| `LLM_RATE_LIMIT_ENABLED` | `false` | Turn on a requests-per-interval limit |
| `LLM_RATE_LIMIT_REQUESTS` / `LLM_RATE_LIMIT_INTERVAL` | 60 / 60 s | 10 requests for local providers (Ollama, llama.cpp, LM Studio) |
| `AUTO_RATE_LIMIT` | `true` | On a 429/503/529 or timeout, switches the limiter on for 15 minutes (extended while errors continue) |
| `LLM_EXTRACTION_MODEL` / `_PROVIDER` / `_ENDPOINT` / `_API_KEY` / `_API_VERSION` | the main model | Use a cheaper or faster model just for graph extraction |
| `LLM_SUMMARIZATION_*`, `LLM_QUERY_*` | the main model | Same, for summaries and for answering queries |
| `LLM_MAX_COMPLETION_TOKENS` | 16384 | Also caps the default chunk size (half of it) |
| `LLM_ARGS` | — | Extra litellm arguments merged into every call, e.g. a request `timeout` |

Structured-output calls retry with exponential backoff for at least two
attempts and four minutes before failing; auth, not-found, and quota errors
are not retried.

The rate limiter is built from the main model's settings. A stage routed to
a local model does not get the local default of 10 requests; set
`LLM_RATE_LIMIT_REQUESTS` yourself.

### 4. Embeddings

| Env var | Default |
|---|---|
| `EMBEDDING_BATCH_SIZE` | 36 texts per request |
| `EMBEDDING_MAX_CONCURRENT_DATA_POINTS` | 150 (so 150 / 36 = 4 concurrent requests) |
| `EMBEDDING_RATE_LIMIT_ENABLED` / `_REQUESTS` / `_INTERVAL` | off / 60 / 60 s |

There is no automatic overload back-off for embeddings. Without any
credentials cognee embeds locally with fastembed (`BAAI/bge-small-en-v1.5`),
which runs on the CPU and blocks while it works.

### 5. Skip the LLM for extraction

`remember(data, extractor="gliner")` builds the graph and summaries with a
local GLiNER model: no LLM calls, embeddings still run (`pip install
"cognee[gliner]"`). See the `cognee-ingestion` skill for its limits.

### 6. Read latency

- **`AUTO_FEEDBACK=false`** is the biggest win for chat-style use: by default
  every answered turn with a session makes one extra LLM call to analyze
  feedback. Keep `CACHING=true` so session memory still works.
- **Pick a type without an LLM** when you need passages, not an answer:
  `CHUNKS`, `CHUNKS_LEXICAL`, `SUMMARIES`, `CODE`, `SKILLS`. Completion types
  make one LLM call; `GRAPH_COMPLETION_COT` and `_DECOMPOSITION` make several;
  `FEELING_LUCKY` adds one to pick the type.
- **Narrow the search**: pass `datasets=[...]` (one search runs per dataset)
  and a smaller `top_k` (per dataset, default 15).
- `SESSION_SEARCH_MODE=concurrent` (default) overlaps the feedback analysis
  with the answer; `sequential` runs them back to back.

### 7. Several datasets at once

With access control on (the default), each dataset has its own databases.
`DATASET_QUEUE_MAX_CONCURRENT` (default 6, from `DATABASE_MAX_LRU_CACHE_SIZE`)
caps how many datasets are processed at once in one process, and
`SUBPROCESS_IDLE_TTL_SECONDS` (600) keeps idle database workers warm. Keep
`DATASET_QUEUE_ENABLED=true` for parallel multi-dataset load.

### 8. Scaling a deployment

Open-source cognee scales **vertically, in one process**:

- The API server runs one worker (`gunicorn -w 1`), and the improve and
  session locks, caches and semaphores are in-process only. More workers or
  replicas against the same data are not coordinated (the Helm chart README
  says to validate before scaling past one replica).
- The default embedded databases (SQLite, LanceDB, Ladybug) live on local
  disk. For production, use Postgres + PGVector and a graph-native backend
  such as Neo4j.
- `distributed/deploy/` holds one-click deploy templates (Modal, Fly,
  Railway, Render, Daytona), not a distributed runner.

> **For production scale:** horizontal scaling (distributed ingestion across
> workers), cognee's enterprise GLiNER extraction, and the production
> Postgres graph adapter are part of cognee's proprietary offering. Contact
> social@cognee.ai.

## Pitfalls

- **`CHUNK_SIZE`, `CHUNK_OVERLAP`, `CHUNK_STRATEGY`** (in `.env.template`) and
  `cognee.config.set_chunk_size()` etc. **have no effect on ingestion.** Pass
  `chunk_size=` to `remember()` instead.
- **`LLM_RATE_LIMIT_TOKENS` and `EMBEDDING_RATE_LIMIT_TOKENS` do nothing**;
  only request-count limits are enforced.
- **`FALLBACK_MODEL`** is used only when the main model rejects content on
  policy grounds, not when it is slow, down, or rate-limited.
- **`data_per_batch=None` crashes.** Omit the argument to get the default.
- **A background `cognify()`** is not tracked by
  `cognee.wait_for_background_tasks()` (a background `remember()` or
  `improve()` is). Await its pipeline status yourself.
- **Big default chunks with a small local embedder.** The default chunk size
  follows the configured embedding limit (8191), even for fastembed's
  512-token model. Pass a smaller `chunk_size` for local embedding models.
- Benchmark with `CACHING=true`: with it off you measure cognee without its
  memory layer.

## How it works

`remember()` → `add()` → `cognify()`: every document of the dataset is
scheduled at once, `data_per_batch` of them run concurrently, and each runs
the task chain (classify, chunk, extract graph + summarize, store). Tasks
receive their input in batches of `chunks_per_batch`.
`extract_graph_and_summarize` runs extraction and summarization together,
each gathering one LLM call per chunk in the batch. Every LLM call passes
through one process-wide rate limiter.

- Cognify task list and defaults: `cognee/api/v1/cognify/cognify.py`
- Concurrency (`data_per_batch` semaphore): `cognee/modules/pipelines/operations/run_tasks.py`
- Task batching: `cognee/modules/pipelines/tasks/task.py`, `run_tasks_base.py`
- LLM calls per chunk: `cognee/tasks/graph/extract_graph_and_summarize.py`
- Chunk-size default: `cognee/infrastructure/llm/utils.py:get_max_chunk_tokens`
- LLM config, rate limits, stage routing: `cognee/infrastructure/llm/config.py`,
  `cognee/shared/rate_limiting.py`, `cognee/infrastructure/llm/overload_policy.py`,
  `cognee/infrastructure/llm/retry_config.py`
- Embedding config: `cognee/infrastructure/databases/vector/embeddings/config.py`
- Dataset queue: `cognee/infrastructure/databases/dataset_queue/queue.py`
- Cost estimate: `cognee/modules/cognify/estimator.py`

## Extending it

- Benchmarks: `cognee/tests/performance/` (`batch_add_cognify_test.py`,
  `locust_performance_analysis.py`, and `statistics_percentile/` for p50–p99
  runs with mocked or replayed LLM calls). Measure before and after a change.
- Retrieval quality and cost: `cognee/eval_framework/` (`python -m
  cognee.eval_framework`; Modal runs cost money).
- A new pipeline task that calls an LLM should keep `needs_llm=True` and go
  through `LLMGateway`, so it shares the rate limiter and retry policy.
