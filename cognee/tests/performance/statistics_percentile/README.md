# Nightly performance corpora

The nightly percentile benchmark (`.github/workflows/nightly_tests.yml`, dispatched by
`.github/workflows/nightly_scheduler.yml` — dev daily, main weekly) runs each
corpus through `statistics_percentile_report.py` on four backends. A corpus is two
S3 objects under
`s3://github-runner-cognee-tests/nightly_ci_artifacts/performance_test_artifacts/`:

| Object | What it is |
|---|---|
| `<label>.json` | the corpus — a JSON array of `{"title", "content"}` |
| `mock_<label>.json` | the replay cassette, keyed by the exact chunk text the LLM saw |

The Rust arms read their own copies from `topoteretes/cognee-rs` at
`scripts/perf/fixtures/<label>/{memories.json,cassette.json}` — a separate repo,
so adding a corpus there is a separate PR.

## Current corpora

| Label | Shape | Size |
|---|---|---|
| `50_small_documents` | short synthetic documents | 21 KB |
| `war_and_peace` | one very long document | 3.2 MB |
| `war_and_peace_large` | the War and Peace corpus replayed against a 27×-inflated graph (~100k nodes) | 39 MB cassette |
| `datasheets` | 164 medium-sized real product datasheets | 1.4 MB |

`datasheets` exists because neither of the first two covers the common case: many
medium documents rather than one long one or a handful of short ones.

## Rebuilding a corpus

Both steps are reproducible; neither was, before `build_corpus.py` landed. Note
that `mock_ingestion.py` still refers to a `generate_large_mock.py` that was never
committed — `war_and_peace_large` cannot currently be rebuilt from source.

### 1. Corpus, from a directory of documents

```bash
python build_corpus.py --from-dir ~/path/to/datasheets --output datasheets.json
```

Handles `.pdf` (via pypdf), `.txt`, `.md`. It also **enforces title uniqueness**,
which is a correctness requirement rather than cosmetics: the replay in
`mock_ingestion.install_mocks` matches a chunk to its recorded response with
`title in chunk_text`, so a title contained in another title silently replays the
wrong knowledge graph and the benchmark measures the wrong thing.

### 2. Cassette, from one real LLM run

```bash
# Smoke-test on two documents before spending tokens on the whole corpus
python capture_mock.py --memories datasheets.json --num-memories 2 --output /tmp/probe.json

python capture_mock.py --memories datasheets.json --output mock_datasheets.json
```

`capture_mock.py` **prunes the configured cognee instance** before ingesting. Point
it at a throwaway one first, or it will delete your local data:

```bash
export DATA_ROOT_DIRECTORY=/tmp/capture/.data_storage
export SYSTEM_ROOT_DIRECTORY=/tmp/capture/.cognee_system
```

Check the summary it prints: `Substring-match collisions` must be `0`, and
`Incomplete chunks skipped` should be `0`. A non-zero collision count means two
chunks would replay each other's graph.

### 3. Upload

```bash
BUCKET=github-runner-cognee-tests
PREFIX=nightly_ci_artifacts/performance_test_artifacts
aws s3 cp datasheets.json      "s3://$BUCKET/$PREFIX/datasheets.json"
aws s3 cp mock_datasheets.json "s3://$BUCKET/$PREFIX/mock_datasheets.json"
```

Needs `s3:PutObject` on that bucket, which lives in AWS account `463722570299`.

## Adding a corpus to the nightly

Five caller jobs in `nightly_tests.yml` — `perf-<label>-llm`, `perf-<label>-mock`,
`perf-<label>-cloud`, `perf-rust-<label>-llm`, `perf-rust-<label>` — plus, in the
`notify` job, seven `REPORT_KEYS` entries, seven `ARMS` lines, the `needs` list and
the status gate. Copy the `war_and_peace` block; it is the closest template.

Nothing is needed for MotherDuck: `motherduck_nightly_etl.py` discovers labels by
globbing the S3 report keys, so a new label appears in `ci_analytics.nightly` by
itself.
