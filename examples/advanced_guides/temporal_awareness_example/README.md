# Temporal awareness examples

## `temporal_hybrid_demo.py`

Ordinary graph extraction with a timestamp-promotion task inserted into the default
cognify pipeline. Builds on [`temporal_recall.py`](../../guides/temporal_recall.py),
which uses `temporal_cognify=True` and `SearchType.TEMPORAL`. This example adds the
custom extraction task and, later, direct temporal hybrid retrieval.

### Compared to `temporal_awareness_example.py`

The incumbent in this folder uses `temporal_cognify=True`, which swaps the whole
cognify task list for an event-centric pipeline, and answers through
`SearchType.TEMPORAL` — a separate retriever that never sees chunks or entity
neighbourhoods. This demo keeps the ordinary extraction pipeline and filters hybrid
candidates by time instead: chunks and entity neighbourhoods stay in the answer, at
the price of no event timeline and no relative-date handling at query time. One more
price: the filter only removes candidates, so a time-relevant chunk the vector search
misses never surfaces. The oversized candidate budget (40 fetched for 5 kept) softens
this; a real replacement should pull chunks from the time index directly, the way the
incumbent's `collect_time_ids` does.

### Prerequisites

Set `LLM_API_KEY` (and an embedding provider if you are not using the OpenAI default)
in `.env` or the environment.

Install `dateparser`, or run the demo with `uv run --with dateparser`. Partial dates
("of 27 April") are resolved against the last stated date earlier in the document and
appended to the extraction prompt as normalization hints. Only spans anchored to a
month name, a time of day, or a relative word ("four weeks later") are hinted —
scores, ordinals, and bare durations are dropped. dateparser also normalizes
timestamp names the LLM left unnormalized before promotion. The extraction hook and
the promotion fallback raise with an install hint when dateparser is missing.

### Run

```bash
uv run python examples/advanced_guides/temporal_awareness_example/temporal_hybrid_demo.py
```

Pass other ingest files and a query list (one question per line, `#` comments skipped):

```bash
uv run python examples/advanced_guides/temporal_awareness_example/temporal_hybrid_demo.py \
    --data examples/advanced_guides/temporal_awareness_example/data/captain-midnight-jamming.md \
    --queries examples/advanced_guides/temporal_awareness_example/data/captain_midnight_queries.txt
```

Uses the graph, vector, and relational backends from `.env`. It starts with
`prune_data()` and `prune_system(metadata=True)`, which clears that configured storage.

### Expected output

Promoted `Timestamp` nodes with `timestamp_str`, every `*_at` edge, and any timestamp
candidates that were left as ordinary entities (unparseable names or outgoing edges).

### Limitations

- Timestamp names must be `YYYY`, `YYYY-MM`, `YYYY-MM-DD`, or `YYYY-MM-DD HH:MM:SS`;
  other absolute dates ("23 March 1947") are normalized by a dateparser fallback at
  promotion time, at the precision the name states. Year-less or relative names still
  skip — and a skipped candidate leaves its chunk invisible to the time filter, since
  eligibility needs a matching timestamp in the chunk.
  A closed period needs one owner with one `begins_at` and one `ends_at` in the same
  chunk. That is why the demo chunks on blank lines (`RegexChunker`): one section per
  chunk keeps a period's owner and bounds together, and keeps the per-chunk time
  filter sharp — the default chunker can pack several sections into one chunk.
- Precision lives in the shape of `timestamp_str` (`1950` vs `1950-03-15`). Core's
  `generate_timestamp_datapoint` always writes the full form, so graphs produced by
  the two cannot be mixed; productionizing needs an explicit precision field.
- A bare-year date advances the rolling hint base with an arbitrary month and day, so
  a month-less expression like "that spring" can inherit a wrong month.
- `promote_timestamps` places `Timestamp` objects into `DocumentChunk.contains`,
  which does not declare that type; the write works because pydantic skips validation
  on in-place list mutation. Widening the union is the core prerequisite.
- Core's hybrid renderer labels nodes by `name`, which `Timestamp` lacks; the
  filtered context relabels timestamp edge bullets from `timestamp_str` itself.

### Retrieval

The demo then asks the questions in `--queries` (default
`data/biography_queries.txt`) with candidate/final budgets of 40/5. The biography
list covers an interior year, a multi-year window, an undated question, and a date
absent from the corpus. `data/captain_midnight_queries.txt` is looser for the
jamming article: the year 1986, after the HBO interruption, an undated identity
question, and the same absent date.

Retrieval is `TemporalHybridRetriever` (`temporal_hybrid_retriever.py`), a
`HybridRetriever` subclass that fetches candidates and extracts the query interval
concurrently, then filters the candidates by temporal overlap before the final limit.
The matching lives in `temporal_matching.py`; its index over the graph snapshot is
built once per run, not per query.

The demo prints the extracted interval, matching timestamps and periods, eligible and
retained chunk IDs, the fallback reason, and the filtered context and answer; pass
`--compare` to also format and answer the unfiltered baseline. No usable constraint,
or no surviving passage, falls back to ordinary hybrid retrieval with a visible
reason. A retained passage does not mean every fact in it holds throughout the
requested time.
