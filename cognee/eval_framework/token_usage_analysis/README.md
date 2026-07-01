# token_usage_analysis

Estimate the token cost of **cognee persistent memory** versus **full-context
prompting**, from a few representative chunks of any text.

This is a follow-up to the earlier token-cost probe, which measured that cost on
a single synthetic corpus. Here we characterize how it varies with the input:
running the same measurement across a spectrum of real text — fiction, news,
encyclopedic — and across chunk sizes, so the break-even can be read for text
that resembles a given workload rather than one fixed corpus.

The measurement centers on ingestion cost, which is dominated by graph extraction.
Its output-tokens-per-input-token ratio scales with how densely entities and
relations are packed into the text, and that ratio in turn drives the break-even.
Running it over a density spectrum shows the two moving together — a small,
concrete extension of the note's cost model to a wider range of inputs.

## The cost model

Two strategies answer the same repeated queries over a fixed corpus. Each one's
cumulative token cost is a straight line in the number of queries:

```
full-context:  queries × (corpus_tokens + query_overhead)
cognee:        ingestion_tokens + queries × retrieved_context
```

- `ingestion_tokens` is the one-time cost of `cognee.remember()` — summarizing and
  graph-extracting every chunk. We measure it on a few representative chunks and
  scale by the corpus size: `ingestion_tokens = multiplier × corpus_tokens`, where
  `multiplier` is measured ingestion tokens per content token.
- `retrieved_context` is the (roughly constant) context `cognee.recall()` packs
  per query.

Reading off this model, each **reduction milestone** answers: after how many
repeated queries does the cumulative full-context cost become `factor`× the
cumulative cognee cost, including cognee's one-time ingestion cost?

```
queries(factor) =
    factor × ingestion_tokens
    / ((corpus_tokens + query_overhead) − factor × retrieved_context)
```

Parity, the cross-over where both cumulative costs are equal, is `factor = 1`.
For example, the `7` milestone is the query count where full-context has spent
7× as many tokens as cognee overall, so cognee is 7× cheaper at that point. No
milestone is privileged — break-even is just one of them.

Token usage is read from the **real LLM responses** (prompt + completion), so the
instruction wrapper and the Pydantic graph schema are included with no estimated
constants.

## Results: the density spectrum

Measured with `openai/gpt-5-mini`, 3 sampled chunks per corpus,
`retrieved_context = 1118`, `query_overhead = 32`. Each corpus is ~10–12k tokens.
We ran the whole spectrum at **two chunk sizes** to show that chunk size, not just
density, moves the numbers: cognee's `8191` default and `4095` (half). Values are
`graph output ÷ input` / `ingestion multiplier` / `parity (queries)`.

| Tier | Source | Corpus tokens | at chunk 8191 | at chunk 4095 |
| --- | --- | ---: | :--- | :--- |
| Fiction | *War and Peace* excerpt | 10,504 | 2.8× / 5.7× / ~6 | 4.5× / 8.0× / ~9 |
| News | Wikinews (23 articles) | 10,161 | 4.2× / 7.3× / ~8 | 5.9× / 9.8× / ~11 |
| Encyclopedic | Wikipedia — *Apollo 11* | 10,037 | 3.4× / 6.5× / ~7 | 5.9× / 10.3× / ~12 |
| Dense synthetic | benchmark `people_raw` | 12,009 | 9.3× / 12.0× / ~13 | 10.1× / 14.0× / ~15 |

Three things stand out:

- **The ingestion multiplier tracks the graph-output ratio**, and the break-even
  in queries tracks the multiplier — exactly as the cost model predicts. The dense
  synthetic corpus produces ~9–10× graph output per input token; ordinary prose
  produces ~3–6×.
- **Realistic text is clustered and lower.** Fiction, news, and encyclopedic prose
  break even in the single digits to low teens; the dense synthetic corpus is the
  outlier. The technical note's ~23–26 queries reflects that dense outlier (and
  larger ingestion chunks), not typical input.
- **Chunk size matters too.** Halving the chunk size raises every ratio and
  multiplier, because smaller chunks carry proportionally more fixed prompt/schema
  overhead per content token. Same corpus, same model — different break-even.

Caveat: these are single runs, and graph extraction is not fully deterministic —
its output length varies between runs (most on the dense corpus, where a repeat of
the 8191 run moved the multiplier by several points). Treat the numbers as
representative, not exact; the ordering is the robust result.

Per-tier cumulative-cost plots and full per-chunk JSON are in
`results/chunk_8191/` and `results/chunk_4095/`.

## Usage

Install the project with the eval dependencies, then run the commands from this
directory:

```bash
uv sync --dev --all-extras
cd cognee/eval_framework/token_usage_analysis
```

The script loads the repo-root `.env`, so make sure it contains a working
`LLM_PROVIDER`, `LLM_MODEL`, and API key. If `--llm-models` is omitted, the
configured `LLM_MODEL` is used.

```bash
# one representative file (the input is treated as the corpus)
uv run python analyze.py --file data/wikipedia_article.txt --plot

# a folder of .txt files, pooled then sampled
uv run python analyze.py --dir some_corpus/ --out report.json

# a single representative chunk (corpus size must be given explicitly)
uv run python analyze.py --text "$(cat one_chunk.txt)" --corpus-tokens 200000
```

The script chunks the input with cognee's `TextChunker`, samples a few chunks
(`--samples`, default 3), measures each through `cognee.remember()`'s summary and
graph-extraction calls, and writes a JSON report. `--plot` additionally writes the
cumulative-cost figure (`matplotlib` is included in the `evals` extra).

Key options:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--file` / `--dir` / `--text` | — | input form (exactly one) |
| `--samples` | 3 | chunks to measure |
| `--llm-models` | the `.env` model | comma list; runs each, switching cognee's config |
| `--corpus-tokens` | token count of the input | corpus size for the comparison (required with `--text`) |
| `--retrieved-context` | 1118 | recall context per query |
| `--query-overhead` | 32 | instruction + question tokens per query |
| `--reduction-factors` | `1,2,7,10` | milestones to report (1 = parity) |
| `--plot` / `--plot-dir` | off / `.` | also write the cross-over figure |

## Reproducing the spectrum

The spectrum is four single-input runs per chunk size, collated by hand. Each
chunk size gets its own results folder:

```bash
for size in 8191 4095; do
  for tier in war_and_peace wikinews wikipedia dense; do
    uv run python analyze.py --file "data/${tier}"*.txt --max-chunk-size "$size" \
      --out "results/chunk_${size}/${tier}.json" --plot --plot-dir "results/chunk_${size}"
  done
done
```

`--max-chunk-size` defaults to `4095`; pass `8191` for cognee's default.

`--llm-models` runs only models you have keys for. To compare providers, add
`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` to `.env`; the script switches
`LLM_PROVIDER`, `LLM_MODEL`, and `LLM_API_KEY` before each measurement. The runs
above used the configured `openai/gpt-5-mini`.

## Modules

`analyze.py` is pure orchestration; each step is one call into a focused module:

- `cli.py` — argument definitions and default resolution.
- `corpus.py` — input → sampled chunks (cognee `TextChunker`). No LLM, no math.
- `measure.py` — the only LLM/IO surface: run a chunk through cognee and record
  real token usage per model.
- `cost_model.py` — pure arithmetic: the two strategy cost objects, the average
  chunk, ingestion scaling, and reduction milestones.
- `report.py` — per-model orchestration and JSON assembly.
- `plot.py` — optional cumulative-cost figure (matplotlib, imported lazily).

## Assumptions and caveats

- **Constant retrieved context.** Recall context is treated as fixed per query
  (it grows only modestly with corpus size under a fixed `top_k`).
- **Embeddings excluded.** Only language-model tokens are counted; embeddings are
  computed locally.
- **Representative-chunk extrapolation.** Ingestion is scaled from the average of
  the sampled chunks, assuming the rest of the corpus has similar density.

## Data and attribution

The `data/` excerpts are the raw text that actually gets ingested:

- `war_and_peace_excerpt.txt` — *War and Peace*, Project Gutenberg eBook #2600
  (public domain).
- `wikipedia_article.txt` — *Apollo 11*, English Wikipedia
  (CC BY-SA 4.0). https://en.wikipedia.org/wiki/Apollo_11
- `wikinews_article.txt` — 23 recent English Wikinews articles, concatenated to a
  comparable corpus size (CC BY 2.5). Titles and URLs in `data/wikinews_sources.json`.
- `dense_synthetic.txt` — an excerpt of the benchmark `people_raw` corpus shipped
  in this repo.
