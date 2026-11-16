# cognee-connector-arxiv

An arXiv data-source connector for [cognee](https://github.com/topoteretes/cognee):
sync the papers of the categories and authors you select into memory, incrementally by
submission date, and forget a paper when it disappears from the re-queried window.

It exposes a `dlt` source you hand to `cognee.remember(...)`. Each paper becomes a
**normal document** in cognee (it flows through chunking and LLM entity extraction),
via cognee's document-mode marker: the title, authors, abstract and metadata of one
paper render into one document. One source is one selection of arXiv categories
(`cs.AI`) and/or authors (`lecun_y`); the arXiv API is public, so there is nothing to
connect and no credential anywhere in the package.

## Install

```bash
pip install cognee-connector-arxiv             # the connector and its DLT source
pip install "cognee-connector-arxiv[cognee]"   # plus cognee itself, which runs the sync
```

## Requirements

cognee 1.6.5 or later whose ingestion reads per-row node sets (`DOCUMENT_SYNC_VERSION`
3 or later; the source refuses an older one before it makes a request). The REST client
uses `requests`, which the package declares and `dlt` (a dependency of
`cognee-connectors-core[dlt]`) requires anyway.

## Configuration

This package reads no environment variable: the arXiv API needs no key, and the token
is never read from the environment anywhere. Everything is configured on the
`arxiv_source(...)` call, see the argument table under Usage.

## Setup

1. Pick the selection: arXiv category codes (`cs.AI`, `stat.ML`, ...), author names in
   arXiv's normalized form (`lecun_y`, `hinton_g_e`, ...), or both (combined with AND).
   At least one is required; an empty selection can never mean "everything".
2. Optionally bound the first run with `submitted_after` (an ISO date like
   `"2024-01-01"`); without it, the first run ingests the whole selection.
3. Set your `LLM_API_KEY` like any other cognee run.

## Usage

```python
import cognee
from cognee_connector_arxiv import arxiv_source

await cognee.remember(
    arxiv_source(
        categories=["cs.AI"],
        resource_name="arxiv_cs_ai",
        submitted_after="2024-01-01",  # first run only; later runs use the stored cursor
    ),
    dataset_name="arxiv",
    primary_key="id",
    write_disposition="merge",  # required: incremental upserts and deletes
    max_rows_per_table=0,  # no row cap; forget-on-delete must see the whole window
    self_improvement=False,
)

answer = await cognee.recall("What papers came out on diffusion models?", datasets=["arxiv"])
```

Run the same `remember` call again whenever you want to sync. Each run asks arXiv only
for the papers submitted since the previous completed run (`submittedDate` range), so
re-running is cheap; the client waits 3 seconds between requests, which is the rate
arXiv's fair-use terms ask for, built in.

| Parameter | Meaning |
|---|---|
| `categories` | arXiv category codes, like `["cs.AI", "cs.LG"]` |
| `authors` | normalized author names, like `["lecun_y"]` |
| `resource_name` | staging table and sync-state name; two selections in one dataset need different names |
| `submitted_after` | first-run bound, ISO date/datetime; later runs ignore it and use the cursor |
| `max_results` | page size of the listing requests (default 100) |
| `request_delay` | seconds between API requests (default 3.0, arXiv's fair-use rate) |
| `service` | a pre-built client for tests or a host; normally omitted |
| `check_active` | host authorization checkpoint; normally omitted |

A runnable script lives under `examples/`:

```bash
pip install "cognee-connector-arxiv[cognee]"
export LLM_API_KEY=...
python examples/arxiv_papers.py
```

## Running on a cognee server

There is nothing to register: `cognee_connector_arxiv.cognee_host.register()` is a
no-op and `routers()` returns `[]`, on purpose. cognee's integration registry serves
OAuth providers on `/api/v1/integrations/{provider}/...`, and arXiv has no account, no
token and no consent screen — the SDK path above is the whole surface.

## What is synced

One document per paper: the arXiv id (without version), title, authors, abstract,
submission and update dates, categories, and — when arXiv reports them — the comment,
journal reference and DOI. The full text (PDF/source) is not fetched. Every document
carries the node sets `arxiv:category:<code>` of its categories, so a `recall` can be
scoped to one category.

Deletion is settled only inside the window a run re-queried: a paper that disappears
from the listing of the current `submittedDate` window is forgotten on that run, while
a paper deleted behind the stored cursor is not re-checked (arXiv almost never removes
papers, and its API has no deletion feed). Narrowing or renaming the selection under
the same `resource_name` forgets nothing by itself; the members of the old selection
age out of the window untouched.

Limits: the listing pages at `max_results` per request through the whole window, at
one request every `request_delay` seconds — a first run over a large category is
network-bound by design. A failed request aborts the run with nothing forgotten, and
the next run starts the window over.

## Development

```bash
export COGNEE_ENV_FILE=<an empty file> TELEMETRY_DISABLED=1 LITELLM_MODE=PRODUCTION LLM_API_KEY=sk-test-dummy
PYTHONPATH=core:connectors/arxiv python -m pytest connectors/arxiv/tests -q
python scripts/check_cognee_free.py cognee_connector_arxiv
```
