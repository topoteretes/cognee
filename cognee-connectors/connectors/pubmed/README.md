# cognee-connector-pubmed

A PubMed data-source connector for [cognee](https://github.com/topoteretes/cognee): sync
article metadata and abstracts matching a query into memory, incrementally, and forget
articles when they are removed from the PubMed index.

It exposes a `dlt` source you hand to `cognee.remember(...)`. Each article becomes a
**normal document** in cognee (it flows through chunking and LLM entity extraction), via
cognee's document-mode marker. One source is one PubMed query scoped to an *entry-date*
window (`edat`, the date a record entered the PubMed index). The package also carries the
provider's connection contract (`CONNECTOR`, an API-key descriptor on
`cognee-connectors-core`) for a cognee server or Cognee Cloud.

## Install

```bash
pip install cognee-connector-pubmed             # the connector and its DLT source
pip install "cognee-connector-pubmed[cognee]"   # plus cognee itself, which runs the sync
```

## Requirements

cognee 1.6.5 or later whose ingestion reads per-row node sets (`DOCUMENT_SYNC_VERSION` 2 or
later; the source refuses an older one before it makes a request). The E-utilities client
uses `requests`, which the package declares and `dlt` (a dependency of
`cognee-connectors-core[dlt]`) requires anyway.

PubMed's E-utilities need no account and no key for light use. An [NCBI API
key](https://www.ncbi.nlm.nih.gov/account/settings/) raises the rate limit from 3 requests
a second to 10; without one the client spaces its requests to stay under the lower limit.

## Configuration

This package reads no environment variable. cognee has no `PUBMED_*` settings and no
settings class for PubMed, and the key is never read from the environment: it is a
parameter of the source factory (`api_key=`, or a pre-built `service=`). Everything is
configured on the `pubmed_source(...)` call, see the argument table under Usage.

## Setup

1. Optional: create an API key at [NCBI account settings](https://www.ncbi.nlm.nih.gov/account/settings/).
2. Pass it as `api_key` if you have one. The connector only searches and reads; it never
   writes anything to NCBI. Set your `LLM_API_KEY` like any other cognee run.

## Usage

```python
import cognee
from cognee_connector_pubmed import pubmed_source

await cognee.remember(
    pubmed_source(
        api_key="<ncbi api key, optional>",
        resource_name="pubmed_alzheimers",
        query='"alzheimer disease"[title/abstract]',
    ),
    dataset_name="pubmed",
    primary_key="id",
    write_disposition="merge",  # required: incremental upserts and deletes
    max_rows_per_table=0,  # no row cap; orphan cleanup must see the whole corpus
)

answer = await cognee.recall(
    "What does the literature say about amyloid clearance?",
    datasets=["pubmed"],
)
```

A runnable version is [examples/example.py](examples/example.py) (`python
examples/example.py` from the package directory, with `LLM_API_KEY` set). Run the same
`remember` call again whenever you want to sync.

| Argument | Meaning | Default |
|---|---|---|
| `resource_name` | Names this connection: it keys the sync state and the cleanup of deleted records, one per query | required |
| `api_key` | Optional NCBI API key; raises the rate limit from 3/s to 10/s | none |
| `query` | PubMed search term selecting what to ingest, in esearch syntax | whole index the dates reach |
| `start_date` | Lower bound of the entry-date window, `YYYY/MM/DD` | open downwards |
| `end_date` | Upper bound of the window, `YYYY/MM/DD` | today |
| `today` | The clock the default `end_date` reads (`YYYY/MM/DD`); mainly for tests | wall clock |
| `max_requests` | Budget of API calls in one run; a call the client retries counts once | 500 |
| `check_active` | Host checkpoint, called before every request | none |
| `service` | Pre-built client, from `build_pubmed_service(api_key)` | built from `api_key` |

After a run, `source.cognee_sync_stats` holds counts only: `scanned` (ids the window
listed), `skipped` (already fingerprinted), `fetched` (articles rendered and emitted),
`deleted` (tombstones), `failed`, `requests` (API calls the run made), plus the reason a
run stopped early: `failed_rate_limit`, `failed_budget`, `failed_auth`, `failed_cap`, or
`failed_deletion_check` (one record's removal check errored; the record is kept).

## Running on a cognee server

```python
from cognee_connector_pubmed.cognee_host import register

register()  # a no-op: there is nothing to register for PubMed
```

PubMed has no OAuth flow, so cognee ships no adapter, no credential routes and no settings
for it: `register()` does nothing, `routers()` is empty, and the package declares no
`cognee.integrations` entry point. A server syncs PubMed the way the SDK does, by handing
`pubmed_source(...)` to `remember`; a host that stores keys builds the source with
`service=build_pubmed_service(key)` and names the connection with `resource_name`, as
cognee's `modules.integrations.ingestion` does. `CONNECTOR` (an API-key
`ConnectorDescriptor` served by `DescriptorConnector`) names the provider for a host's
catalog, derives a connection's dataset from the account id it stores (a `pubmed_<id>`
dataset) and hands a stored key out as the sync credential, read from
`tokens["access_token"]`; it does not enumerate a query, the source does.

## What is ingested

| Row | Id | Title | Content |
|---|---|---|---|
| Article | `pmid:<pmid>` | article title | a sentence naming the kind, then `Label: value` lines and the abstract |

```
PubMed article "Sleep and the consolidation of declarative memory".
Authors: Smith J, Doe A
Journal: Journal of Memory Studies
Year: 2026
DOI: 10.1000/example
Publication types: Journal Article
Keywords: sleep, memory consolidation
Abstract: BACKGROUND: ... RESULTS: ...
```

Rendered is what `efetch` returns as article metadata: title, authors (`LastName
Initials`, collective names as-is), journal title, publication year, DOI, publication
types, author keywords and the abstract (labeled sections keep their labels). Not
rendered: MeSH headings, chemicals, grants, references, affiliation addresses and
anything else on the record. Each article is its own document; the graph joins two of
them only when their texts use the same name, like any cognee document.

## Node set and scoping a recall

Each row carries `cognee_node_set`: `["pubmed:<resource_name>"]`. The name already starts
with the `pubmed:` source tag, so cognee keeps it as it is, and the article's document,
chunks and entities belong to that node set. Scope a recall to this connection with the
name:

```python
await cognee.recall(
    "What does the literature say about amyloid clearance?",
    datasets=["pubmed"],
    node_name=["pubmed:pubmed_alzheimers"],
)
```

The node set is part of a document's identity, like every other column this source emits.
It is fixed at the first release: changing it, or the wording of the text above, gives
every article a new document and one more extraction on your LLM key. A tombstone carries
no node set.

## How sync and forget-on-delete work

- **The window.** `esearch` lists the PMIDs whose `edat` lies in
  `[start_date, end_date]` (`datetype=edat`, both bounds inclusive). `end_date` defaults
  to today, so the window grows with the calendar; `start_date` defaults to open
  downwards, which for a wide query is the whole index — narrow it.
- **Incremental fetch.** The listing is ids only and cheap; the articles themselves come
  from `efetch`, and only for PMIDs the run has no fingerprint for. A fingerprint (a hash
  of the rendered row) is kept per article in dlt's per-resource state, so a re-run skips
  everything it has synced and fetches only what entered the index since. The state is
  the incremental cursor: it lives in the resource, keyed by `resource_name`.
- **Deleted articles.** The window is listed in full on every run, so an id that
  disappeared is spotted at any age. Before a tombstone is emitted, one
  `esearch term=<pmid>[uid]` lookup confirms the index really dropped it (count 0); a
  count above 0 or an errored check keeps the record and counts the failure. Forgotten
  articles are emitted as `{"id": ..., "_deleted": True}` on a `hard_delete` column: dlt
  removes them on `merge`, and cognee's orphan cleanup removes them from the graph and
  vector stores.
- **Deselected articles.** Narrowing `start_date` (or changing `query`) restarts the
  listing; articles whose `edat` falls outside the new window are forgotten at once, like
  a filter a user dropped — no confirming lookup, since they left the selection, not the
  index.
- **Resumable runs.** A run cut by a 429 or a spent `max_requests` budget stops cleanly,
  keeps its listing offset and fetch queue, reports the reason in `cognee_sync_stats`, and
  the next run resumes where it stopped. A key rejected after rows were emitted ends the
  run the same way (`failed_auth`); one rejected before any row raises `PubMedAuthError`,
  and dlt drops the state of that run.
- **Capped listings.** `esearch` deep-paging answers at most 10,000 ids. A bigger window
  is listed up to the cap, reported as `failed_cap`, and **never forgets anything**,
  because a capped listing cannot be proven complete.
- **Polite client.** Without an API key the client waits ~0.4s between requests, with one
  ~0.12s (NCBI's 3/s and 10/s limits). It never waits out a 429: it stops and resumes
  next run. Retries of 5xx answers and network errors back off 1s then 2s, three attempts
  per call, and a call that retries still counts once against `max_requests`.
- **A redirect is an error.** An answer with a 3xx status raises `PubMedAPIError` and is
  not followed: the API key rides in the query string, and a hop would carry it to
  whatever host the answer names.
- **A PMID is checked before it is read.** PMIDs come out of NCBI's own answers, and a
  deletion check builds a `pmid[uid]` term with the id: anything but digits is refused,
  the check counts as failed, and the record is kept.
- `check_active` (for hosts) runs before every API request.

A run costs one `esearch` call per 500 ids of the window (plus one for the count), one
`efetch` call per 200 new articles, and one confirming `esearch` per article that left a
complete listing.

## Rate limits

NCBI's E-utilities allow about 3 requests a second without an API key and 10 with one,
counted per IP address and key. The client enforces the spacing itself; bursts and
parallel jobs against the same key can still draw a 429, which the run reports and
resumes from. There is no daily quota for the volume this connector reaches.

## Privacy

This connector reads the public scientific literature and sends it to your configured LLM
and embedding providers during cognify, like any document you add to cognee: check that
you may send it there, or use local models. Be clear about what it does:

**What is read**

- Article metadata and abstracts of the public PubMed index, for the query and window you
  name — nothing else is requested, and the connector never creates, changes or deletes
  anything at NCBI.
- Received but never rendered: affiliation addresses, author emails when present in
  affiliations, MeSH headings, chemicals, grants, references.

**Where it goes**

- The text above becomes cognee documents. During cognify it is sent to your configured
  LLM and embedding providers, like any document you add to cognee.
- dlt state stores each article's id with a fingerprint (a hash of its rendered text),
  its entry date, the ids waiting for a deletion check and the listing offset. It stores
  no titles or abstracts. Document ids contain the PMID.
- dlt's staging database for the dataset (`dlt_database_<dataset>`) holds the whole
  rendered text of every article next to that state.
- dlt also keeps every completed load package in its data directory (`~/.dlt` of the
  server process unless `DLT_DATA_DIR` moves it): it does not delete completed jobs by
  default, so the text of a deleted article stays in those packages. That is a known
  limit of cognee's dlt ingestion, not of this connector; see the Apollo README for the
  full discussion of what deletion does and does not erase.

**How deletion is honoured**

- An article removed from the PubMed index is forgotten on the **next sync** that lists a
  window covering its entry date. Run syncs in the foreground: forgetting happens as part
  of `remember`.
- A window of **more than 10,000 matching articles never detects deletions** until it is
  narrowed (`failed_cap`).
- An article whose removal check keeps failing stays in memory until a check succeeds.

## Limitations

- Corrections are not re-synced. PubMed does not move `edat` when a record is revised,
  and this connector does not re-read what it has fingerprinted; a corrected abstract
  stays as first synced. Widen nothing: start a new `resource_name` to re-pull.
- At most 10,000 matching articles per listing.
- Author lists can be very long for consortium papers; every author is rendered, by
  design.

## Development

From a `cognee-connectors` checkout, with cognee installed (the workspace's `dev` group
has it; the integration tests run its pipeline):

```bash
export COGNEE_ENV_FILE=<an empty file> TELEMETRY_DISABLED=1 LITELLM_MODE=PRODUCTION LLM_API_KEY=sk-test-dummy
PYTHONPATH=core:connectors/pubmed python -m pytest connectors/pubmed/tests -q
python scripts/check_cognee_free.py cognee_connector_pubmed
```

No NCBI account or API key is needed. `tests/unit/fake_pubmed.py` is an in-memory
E-utilities stand-in: paged `esearch` answers, `efetch` article XML, per-PMID lookups and
rate-limit errors. It answers no endpoint the connector does not use, so a call to one
fails the test that made it. `tests/integration/` runs the real cognee pipeline with the
LLM and embeddings mocked.
