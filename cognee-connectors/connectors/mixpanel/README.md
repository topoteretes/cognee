# cognee-connector-mixpanel

A Mixpanel data-source connector for [cognee](https://github.com/topoteretes/cognee): sync your
project's analytics definitions — annotations, cohort definitions, saved funnel reports and the
Lexicon event schema — into memory, incrementally, and forget records when they are deleted in
Mixpanel.

It exposes a `dlt` source you hand to `cognee.remember(...)`. Each record becomes a **normal
document** in cognee (it flows through chunking and LLM entity extraction), via cognee's
document-mode marker. One source is one Mixpanel project. The package also carries the provider's
connection contract (`CONNECTOR`, an API-key descriptor on `cognee-connectors-core`) for a cognee
server or Cognee Cloud.

**Definitions are knowledge; raw events are noise.** The connector syncs the definitions a team
wrote about its metrics, never raw event streams, unless you explicitly opt into the `events`
kind, which exports raw events as **daily per-event counts** over a date window — one small
document per day and event instead of one document per event.

## Install

```bash
pip install cognee-connector-mixpanel             # the connector and its DLT source
pip install "cognee-connector-mixpanel[cognee]"   # plus cognee itself, which runs the sync
```

## Requirements

cognee 1.6.5 or later whose ingestion reads per-row node sets (`DOCUMENT_SYNC_VERSION` 2 or
later; the source refuses an older one before it makes a request). The REST client uses
`requests`, which the package declares and `dlt` (a dependency of `cognee-connectors-core[dlt]`)
requires anyway.

## Configuration

This package reads no environment variable. cognee has no `MIXPANEL_*` settings and no settings
class for Mixpanel, and the credential is never read from the environment: it is a parameter of
the source factory (a service account, a legacy API secret, or a pre-built `service=`).
Everything is configured on the `mixpanel_source(...)` call, see the argument table under Usage.

## Setup

1. Create a **service account** in Mixpanel under **Organization Settings > Service Accounts**
   with a role that can read the project you want to sync. Copy its username and secret — the
   secret is shown once. A legacy **project API secret** (Project Settings > Access Keys) works
   too via `api_secret=`; a service account is the recommended shape.
2. Pass the credential to `mixpanel_source`. The connector only lists and reads definitions.
   Set your `LLM_API_KEY` like any other cognee run.
3. If the credential can access more than one project, pass `project_id` explicitly. Without it,
   the connector lists the accessible projects and requires exactly one, so a sync never binds
   to a guess.

## Usage

```python
import cognee
from cognee_connector_mixpanel import mixpanel_source

await cognee.remember(
    mixpanel_source(
        resource_name="mixpanel_product",  # one per Mixpanel project
        service_account_username="<service account username>",
        service_account_secret="<service account secret>",
    ),
    dataset_name="mixpanel",
    primary_key="id",
    write_disposition="merge",  # required: incremental upserts and deletes
    max_rows_per_table=0,  # no row cap; orphan cleanup must see the whole corpus
)

answer = await cognee.recall(
    "Which Mixpanel cohorts exist and what do they track?", datasets=["mixpanel"]
)
```

Run the same `remember` call again whenever you want to sync. A runnable version is
[`examples/mixpanel.py`](examples/mixpanel.py).

| Argument | Meaning | Default |
|---|---|---|
| `resource_name` | Names this connection: it keys the sync state and the cleanup of deleted records | required, one per Mixpanel project |
| `service_account_username`, `service_account_secret` | A Mixpanel service account | required unless `api_secret` or `service` is given |
| `api_secret` | A legacy project API secret, instead of a service account | none |
| `region` | `"us"` or `"eu"`, the project's data residency | `"us"` |
| `project_id` | The project to sync; without it, the credential must see exactly one | discovered |
| `include` | Which kinds to sync: `"annotations"`, `"cohorts"`, `"funnels"`, `"schemas"`, and opt-in `"events"` | the four definition kinds |
| `events_from` | First day of the first `events` window (`YYYY-MM-DD`) | required on the first run that includes `"events"` |
| `today` | The run's date: a `date`, a callable returning one, or None for today | None |
| `service` | Pre-built client, from `build_mixpanel_service(...)` | built from the credential |
| `check_active` | Host checkpoint, called before every request | none |

**One connection per Mixpanel project.** The first run resolves and stores the project id. A run
under another project with the same `resource_name` is refused, because it would forget the first
project's records. Give every project its own `resource_name`, even in one dataset. A host that
keeps one connection per project can read the id with one request:

```python
from cognee_connector_mixpanel.source import build_mixpanel_service, mixpanel_project_id

project_id = mixpanel_project_id(
    build_mixpanel_service(service_account_username="...", service_account_secret="...")
)
```

It raises when the credential can access zero or several projects and no `project_id` picks one,
and so does a sync.

After a run, `source.cognee_sync_stats` holds counts only: `scanned`, `skipped`, `deleted`,
`failed` and `requests`. A run costs one `GET /projects` on the first run (or when `project_id`
is not pinned yet), one request per included definition kind, and one request per `events` run.

- `failed_auth`: Mixpanel rejected the credential (HTTP 401). The run raises, dlt drops its
  state, and the next run starts over.
- `failed_rate_limit`: Mixpanel answered HTTP 429. The run raises and changes nothing; run it
  again later. A definitions sync is a handful of requests, well under the Query API's limit of
  60 queries an hour.

## Running on a cognee server

```python
from cognee_connector_mixpanel.cognee_host import register

register()  # a no-op: there is nothing to register for Mixpanel
```

Mixpanel has no OAuth flow, so cognee ships no adapter, no credential routes and no settings for
it: `register()` does nothing, `routers()` is empty, and the package declares no
`cognee.integrations` entry point. A server syncs Mixpanel the way the SDK does, by handing
`mixpanel_source(...)` to `remember`; a host that stores credentials builds the source with
`service=build_mixpanel_service(...)` and names the connection with `resource_name`, as cognee's
`modules.integrations.ingestion` does. `CONNECTOR` (an API-key `ConnectorDescriptor` served by
`DescriptorConnector`) names the provider for a host's catalog, derives a connection's dataset
from its account id (`mixpanel_<project id>`) and hands a stored secret out as the sync
credential, read from `tokens["access_token"]`; it does not enumerate a project, the source does.

## What is ingested

| Row | Id | Title | Content |
|---|---|---|---|
| Annotation | `annotation:<id>` | the note | a sentence with its date and the note |
| Cohort | `cohort:<id>` | name | a sentence with its name, description, creation date |
| Saved funnel report | `funnel:<id>` | name | a sentence with its name, its steps in order |
| Event schema (Lexicon) | `schema:<name>` | event name | a sentence with the description, sorted property names |
| Daily event count (opt-in) | `events:<date>:<event>` | `<event> on <date>` | one sentence with the day's count |

Every text opens with one sentence that names its kind, and the rest is `Label: value` lines:

```
Mixpanel cohort Power users.
Description: Users with 10+ sessions a week
Created: 2026-01-15
```

Each record is its own document and the graph only joins two of them when their texts use the
same name. Not rendered: cohort membership and sizes (they are volatile and personal), user and
group profiles, computed report results (a saved funnel's conversion numbers change on every
query — its definition does not), and per-event rows for the `events` kind, which aggregate to
daily counts instead. Saved Insights bookmarks are not synced: the Query API can run one by
`bookmark_id` but lists none, so the connector syncs saved funnel reports as the report
definitions Mixpanel can enumerate.

The `events` kind answers "how much" questions at day granularity without drowning memory in
noise: one document per day and event name with that day's count. Its window ends the day before
the run — today's counts are partial — and late events in a closed day raise the count, so the
day's document re-ingests once with the corrected number.

## Node set and scoping a recall

Each row carries `cognee_node_set`: `["mixpanel:<project id>:<kind>"]`, with `<kind>` one of
`annotations`, `cohorts`, `funnels`, `schemas` or `events`. The name already starts with the
`mixpanel:` source tag, so cognee keeps it as it is, and the record's document, chunks and
entities belong to that node set. Scope a recall to Mixpanel, or to one kind, with the names
(several are matched with OR):

```python
await cognee.recall(
    "Which Mixpanel cohorts exist and what do they track?",
    datasets=["mixpanel"],
    node_name=[f"mixpanel:{project_id}:cohorts"],
)
```

The node set is part of a document's identity, like every other column this source emits. It is
fixed at the first release: changing it, or the wording of the text above, gives every record a
new document and one more extraction on your LLM key. A tombstone carries no node set.

## How sync and forget-on-delete work

- **Fingerprints:** Mixpanel's definition endpoints carry no change feed, no updated-since
  filter and no pagination: each run lists every included kind in full — one request per kind —
  and emits only the records whose rendered text changed, using a hash kept in dlt's
  per-resource state. A definitions sync is a handful of requests no matter the project size.
- **Deleted records:** a definition that stops appearing in a complete listing is forgotten.
  There are no filters that could hide a listed record, so no per-record re-check is needed. The
  tombstone is emitted as `{"id": ..., "_deleted": True}` on a `hard_delete` column; dlt removes
  it on `merge`, and cognee's orphan cleanup removes it from the graph and vector stores.
- **Deselected kinds:** leaving a kind out of `include` forgets its records on the next run, the
  same as a deletion.
- **A failed run changes nothing.** Each listing is a single request, so there is no partial
  listing to resume. An error raises, dlt drops the run's state, and the next run lists again.
  Tombstones are computed only after every included kind listed successfully, so a run that
  fails mid-way never forgets.
- **The events cursor:** `events` is incremental by a date window, the only incremental shape
  the export API offers. The first run exports `events_from` through the day before the run; each
  later run exports only the days since the stored cursor, and advances the cursor to the day
  before the run. The cursor and the day fingerprints live in the same dlt resource state.
- **A redirect is an error.** An answer with a 3xx status raises `MixpanelAPIError` naming the
  status and the endpoint, and is not followed: `requests` drops an `Authorization` header on a
  hop to another host, but the check keeps the run honest about what it asked.
- `check_active` (for hosts) runs before every API request.
- **Rate limits:** the Query API allows 60 queries an hour. A definitions sync costs well under
  that; a run never waits out a limit — on HTTP 429 it raises, changes nothing, and you run it
  again.

## Privacy

This connector reads analytics metadata about your product, not personal data:

**What is read**

- Annotations (dated notes the team left on its metrics), cohort definitions (name, description,
  creation date — never membership, never counts), saved funnel reports (name and step
  sequence), and the Lexicon event schema (event names, descriptions, property names).
- With the opt-in `events` kind: raw event names and timestamps for the window, aggregated to
  daily counts before anything is stored or rendered. No event properties, no user identities.
- Not read: cohort membership, user and group profiles, lookup tables, and computed report
  results. The connector never creates, changes or deletes anything in Mixpanel.

**Where it goes**

- The text above becomes cognee documents. During cognify it is sent to your configured LLM and
  embedding providers, like any document you add to cognee.
- dlt state stores the Mixpanel project id, each record's id with a fingerprint (a hash of its
  rendered text), and the `events` date cursor. It stores no descriptions or text. Document ids
  contain the Mixpanel record id.
- dlt's staging database for the dataset (`dlt_database_<dataset>`) holds the whole rendered text
  of every record next to that state, and dlt keeps completed load packages in its data
  directory (`~/.dlt` of the server process unless `DLT_DATA_DIR` moves it), outside
  `DATA_ROOT_DIRECTORY` and `SYSTEM_ROOT_DIRECTORY`.
- Logs and `cognee_sync_stats` contain counts, statuses and endpoints. Credentials are never
  logged, stored or put in an error: a credential `requests` would reject is refused before the
  first request, and a redirect is never followed.

**How deletion is honoured**

- A definition deleted in Mixpanel is forgotten on the **next sync**. Run syncs in the
  foreground: forgetting happens as part of `remember`.
- Every record of a kind you leave out of `include` is forgotten on the next sync.
- A deleted record stays in dlt's completed load packages (cognee's dlt ingestion limit, not
  this connector's); `cognee.forget` does not erase it for good either — the next sync rebuilds
  every document the source still lists. To stop a record coming back, delete it in Mixpanel and
  sync.

## Limitations

- Saved Insights bookmarks are not synced: the Query API runs one by `bookmark_id` but lists
  none. Saved funnel reports are the enumerable report definitions.
- `events` is day-granular counts, not event rows: properties, user identities and intra-day
  patterns are out by design.
- A credential that can access several projects must be told which one with `project_id`.

## Development

From a `cognee-connectors` checkout (the unit tests need no cognee and no Mixpanel account):

```bash
export COGNEE_ENV_FILE=<an empty file> TELEMETRY_DISABLED=1 LITELLM_MODE=PRODUCTION LLM_API_KEY=sk-test-dummy
PYTHONPATH=core:connectors/mixpanel python -m pytest connectors/mixpanel/tests -q
python scripts/check_cognee_free.py cognee_connector_mixpanel
```

`tests/unit/fake_mixpanel.py` is an in-memory Mixpanel API shaped like its responses: the
single-request listings, project discovery, and the newline-delimited event export. It answers
no endpoint the connector does not use, so a call to one fails the test. The events window and
the run's date are injected, so the cursor's day arithmetic is tested without sleeping or
freezing time.
