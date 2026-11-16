# cognee-connector-bamboohr

A BambooHR data-source connector for [cognee](https://github.com/topoteretes/cognee): sync your
company's employees into memory, incrementally, and forget an employee when BambooHR no longer
serves them.

It exposes a `dlt` source you hand to `cognee.remember(...)`. Each employee becomes a **normal
document** in cognee (it flows through chunking and LLM entity extraction), via cognee's
document-mode marker. One source is one BambooHR company (one subdomain). The package also carries
the provider's connection contract (`CONNECTOR`, an API-key descriptor on
`cognee-connectors-core`) for a cognee server or Cognee Cloud.

Employee data is sensitive: **the connector only ever reads the fields you name in an explicit
allowlist**, on every request, and there is no full-dump mode.

## Install

```bash
pip install cognee-connector-bamboohr             # the connector and its DLT source
pip install "cognee-connector-bamboohr[cognee]"   # plus cognee itself, which runs the sync
```

## Requirements

cognee 1.6.5 or later whose ingestion reads per-row node sets (`DOCUMENT_SYNC_VERSION` 2 or
later; the source refuses an older one before it makes a request). The REST client uses
`requests`, which the package declares and `dlt` (a dependency of `cognee-connectors-core[dlt]`)
requires anyway.

The API key needs access to `GET /meta/fields` and `POST /employees/custom-report` (and
`GET /employees/{id}/` for deletion checks). BambooHR answers HTTP 403 when the key's permissions
do not cover an endpoint, and a sync that needs that endpoint fails naming it.

## Configuration

This package reads no environment variable. cognee has no `BAMBOOHR_*` settings and no settings
class for BambooHR, and the key is never read from the environment: it is a parameter of the
source factory (`api_key=`, or a pre-built `service=`). Everything is configured on the
`bamboohr_source(...)` call, see the argument table under Usage.

## Setup

1. Create an API key in BambooHR under **Settings > API Keys** (a user with sufficient
   permissions; the key inherits that user's field access, so keep both narrow).
2. Know your company subdomain: the `acme` in `https://acme.bamboohr.com`.
3. Pass the key as `api_key`. The connector only reads employee records. Set your `LLM_API_KEY`
   like any other cognee run.

## Usage

```python
import cognee
from cognee_connector_bamboohr import bamboohr_source

await cognee.remember(
    bamboohr_source(
        resource_name="bamboohr_acme",
        subdomain="acme",
        api_key="<bamboohr api key>",
        # The explicit allowlist: the only fields ever requested or rendered.
        fields=["firstName", "lastName", "jobTitle", "department", "location", "status"],
    ),
    dataset_name="bamboohr_hr",
    primary_key="id",
    write_disposition="merge",  # required: incremental upserts and deletes
    max_rows_per_table=0,  # no row cap; orphan cleanup must see the whole corpus
)

answer = await cognee.recall("Who works in the Research department?", datasets=["bamboohr_hr"])
```

Run the same `remember` call again whenever you want to sync.

| Argument | Meaning | Default |
|---|---|---|
| `resource_name` | Names this connection: it keys the sync state and the cleanup of deleted employees | required, one per BambooHR company |
| `subdomain` | The company's BambooHR subdomain (the `acme` of `acme.bamboohr.com`) | required |
| `fields` | The explicit field allowlist, as `GET /meta/fields` lists the field aliases. Only these fields are ever requested | required, no default |
| `api_key` | BambooHR API key | needed unless `service` is given |
| `max_requests` | Budget of API calls in one run; a call the client retries counts once | 500 |
| `check_active` | Host checkpoint, called before every request | none |
| `service` | Pre-built client, from `build_bamboohr_service(api_key, subdomain)` | built from `api_key` |

Every argument that takes a list takes a list or tuple of strings; anything else, a bare string
included, is refused. The allowlist is sorted and repeats dropped. Changing `fields` re-renders
every employee once.

The allowlist is checked against BambooHR's own field list on every run. A name BambooHR does not
list — a typo, a deleted custom field — stops the run with a `BambooHRSourceError` naming it,
before it emits or forgets anything.

**One connection per BambooHR company.** The first run stores the subdomain. A run under another
subdomain with the same `resource_name` is refused, because it would forget the first company's
employees. Give every company its own `resource_name`, even in one dataset.

After a run, `source.cognee_sync_stats` holds counts only: `scanned`, `skipped`, `deleted`,
`failed`, `kept`, `pending`, `requests` and `held_back`, plus the reason a run or a check failed.
`requests` counts the API calls the run made; a call the client retries after a 5xx or a network
error counts once, so BambooHR's own count can be higher.

- `failed_rate_limit`, `failed_budget`: the run stopped early and resumes next time
- `failed_auth`: BambooHR rejected the key. After the run emitted rows it stops early, keeps its
  state and resumes next time. Before any row it raises, dlt drops the state of that run
- `failed_deletion_check`: the check of one employee failed (a 5xx, an unreadable answer)
- `failed_access`: the key cannot call an endpoint (HTTP 403)

`held_back` is the number of employees whose deletion check keeps failing and who sit out in
back-off (see below). It is not a failure and not part of `failed`.

## Running on a cognee server

```python
from cognee_connector_bamboohr.cognee_host import register

register()  # a no-op: there is nothing to register for BambooHR
```

BambooHR has no OAuth flow, so cognee ships no adapter, no credential routes and no settings for
it: `register()` does nothing, `routers()` is empty, and the package declares no
`cognee.integrations` entry point. A server syncs BambooHR the way the SDK does, by handing
`bamboohr_source(...)` to `remember`; a host that stores keys builds the source with
`service=build_bamboohr_service(key, subdomain)` and names the connection with `resource_name`.
`CONNECTOR` (an API-key `ConnectorDescriptor` served by `DescriptorConnector`) names the provider
for a host's catalog, derives a connection's dataset from its account id
(`bamboohr_<account id>`) and hands a stored key out as the sync credential, read from
`tokens["access_token"]`; it does not enumerate a company, the source does.

## What is ingested

One row per employee, with the allowlisted fields only:

| Row | Id | Title | Content |
|---|---|---|---|
| Employee | `employee:<id>` | first and last name, else `Employee <id>` | one sentence naming the employee, then one `Label: value` line per allowlisted field that has a value |

```
BambooHR employee Ada Lovelace.
Department: Research
First Name: Ada
Job Title: Engineer
Last Name: Lovelace
Location: London
Status: Active
```

Lines follow the sorted allowlist order; a field with no value renders no line. Fields outside
the allowlist are never rendered, and nothing volatile (no sync timestamp) is added. A field
BambooHR lists without a display name is labeled by its alias.

**The allowlist is the privacy control.** It decides what is read from BambooHR, what is rendered
into text and what reaches your LLM and embedding providers. Do not list fields you would not
want in a document — compensation, government identifiers, personal contact fields — unless your
deployment may hold them. Multi-value fields render sorted, so their text is deterministic.

## Node set and scoping a recall

Each row carries `cognee_node_set`: `["bamboohr:<subdomain>"]`. The name already starts with the
`bamboohr:` source tag, so cognee keeps it as it is, and the employee's document, chunks and
entities belong to that node set. Scope a recall to one company with the name:

```python
await cognee.recall(
    "Who works in the Research department?",
    datasets=["bamboohr_hr"],
    node_name=["bamboohr:acme"],
)
```

The node set is part of a document's identity, like every other column this source emits: changing
it gives every employee a new document and one more extraction on your LLM key. A tombstone
carries no node set.

## How sync and forget-on-delete work

- **One report per run.** BambooHR has no change feed. Each run calls `GET /meta/fields` (to
  validate the allowlist and label the lines) and `POST /employees/custom-report` with exactly the
  allowlisted fields. The report is one request and one answer; there is no pagination.
- **Fingerprints:** a row is emitted only when the hash of its rendered text changed. The hashes
  live in dlt's per-resource state.
- **Deleted employees:** an id that stopped appearing in a complete report is read once more
  (`GET /employees/{id}/`) before it is forgotten. A 404 is a tombstone. Any other answer keeps
  the record: a 200 (a report setup can miss an employee the API still serves), a 403, a 5xx or
  no answer.
- **Tombstones:** forgotten employees are emitted as `{"id": ..., "_deleted": True}` with a
  `hard_delete` column. dlt removes them on `merge`, and cognee's orphan cleanup removes them from
  the graph and vector stores.
- **A check that keeps failing backs off.** After five failed checks in a row an employee is left
  alone for ten cycles: no request, no `failed` count, but counted in `held_back`. Then it is
  tried once more. An employee listed again drops its count.
- **Rate limits and pacing.** BambooHR asks callers to limit themselves to one request per
  second; the client paces every request (a retry included) at least 1.1 seconds after the
  previous one. On HTTP 429, or when `max_requests` is spent, the run stops cleanly, keeps its
  progress and reports why in `cognee_sync_stats`; the next run finishes the pending deletion
  checks first. A key rejected after rows were emitted ends the run the same way (`failed_auth`);
  one rejected before any row raises `BambooHRAuthError`, and dlt drops the state of that run.
- **An employee id is checked before it is read.** A deletion check reads `/employees/{id}/` with
  an id that came out of a BambooHR report. An id with anything but letters and digits (`..`, a
  slash, `?`) is not requested: the check fails like any other failed check and the record is
  kept.
- **A redirect is an error.** An answer with a 3xx status raises `BambooHRAPIError` naming the
  status and the endpoint, and is not followed.
- `check_active` (for hosts) runs before every API request.

A run costs two requests (the field list and the report) plus one per employee that stopped
appearing. With the one-request-per-second pace, a run with many deletion checks is slow: raise
`max_requests` only together with patience, or let the sync finish over several runs.

## Privacy

This connector reads personal data about your own organisation's employees. Be clear about what
it does:

**What is read**

- Only the fields named in the allowlist, from your own BambooHR company. The custom report and
  every deletion check ask for exactly those fields; fields outside the allowlist are never
  requested, and the source has no full-dump mode.
- The field list (`GET /meta/fields`) is read to validate the allowlist and to label the lines.

**Where it goes**

- The text above becomes cognee documents. During cognify it is sent to your configured LLM and
  embedding providers, like any document you add to cognee. Check that you may send HR data
  there, or use local models.
- dlt state stores the subdomain, each employee's id with a fingerprint (a hash of the rendered
  text), the ids waiting for a deletion check, and the failure counts of checks in back-off. It
  stores no names or text. Document ids contain the BambooHR employee id.
- dlt's staging database for the dataset (`dlt_database_<dataset>`) holds the whole rendered text
  of every employee, next to that state.
- dlt also keeps every completed load package, with the rendered text of each row it loaded, in
  its data directory (`~/.dlt` of the server process unless `DLT_DATA_DIR` moves it): dlt does not
  delete completed jobs by default. That directory is outside `DATA_ROOT_DIRECTORY` and
  `SYSTEM_ROOT_DIRECTORY`.
- Logs and `cognee_sync_stats` contain counts, statuses and endpoints. Employee ids are replaced
  by `{id}` in error messages, and the key is never logged, stored or put in an error: the client
  refuses a key `requests` would reject (it names a header value it rejects), and a redirect is
  never followed. The dataset itself holds the names above: anyone with read access to it can see
  them, so keep it in its own dataset and share it only with people allowed to see the HR data.

**How deletion is honoured**

- An employee deleted in BambooHR (a 404 on the check) is forgotten on the **next sync**. Run
  syncs in the foreground: forgetting happens as part of `remember`.
- An employee whose check keeps failing stays in memory until a check succeeds.
- **A deleted employee stays on disk.** The sync removes the document, the graph content, the
  vectors and the staged row (the tombstone replaces it), but not the completed load packages
  above, which keep the text of every load. This is a known limit of cognee's dlt ingestion, not
  of this connector.
- **`cognee.forget` does not erase an employee for good.** It removes the documents, the graph
  and the vectors, but not the staged copy of the rendered text or the dlt state. The next sync
  under the same `resource_name` reads the staged rows back and builds every document again. To
  stop an employee coming back, delete them in BambooHR and sync.

## Limitations

- No incremental delta: every run re-reads the whole company in one report. Fine for the company
  sizes BambooHR serves; the per-run fingerprint skip avoids re-extracting unchanged employees.
- No employee photos, documents, tables (job info history, compensation), time off or recruiting
  data: the report endpoint reads scalar employee fields only.
- Deletion checks are 404-based; BambooHR keeps terminated employees readable (their `status`
  field is `Inactive`), so termination alone does not remove anyone from memory.
- The one-request-per-second pace makes large backlogs of deletion checks slow by design.

## Development

From a `cognee-connectors` checkout, with cognee installed (the workspace's `dev` group has it):

```bash
export COGNEE_ENV_FILE=<an empty file> TELEMETRY_DISABLED=1 LITELLM_MODE=PRODUCTION LLM_API_KEY=sk-test-dummy
PYTHONPATH=core:connectors/bamboohr python -m pytest connectors/bamboohr/tests -q
python scripts/check_cognee_free.py cognee_connector_bamboohr
```

No BambooHR account or API key is needed. `tests/unit/fake_bamboohr.py` is an in-memory BambooHR
API shaped like its responses: the field list, the custom report, per-employee reads that answer
like a deleted employee, and per-path failures. It answers no endpoint the connector does not
use, so a call to one fails the test.
