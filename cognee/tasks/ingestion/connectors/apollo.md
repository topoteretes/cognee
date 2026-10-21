# Apollo SDK connector

An Apollo.io data-source connector for [cognee](https://github.com/topoteretes/cognee): sync
your team's contacts, accounts and sequences into memory, incrementally, and forget records
when they are deleted in Apollo.

It exposes a `dlt` source you hand to `cognee.remember(...)`. Each record becomes a **normal
document** in cognee (it flows through chunking and LLM entity extraction), via cognee's
document-mode marker. It reads your team's own CRM records, never Apollo's enrichment
database. One source is one Apollo team.

## Requirements

Bundled with the Cognee SDK, including per-row node sets (`DOCUMENT_SYNC_VERSION` 2 or later; a
core without them is refused). No extra and no separate package is required. The REST client uses
`requests`, which `dlt` (a core dependency) already requires.

The key needs API access to the contact, account and sequence search endpoints. Apollo answers
HTTP 403 for a team or plan that lacks it, and the run stops with an error that names the refused
endpoint. Per the community README that access comes with a paid plan or a trial, not with a
free plan; this has not been tested live.

## Setup

1. Create an API key in Apollo under **Settings > Integrations > API**. A master key works. A
   scoped key needs the endpoints below, and only those of the kinds you sync:

   | Endpoint | Needed by |
   |---|---|
   | `api/v1/contact_stages/index` | every run: it names contact stages and carries the team id |
   | `api/v1/contacts/search`, `api/v1/contacts/show` | contacts |
   | `api/v1/account_stages/index` | accounts |
   | `api/v1/accounts/search`, `api/v1/accounts/show` | accounts |
   | `api/v1/emailer_campaigns/search` | sequences, and contacts (they name their sequences) |
   | `api/v1/labels/index` | contacts and accounts |
   | `api/v1/fields/index` | only with `custom_field_ids` |

   `contacts/show` and `accounts/show` are read only to check a record that stopped appearing.
2. Pass the key as `api_key`. The connector only lists and reads records. Set your
   `LLM_API_KEY` like any other cognee run.

## Usage

```python
import cognee
from cognee.tasks.ingestion.connectors import apollo_source

await cognee.remember(
    apollo_source(api_key="<apollo api key>", resource_name="apollo_sales"),
    dataset_name="apollo_crm",
    primary_key="id",
    write_disposition="merge",  # required: incremental upserts and deletes
    max_rows_per_table=0,  # no row cap; orphan cleanup must see the whole corpus
)

answer = await cognee.recall(
    "Which contacts at Acme are in an active sequence?", datasets=["apollo_crm"]
)
```

Run the same `remember` call again whenever you want to sync.

| Argument | Meaning | Default |
|---|---|---|
| `resource_name` | Names this connection: it keys the sync state and the cleanup of deleted records | required, one per Apollo team |
| `api_key` | Apollo API key | needed unless `service` is given |
| `include` | Which kinds to sync: `"contacts"`, `"accounts"`, `"sequences"` | all three (`None`) |
| `contact_stage_ids`, `contact_label_ids` | Only contacts in these stages or lists | no filter |
| `account_stage_ids`, `account_label_ids` | Only accounts in these stages or lists | no filter |
| `custom_field_ids` | Custom fields to render, as `GET /fields?source=custom` lists their ids: `contact.<id>` or `account.<id>`, nothing else. Not picklist or multi-select fields, see below | none |
| `max_requests` | Budget of API calls in one run; a call the client retries counts once | 500 |
| `check_active` | Host checkpoint, called before every request | none |
| `service` | Pre-built client, from `build_apollo_service(api_key)` | built from `api_key` |

Labels are what Apollo's UI calls lists. Every argument that takes ids takes a list or tuple of
id strings; anything else, a bare string included, is refused. The ids are sorted and repeats
dropped, so the same filter in another order resumes an interrupted listing. Changing a filter,
`include` or `custom_field_ids` changes what is rendered or selected: records that leave the
selection are forgotten, and changing `custom_field_ids` re-renders every record once.

A stage or list filter is checked against Apollo's own lists on every run. An id Apollo does not
list, whether a typo, a blank, a deleted stage or a list that `GET /labels` does not return (a
private list of another user, for instance), stops the run with an `ApolloSourceError` that
names the ids, before it emits or forgets anything. A filter that matches nothing would empty the
listing, and every record you synced would look deleted. Only the filters of the kinds in
`include` are checked.

**One connection per Apollo team.** The first run stores the team id Apollo reports. A run under
another team's key and the same `resource_name` is refused, because it would forget the first
team's records. Give every Apollo team its own `resource_name`, even in one dataset. A host that
keeps one connection per team can read the id with one request:

```python
from cognee.tasks.ingestion.connectors.apollo import apollo_team_id, build_apollo_service

team_id = apollo_team_id(build_apollo_service("<apollo api key>"))
```

It raises when Apollo's answer carries no team id, and so does a sync.

After a run, `source.cognee_sync_stats` holds counts only: `scanned`, `skipped`, `deleted`,
`failed`, `requests` and `held_back`, plus the reason a run or a record failed. `requests` counts
the API calls the run made. A call the client retries after a 5xx or a network error counts
once, so Apollo's own count can be higher: up to three attempts per call.

- `failed_rate_limit`, `failed_budget`: the run stopped early and resumes next time
- `failed_auth`: Apollo rejected the key. After the run emitted rows it stops early, keeps its
  state and resumes next time. Before any row it raises, dlt drops the state of that run and the
  next run starts the cycle over
- `failed_cap`: a listing hit Apollo's 50,000 record limit, so deletions were not detected
- `failed_deletion_check`: the check of one record failed (a 5xx, an answer cognee cannot read)
- `failed_access`: the key or plan cannot call an endpoint (HTTP 403). For one record read it is one
  record kept; for a listing or a lookup it is the run that raised, see below

`held_back` is the number of records in back-off when the run started, because their deletion
check keeps failing (see below). Every run reports it, one that is cut or capped before it
closes the listing included. It is not a failure and not part of `failed`. The syncs that hosts
run for Drive, Notion and Linear mark a sync degraded only when `failed` is not 0, so a sync that
only holds records back reads as ok: watch `held_back` if you need to know.

## What is ingested

| Row | Id | Title | Content |
|---|---|---|---|
| Contact | `contact:<id>` | name | a sentence with its account and job title, the account, contact stage, lists, listed custom fields, sequences |
| Account | `account:<id>` | name, else domain, else "Unnamed account" | a sentence with its domain, account stage, lists, listed custom fields |
| Sequence | `sequence:<id>` | name | a sentence with its status (active, inactive, archived), number of steps |

Every text opens with one sentence that names its kind, and the rest is `Label: value` lines:

```
Apollo contact Dana Whitfield works at Acme as Chief Financial Officer.
Account: Acme (acme.com)
Contact stage: Interested
Sequences: Q4 Outreach (active); Renewals (paused)
```

A contact names its account by the account's own title, read from the accounts listing, and falls
back to its own `organization_name` when the account names nothing: the contact has none, it was
filtered out of the selection, `accounts` is not in `include`, or Apollo lists it with neither a
name nor a domain. An account with only a domain is titled by the domain, which is then written
once, in its sentence and in the contact's `Account:` line. Two accounts with neither would share
the title "Unnamed account" and so one entity in the graph; it is not confirmed that Apollo can
return such an account. Each record is its own document and the graph only joins two of them when
their texts use the same name. The stage lines say `Contact stage` and
`Account stage` because entities merge by name, so a stage called "Cold" on both would become one.
Ids and stages are written as names. Not rendered: emails, phone numbers, social profile links,
anything from Apollo's enrichment data, the dates and steps of a contact's sequence activity, the
open and reply rates of a sequence and **who owns a record**. The per-contact activity feed is not
read.

**Who owns X has no answer in memory.** Apollo names an owner by user id, and turning that into a
name needs `/users/search`, which a scoped key probably cannot call (not tested, it would likely
need a master key). The owner is left out of the text instead of rendered as an id nobody can read.

Two people with the same name are one entity in the graph, whatever company they work for. That is
how cognee merges entities and a known limit of rendering each record as text.

The titles contacts name their accounts by are kept in the sync state as `account_names`, account
id to title and domain. Measured with 12,000 accounts (what 50,000 contacts reach): 72 bytes an
account, 0.87 MB in all, next to 3.5 MB of fingerprints for 62,000 records. Renaming an account
changes its text and that of every one of its contacts, so each is extracted again; renames are
rare, and the cost is one extraction per contact of that account plus one for the account.

## Node set and scoping a recall

Each row carries `cognee_node_set`: `["apollo:<team id>:<kind>"]`, with `<kind>` one of `contacts`,
`accounts` or `sequences`. The name already starts with the `apollo:` source tag, so cognee keeps
it as it is, and the record's document, chunks and entities belong to that node set. Scope a
recall to Apollo, or to one kind, with the names (several are matched with OR):

```python
from cognee.tasks.ingestion.connectors.apollo import apollo_team_id, build_apollo_service

team_id = apollo_team_id(build_apollo_service("<apollo api key>"))
await cognee.recall(
    "Which contacts at Acme are in an active sequence?",
    datasets=["apollo_crm"],
    node_name=[f"apollo:{team_id}:contacts"],
)
```

The node set is part of a document's identity, like every other column this source emits. It is
fixed at the first release: changing it, or the wording of the text above, gives every record a
new document and one more extraction on your LLM key. A tombstone carries no node set.

**Custom fields are not rendered unless you list them in `custom_field_ids`**: teams keep phone
numbers and personal emails in them. Do not list picklist or multi-select fields: Apollo returns
option ids for them, not labels, so they render as ids that mean nothing to a reader.

## How sync and forget-on-delete work

- **Fingerprints:** Apollo has no change feed and no deleted flag. Each run lists every included
  kind in full, oldest first, and emits only the records whose rendered text changed, using a
  hash kept in dlt's per-resource state. Contact `updated_at` is not used: sequence activity does
  not bump it, and accounts have none.
- **Deleted records:** a record that stops appearing in a complete listing is read once more
  before it is forgotten. It is forgotten when Apollo answers 404 or its legacy "this contact has
  been deleted" text (HTTP 422), or when it is still there but left the selected stages or
  lists. Any other answer, a 403 or a 5xx included, keeps it. A sequence that is gone from the
  sequence list is forgotten at once.
- **Deselected kinds:** leaving a kind out of `include` forgets its records on the next run.
- **Tombstones:** forgotten records are emitted as `{"id": ..., "_deleted": True}` with a
  `hard_delete` column. dlt removes them on `merge`, and cognee's orphan cleanup removes them
  from the graph and vector stores.
- **Capped listings:** Apollo search returns at most 50,000 records per listing. A bigger
  workspace has to be narrowed with stage or list filters. Until then the run reports
  `failed_cap` and **never detects deletions**, because a capped listing is incomplete.
- **A record whose check keeps failing backs off.** A record the key cannot read, or one Apollo
  answers with an error it does not call "deleted", is checked again every cycle (one cycle is
  one complete listing, normally one run). After five failed checks in a row it is left alone for
  ten cycles: no request, no `failed` count, but it is counted in `held_back`. Then it is tried
  once more. Another failure starts the ten cycles again and a success clears it. The record
  stays in memory the whole time, unless you leave its kind out of `include`: that forgets it at
  once. One that is listed again drops its count.
- **Rate limits:** a run never waits out a rate limit. When Apollo answers 429, fewer than 10
  calls are left in any window of the endpoint the run is about to call, or `max_requests` is
  spent, the run stops cleanly, keeps its progress and reports why in `cognee_sync_stats`. The next
  run resumes at the page it stopped on. A key that is rejected after the run emitted rows ends it
  the same way (`failed_auth`); one rejected before any row raises `ApolloAuthError`, and dlt drops
  the state of that run.
- **A refused listing raises:** an HTTP 403 on a search or a lookup (a team without access to that
  endpoint) raises `ApolloAccessError` and names the endpoint, instead of cutting the run. A cut
  run keeps its state, so a listing that already finished would stay finished, the refused one
  would never finish, the cycle would never close and no later delete would be seen. dlt drops
  the state of a run that raises, so the next run lists again, and nothing is forgotten.
- **What a host reads after a run that raises:** every raise adds one to `failed`. A 403 also adds
  one to `failed_access` and a key rejected before any row one to `failed_auth`. Any other error
  that raises (a 5xx, a filter id Apollo does not list, another team's state) sets no reason key.
  A run that is cut adds one to `failed` too, so one cut after a failed record check counts both.
- **A redirect is an error.** An answer with a 3xx status raises `ApolloAPIError` naming the
  status and the endpoint, and is not followed: the request carries the API key in a header that
  `requests` would send to whatever host the answer names, unlike an `Authorization` header, which
  it drops on a hop to another host.
- **A record id is checked before it is read.** A deletion check reads `/contacts/{id}` or
  `/accounts/{id}` with an id that came out of an Apollo answer. An id with anything but letters
  and digits (`..`, a slash, `?`) is not requested: the check fails like any other one that fails,
  and the record is kept. The check protects the request only. The id stays as Apollo sent it in
  the row id and in the `url` column, where it is never requested.
- `check_active` (for hosts) runs before every API request.

A run reads only the lookups its kinds render. An accounts-only run needs no sequence list, so a
plan or a scoped key without `emailer_campaigns/search` still syncs accounts. Contacts name their
sequences, so a contacts run without that endpoint stops with an `ApolloAccessError`. A run costs
a few lookup requests (stages, lists, the sequence list), one request per 100 records of each
included kind, and one per record that stopped appearing. `cognee_sync_stats["requests"]` says
how many calls a run made.

## Rate limits

Apollo limits requests per team and per endpoint in three windows: minute, hour and day. All
keys and users of a team share them. On the plans Apollo lists today a paid plan allows 200
requests a minute, 400 to 600 an hour and 2,000 to 6,000 a day for the endpoints this connector
reads (Apollo's rate-limits page, August 2026; legacy plans differ). The client reads the windows
from every answer that carries them, an error answer included, and keeps them per endpoint:
`/contacts/search` and `/contacts/{id}` are two. An answer without them leaves the windows last
seen for its endpoint. It is not confirmed that Apollo sends them on a 404 or a 422; if it does
not, record reads are limited by `max_requests` alone. A run stops when the endpoint it is about
to call has fewer than 10 calls left in any of the three windows, or at `max_requests`, and what
another endpoint has left does not matter. A large workspace takes several runs.

## Privacy

This connector reads personal data about people outside your organisation. Be clear about what
it does:

**What is read**

- From your team's own Apollo records only: contact names, job titles and companies; account
  names and domains; the stages and lists of both; the names and statuses of the sequences a
  contact is in; sequence names, status and step counts; and the custom fields you list in
  `custom_field_ids`.
- Received but never rendered or stored: Apollo's contact search answers with each contact's
  email, phone numbers and LinkedIn URL, and the connector drops them before it renders a row.
- Not read: Apollo's enrichment database (people and organization search) and the activity feed.
  The connector never creates, changes or deletes anything in Apollo.

**Where it goes**

- The text above becomes cognee documents. During cognify it is sent to your configured LLM and
  embedding providers, like any document you add to cognee. That is third-party prospect data,
  so check that you may send it there, or use local models.
- dlt state stores the Apollo team id, each record's id with a fingerprint (a hash of its
  rendered text), the ids waiting for a deletion check, and the failure counts of records in
  back-off. It stores no names or text. Document ids contain the Apollo record id.
- dlt's staging database for the dataset (`dlt_database_<dataset>`) holds the whole rendered text
  of every record, names, titles and companies included, next to that state.
- dlt also keeps every completed load package, with the rendered text of each row it loaded, in
  its data directory (`~/.dlt` of the server process unless `DLT_DATA_DIR` moves it): dlt does not
  delete completed jobs by default. That directory is outside `DATA_ROOT_DIRECTORY` and
  `SYSTEM_ROOT_DIRECTORY`.
- Logs and `cognee_sync_stats` contain counts, statuses and endpoints. Record ids are replaced
  by `{id}` in error messages, and the key is never logged, stored or put in an error: the client
  refuses a key that `requests` would reject (it names a header value it rejects), and a
  redirect is never followed. `raise ... from None` hides the cause of an error but not its
  `__context__`, which dlt's trace file records, so that check is what protects the key. The
  dataset itself holds the names above: anyone with read access to it can see them, so keep it in
  its own dataset and share it only with people allowed to see the CRM.

**How deletion is honoured**

- A contact or account deleted in Apollo (404 or the legacy "deleted" answer), or one that left
  the selected stages or lists, is forgotten on the **next sync** that checks it. Run syncs in
  the foreground: forgetting happens as part of `remember`.
- A sequence that is gone, and every record of a kind you leave out of `include`, is forgotten
  on the next sync.
- A workspace of **more than 50,000 records in one listing never detects deletions** until it is
  narrowed with filters (`failed_cap`).
- A record whose check keeps failing stays in memory until a check succeeds.
- **A deleted prospect stays on disk.** The sync removes the document, the graph content, the
  vectors and the staged row (the tombstone replaces it), but not the completed load packages
  above, which keep the text of every load. Checked with dlt 1.30.0 on a SQLite staging
  database: after a contact was deleted and synced, the package of the sync that first loaded it
  still held its name, and SQLite had not overwritten the freed pages of the staging file. This
  is a known limit of cognee's dlt ingestion, not of this connector. To erase a prospect from the
  server, delete the load packages under dlt's data directory and clear the staging database
  yourself.
- **`cognee.forget` does not erase a prospect for good.** It removes the documents, the graph
  and the vectors, but not the staged copy of the rendered text or the dlt state. The next sync
  under the same `resource_name` reads the staged rows back and builds every document again,
  also after `forget(dataset=...)`. To stop a prospect coming back, delete it in Apollo, or move it
  out of the selected stages or lists, and sync.

## Limitations

- Needs API access to the search endpoints, as above.
- At most 50,000 records per kind and filter selection.
- Emails, phone numbers and social profiles are never ingested, and attachments, notes, tasks and
  calls are not read.
- Apollo reports a deleted record with a 404 or a legacy 422 text. The legacy text is matched
  whole, and until Apollo documents a deleted-record code, nothing else counts as deleted.

## Testing

```bash
uv run pytest cognee/tests/unit/tasks/connectors/ cognee/tests/integration/tasks/connectors/
```

No Apollo account or API key is needed. `cognee/tests/unit/tasks/connectors/fake_apollo.py` is
an in-memory Apollo API shaped like its responses: paged searches, per-record reads that answer
like a deleted record, rate-limit headers and stage rows with a team id. It answers no endpoint
the connector does not use, so a call to one fails the test. `test_apollo_forget.py` runs the
real cognee pipeline with the LLM and embeddings mocked.
