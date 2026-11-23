# Granola SDK connector

`granola_source` turns the meeting notes a Granola API key can see in one scope into memory,
one document per note. It is a `dlt` resource you hand to `cognee.remember`, and the source a
host reaches through `_PROVIDERS["granola"]`.

Ported from the Cognee Cloud pod source (cloud-backend CLO-504,
`cognee-saas-pod/modules/integrations/sources/granola.py` at aa05be08, shipped in Cloud
v1.6.8-001 and v1.6.9). The row id, the rendered text, the node set, the state keys and the
bounds are the pod's, unchanged, so the pod can switch to this source without re-extracting a
note. What changed is listed under "Changes from the Cloud pod source".

## Requirements

- Cognee with `DOCUMENT_SYNC_VERSION >= 3`. No extra install: the client uses `requests`,
  which `dlt` already brings.
- A Granola API key (`grn_...`, Granola: Settings, Connectors, API keys). Granola's REST API
  is for Business and Enterprise plans. A personal key and a workspace key both work; the
  source sends either as `Authorization: Bearer` and does not care which. Granola has no
  OAuth for its REST API (only for its MCP server).
- The key is passed through an auth object, so a `~/.netrc` entry for
  `public-api.granola.ai` never replaces it. Proxies and CA bundles still apply as requests
  reads them (`HTTPS_PROXY`, `REQUESTS_CA_BUNDLE`).

## Usage

```python
import cognee
from cognee.tasks.ingestion.connectors import granola_source

source = granola_source(
    scope="fol_...",  # "all", a folder id (fol_...) or a space id (spa_...)
    api_key="grn_...",
    resource_name="granola_notes_acme_planning",  # one per connection and scope
)
await cognee.remember(
    source,
    dataset_name="granola_acme",
    write_disposition="merge",  # REQUIRED: the add pipeline defaults to "replace"
    primary_key="id",
    max_rows_per_table=0,
    self_improvement=False,
)
print(source.cognee_sync_stats)
```

Pass the same `scope`, `resource_name` and `remember` arguments on every sync. Several scopes
go into one dataset with one source and one `resource_name` each.

Arguments, all keywords:

| Argument | Default | Meaning |
|---|---|---|
| `scope` | required | `all` (every note the key sees, unfiled ones included), a folder id (its child folders included) or a space id. Anything else is a `ValueError` at construction |
| `resource_name` | `granola_notes_<scope lowercased>` | the staging table and the sync state. The default is shared by two keys that sync the same scope into datasets of the same name, so hosts pass their own (Cloud: `granola_notes_<identity>_<scope>`) |
| `api_key` | none | builds the client when `service` is omitted; never read from the environment |
| `service` | none | a pre-built client (`build_granola_service(api_key)`); it must accept `get(path, params, *, check_active, time_left)` and call `check_active` before each attempt |
| `key_revision` | none | any string that changes when the key is replaced; a change starts the walk over |
| `include_transcript` | `True` | `False` never asks Granola for a transcript and renders the note without its `Transcript:` block. A change from the previous sync lists the whole scope again, so every synced note is rendered again with or without its transcript |
| `check_active` | none | the host's checkpoint, called before every request attempt (by the client) and around every row |
| `max_requests`, `deadline_seconds`, `max_rows` | 200, 240, 200 | bounds of one run (below) |

Account id: Granola has no endpoint that names the account a key belongs to (no whoami), so
this source offers no `granola_account_id`. A host mints the connection's identity itself;
Cloud uses the Cognee user id, scoped by workspace.

## What is read

| Call | When |
|---|---|
| `GET /v1/notes?page_size=10[&folder_id\|space_id][&updated_after][&cursor]` | the incremental walk, every run |
| `GET /v1/notes/{id}?include=transcript` | each listed note (without `include` when `include_transcript=False`) |
| `GET /v1/notes/{id}` then `GET /v1/notes/{id}/transcript?page_size=100[&cursor]` | a transcript too large to inline (413) or refused inline (403) |
| `GET /v1/notes?page_size=1` | once, to tell a failing note from an outage, when a note answers 5xx before anything else did |
| `GET /v1/notes?page_size=30[&folder_id\|space_id][&cursor]` | the reconcile listing, at most once a day |
| `GET /v1/notes/{id}` | once per known note the reconcile listing did not show, to confirm it before it is forgotten |

Granola lists only notes with a generated summary, newest first by `created_at`.

Each note is one row in the table named `resource_name`:

| Column | Value |
|---|---|
| `id` | the note id (`not_...`) |
| `title` | the note's title, `Untitled meeting` when it has none |
| `content` | the title, `Meeting date: <created_at>`, a blank line and the summary (Markdown, else text), then a blank line, `Transcript:` and one `<speaker>: <text>` line per transcript part (the speaker is the diarization label, else the attribution, else the source, else `Speaker`) |
| `url` | the note's `web_url`, only when it starts with `https://notes.granola.ai/` |
| `cognee_node_set` | `["granola:<scope>"]` |

### Frozen names

These are the Cloud pod's and must not change: a change re-extracts every note on the tenant's
LLM key. Row id = note id; the render above; node set `granola:<scope>`; state keys
`watermark`, `cursor`, `pending`, `next_cursor`, `cycle_start`, `note_failures`,
`key_revision`, and since this port `known`, `reconciled_at`, `reconcile_cursor`,
`reconcile_seen`, `reconcile_candidates` and `include_transcript`; the stats keys below; the
`_PROVIDERS` prefix `granola`.

## How sync and forget-on-delete work

**Incremental walk** (the pod's, unchanged). A cycle starts by fetching again, by id, the
notes in `note_failures`; a retried note is imported only when its `folder_membership` or
`space_membership` still places it in the scope (directly, or in a direct child folder).
Then it lists what changed since `watermark` (`updated_after`), one page at a time; the page's
ids and stamps are kept in `pending`, so a cut run resumes at the unread note and never reads
a finished one again. When the walk ends, `watermark` becomes the cycle's start minus 5
minutes. A note that fails to read leaves the page for a retry by id; only its own errors (a
4xx, a malformed answer, the size cap) use up its 3 attempts, a 5xx never does. An expired
listing cursor (HTTP 400) restarts the listing once from the watermark; a cursor that comes
back twice stops the run.

**Deletions.** A note listed with `deleted_at` (only workspace-wide keys see these) is
tombstoned at once. Everything else is caught by the **reconcile**: at most once every 24
hours, and only in a run whose incremental walk completed, the source lists the whole scope
(ids only, 30 a page, no `updated_after`). Every note in `known` (the note ids this table
holds) that the listing no longer shows is only a candidate, because a listing that pages
while notes change can skip a live note. Each candidate is read by id (`GET /v1/notes/{id}`,
no transcript) and tombstoned only when Granola confirms it: a 404, `deleted_at`, or a
`folder_membership` / `space_membership` that no longer places it in the scope (a note
deleted, moved out, or no longer visible to the key). A note that reads and is still in the
scope stays in `known`. The listing is carried across runs (`reconcile_cursor`,
`reconcile_seen`), so a budget cut resumes it, and an expired reconcile cursor restarts it
from the first page once. The candidates are carried too (`reconcile_candidates`): any other
answer to a confirmation (401, 429, 5xx, a malformed answer, a spent bound) stops the run as
usual, deletes nothing that is still unconfirmed, and the next run continues the
confirmations without listing again.

- Only a listing that finished names candidates; a cut or failed one deletes nothing.
- A complete listing that returns zero notes while notes are known deletes nothing and
  reports `failed_access` (a key that lost access looks exactly like that); the next run lists
  again.
- A note the walk emits while a reconcile is open counts as listed, and is no longer a
  candidate: it may have been created or moved in after the listing passed its place.
- A table synced before this port has an empty `known`, so its first reconcile deletes
  nothing and only fills `known` from the listing. Notes removed before that are not
  tombstoned.

**Key replaced.** A new `key_revision` clears the walk's and the reconcile's positions and
lists from the start (a new key may see older notes). Candidates the old key had not
confirmed go back into `known`. It deletes nothing by itself; a later reconcile with the new
key forgets what that key confirms is gone.

**Transcripts switched on or off.** The source keeps the `include_transcript` it last synced
with (a table synced before the option existed counts as `True`, as the pod always rendered
transcripts). When the option changes, the walk's positions (`watermark`, `cursor`,
`pending`, `next_cursor`, `cycle_start`) start over, so every note in the scope is listed and
rendered again: turning it off drops every synced transcript from memory, turning it on adds
them. Nothing is deleted by the change, and the reconcile state is left as it is. The cost is
a first sync's (one request per note).

**Errors.** No Granola failure raises: the run stops, keeps what it emitted and its state,
and the next run resumes. The reason is in the stats: 401 is `failed_auth`, 429 is
`failed_rate_limit` (with `retry_after_seconds`), an outage (5xx or no answer, twice) is
`failed_unavailable`, a spent bound is `failed_budget`. A listing Granola refuses (another
4xx) or answers malformed counts in `failed` with no reason key. What `check_active` raises
stops the run as it is.

### Stats

`cognee_sync_stats` has every key from the start of a run:

| Key | Meaning |
|---|---|
| `scanned` | rows emitted, tombstones included |
| `skipped` | notes left for a retry by id, or read and found outside the scope |
| `deleted` | tombstones emitted |
| `failed` | failed notes plus one when the run stopped early |
| `skipped_unavailable` | notes given up after their retries (an edit retries them) |
| `failed_auth`, `failed_rate_limit`, `failed_unavailable`, `failed_budget` | 1 when the run stopped for that reason |
| `failed_access` | 1 when a complete reconcile listing was empty while notes are known |
| `retry_after_seconds` | the wait a 429 asked for (0 when none) |
| `requests` | requests the run made, a retried one once |

## Limits

All bounds are the Cloud pod's (CLO-504); changing one is a decision. None drops data: a run
that reaches one stops with `failed_budget` and the next run continues.

| Bound | Value | Source |
|---|---|---|
| requests a run | 200 | CLO-504 |
| run deadline | 240 s | CLO-504 (inside the pod's 900 s lock wait) |
| rows a run | 200 | CLO-504 |
| pacing | 0.25 s between requests | Granola: 5 requests a second sustained, burst 25 in 5 s, per user or workspace |
| response body | 8 MiB | CLO-504 |
| transcript | 2,000,000 characters | CLO-504; a longer transcript fails its note (retried, then given up) |
| retries of a note | 3 | CLO-504 |
| notes being retried | 500 | CLO-504 |
| retry of a request | once, at once, on a 5xx or no answer | CLO-504 |
| `Retry-After` honoured | never slept; reported, capped at 7 days | CLO-504 |

Cost: a first sync is one listing request per 10 notes plus one per note (plus transcript
pages): 1,500 notes are about 1,650 requests, about 8 runs at the 200-request budget and
about 7 minutes at 4 requests a second. The first reconcile runs right after the first
complete walk. An idle sync is one request. A reconcile is `ceil(N / 30)` requests (50 for
1,500 notes) plus one per known note the listing no longer shows, no LLM.

Known limitations:

- Overlapping scopes (a folder and its space, two folders sharing a note) import a note
  once per table.
- A note Granola moves to the trash may stay listed until it is purged (up to 30 days); then
  it stays in memory until that reconcile. Not verified on a live account.
- Whether Granola moves `updated_at` when it deletes a transcript on its own schedule is not
  verified; if it does not, the old transcript stays in memory until the note changes.
- A note in a folder deeper than a direct child of the selected folder is listed by Granola,
  but a retry by id of such a note treats it as outside the scope.

## Forget and its gaps

`forget(data_id=...)`, `forget(dataset=...)` and `forget(memory_only=True)` remove memory, not
the dlt staging copy or the state, so the next sync brings back what Granola still lists. To
erase, remove the note at Granola or narrow the selection (the host retires the table with
`ingestion.empty_resource`).

## Privacy

Only the fields of the render are read into a row: private scratch notes (`private_notes_*`),
attachments, owners and attendee emails never are. The note text goes to the configured LLM
and embedding providers when cognify runs. The key lives only in the client object; it is
never in the dlt state, a row, an error message or a log line, and no error or log line
carries a note id, a folder id or a URL. The pending page and `known` hold note ids only.

## Changes from the Cloud pod source

Each with its test in `cognee/tests/unit/tasks/connectors/test_granola_source.py` or
`test_granola_client.py`:

- The daily reconcile and the `known` set (contract P3): `test_a_note_moved_out_of_the_folder_is_tombstoned_by_the_next_reconcile` and the reconcile tests next to it.
- The reconcile confirms each candidate by id before it forgets it, and carries unconfirmed
  candidates across runs: `test_a_live_note_the_paged_listing_skipped_is_confirmed_and_kept`,
  `test_a_401_while_confirming_deletes_nothing_unconfirmed_and_the_next_run_resumes` and the
  tests next to them.
- `include_transcript`: `test_include_transcript_off_never_asks_for_one_and_renders_without_it`;
  a change renders every synced note again:
  `test_turning_transcripts_off_and_on_renders_every_synced_note_again`.
- The request budget and the deadline sit in front of the client, so a host-built `service`
  is bounded too (the pod ignored its budgets for an injected client):
  `test_a_run_cut_by_a_budget_resumes_without_losing_a_note`.
- `check_active` before every request attempt, a retry included, through
  `dlt_utils.with_retries`: `test_check_active_runs_before_every_attempt_retries_included`.
- The key goes through a requests auth object, so `~/.netrc` cannot replace it:
  `test_a_netrc_entry_for_the_host_never_replaces_the_bearer_key`.
- Log lines no longer name a note or a scope: `test_no_log_line_names_a_note_or_the_scope`.
- Every stats key is present from the start, with `failed_access` and `requests` added:
  `test_the_factory_names_the_table_from_the_scope_and_tags_the_source`; a stopped run adds
  one to `failed` instead of `max(1, failed)`: `test_a_cut_after_a_failed_note_counts_both_in_failed`.
- A nullable `url` column hint, so the first note with a URL does not change the identity of
  older rows without one: `test_the_url_column_is_declared_so_a_first_url_changes_no_older_row`.
- A first 5xx response is closed before the retry: `test_a_5xx_answer_is_closed_before_the_retry`.

## Testing

```bash
uv run pytest cognee/tests/unit/tasks/connectors/test_granola_source.py \
  cognee/tests/unit/tasks/connectors/test_granola_client.py \
  cognee/tests/unit/modules/integrations/test_ingestion_contract.py
uv run pytest cognee/tests/integration/tasks/connectors/test_granola_ingestion.py \
  cognee/tests/integration/tasks/connectors/test_granola_forget.py
```

The integration tests mock the LLM and the embeddings and need no key. No test contacts
Granola; a live run (connect, sync, edit, move, delete, revoke) is still to be done.
