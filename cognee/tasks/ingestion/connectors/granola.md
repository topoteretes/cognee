# Granola SDK connector

`granola_source` turns the meeting notes a Granola API key can see in one scope into memory,
one document per note. It is a `dlt` resource you hand to `cognee.remember`, and the source a
host reaches through `_PROVIDERS["granola"]`.

Ported from the Cognee Cloud pod source (cloud-backend CLO-504,
`cognee-saas-pod/modules/integrations/sources/granola.py` at aa05be08, shipped in Cloud
v1.6.8-001 and v1.6.9). The row id, the rendered text, the node set and the bounds are the
pod's. The table names are not: the pod names its tables `granola_notes_<digest>`, the
`_PROVIDERS` prefix here is `granola`, so a host that moves from the pod source to this one
syncs every connection once more from the start. What changed is listed under "Changes from
the Cloud pod source".

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
| `scope` | required | `all` (every note the key sees, unfiled ones included), a folder id (every folder below it included) or a space id. Anything else is a `ValueError` at construction |
| `resource_name` | required | the staging table and the sync state, one per connection and scope. Two keys or two users must never share one: the dlt pipeline is named from the dataset name and this name only, so a shared name shares the state and one sync tombstones the other's notes. The state remembers its `scope`: another scope on the same name stops the extract before any request, with a `ValueError` as the cause (dlt raises it wrapped in `PipelineStepFailed`) |
| `api_key` | none | builds the client when `service` is omitted; surrounding whitespace is stripped; never read from the environment |
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
| `GET /v1/notes/{id}` then `GET /v1/notes/{id}/transcript?page_size=100[&cursor]` | a transcript too large to inline (413, or an answer over 8 MiB), or any 403 on the inline read (the bare note still carries the summary) |
| `GET /v1/notes?page_size=1` | to tell a failing note from an outage, when a note answers 5xx before anything else did; and after a 403 on a note, to tell the note from the account |
| `GET /v1/notes?page_size=30[&folder_id\|space_id][&cursor]` | the reconcile listing, at most once a day |
| `GET /v1/notes/{id}` | once per known note the reconcile listing did not show, to confirm it before it is forgotten |
| `GET /v1/notes/{id}?include=transcript` | once per note the reconcile listing shows and the table lacks, to import it |
| `GET /v1/notes?page_size=30&folder_id\|space_id&created_after&created_before` | when a note read by id (a retry, a reconcile candidate) cannot be placed by its own membership: the scope's listing for the seconds around its `created_at` |

Granola lists only notes with a generated summary, newest first by `created_at`. A folder's
listing includes every folder below it. When transcripts are disabled by the workspace admin,
Get note answers 200 with `transcript: null` (the note is imported with its summary) and the
transcript endpoint answers 403 (the same).

Each note is one row in the table named `resource_name`:

| Column | Value |
|---|---|
| `id` | the note id (`not_...`) |
| `title` | the note's title, `Untitled meeting` when it has none |
| `content` | the title, `Meeting date: <created_at>`, a blank line and the summary (Markdown, else text), then a blank line, `Transcript:` and one `<speaker>: <text>` line per transcript part (the speaker is the diarization label, else the attribution, else the source, else `Speaker`) |
| `url` | the note's `web_url`, only when it starts with `https://notes.granola.ai/` |
| `cognee_node_set` | `["granola:<scope>"]` |

### Frozen names

These freeze with the first SDK release: a change re-extracts every note on the user's LLM
key. Row id = note id; the render above; node set `granola:<scope>`; state keys `scope`,
`watermark`, `cursor`, `pending`, `next_cursor`, `cycle_start`, `note_failures`,
`key_revision`, `known`, `reconciled_at`, `reconcile_cursor`, `reconcile_seen`,
`reconcile_candidates`, `reconcile_new` and `include_transcript`; the `_PROVIDERS` prefix
`granola`. The stats keys below are read by hosts: a key may be added, never renamed or
removed.

## How sync and forget-on-delete work

**Incremental walk** (the pod's). A cycle starts by fetching again, by id, the notes in
`note_failures`; a retried note is imported only when it is still in the scope (see "In the
scope" below).
Then it lists what changed since `watermark` (`updated_after`), one page at a time; the page's
ids and stamps are kept in `pending`, so a cut run resumes at the unread note and never reads
a finished one again. When the walk ends, `watermark` becomes the cycle's start minus 5
minutes. A note that fails to read leaves the page for a retry by id; only its own errors (a
4xx, a malformed answer, a bare note over 8 MiB) use up its 3 attempts, a 5xx never does. An expired
listing cursor (HTTP 400) restarts the listing once from the watermark; a cursor that comes
back twice stops the run.

**Deletions.** A note listed with `deleted_at` (only workspace-wide keys see these) is
tombstoned at once. Everything else is caught by the **reconcile**: at most once every 24
hours, and only in a run whose incremental walk completed, the source lists the whole scope
(ids only, 30 a page, no `updated_after`). Every note in `known` (the note ids this table
holds) that the listing no longer shows is only a candidate, because a listing that pages
while notes change can skip a live note. Each candidate is read by id (`GET /v1/notes/{id}`,
no transcript) and tombstoned only when Granola confirms it: a 404, `deleted_at`, or a
membership that places it elsewhere (a note deleted, moved out, or no longer visible to the
key). A candidate whose membership cannot place it (a deeper folder, no membership in the
answer) is kept even when the scope's narrow listing does not show it, and counted in
`reconcile_unconfirmed`: that absence would rest on Granola's `created_at` filter alone. A note
that reads and is still in the scope stays in `known`. A note-level failure of the read (a 403
while the account still lists, another 4xx, a malformed answer, or a 5xx or timeout while
Granola answers the cheapest listing) keeps the note in `known` and the reconcile goes on, so
one note can never stall it; the next reconcile asks again. The
listing is carried across runs (`reconcile_cursor`, `reconcile_seen`), so a budget cut
resumes it, and an expired reconcile cursor restarts it from the first page once. The
candidates are carried too (`reconcile_candidates`): a 401, a 429, an account-level 403, an
outage or a spent bound stops the run as usual, deletes nothing still unconfirmed, and the
next run continues without listing again.

The listing also names the notes the table lacks (`reconcile_new`): a note shared with the
key's owner or moved into the scope without an edit keeps its `updated_at`, so the incremental
walk never lists it. Each is read and emitted like a walked note (a note-level failure leaves
it out of `known`, so the next reconcile names it again). This counts against the run's row
budget.

- Only a listing that finished names candidates; a cut or failed one deletes nothing.
- A complete listing that returns zero notes while notes are known deletes nothing and
  reports `failed_access` (a key that lost access looks exactly like that); the next run lists
  again.
- A note the walk emits while a reconcile is open counts as listed, and is no longer a
  candidate: it may have been created or moved in after the listing passed its place.
- A table synced before this port has an empty `known`, so its first reconcile deletes
  nothing and reads every listed note once (unchanged rows keep their identity, so nothing is
  extracted again). Notes removed before that are not tombstoned.

**In the scope.** A note read by id (a retry, a reconcile candidate) is placed by its own
membership when that can tell: `folder_membership` names the note's folder and that folder's
parent, which proves the note is directly in the selected folder or in a direct child, or in
no folder at all; `space_membership` names the spaces. A note in a deeper folder, or an answer
without the membership, is asked of the scope's own listing, filtered to the seconds around the
note's `created_at`: the note is in the scope when that listing shows it, so membership never
depends on the folder's depth.

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
`failed_rate_limit` (with `retry_after_seconds`), a 403 the account gets on every request
(subscription inactive, account suspended, IP allowlist) is `failed_access`, an outage (5xx or
no answer, twice) is `failed_unavailable`, a spent bound is `failed_budget`, and a listing
Granola refuses (another 4xx) or answers malformed (a page of no documented shape, a repeated
cursor, an id of no known shape) is `failed_listing`. A 403 on one note is that note's only
when the cheapest listing still answers; when that listing answers 401, 403 or 429, the run
stops for that reason and no note is charged.
What `check_active` raises stops the run as it is.

### Stats

`cognee_sync_stats` has every key from the start of a run:

| Key | Meaning |
|---|---|
| `scanned` | rows emitted, tombstones included |
| `skipped` | notes left for a retry by id, read and found outside the scope, or kept by the reconcile after a note-level failure |
| `deleted` | tombstones emitted |
| `failed` | notes the walk failed to read (a retry by id included) plus one when the run stopped early |
| `skipped_unavailable` | notes given up after their retries (an edit retries them) |
| `failed_auth`, `failed_rate_limit`, `failed_unavailable`, `failed_budget`, `failed_listing` | 1 when the run stopped for that reason |
| `failed_access` | 1 when the account is refused (a 403 on every request), or a complete reconcile listing was empty while notes are known |
| `transcript_too_large` | notes imported with their summary and without a transcript over a cap |
| `reconcile_unconfirmed` | reconcile candidates kept because nothing placed them either way (their membership cannot, and the narrow listing does not show them) |
| `retry_after_seconds` | the wait a 429 asked for (0 when none) |
| `requests` | requests the run made, a retried one once |

## Limits

All bounds are the Cloud pod's (CLO-504); changing one is a decision. The run bounds drop
nothing: a run that reaches one stops with `failed_budget` and the next run continues. Three
bounds drop data, and are counted: a transcript over a cap leaves the note in memory with its
summary and without that transcript (`transcript_too_large`); a note whose bare answer is over
8 MiB, like any note that fails 3 times on its own, is given up (`skipped_unavailable`) until
an edit lists it again; and while 500 notes already wait for a retry, a note that fails on its
own is given up at its first failure (`skipped_unavailable`), also until an edit.

| Bound | Value | Source |
|---|---|---|
| requests a run | 200 | CLO-504 |
| run deadline | 240 s | CLO-504 (inside the pod's 900 s lock wait) |
| rows a run, tombstones included | 200 | CLO-504 |
| pacing | 0.25 s between requests | Granola: 5 requests a second sustained, burst 25 in 5 s, per user or workspace |
| response body | 8 MiB | CLO-504; an inline transcript answer over it is read page by page, a bare note over it fails the note |
| transcript | 2,000,000 characters | CLO-504; a longer transcript is left out and the summary imported |
| retries of a note | 3 | CLO-504 |
| notes being retried | 500 | CLO-504 |
| retry of a request | once, at once, on a 5xx or no answer | CLO-504 |
| `Retry-After` honoured | never slept; reported, capped at 7 days | CLO-504 |

Cost: a first sync is one listing request per 10 notes plus one per note (plus transcript
pages): 1,500 notes are about 1,650 requests, about 8 runs at the 200-request budget and
about 7 minutes at 4 requests a second. The first reconcile runs right after the first
complete walk. An idle sync is one request. A reconcile is `ceil(N / 30)` requests (50 for
1,500 notes) plus one per known note the listing no longer shows and one per listed note the
table lacks (plus one scope listing for a note its membership cannot place), no LLM.

Known limitations:

- Overlapping scopes (a folder and its space, two folders sharing a note) import a note
  once per table.
- A note Granola moves to the trash may stay listed until it is purged (up to 30 days); then
  it stays in memory until that reconcile. Not verified on a live account.
- Whether Granola moves `updated_at` when it deletes a transcript on its own schedule is not
  verified; if it does not, the old transcript stays in memory until the note changes.
- A note that fails 3 times on its own (a 4xx, a malformed answer, a bare note over 8 MiB) is
  given up, as in the pod, until an edit lists it again with a fresh allowance; a reconcile
  imports it only while the table lacks it.
- With `urllib3` logging at DEBUG, its request lines would name each path, and so a note id;
  the client drops urllib3's lines about Granola's host. Other HTTP debugging (a proxy, a
  packet capture) still sees them.

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
carries a note id, a folder id or a URL (no error carries a response body, not even as its
`__context__`). The pending page, `known` and the reconcile queues hold note ids only.

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
- `resource_name` is required, as in the pod: `test_resource_name_is_required_so_two_keys_never_share_a_table`.
- Scope membership does not depend on folder depth (the scope's listing decides what the
  membership cannot): `test_a_skipped_candidate_whose_membership_cannot_tell_is_asked_of_the_scope_listing`,
  `test_a_retry_by_id_of_a_grandchild_folder_note_imports_it`.
- A note-level failure keeps a reconcile candidate and the reconcile finishes:
  `test_a_candidate_refused_alone_is_kept_and_the_reconcile_still_finishes`.
- An account-level 403 stops the run with `failed_access` and charges no note:
  `test_a_refused_account_mid_walk_stops_with_failed_access_and_charges_no_note`.
- The reconcile imports the notes it finds the table lacks:
  `test_a_note_moved_in_without_an_edit_is_read_and_emitted_by_the_reconcile`.
- A transcript over a cap leaves the summary in: `test_a_transcript_over_the_cap_imports_the_summary_and_is_counted`.
- No error carries a body as `__context__`, and urllib3's DEBUG lines about Granola are
  dropped: `test_an_unreadable_body_is_no_errors_context`,
  `test_urllib3_debug_lines_about_granola_never_name_a_note`.
- A refused or malformed listing is `failed_listing`:
  `test_a_listing_granola_answers_malformed_stops_with_failed_listing`.
- A reconcile note that keeps answering 5xx while Granola otherwise answers is kept and the
  reconcile finishes; during an outage the run still stops:
  `test_a_candidate_that_keeps_failing_with_a_5xx_is_kept_and_the_reconcile_finishes`,
  `test_a_new_note_that_keeps_failing_with_a_5xx_waits_for_the_next_reconcile`,
  `test_a_candidate_5xx_during_an_outage_still_stops_the_run_and_keeps_the_queue`.
- The state remembers its scope, so two scopes on one `resource_name` cannot tombstone each
  other's notes: `test_a_state_written_for_another_scope_is_refused`.
- Scope and note ids are ASCII with a lowercase prefix:
  `test_scope_ids_are_ascii_with_a_lowercase_prefix`.
- A repeated cursor on a note's own narrow listing is charged to that note:
  `test_a_retried_note_whose_scope_listing_repeats_a_cursor_is_charged`.
- `max_rows` bounds the reconcile's tombstones too: `test_max_rows_bounds_the_reconcile_tombstones_too`.
- A note's 403 while the key is revoked stops with `failed_auth`:
  `test_a_note_403_while_the_key_is_revoked_stops_with_failed_auth_and_charges_nothing`.
- `transcript_too_large` counts imported notes only:
  `test_a_transcript_over_the_cap_counts_only_a_note_that_is_imported`.
- The streamed read is tested over a real socket: `test_the_streamed_read_over_a_real_socket`.

## For the Cloud ticket: counters Cloud drops

Cloud's `_COUNT_KEYS` (cognee-saas-backend `resource_health.py`) keeps only the keys it knows;
these are dropped today and each needs a line in the Cloud ticket:

| Key | What it should tell the user |
|---|---|
| `failed_access` | "Granola refused this key's account (subscription, IP allowlist), or listed no note in this folder while notes are synced; nothing was removed" |
| `failed_listing` | "Granola refused or garbled the folder's listing; the next sync tries again" |
| `transcript_too_large` | "N meetings were imported without their transcript, which is over the size limit" |
| `reconcile_unconfirmed` | "N notes no longer in this folder's listing could not be confirmed as moved or deleted, so they stay in memory" |
| `requests` | diagnostic only (the run's request count) |

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

To verify on a live account: that `GET /v1/notes?folder_id=...&created_after=...&created_before=...`
around a note's `created_at` (as Get note returns it) lists that note, for a note in a
grandchild folder and for one directly in the folder; until then a candidate the narrow
listing misses is kept (`reconcile_unconfirmed`), never tombstoned.
