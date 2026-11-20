# Evernote SDK connector

`evernote_source` turns the notes of one Evernote account into cognee memory: one document per
note, re-read only when the note changes, forgotten when it is trashed or deleted at Evernote.

Ported from the cognee-community connector (topoteretes/cognee-community#559, commit 25dc254,
blvckShawn). The OAuth helper of that package is not part of this port.

## Requirements

- `pip install "cognee[evernote]"` for Evernote's Thrift client. It is imported only when a sync
  makes its first call; importing `cognee.tasks.ingestion.connectors` loads nothing.
- A token: a developer token for your own account, or an OAuth 1.0a access token your application
  obtained from Evernote. This connector runs no OAuth flow and never reads the environment.

### SDK only

Evernote authorizes third-party applications with OAuth 1.0a. Cognee Cloud connects providers
through OAuth 2 only, so Evernote has no Cloud connect flow. `_PROVIDERS["evernote"]` exists (the
service builder takes the token as it is), but no Cloud host can obtain one until Cloud supports
OAuth 1.0a.

## Usage

```python
import cognee
from cognee.tasks.ingestion.connectors import evernote_source

source = evernote_source(
    resource_name="evernote_notes",      # one per account, required
    token="S=s1:U=...:H=...",
    notebook_names=["Work"],             # optional, by name
    tag_names=["project-x"],             # optional, by name
)
await cognee.remember(
    source,
    dataset_name="evernote",
    primary_key="id",
    write_disposition="merge",           # REQUIRED: the default "replace" wipes the corpus
    max_rows_per_table=0,
    self_improvement=False,
)
print(source.cognee_sync_stats)
```

Other arguments: `sandbox=True` (sandbox.evernote.com), `full_resync=True` (read the whole account
again and forget what is gone), `max_notes_per_run` (note bodies per run, default `None`, no limit),
`service` (a pre-built client, `build_evernote_service(token)`), `check_active` (a host checkpoint
called before each request attempt; what it raises stops the run as it is). Example:
`examples/guides/evernote.py`.

## What is ingested

One row per note: `id` (`note:<guid>`), `title`, `content`, `_deleted`. The content is the note's
ENML as text (headings as `#`, list items as `- `, to-dos as `[x]` / `[ ]`, the cells of a
table row joined by ` | `), after a
`Notebook: <name>` line and a `Tags: <names>` line with the tags sorted case-insensitively. An
attachment becomes `[attachment: <mime type>]`, an encrypted section `[encrypted content]`;
attachment bodies, resources and note links are not read. A note with no text uses its title; a
note with neither is not stored. There is no `url` column: a note link needs the account's shard
and is not built.

Frozen at the first release: the row id (`note:<guid>`), the resource name the caller chose, the
state keys and the stats keys below.

## How sync and forget-on-delete work

Evernote numbers every change in an account with an update sequence number (USN). A run:

1. binds the resource to the account (`production:<user id>` or `sandbox:<user id>`); a state bound
   to another account or environment raises, so two accounts never share one table;
2. reads `getSyncState`; when `updateCount` equals the stored `last_usn`, nothing changed and the
   run goes straight to step 6;
3. pass 1: reads `getFilteredSyncChunk` from `last_usn` (from 0 on a full pass), 250 entries a
   call: notes' metadata, notebooks, tags and the expunged guids. Progress is kept after every
   chunk, so a run cut here continues from the last chunk next time;
4. once pass 1 is complete, emits the tombstones it proves and resolves the selection;
5. pass 2: fetches every changed note in the selection with its body (`getNote`, no resource
   data) in USN order and emits it.
   A cut run keeps what it emitted and the notes still to fetch; the next run continues there
   without reading pass 1 again;
6. the retry tail: reads the notes that failed in earlier runs (see Limits), after the pass is
   stored, so they never hold new notes and edits back.

Only when pass 2 is done are `last_usn`, `last_sync_time`, the notebook and tag maps and the
selection stored. A note edited while a sync is unfinished is read again by the next sync.

A full pass runs on the first sync, when the selection changes, when its names resolve to other
notebooks or tags, when a notebook was expunged, with `full_resync=True`, when `updateCount` is below the stored `last_usn`,
and when `fullSyncBefore` is newer than the last sync. **`last_usn` cannot expire on its own:
`fullSyncBefore` is Evernote's signal that it is no longer usable, and the run then lists the
whole account.** A saved pass 1 older than `fullSyncBefore`, or for another selection, is dropped
and started again. A full pass that is cut leaves no stored selection behind, so the next run is
full too, whatever selection it is given: a cut pass may already have forgotten notes the
previous selection held.

Deletion, only from a complete pass 1 and only for notes this resource emitted:

| At Evernote | In memory |
|---|---|
| note trashed or expunged | tombstone |
| notebook expunged | a full pass: tombstone for every emitted note it no longer lists |
| note moved out of the selected notebooks or tags | tombstone |
| note gone when its body is read | tombstone |
| note edited to no text and no title | tombstone |
| on a full pass, any emitted note not live and selected | tombstone |

A cut or failed pass 1 never deletes. `cognee.forget()` does not reach Evernote: a note forgotten
by hand comes back on the next sync while it is still at Evernote and its staging row is kept.

## Limits

- Rate limit: Evernote limits calls per API key and user per hour and does not publish the number.
  A `RATE_LIMIT_REACHED` is waited out while the waits of one call stay within 30 seconds (the
  community connector's bound; the sync holds cognee's process-wide dlt staging lock); a longer
  one ends the run cleanly with `failed_rate_limit`, and the next run continues.
- Transport errors are retried three times, after 1, 2 and 4 seconds.
- Transport errors that outlast the retries, and Evernote system errors (`INTERNAL_ERROR`,
  `SHARD_UNAVAILABLE`, ...), are "unavailable" answers. In pass 1 one ends the run cleanly with
  `failed_unavailable`, and the next run continues where it stopped. In pass 2 one on a single
  note sends that note to `retry`; two in a row are an outage: the run ends cleanly with
  `failed_unavailable`, those two notes go to `retry` and the rest stay pending for the next run.
  A run whose only read was one note that answered unavailable is an outage too: that note goes
  to `retry` the same way and the run reports `failed_unavailable` (so does a retry tail whose
  only read did so). A single note that always answers so therefore reads as an outage on runs
  with nothing else to read. So at most two notes a run move to `retry` during an outage, every
  outage run reports it, and two notes that always fail cannot keep the pass from completing. Nothing is lost: once
  Evernote answers, the retry tail reads them.
- First sync cost: one `getFilteredSyncChunk` per 250 changed entries plus one `getNote` per note
  in the selection, then cognify of every note.
- A note Evernote will not hand over (any user error on its `getNote` but a rejected token:
  `PERMISSION_DENIED` after a withdrawn share, `TAKEN_DOWN`, malformed data, an answer that
  cannot be decoded) is skipped and counted in `failed`; the sync moves on and the note's old
  document stays. Its guid is kept in `retry`.
- The retry tail runs after the pass is stored, every run, idle runs included, and reads the
  notes that were in `retry` when the run started. A note read there leaves `retry`; a note gone
  there is deleted. Any failure keeps it in `retry` and counts in `failed`; two unavailable
  answers in a row end the tail with `failed_unavailable`; a rate limit or a rejected token ends
  it cleanly with `failed_rate_limit` / `failed_auth`, never a raise, since the pass is already
  stored. The tail counts against `max_notes_per_run`; with notes waiting in `retry` and a cap of 2 or
  more, the pass stops one read short of the cap and that read goes to the tail, so a run whose
  new notes fill the cap still retries one note. The tail starts with the notes tried least
  recently (a note that fails again goes to the back), so failing notes cannot use up the budget
  ahead of one that would be read. A pass that lists a waiting note again (it
  changed) or proves it gone settles it instead, and a full pass settles them all. A note in
  `retry` is deleted only on evidence: Evernote reports it expunged or not found, or a full pass
  no longer lists it. The failure count has no reason key of its own: Cloud's `_COUNT_KEYS` has
  none that fits.
- State size: while a sync is unfinished, the state keeps a guid, a USN and a notebook guid per
  pending note, no text: about 8 MB of JSON for 100,000 pending notes (16.6 MB when titles were
  kept), and the guid to notebook map of every emitted note once it is done.
- `resource_name` must be a dlt table name (`work_notes`, not `Work Notes`): dlt would store both
  spellings in one table under two account binds. A `service` built for the sandbox needs
  `sandbox=True` here too: the bind names the environment.

### Known limitations

- **Selection by name.** Renaming a selected notebook or tag fails the next run with a message
  that names the missing name; update `notebook_names` / `tag_names`.
- **Renamed notebooks and tags stay in the text.** The `Notebook:` and `Tags:` lines carry names,
  so after a rename a note keeps the old name until the note itself changes.
- **SDK only** (see above).

## Stats

`source.cognee_sync_stats` after a run, every key present on every exit: `emitted` (note
documents), `deleted` (tombstones), `failed`, `failed_rate_limit`, `failed_auth`,
`failed_unavailable`. A note that could not be read adds one to `failed` and nothing else. A run
cut by the rate limit sets `failed` and `failed_rate_limit`, one cut by an outage `failed` and
`failed_unavailable`. A token Evernote rejects (`INVALID_AUTH`, `AUTH_EXPIRED`) sets `failed` and
`failed_auth`; a `PERMISSION_DENIED` on any call but a note's body sets `failed` alone. Before any
row both raise (`EvernoteAuthError`, `EvernotePermissionError`); after rows the run ends cleanly
and keeps what was emitted. A cut by `max_notes_per_run` is no failure.

## Privacy

The notes' text goes to your LLM and embedding providers during cognify. Errors and log lines carry
no token, note guid, URL or response text. dlt keeps the rendered rows in its staging database and
load packages under `~/.dlt` of the process, and `cognee.forget()` does not remove them; a note
deleted at Evernote leaves memory with the next sync.

## Testing

Unit (no network, no Evernote library; the client tests replace the Thrift classes):
`cognee/tests/unit/tasks/connectors/test_evernote_source.py`, `test_evernote_client.py`,
`test_evernote_content.py`. Integration through the real pipeline with the account faked:
`cognee/tests/integration/tasks/connectors/test_evernote_ingestion.py` and
`test_evernote_forget.py`.
