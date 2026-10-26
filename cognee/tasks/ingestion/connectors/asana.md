# Asana SDK connector

An Asana data-source connector for [cognee](https://github.com/topoteretes/cognee): sync the
tasks of the projects you select into memory, incrementally, and forget a task when it leaves
the project.

It exposes a `dlt` source you hand to `cognee.remember(...)`. Each task and each selected
project becomes a **normal document** in cognee (it flows through chunking and LLM entity
extraction), via cognee's document-mode marker. One source is one access token and a selection
of projects.

## Requirements

Bundled with the Cognee SDK (a core that does not read per-row node sets,
`DOCUMENT_SYNC_VERSION` below 2, is refused). No extra and no separate package is required.
The REST client uses `requests`, which `dlt` (a core dependency) already requires.

The token needs read access to the projects and their tasks, comments (stories), subtasks and
events. A personal access token has the access of the user who created it. This port has not been
tested against a live Asana workspace: the fake it is tested with follows the behaviour the
community connector recorded (see Testing).

## Setup

1. Create a personal access token in the Asana developer console, or use an OAuth access token.
   The connector never refreshes a token: a host hands over a fresh one.
2. Collect the gids of the projects to sync (the number in a project's URL). A workspace-wide
   sync is a list of its project gids; there is no "whole workspace" mode, so an empty selection
   can never mean everything.
3. Pass them as `project_gids` and the token as `access_token`. Set your `LLM_API_KEY` like any
   other cognee run.

| Endpoint | Used for |
|---|---|
| `GET /projects/{gid}` | the project document, once for each project a run lists or resumes (not again for a project a kept pass already finished) |
| `GET /events?resource=<project>` | what `modified_at` does not show, see below |
| `GET /tasks?project=<project>` | the sweep: every task with its `modified_at` |
| `GET /tasks/{gid}`, `/tasks/{gid}/stories`, `/tasks/{gid}/subtasks` | a task that changed |
| `GET /tasks/{gid}` (memberships only) | the check of a known task the listing lacks, see "Checking a missing task" |

## Usage

```python
import cognee
from cognee.tasks.ingestion.connectors import asana_source

await cognee.remember(
    asana_source(
        resource_name="asana_website",
        access_token="<personal access token>",
        project_gids=["1201234567890123"],
    ),
    dataset_name="asana",
    primary_key="id",
    write_disposition="merge",  # required: incremental upserts and deletes
    max_rows_per_table=0,  # no row cap; orphan cleanup must see the whole corpus
    self_improvement=False,
)

answer = await cognee.recall("Who is working on the launch post?", datasets=["asana"])
```

Run the same `remember` call again whenever you want to sync.

| Argument | Meaning | Default |
|---|---|---|
| `resource_name` | Names this connection: it names the staging table and keys the sync state. A non-blank string that does not start with `_dlt_` (dlt's own tables) | required, one per connection |
| `project_gids` | The projects to sync: a non-empty list or tuple of gid strings | required |
| `access_token` | Asana personal access token or OAuth access token. Never read from the environment | needed unless `service` is given |
| `max_requests` | Budget of all the requests of one run, the listing's and the task reads' (three a task). An int of at least 3 | 1500 |
| `check_active` | Host checkpoint, called before each attempt of each request, a retry included. What it raises stops the run as it is | none |
| `service` | Pre-built client, from `build_asana_service(access_token)`. A client of your own must accept `request(method, path, *, params, check_active)` and call `check_active` before each attempt it makes | built from `access_token` |

`project_gids` is sorted and repeats are dropped, so the same selection in another order is the
same selection. A bare string, an empty list, a blank gid or a gid with anything but letters and
digits raises `ValueError` when the source is built. Changing the selection is allowed, see
below. Two connections in one dataset need two `resource_name`s: they would share state and each
sync would forget the other's tasks. The hosted names are `asana_<24 hex>`.

## What is ingested

| Row | Id | Title | Content | Node set |
|---|---|---|---|---|
| Task | `task:<gid>` | the task name | completed or not and the day it was completed, assignee name, due date, the selected projects it is in with its section, custom fields, the description, subtasks, comments | `asana:<gid>` for every selected project it is in, sorted |
| Project | `project:<gid>` | the project name | the project description | `asana:<its own gid>` |

```
Completed: yes
Completed on: 2026-10-14
Assignee: Ada
Due: 2026-10-20
Project: Website launch (section: In progress)
Priority: High

Draft, then review with marketing.

Subtasks:
- [x] Outline
- [ ] Screenshots
  Assignee: Grace

Comments:
- Grace: Looks good
```

A task is one document with its comments (oldest first) and its subtasks (in creation order) in
it, one level of subtasks. A task in two selected projects is one document that names both;
only **selected** projects are named, because a project outside the selection is never synced and
its rename would never be seen. Tasks and projects are separate rows of one table.

Every list is sorted and nothing volatile is rendered (no `modified_at`, no counters, no
time of day), so an unchanged task keeps its content hash and is not extracted again. People are
text: an assignee or a comment author is a name, not an entity, and no email address is requested
or rendered (a person can still type one into a comment or a description). System entries in a
task's history ("added to project") are not comments and are left out.

A completed task is synced like any other. While it is completed, the line `Completed on:
YYYY-MM-DD` follows `Completed:`: the UTC day of Asana's `completed_at`, never a time. Reopening
the task removes the line, which is a change of text and is extracted again. A `completed_at`
that cannot be read renders no line.

**Node set.** Every row carries `cognee_node_set`, so a `recall` can be scoped to one project
(`node_name=["asana:<project gid>"]`). A task gets `asana:<gid>` for every **selected** project
it is in, sorted; a project row gets `asana:<its own gid>`. Only gids are used, never names, so
renaming a project costs nothing: no node set changes and no row is extracted again. The node set
is part of the row's identity, so a task that joins or leaves a selected project, or whose
selection changes (deselecting one of two projects a task is in), is extracted once more. A task
that is in no selected project when it is read gets an empty list, which is not hashed. A
tombstone carries no node set. `cognee_structure` is not set.

### Frozen names

These are permanent once anyone has synced, because every change re-extracts every task on your
LLM key or orphans documents: the provider and table prefix `asana`, the row ids `task:<gid>` and
`project:<gid>`, the node set names `asana:<project gid>`, the state keys and the stats keys
below, and the rendered text, including the `Completed on:` line.

## How sync and forget-on-delete work

**Finding what changed.** Every run lists each selected project's tasks with their
`modified_at`. A task is read again when it is new to the dataset, or when its `modified_at` is
inside a window that starts a minute before the project's stored cursor and differs from the one
stored in `seen`. What `modified_at` does not show comes through the Events API, which keeps one
sync token per project: a comment that was added or edited (Asana leaves `modified_at` alone for
both), a subtask that was renamed or removed (the parent is found through a subtask-to-parent
index), and the name of a project or a section that changed, which re-reads every task of that
project. An expired or missing sync token (HTTP 412) also re-reads the whole project, as does
the very first run. Events are followed page by page until Asana says there are no more. A sync
token that comes back twice in one read of the feed while it says there are more raises instead
of looping, as a pagination offset that comes back twice does.

**Reading a task** costs three requests: the task, one page of comments and one page of
subtasks (more pages cost more). Fresh tasks are read first, sorted by gid as text, then the tasks
kept for a retry.

**Forgetting.** A task is emitted as `{"id": ..., "_deleted": True}` only when a complete listing
of the selected projects lacks it **and** Asana confirms it is gone, see "Checking a missing
task": deleted in Asana, moved out of the selected projects, or its project deselected. dlt
removes the row on `merge`, and cognee's orphan cleanup removes the document, graph content and
vectors. A run that fails, or is cut inside the listing, forgets nothing.

**A task that cannot be read.** A task that is still listed but answers 403, 404, a 5xx or
nothing (after three attempts, backing off 1 s then 2 s), or a 200 of the wrong shape, keeps its
**old document**. It is kept for a retry (`state["retry"]`), asked again at the end of the next
run's order, never forgotten and never given up on; it fails at most once per run. A task
answering 403 or 404 never stops another task from being read and never counts toward an outage.
An answer of 5xx or nothing counts in the streak of its own part of the run (the retry tail for a
retried task) and, three in a row, stops the run, see "A run that stops" and "Known gaps". A 200
of the wrong shape is told the same way and never counts toward an outage: a task body that is not
an object, a comment or a subtask that is not an object, a broken pagination of those two lists
(`next_page` without an offset, an offset that comes back twice), a `memberships` that is
malformed (see below) or a `custom_fields` that is not a list of fields. The task is counted in
`failed`, kept for a retry and never skipped silently, and the run does not raise, so one
malformed task never blocks the others. Only a task that leaves a complete listing and is
confirmed gone is forgotten. Known gap: a task Asana will not show you, or keeps answering with a
body of the wrong shape, keeps a stale document for as long as it stays listed, and one that
becomes readable again is read again with no change to `modified_at`.

**A task that left the selection** between the listing and its read (it was moved to a project
you did not select) is not rendered: no document is built for it. It stays known until the next
pass finds it missing and forgets it.

**A run that stops.** On a rate limit (HTTP 429), a spent request budget, an **outage** (three
tasks in a row answering 5xx or nothing; the fresh tasks, the retry tail and the checks of
missing tasks each count their own streak, a success resets it, and a 403 or 404 never counts,
so a task that keeps answering 403 or 404 never stops a run) or a rejected token after rows
were emitted, the run ends cleanly. It never sleeps on a 429: it does not read `Retry-After`,
because the walk holds cognee's process-wide staging lock and dlt rolls back the state of a
failed run.

- If a task read or a check of a missing task got an answer (any answer, a failure or a task
  that left the selection included), what the run emitted is committed, the tasks it did not
  reach go to `state["pending"]` and are read on the next run (before the retry tasks, sorted by
  gid with the tasks that changed), and the cursors and sync tokens advance. Tombstones it
  already emitted stay out; nothing else is forgotten.
- If it was cut before any answer (inside the listing, or right after it), it only keeps the
  listing, see below: `pending`, `retry`, the cursors and the tokens stay as the run before left
  them. Its `pending` and `kept` counts are then 0 while `state["pending"]` and
  `state["retry"]` may still hold tasks: the counts are the queue a run committed, 0 when it
  kept only the listing.

Errors raise instead of stopping the run. A rejected token before any row (HTTP 401) raises
`AsanaAuthError`. Any other failing answer raises `AsanaAPIError` with its status: a 5xx or an
unanswered request after the retries in the listing, the events feed or a project read, a
project read other than 403 or 404, a task check other than 403, 404, 5xx or nothing, and a
pagination offset Asana rejects (HTTP 400, see "Known gaps"). A task read or a check answering
5xx or nothing is kept instead, as described above. A 200 that is not the expected object or
list in a listing, the events feed or a project read raises `AsanaResponseError`, so does a check
of a missing task whose body is of the wrong shape or whose `memberships` is malformed: none of
these is read as an empty project. Only the read of one task keeps the task instead, see "A task
that cannot be read". A pagination offset that comes back twice in one listing and a sync token
that comes back twice in one read of the events feed raise `AsanaResponseError` as well. A budget
the run spent without any progress raises `AsanaSourceError` too, see "A budget too small". dlt
drops the state of a run that raises, so the next run starts from the last good state and forgets
nothing.

**A listing cut by a rate limit or the budget.** The listing is a pass over the selected
projects, and a pass can take several runs. When a run is cut inside it (HTTP 429 or a spent
`max_requests`), its progress stays in `state["listing"]`: per finished project the gids listed,
the gids its window or events flagged and its next cursor, sync token and `seen`; for a project
cut inside, its events so far, the task gids with their timestamps listed so far and the **page
offset** of the next page. Ids, timestamps and offsets, never text. The state can hold more ids
than `known_ids` until the pass completes. The finished projects' rows are emitted at once. The
next run continues at the page it stopped at (the project is read again first, one request). A
run cut inside the listing reads no task and forgets nothing, and neither cursors nor sync
tokens nor the stored selection move: the pass renders, forgets and advances only when every
selected project is listed, and only then is `state["listing"]` cleared (a project's offset goes
when the project finishes). A changed selection drops the pass and every offset, because what
was kept described another set of projects. A run that raises keeps the state it started with. A
run that completes the listing but is cut before any task read or check got an answer keeps the
whole listing too, so the next run starts from it instead of listing again first. A pass that is
not cut commits at its end and keeps no `listing`.

**Checking a missing task.** A listing can miss a task that still exists: it moved from one
selected project to another while the pass ran or between the runs that made it, it was dragged
to the top of a project list, or a resumed page shifted. So every known task the complete
listing lacks, in any pass, is asked for its memberships (`GET /tasks/{gid}`, one request each,
counted in `max_requests`). A 403 or 404, or an empty `memberships` list, or no selected
project in it, means it is gone and it is forgotten at once; a selected project keeps it known,
and its project's events or listing render it on a later pass. An answer of 5xx or nothing is
unknown, not gone: the task stays known, it counts in `failed` like a read that failed and in
the outage streak (three in a row stop the run cleanly with `failed_unavailable`, so a hung
Asana does not spend the budget on checks); one such task alone never stops the others. A 200
of the wrong shape raises `AsanaResponseError`, and so does one whose `memberships` is malformed,
instead of being read as "no project": unlike a task read, a check does not keep the task and go
on, because it decides whether a task is forgotten. A malformed `memberships` is one that is not
a list, an entry that is not an object, a `project` that is not an object, or a project without a
string gid. An entry with no `project` (absent or null, for example a task on someone's My Tasks
list) is not a project membership and is skipped: a task with one selected project and one such
entry is selected, and a task with such entries only is in no project (a render read does not
build a document for it, a check forgets it).
A confirmed-live task is not remembered: the next pass lists again and finds it in the fresh
listing, and only a pass that misses it again asks again.

Every answer to a check, like every answer to a task read, counts as progress for the run: a
run that got one commits (cursors, `pending`, tombstones). A run cut before any answer keeps the
complete listing, so the next run starts from it without listing again. Verdicts are
independent, so a run cut by a 429 or the budget keeps the tombstones it already emitted and
the rest stay known. Forgetting is delayed by the budget, and a run that can afford a check moves
it forward; a budget that cannot raises, see Limits. Tasks that belong to an unresolved project
are never asked.

**A budget too small.** A run cut by the budget that kept nothing new, because every run would
spend the same requests the same way and stop at the same place, raises `AsanaSourceError`
saying that `max_requests` is too small and giving the numbers (the budget, the requests the run
spent, how many of them re-read projects Asana would not show). No ids or names are in the
message. A 429 never raises this: it is the provider's own limit and clears with time. The state
is left as it was.

**A project Asana will not show.** A 403 or 404 on `GET /projects/{gid}` (that call only; a 401
and every other error still raise) makes the project **unresolved** for the pass: it is skipped
and every other project syncs. Its stored cursor, sync token, `seen` and the gids of the tasks
last listed for it (`state["projects"][gid]["tasks"]`) stay untouched, its project row and those
tasks are not forgotten, and the tasks of it that were queued in `pending` or `retry` stay
queued, so an edit waiting for a read is still read when the project returns. A task that left a
resolved project but is still in an unresolved project's stored `tasks` stays protected too: it
is not asked for, until that project resolves again or is deselected. The project is asked
again on every run, so it syncs again, with what changed meanwhile, as soon as access returns.
It is counted in `unresolved` and logged as one warning per cause (403, 404) with a count and no
gid. When every selected project is unresolved the run ends cleanly, emits nothing and forgets
nothing. Deselecting an unresolved project removes it like any other: its row and tasks are
forgotten as they leave the selection.

Known gaps:
- A project truly deleted in Asana (HTTP 404) keeps its row and its tasks, unchanged, until it is
  removed from the selection. The only signal is `unresolved` in the stats and the warning.
- A mistyped project gid also ends as unresolved (404): the run is clean, and the project never
  syncs. Nothing here tells a typo, a deleted project and a missing access apart: the log tells
  403 from 404, and everything else treats them alike.
- A page offset is stored between runs. If Asana rejects a stored offset (HTTP 400), the run
  raises `AsanaAPIError` on every run until the selection changes or the project is removed;
  there is no restart path. Whether Asana expires offsets is not verified.
- Three tasks that keep answering 5xx or nothing in the retry tail stop every run with
  `failed_unavailable`, so the check phase never runs and tasks deleted in Asana stay known; and
  one check that keeps answering 5xx, with a budget of just the listing plus one request, makes
  every run spend the listing again before it stops at that check. Both need Asana to fail
  persistently for those specific tasks. They resolve when the tasks answer again, or when the
  project they belong to is deselected.

**Changing the selection.** The selection is stored. A project you add is read in full. A project
you remove forgets the tasks that were only in it and its project document, and the tasks that
were also in a project you kept are read again so the removed project's line leaves their
text. (Only tasks that were in several selected projects are tracked for this, in `state["multi"]`.)

After a run, `source.cognee_sync_stats` holds counts only, all non-negative integers, present on
every way a run can end:

| Key | Meaning |
|---|---|
| `scanned` | tasks listed |
| `skipped` | listed tasks that did not need reading |
| `deleted` | tombstones emitted |
| `failed` | tasks that could not be read, plus the one request that stopped a stopped run, plus one for a run that raises |
| `kept` | the retry queue the run committed (tasks carried over from an unresolved project or an earlier run included); 0 when it kept only the listing; not part of `failed` |
| `pending` | the pending queue the run committed (tasks of an unresolved project included); 0 when it kept only the listing; not part of `failed` |
| `requests` | API calls made (a call the client retries counts once) |
| `unresolved` | selected projects Asana would not show (403 or 404) this run, skipped and asked again next run |

and exactly one reason key when a run stopped or raised for one: `failed_rate_limit`,
`failed_budget`, `failed_unavailable` or `failed_auth`.

State (version 1, `state["v"]`): `selection`, `known_ids`, `subtasks`, `multi`, `retry`,
`pending`, `listing` (only while a listing pass is unfinished: `selection`, `done` per finished
project and `partial` per project cut inside, with its page offset) and, per project, `cursor`,
`seen`, `sync` and `tasks` (the gids last listed). It holds ids, timestamps and offsets, never
text. `known_ids` grows with the number of rows; `listing` and the per-project `tasks` hold the
ids and timestamps of the listed tasks again, so the state can exceed `known_ids` until a pass
completes.

## Limits

- **Budget.** `max_requests` bounds every request of a run: the listing's, the task reads' and the
  checks of missing tasks (a task costs three, so the minimum is 3). It must also cover the
  re-read of every unresolved project each run (one request each) plus one task read (three
  requests) or one check (one request): a budget that cannot raises, see "A budget too small".
  1500 by default is a choice, not an Asana limit (Asana documents per-minute limits). The listing
  costs `projects x (1 + events pages + ceil(tasks / 100))` requests a pass, plus one project read
  for each project resumed, and rendering waits for a complete pass, so a run reads about
  `(max_requests - listing) / 3` tasks. A bigger selection finishes over several runs, each
  starting from the kept listing or from `pending`. A first sync of N tasks costs about `3N` plus
  the listing: 3,000 tasks in one project is 32 listing requests and about 9,000 task requests, so
  the first run reads 489 tasks and each later run lists again (32) and reads 489 more, seven runs
  in all.
- **Deselecting a big project** costs one check per task of it: 5,000 tasks are about 5,012
  requests (measured against the fake) over about four runs at the default budget, instead of the
  three requests of a listing alone. Every run forgets some of them, so it finishes, but it is
  not cheap.
- **When the listing alone exceeds one run's budget** every pass spans several runs. The run
  that completes the listing renders with what is left of its budget; if nothing is left, the
  next run starts from the kept listing and renders or checks first. So a pass either gets an
  answer to a task read or a check before it commits, or, when the budget cannot afford even
  that, the run raises. Tasks and deletions are delayed by a small budget, not stalled. The cost
  is time: size the budget well above one pass, or each pass settles only a few tasks before the
  next one lists again (measured in the tests: two projects of six tasks at two a page, ten
  listing requests a pass, take five runs at a budget of 3 to forget one deleted task).
- **Lock hold time.** A run holds cognee's process-wide staging lock while it walks. The listing
  and the reads share `max_requests`, so a run holds the lock for at most that many requests plus
  the client's retries of each (three attempts, backing off 1 s then 2 s). There is no wall-clock
  deadline, no cap on projects or pages and no other limit. `check_active` is called before each
  attempt and before each backoff, so a host that revokes the run is noticed before the next
  attempt, not after the retries; a revoked run still finishes the attempt in flight (10 s to
  connect, 30 s to read) and sends the one that follows a backoff already begun.
- **Renaming a project or a section** changes the text of every task in it (each task names its
  projects and sections), so each of those tasks is read again (three requests each) and
  cognified again: LLM extraction and a summary on your key. The node set uses the project gid,
  so it does not change, and neither does anything else. Rename sparingly in a big project.
- **Project rows** are emitted again, unchanged, on every pass. They cost no request (read during
  the listing) and no cognify: their content hash is the same, so cognee builds no document.
- **An unchanged run** costs, per project, the project, the events page(s) and the listing pages.
  An edited task costs three. A renamed project or section, an expired sync token or a changed
  selection costs three requests per task of the project that is read again, and one more
  cognify for each.
- **No sleeping.** A 429 stops the run at once; whatever `Retry-After` says. Asana limits requests
  per minute, so the next run (a minute later) usually continues.
- **A plan with a low per-minute limit** ends a run at the limit, long before `max_requests`: the
  run stops with `failed_rate_limit`, keeps what it rendered and the listing it finished, and
  the next run continues. A first sync of a big selection then takes as many runs as the limit
  needs (a task costs three requests, and every pass lists its projects again). The connector does
  not pace its requests, and `max_requests` only bounds a run, it is not tuned to a plan. The
  per-plan limits are not checked against Asana here.
- Subtasks of subtasks, attachments, task history other than comments, tags, followers and
  dependencies are not read. A subtask is only part of its parent's document.
- If the token loses access to every task but the listing still works, a run ends with no reason
  key: only `failed=N` and `kept=N` show it. The community connector's `failed_permission`
  signal was dropped; this is a known gap.
- A project Asana answers 403 or 404 for is skipped and asked again every run, see "A project
  Asana will not show"; it does not stop the other projects.

## Privacy

This connector reads the **names and text of your Asana tasks**. It is opt-in: nothing is fetched
until you construct a source and call `remember`.

- Read: task names, descriptions, comments with their authors' names, assignee names, due dates,
  section and project names, subtask names and notes, custom field values and project
  descriptions of the selected projects. Not requested: email addresses and any user profile.
- That text goes to your configured LLM and embedding providers during cognify, like any document
  you add. People type emails, phone numbers and customer details into comments and descriptions,
  and they are sent too. Anyone with read access to the dataset can read all of it.
- The access token lives in the client object only. Errors carry the HTTP status and an endpoint
  with ids replaced (`/tasks/{gid}`); Asana's own explanation of an error is on the error's
  `detail` attribute with the token cut out and is never logged. The client strips leading and
  trailing whitespace (a trailing newline in an env token is harmless) and refuses a token with
  whitespace, control or non-ASCII characters inside it, without echoing it. A task id is never
  logged: a run with unreadable tasks logs one warning with their count.
- dlt keeps the staging copy of the rendered text (`dlt_database_<dataset>`) and every completed
  load package, with the text of each row, under its data directory (`~/.dlt` of the server
  process unless `DLT_DATA_DIR` moves it); a task deleted in Asana is removed from the graph, the
  vectors and the staged row, not from those packages.
- `cognee.forget(dataset=...)` and `forget(data_id=...)` remove the documents, the graph and the
  vectors, but not the staged copy or the sync state: the next sync under the same dataset and
  `resource_name` builds the documents again. To keep a task out, delete it in Asana or take its
  project out of the selection, and sync.

## Testing

```bash
uv run pytest cognee/tests/unit/tasks/connectors/test_asana_source.py \
  cognee/tests/unit/tasks/connectors/test_asana_client.py \
  cognee/tests/integration/tasks/connectors/test_asana_ingestion.py \
  cognee/tests/integration/tasks/connectors/test_asana_forget.py
```

No Asana account or token is needed. `cognee/tests/unit/tasks/connectors/fake_asana.py` is an
in-memory Asana API: paged listings whose offsets are opaque tokens standing for a position
in one listing (items added or removed meanwhile shift them, a token it never issued, or issued
for another listing, answers 400), per-task reads, sync tokens that answer 412 when stale, and
a comment edit or a new comment that leaves `modified_at` alone and arrives only as an event. It
answers no endpoint the connector does not use. The walk is tested with a plain dict as state,
through a real `dlt` pipeline into SQLite, and through the real client with only the transport
faked. `test_asana_forget.py` runs the real cognee pipeline with the LLM and embeddings mocked.

Ported from the community connector (`packages/connector/asana`, by AbdulWasih05), without its
workspace mode, its `include_completed` option, its environment-variable token, its sleeping on
`Retry-After` and its give-up rules for unreadable tasks.
