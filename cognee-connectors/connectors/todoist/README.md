# cognee-connector-todoist

A Todoist data-source connector for [cognee](https://github.com/topoteretes/cognee): sync the
tasks, comments and projects of a Todoist account into memory, incrementally, and forget a task
or project when it is deleted in Todoist.

It exposes a `dlt` source you hand to `cognee.remember(...)`. Each task (with its comments
folded in) and each project becomes a **normal document** in cognee (it flows through chunking
and LLM entity extraction), via cognee's document-mode marker. One source is one API token and
an optional selection of projects. The package also carries the provider's connection contract
(`CONNECTOR`, an API-key descriptor on `cognee-connectors-core`) for a cognee server or Cognee
Cloud.

## Install

```bash
pip install cognee-connector-todoist             # the connector and its DLT source
pip install "cognee-connector-todoist[cognee]"   # plus cognee itself, which runs the sync
```

## Requirements

cognee 1.6.5 or later whose ingestion reads per-row node sets (`DOCUMENT_SYNC_VERSION` 2 or
later; the source refuses an older one before it makes a request). The Sync API client uses
`httpx`, which the package declares.

The token needs read access to the account's tasks, comments and projects. An API token has
the access of the user who created it. This connector has not been tested against a live
account: the fake it is tested with follows the behaviour the Todoist Sync API documents (see
Development).

## Configuration

This package reads no environment variable. cognee has no `TODOIST_*` settings and no settings
class for Todoist, and the token is never read from the environment: it is a parameter of the
source factory (`access_token=`, or a pre-built `service=`). Everything is configured on the
`todoist_source(...)` call, see the argument table under Usage.

## Setup

1. Create an API token in Todoist: **Settings → Integrations → Developer** (or
   <https://todoist.com/app/settings/integrations/developer>). The connector never refreshes a
   token: a host hands over a fresh one.
2. Pass it as `access_token`. Set your `LLM_API_KEY` like any other cognee run.
3. Optionally collect the ids of the projects to sync (Todoist does not show them in the UI;
   the sync's own `projects` answer carries them) and pass them as `project_ids` to limit what
   is ingested. Omit it to sync every project the token sees.

## Usage

```python
import cognee
from cognee_connector_todoist import todoist_source

await cognee.remember(
    todoist_source(
        resource_name="todoist_me",
        access_token="<API token>",
    ),
    dataset_name="todoist",
    primary_key="id",
    write_disposition="merge",  # required: incremental upserts and deletes
    max_rows_per_table=0,  # no row cap; forget-on-delete must see the whole corpus
    self_improvement=False,
)

answer = await cognee.recall("What did I promise to ship this week?", datasets=["todoist"])
```

Run the same `remember` call again whenever you want to sync.

| Argument | Meaning | Default |
|---|---|---|
| `resource_name` | Names this connection: it names the staging table and keys the sync state. A non-blank string that does not start with `_dlt_` (dlt's own tables) | required, one per connection |
| `access_token` | Todoist API token. Never read from the environment | needed unless `service` is given |
| `project_ids` | The projects to sync: `None` syncs every project the token sees, a non-empty list or tuple of id strings limits the sync. Tasks and projects that leave the selection are forgotten on the next run | `None` |
| `max_requests` | Budget of the requests of one run: the sync call (1), one per task whose body the feed did not carry, one per rendered task's comments. A positive int | 1000 |
| `check_active` | Host checkpoint, called around every row. What it raises stops the run as it is | none |
| `service` | Pre-built client, from `build_todoist_service(access_token)`. A client of your own must offer `sync(sync_token, resource_types)`, `get_task(task_id)` and `list_comments(task_id)` and call `check_active` before each attempt it makes | built from `access_token` |

`project_ids` is sorted and repeats are dropped, so the same selection in another order is the
same selection. A bare string, an empty list or a blank id raises `ValueError` when the source
is built. Changing the selection is allowed, see below. Two connections in one dataset need two
`resource_name`s: they would share state and each sync would forget the other's tasks.

A runnable example lives in [`examples/todoist.py`](examples/todoist.py).

## Running on a cognee server

```python
from cognee_connector_todoist.cognee_host import register

register()  # a no-op: there is nothing to register for Todoist
```

Todoist has no OAuth flow in cognee, so cognee ships no adapter, no credential routes and no
settings for it: `register()` does nothing, `routers()` is empty, and the package declares no
`cognee.integrations` entry point. A server syncs Todoist the way the SDK does, by handing
`todoist_source(...)` to `remember`; a host that stores Todoist tokens builds the source with
`service=build_todoist_service(token)`, passes the selection as `project_ids` and names the
connection with `resource_name`, as cognee's `modules.integrations.ingestion` does. `CONNECTOR`
(an API-key `ConnectorDescriptor` served by `DescriptorConnector`) names the provider for a
host's catalog, derives a connection's dataset from its account id (`todoist_<account id>`) and
hands a stored token out as the sync credential, read from `tokens["access_token"]`, with the
project ids a host passes as `resources` as the scope's `items`; it does not enumerate an
account, the selection does.

## What is ingested

| Row | Id | Title | Content | Node set |
|---|---|---|---|---|
| Task | `task:<id>` | the task's content | completed or not and the day it was completed, priority, due date, labels, the description, comments | `todoist:<project id>` |
| Project | `project:<id>` | the project name | empty | `todoist:<its own id>` |

```
Completed: no
Priority: high
Due: 2026-10-20
Labels: launch, website

Draft, then review with marketing.

Comments:
- Looks good
- Use the new palette
```

A task is one document with its comments (oldest first) in it; a comment is not a document of
its own. Tasks and projects are separate rows of one table.

Every list is sorted and nothing volatile is rendered (no `updated_at`, no counters, no time of
day), so an unchanged task keeps its content hash and is not extracted again. Comments render
without an author name: the Sync API carries a note's author only as a user id, and resolving
ids into names is not done in this version. Project **names** never appear inside a task's
text: membership is expressed only through the node set, so renaming a project costs nothing —
no task is extracted again.

**Node set.** Every row carries `cognee_node_set`, so a `recall` can be scoped to one project
(`node_name=["todoist:<project id>"]`). A task gets `todoist:<project id>`; a project row gets
`todoist:<its own id>`. Only ids are used, never names. The node set is part of the row's
identity, so a task that moves to another project is extracted once more. A tombstone carries
no node set. `cognee_structure` is not set: subprojects are not placed under their parent
project in this version.

### Frozen names

These are permanent once anyone has synced, because every change re-extracts every touched task
on your LLM key or orphans documents: the provider and table prefix `todoist`, the row ids
`task:<id>` and `project:<id>`, the node set names `todoist:<project id>`, the state keys and
the stats keys below, and the rendered text, including the `Completed on:` line.

## How sync and forget-on-delete work

**Finding what changed** costs one request. Every run makes one Sync API call with the stored
`sync_token` (`"*"`` on the very first run) and receives exactly what changed since that token
was issued: new or edited tasks, projects and comments, and deletions, which the feed marks
with `is_deleted`. This is one of the cleanest cursors of any provider — there is no listing,
no window and no overlap, and an incremental run reads only the tasks that actually changed.

**Rendering a changed task** costs one request: its comments are read in full through
`GET /rest/v2/comments?task_id=...`, because the feed carries only the *changed* comments. A
task whose *comment* changed but whose body did not is not in the feed's `items` at all; its
body is then read through `GET /rest/v2/tasks/{id}` (two requests with the comments). The
rendered text is deterministic, so a task whose comments did not really change keeps its
content hash and is not extracted again.

**Forgetting.** The feed is authoritative about deletions. A task that arrives with
`is_deleted` is emitted as `{"id": ..., "_deleted": True}` (dlt removes the row on `merge`, and
cognee's orphan cleanup removes the document, graph content and vectors). A deleted project
tombstones its row **and** every task that belonged to it. A run tombstones only ids this
connection itself synced (its own state), and only after the sync call that reported the
deletions succeeded — a run that raises or stops before that forgets nothing.

**The cursor.** The new `sync_token` is stored only after the whole answer was processed and
committed. A run cut by the request budget, or one that raises (a 429 after the client's
bounded retries, an API outage, a malformed answer), keeps the old token: the next run
re-fetches the same changes. Rows a cut run already emitted are upserts, which are idempotent
under `merge`, so repeating them costs nothing.

**Changing the selection.** The selection is stored. Widening it (`None`, or a list that adds a
project) syncs the newly selected projects' tasks like any other change. Narrowing it
tombstones the known tasks and projects that left the selection on the next run; nothing about
them is asked of Todoist.

**A run that stops.** On a spent `max_requests` budget the run ends cleanly: `failed_budget`
is set, no tombstones are computed (they come last), the cursor is not advanced, and the next
run repeats the same changes. On a 429 or a 5xx out of the client's retries the run raises:
dlt drops its state, `failed_rate_limit` or `failed` is set, and the next run starts from the
last good token. A rejected token (401/403) raises `TodoistAuthError` with `failed_auth`. The
run never sleeps on a 429 outside the client's bounded retry budget: the walk holds cognee's
process-wide staging lock, and Todoist's per-minute window clears between runs.

After a run, `source.cognee_sync_stats` holds counts only, all non-negative integers, present
on every way a run can end:

| Key | Meaning |
|---|---|
| `changes` | objects (tasks, projects, comments) the sync call returned |
| `rendered` | task rows emitted |
| `deleted` | tombstones emitted |
| `failed` | the request whose failure ended a run, or one count for a run that raised |
| `requests` | API calls made (a call the client retries counts once) |

and exactly one reason key when a run stopped or raised for one: `failed_budget`,
`failed_rate_limit` or `failed_auth`.

State (version 1, `state["v"]`): `sync_token`, `selection` (`None` or the sorted project
ids), `tasks` (task id -> its project id, of every task this connection knows) and `projects`
(the project ids it knows). It holds ids only, never text.

## Limits

- **Rate limits.** Todoist's Sync API is limited per minute; this connector does not pace its
  requests. A 429 out of the client's five bounded retries ends the run (`failed_rate_limit`)
  and the next run re-fetches from the stored token, so the limit only delays a sync, it does
  not lose changes. `max_requests` bounds a run's requests, it is not tuned to a plan.
- **First sync of a large account** renders every task; each costs one comments request, plus
  one task request for tasks whose comment changed. A budget smaller than that stops the run
  cleanly and the next run continues from the same cursor — the same tasks come back and are
  rendered in idempotent upserts until the run completes.
- **Comment edits** re-render the parent task (two requests) even when the task itself did not
  change, because the task's document carries its comments in full.
- **Sections, labels as objects, reminders, collaborators** are not read. A comment renders
  without its author. Subprojects are flat: a project row names itself, not its parent.
- **Lock hold time.** A run holds cognee's process-wide staging lock while it walks. The run
  costs one sync request plus at most two requests per changed task; there is no wall-clock
  deadline and no other limit.

## Privacy

This connector reads the **names and text of your Todoist tasks and comments**. It is opt-in:
nothing is fetched until you construct a source and call `remember`.

- Read: task contents, descriptions, comments, due dates, priorities and labels of the account
  (or the selected projects), and project names. Not requested: email addresses and any user
  profile.
- That text goes to your configured LLM and embedding providers during cognify, like any
  document you add. People type emails, phone numbers and customer details into comments and
  descriptions, and they are sent too. Anyone with read access to the dataset can read all of
  it.
- The access token lives in the client object only. Errors carry the HTTP status and an
  endpoint with ids replaced (`/tasks/{id}`); Todoist's own explanation of an error is on the
  error's `detail` attribute with the token cut out and is never logged. The client strips
  leading and trailing whitespace (a trailing newline in a copied token is harmless) and
  refuses a token with whitespace, control or non-ASCII characters inside it, without echoing
  it. A task id is never logged: a run with skipped tasks logs one warning with the id only in
  debug.
- dlt keeps the staging copy of the rendered text (`dlt_database_<dataset>`) and every
  completed load package, with the text of each row, under its data directory (`~/.dlt` of the
  server process unless `DLT_DATA_DIR` moves it); a task deleted in Todoist is removed from
  the graph, the vectors and the staged row, not from those packages.
- `cognee.forget(dataset=...)` and `forget(data_id=...)` remove the documents, the graph and
  the vectors, but not the staged copy or the sync state: the next sync under the same dataset
  and `resource_name` builds the documents again. To keep a task out, delete it in Todoist (or
  remove its project from the selection) and sync.

## Development

From a `cognee-connectors` checkout, with cognee installed (the workspace's `dev` group has
it; the unit tests read cognee's row content hash from it):

```bash
export COGNEE_ENV_FILE=<an empty file> TELEMETRY_DISABLED=1 LITELLM_MODE=PRODUCTION LLM_API_KEY=sk-test-dummy
PYTHONPATH=core:connectors/todoist python -m pytest connectors/todoist/tests -q
python scripts/check_cognee_free.py cognee_connector_todoist
```

No Todoist account or token is needed. `tests/unit/fake_todoist.py` is an in-memory Todoist
Sync API plus the two REST endpoints the connector uses: a `sync_token` cursor that returns
exactly what changed (including `is_deleted` objects), per-task comment pages and per-task
bodies. It answers no endpoint the connector does not use. The walk is tested with a plain
dict as state and through a real `dlt` pipeline into SQLite.

Implemented independently for cognee issue #4815; not ported from any community connector.
