# GitHub sync settings (SDK)

The GitHub App integration can pause automatic sync, run on pushes, or use a
weekday schedule. These settings belong to a stored installation connection;
they do not change the repositories authorized in GitHub. Existing connections
keep their previous default: enabled, push-driven, default branch, code graph.

This is SDK and self-hosted API support. It does not configure Cognee Cloud or
install a scheduler. Run one SDK worker per installation: its sync queue is
process-local. Separate processes are not a distributed scheduler.

## Configure a connection

See [the runnable example](../examples/guides/github_sync_settings.py). Public
helpers are exported from `cognee.modules.integrations.github`:

```python
from cognee.modules.integrations.github import (
    GithubSyncSettings, GithubRepositorySettings, SyncSchedule,
    save_sync_settings, run_due_github_syncs,
)

settings = GithubSyncSettings(
    enabled=True,
    mode="scheduled",
    first_sync="scheduled",
    schedule=SyncSchedule(
        interval_minutes=30,
        weekdays=[0, 1, 2, 3, 4],
        timezone="Europe/Belgrade",
        start_time="09:00",
        end_time="17:00",
    ),
    repositories={
        "123456": GithubRepositorySettings(
            branch="main",
            include_paths=["src/**"],
            exclude_paths=["**/tests/**"],
        ),
    },
)
await save_sync_settings(credential, settings, acknowledge_removal=True)
await run_due_github_syncs(user_id=credential.user_id)
```

`save_sync_settings` saves only; call `sync_repositories(credential,
trigger="settings")` to reconcile after a Python settings change or resume.
The HTTP settings endpoint requests this automatically when content changes or
the connection resumes. A first scheduled run waits for its slot unless
`sync_now=True` is explicitly requested. `sync_repositories(credential)` is an explicit manual
run and also works while paused. A paused connection keeps its credentials and
indexed data; an already-started sync finishes. Disconnecting revokes access.

For a programmatic initial connection, save the settings after
`complete_installation` and before calling the adapter's `on_installed` hook.
The existing OAuth callback starts that hook immediately; changing settings
afterward does not undo a first sync that already started.

## Schedule semantics

- Modes: `push`, `scheduled`, `manual`. Scheduled/manual modes ignore pushes.
- Intervals are positive whole minutes, anchored to local midnight or window start.
- Weekdays are 0 (Monday) through 6 (Sunday); holidays are not excluded.
- The start is inclusive and the end exclusive. Omit both times for the whole
  day. Windows must start and end on the same day.
- Timezone names use IANA identifiers. Nonexistent daylight-saving wall times
  are skipped; both occurrences of a repeated wall time are eligible.
- A late tick coalesces missed eligible slots into one sync, inside an allowed
  day/window. It does not replay each missed interval.
- `next_sync_at(settings, after)` returns eligibility, not a promised start time.
  A running sync, provider error or a stopped worker may delay processing.

Invoke `run_due_github_syncs()` from an external scheduler. The authenticated
`POST /api/v1/integrations/github/sync/due` starts only the requesting user's
connections in the background and answers `{"accepted": true}` at once: a due sync
clones and builds every selected repository. Merely saving a schedule does not
execute it periodically.

A push is judged against the saved policy before any GitHub call: it syncs only when
the repository is selected, pushes are enabled, and the pushed branch is the
repository's selected branch (its default branch when none is chosen). Pushes to any
other branch are dropped without a token, a listing or a history entry.

## File selection and removal

Repository settings use immutable numeric GitHub repository ids. `repositories=None`
means every repository available to the installation; `{}` selects none.
`branch=None` follows the default branch. One repository has one selected branch
per dataset; changing branch replaces its code graph.

Patterns are POSIX paths relative to the repository root: `*`, `?`, and character
classes match within one component; a whole `**` component matches zero or more
directories. Use `src/**` to include a folder and future files within it.
Absolute paths, traversal, backslashes and embedded `**` are refused. Exclusions
win. `include_paths=None` means all supported paths; `[]` means no files.

The preview and checkout use the same selector. Existing binary, generated
directory, symlink and submodule exclusions still apply. A preview reports a
tree SHA and skipped paths; it is not a promise that the branch will remain
unchanged until the next run. Incomplete GitHub tree responses fail explicitly.

A selected repository that left the installation (an admin removed it in GitHub) is
skipped with a warning and the others still sync. The saved selection and the record
of indexed repositories keep it, so its content is retained until the selection is
changed.

Changing repository settings requires `acknowledge_removal=True`. A subsequent
full sync removes excluded code from the graph, including at an unchanged
commit. Entire repositories are retired only when their ingestion
was recorded by this version; older connections without that record must first
sync their existing selection before narrowing it. A repository is recorded as soon
as its content is stored, before it is built, so one whose build keeps failing can
still be dropped from a selection. Saving a selection change refuses untracked
existing repository content rather than silently retaining it. Pausing does not delete
data.

GitHub settings currently support code graph ingestion only:
`include_documents=True` is rejected. Ordinary documents have content-based
identities that can be shared with other sources; safe removal needs separate
document ownership support. Direct filtered `cognee.add` likewise requires
`codegraph_config={"include_documents": False, ...}`. Filters apply to remote
repository URLs, not local directories or a remote Cognee API client.

## Self-hosted HTTP API

All endpoints require authentication and operate on the requesting user's
connection. Use `?account_id=INSTALLATION_ID` when several are connected.

| Method and path under `/api/v1/integrations` | Purpose |
| --- | --- |
| `GET /github/sync/settings` | Saved settings, next eligibility, last sync and status |
| `PUT /github/sync/settings` | Body: `settings`, `acknowledge_removal`, optional `sync_now` |
| `POST /github/sync/preview` | Body: `repository_id` and proposed repository `settings` |
| `POST /github/sync/due` | Start this user's due schedules in the background |
| `POST /github/sync` | Existing manual sync endpoint |
| `GET /github/sync/progress` | Latest execution and locally active runs |
| `GET /github/sync/history` | Recorded invocations, newest first; all unless paged with `limit` and `offset` |

Request bodies accept camelCase or snake_case keys and responses use camelCase, like
the rest of the API.

`last_synced_at` follows the existing connector contract: time of the last
attempt, including a failed attempt. A sync that selects nothing records no attempt.
Check `sync_status` or history for its outcome.
Progress/history reuse persisted integration pipeline runs. Reporting is
best-effort and does not change ingestion outcomes; unresolved runs from a
different or stopped worker are not presented as confirmed active executions.
