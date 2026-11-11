# Integration sync schedule

A stored connection (a row in `integration_credentials`) can sync on a schedule.
A schedule is a sync policy: an interval, the local weekdays it runs on, and an
IANA timezone. Without any policy a connection syncs only when asked, exactly as
before.

## Policies

- **Intervals:** 15, 30, 60, 120, 240, 360, 480 minutes, 1 day (1440) or 1 week
  (10080), or whole hours from 1 to 23. Every response lists them as
  `intervalPresets` and `customHours`.
- **Slots:** under a day, from local midnight every N minutes on each listed day;
  a day or a week runs once at 10:00 local, and a week takes one weekday. Missed
  slots are one run, and the slot after it is the next one on that grid: a daily
  slot caught up at 09:59 runs again at 10:00. Slots on one grid start together,
  and nothing limits how many syncs start at once.
- **Levels:** the owner's default applies to every connection the owner has; a
  connection's own policy overrides it. The owner of a connection is its
  `workspace_id` when set, else the user who connected it.
- **Pause:** a paused default holds every connection of the owner, those with their
  own policy too; a paused connection holds only itself. Slots pass without a run
  while paused, a run in progress finishes, and resume schedules the first slot
  after now. A manual sync of a paused connection is a 409 that names the level.
  It also holds the syncs a provider webhook would start (GitHub pushes and added
  repositories, Linear team changes) and the first sync after connecting: they are
  dropped with an info log, not queued. Revoking on uninstall and Linear agent
  sessions are not syncs and still run. Pause does not hold the Linear worker that
  continues a sync cut short by the rate limit or the request budget. Saving a changed
  selection of a paused connection keeps the selection for the first sync after the
  resume; an active connection syncs a changed selection at once.
- **Disconnect:** a connection's own policy stays when it is disconnected and
  applies again when the same owner reconnects it. Reconnected by another owner,
  the connection follows that owner's policies only.

## Routes

The caller's own default. The routes take no owner, so they never reach anyone
else's default:

```http
GET|PUT|DELETE /api/v1/integrations/sync/policy
POST /api/v1/integrations/sync/policy/pause
POST /api/v1/integrations/sync/policy/resume
```

One connection (`account_id` as for `POST /{provider}/sync`):

```http
GET|PUT|DELETE /api/v1/integrations/{provider}/sync/policy?account_id=...
POST /api/v1/integrations/{provider}/sync/policy/pause?account_id=...
POST /api/v1/integrations/{provider}/sync/policy/resume?account_id=...
```

`GET` on a connection returns the policy it syncs by, with `level` set to
`connection` or `default`, and `heldBy` set to the level that holds it paused (a
paused default holds it even when its own policy is not paused). `DELETE` on a
connection returns it to the default. A connection without its own policy has
nothing to pause: save one first. The connection routes find the connection among
the caller's own, and the caller must be able to act for its owner (a host may set
`workspace_id`; nothing in the SDK does). Another owner's policy or connection is
the same 404 as a missing one. Two first saves of one policy at once: the second
is a 409, read and save again. `account_id` is snake_case like every query
parameter here; bodies take camelCase or snake_case.

```http
PUT /api/v1/integrations/sync/policy
{"intervalMinutes": 1440, "weekdays": [0, 1, 2, 3, 4], "timezone": "Europe/Belgrade"}
```

## Running the schedule

Set `INTEGRATIONS_SYNC_SCHEDULER_ENABLED=true` and run that API as one process per
database (one worker, one replica). It checks for due slots every 60 seconds and
starts each through the provider's `sync_now`, the same call as a manual sync; the
runs are recorded with origin `background`. On shutdown the API waits for a tick in
flight to finish, so it is not cut between a claim and its starts. A tick stuck on a
database call holds shutdown until that call gives up; on Postgres set a command or
statement timeout through `DATABASE_CONNECT_ARGS`.

The claim of a slot is a conditional update, so even several processes start a
slot once. Two runs of one connection, though, are kept apart only inside one
process. With the scheduler, manual syncs and webhooks spread over several
processes, two runs of a connection can share one dlt working directory (one's
cleanup deletes the other's files), read the same cursor and pay the provider's
API twice, and leave the connection's status of whichever finished last.
GitHub webhook syncs of named repositories are not held by that guard, so a push
during a full sync is still indexed.

Run history stays where it is: see [integration-sync-progress.md](integration-sync-progress.md).
