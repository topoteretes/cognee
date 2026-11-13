# Vercel SDK connector

`vercel_source` turns one Vercel project into memory: the project, each of its deployments
and, when asked for, the build output of its failed deployments, one document each. It is a
`dlt` resource you hand to `cognee.remember`, and the source the Cognee Cloud pod reaches
through `_PROVIDERS["vercel"]`.

Ported from the community connector (`packages/connector/vercel`, PR #286 by itssaharsh,
commit 0d9c0a7). The provider knowledge is kept; the shape is new (see "Changes from the
community connector").

## Requirements

- Cognee with `DOCUMENT_SYNC_VERSION >= 3` (per-row structure). No extra install: the client
  uses `requests`, which `dlt` already brings.
- A Vercel access token that can read the project and its deployments: a personal access token
  (https://vercel.com/account/tokens, scoped to the team, with an expiry) or the OAuth access
  token of a Vercel Integration. Both are sent as `Authorization: Bearer`, through a requests
  auth object: requests consults `~/.netrc` only when no auth object is given (and would then
  replace the header with a netrc entry for `api.vercel.com`). The rest of the environment
  still applies as requests reads it: `HTTPS_PROXY` / `NO_PROXY` and `REQUESTS_CA_BUNDLE`.
- Answers are read as UTF-8 JSON (RFC 8259), never with a charset requests guesses; a body that
  is not UTF-8 JSON is a `200` of no documented shape.
- The project id (`prj_...`, project Settings, General) and the team id (`team_...`), or for a
  personal (Hobby) account your user id as `account_id`.

## Usage

```python
import cognee
from cognee.tasks.ingestion.connectors import vercel_source

source = vercel_source(
    resource_name="vercel_web",  # one per user and project, required
    project_id="prj_...",  # the id, never the name
    team_id="team_...",  # omit for a personal account and pass account_id instead
    access_token="<token>",
)
await cognee.remember(
    source,
    dataset_name="vercel_acme",  # one dataset per team
    write_disposition="merge",  # REQUIRED: the add pipeline defaults to "replace"
    primary_key="id",
    max_rows_per_table=0,
    self_improvement=False,
)
print(source.cognee_sync_stats)
```

Pass the same `resource_name` and `remember` arguments on every sync. Several projects of one
team go into the same dataset with one source and one `resource_name` each. A runnable script
is `examples/guides/vercel.py`.

Arguments: `resource_name` and `project_id` are required. `team_id` is sent as `teamId` on
every call. `account_id` is the account the project must belong to (default `team_id`).
`service` takes a pre-built client instead of `access_token` (see `build_vercel_service`).
`check_active` is the host's checkpoint, called before every request attempt and retry wait and
around each row. `max_requests` bounds the requests of one run (default none).
`lookback_days` bounds which deployments are read (default none, every deployment Vercel
lists). `include_build_logs` reads the output of failed builds (default off).

## What is read

Calls (all `GET`, all with `teamId` when a team is given):

| Call | When |
|---|---|
| `/v9/projects/{id}` | first, every run: the account guard and the project document |
| `/v7/deployments?projectId=...&limit=100[&since][&until]` | the pass over the listing, paged by `until`; a run that continues a pass also reads the newest pages again |
| `/v13/deployments/{uid}` | once the pass has listed everything, for each known deployment it did not list |
| `/v3/deployments/{uid}/events?limit=-1&follow=0` | with `include_build_logs=True`: once per failed deployment, after the listing pages |

Documents, all in one table named `resource_name`:

| Kind | Row id | Title | First lines |
|---|---|---|---|
| project | `project:<prj id>` | `Project web` | project name, framework, Git repository, production branch, root directory, Node.js version, created |
| deployment | `deployment:<uid>` | `web: production deployment 2026-11-10 14:03 UTC (dpl_x)` | project (its name when built), environment, state (and promotion substate), branch, full 40-character commit sha, commit message, commit author name and login, creator username, error code, error message, created, deployment URL |
| build log | `build_log:<uid>` | `Build output of failed deployment dpl_x of web` | project, deployment, state, a blank line, then every line of the build in order |

`url` is the deployment's inspector URL (`system_metadata["url"]`); the project row has none.
Deployment rows are rendered from the listing only; the single `/v13` read answers in another
shape (`id`, `createdAt`, and a minimal variant without `projectId`) and is used only to learn
whether a deployment still exists.

Never read into a row: emails (creator, commit author), environment
variables, deploy hook URLs, protection bypass keys, trusted IPs, build and install commands,
the team invite code, the raw `meta` map, `updatedAt`, build start and end times, aliases.
Every provider string is cleaned of control characters, terminal escapes and lone surrogates;
a value rendered on a `Label: value` line has its line breaks folded, so a commit message cannot
forge a heading or a field line.

### Graph shape

- Node set per kind: `vercel:project`, `vercel:deployment`, `vercel:build_log` (the row says
  `project`, cognee adds the prefix). `recall(node_name=["vercel:deployment"])` narrows to
  deployments.
- Decision (Goran, 2026-11-11): node sets are per kind only, never per project. A
  `node_name` filter is prefixed `vercel:`, so a project name is never what a user types; the
  project is already in the first line of every deployment (`Project: <name>`); and a node set
  is hashed, so adding one later would re-ingest every row.
- Structure (`cognee_structure`, not hashed): project > deployment > build log. A deployment is
  `child_of` its project row and a build log `child_of` its deployment row; the project row is
  the top (`{"ancestors": []}`). Each ancestor carries a `name` (the project's name, or
  `<project> deployment <uid>`), which search copies into a cited chunk as `document_parent`.
- Decision (taken for Goran on 2026-11-11, reversible): no team level, a change
  from the plan's team > project > deployment > build log. A structure container's id is scoped
  by the table, and each project is its own table, so a team container would be one node per
  project and group nothing. The team id is not in any row's text.
- People and repositories are plain text (`Commit author: Ada Lovelace (ada)`, `Git
  repository: github: acme/web`), so entity extraction merges them across deployments and with
  other connectors in the same dataset.
- Decision (Goran, 2026-11-13): the commit author is the name the Git provider reports plus
  the login, `Name (login)`, or whichever of the two is set. Linear, Slack, Notion and Asana
  write a person by name, a Git source by login, so the same person merges with both. Measured
  with keyless extraction, Vercel and Linear in one dataset: with the login alone the two
  providers share no person; with name and login `ana kovac` and `ben ross` are each one node
  reached by both. `Created by` stays the Vercel username (Vercel reports no name for it).

### Frozen names

These are identity and must not change after a release: the row ids above, the title and line
formats, the node set names, the `_PROVIDERS` prefix `vercel` and its resource name rule, the
state keys (`v`, `project_id`, `team_id`, `rows`, `created`, `pass`, `logs`) and the stats keys
below (`scanned`, `skipped`, `deleted`, `unresolved`, `check_errors`, `build_log_errors`,
`failed` and the `failed_*` reasons). The state stays at version 1: nothing has been
released, so every key of it is part of version 1.

The state is bound to its project and team (decision taken for Goran on 2026-11-11): the first run
stores `project_id` and `team_id` in it, and a run whose `project_id` or `team_id` differs
raises `VercelStateError` (`failed_state`) before any request or state write. Reusing one
`resource_name` for another project would otherwise check the first project's deployments
against the second on every pass, forever. The state version was not raised when the binding was
added, because nothing has been released: a state written by an earlier development build,
without `project_id`, is adopted on its next run as the project it is run with (known, and only
possible before the first release). A state that names a project but no `team_id` key is refused
the same way.

## How sync and forget-on-delete work

Vercel has no change feed: `/v7/deployments` filters by creation time only, and a deployment
keeps changing after it is created (BUILDING to READY, promoted, deleted). So the source walks a
**pass** over every deployment created in the window, newest first, paged with `until` one
millisecond after `pagination.next` (a second deployment of the same millisecond would otherwise
fall between pages; the repeat is dropped by uid). A row is emitted only when its text differs
from the fingerprint kept in the resource state, so a quiet re-sync emits nothing and cognifies
nothing.

A pass can span runs. After every page the position reached (`until`) and the deployments the
pass has seen are saved in `state["pass"]`, so a run cut by a budget, a 429 or an outage is
continued by the next run where it stopped, and a `max_requests` smaller than the listing gets
through it over several runs. A run that continues a pass first reads the newest deployments
again (down to the newest one the pass started from), so a new deployment or a state change at
the top lands on the next run, not at the end of the pass. A head read to its end moves the
pass's `top` to the newest deployment it saw, so the next run reads only what is newer than that
(and its first page), not the same block again. Under a budget a head page is read
only while at least one more request is left after it, so the pass still moves on: with
`max_requests=2` a continued run reads no head. Each run, in order: the project, the head (for
a continued pass), the rest of the pass's listing, the queued build output, then the deletion
checks once the pass has listed everything. The resource state is at version 1 (`state["v"]`);
nothing has been released, so the pass and build-output keys are part of version 1.

A deployment is forgotten (a `_deleted` tombstone, hard-deleted from staging and memory) only on
Vercel's word:

- the listing returns it in state `DELETED`;
- a known deployment created inside the window is missing from a pass that listed everything,
  and its own read answers 404 or 410. The single read's `readyState` has no `DELETED` in the
  spec, so a `200` there means the deployment exists and keeps it.

A deletion check that cannot be read (a 5xx or no answer after the retries, another 4xx, or a
200 that is not JSON or of no documented shape) keeps the deployment, counts it in
`check_errors` (not a failure of the run), and is asked again in the next pass; the run goes
on, so one deployment that always fails never holds the pass open. A deletion can therefore
wait a pass. A 401, 403 or 429 on a check is not an item failure (see "Errors"), and the check
stays pending.

Its build output goes with it. Never forgotten: a deployment that only left the window, any
deployment from a pass that has not listed everything, a project that answers 404, a listing
without its list or its pagination (the run raises), a 401 or 403. Turning `include_build_logs`
off forgets the build output already synced, like a deselected scope.

A project the token cannot see (404 or 410 on the project) is not forgotten: Vercel does not
tell a deleted project from lost access. The run reads nothing, sets `unresolved = 1`, and the
documents stay until the project is deselected (a host empties its table). A project of another
account (its `accountId` is not the expected one) stops the run with `failed_auth` before any
row.

Deselecting a project is the host's job: it empties the project's table with
`ingestion.empty_resource("vercel", <table>)` under `replace`, and the documents leave memory.

### Build output

With `include_build_logs=True` each deployment in state `ERROR` is queued once and read after
the listing pages, with `limit=-1` (every line) and `follow=0` (the spec says the endpoint
streams live events when `follow` is on). Each read settles its own deployment:

- lines: one build log document, never read again (a finished build does not change);
- 404, 410 or no lines: recorded as having no output and never asked again;
- a 5xx, no answer (a timeout included), another 4xx, or a 200 that is not JSON or of the wrong
  shape: counted in `build_log_errors` (not a failure of the run), and asked again in the next
  pass, so it
  never holds back the rows and deletions after it;
- a 401, 403 or 429: as on any other call (see "Errors"), and the read stays queued.

An entry waiting to be read again, or recorded as having no output, is dropped when its
deployment leaves the window (or is forgotten): build output is asked for only while its
deployment is in the window. The fingerprint of a build log already synced stays in the state
after its deployment leaves the window, on purpose: switching `include_build_logs` off must
still find it and forget the build output. Its document stays too (the window bounds reading,
never memory), so a deployment deleted after it left the window keeps its build output
(VERCEL-GAP-6).

### Errors

Every answer ends one of four ways:

- **raise**: the run fails with a typed error; dlt rolls back its state and keeps no row of
  it, so the next run starts where the last successful one ended. `failed` and one reason
  count it.
- **cut**: the run returns normally, keeps the rows it emitted and its state, and the next run
  continues the pass from the last page it completed. `failed` and one reason count it.
- **count**: one item could not be read; it is counted in `check_errors` or `build_log_errors`
  (not in `failed`: the project's run did not fail), kept as it was,
  asked again in the next pass, and the run goes on.
- **none of these**: a documented answer that is not a failure.

A 401 or 403 is a raise while the run has emitted no row yet, so an SDK user with a bad, expired
or revoked token gets an error and not an empty success, and a host sees the error with
`failed_auth`. Once a row was emitted, the same answer is a cut (`failed_auth`): the rows are
kept and the next run resumes. A project of another account (its `accountId` is not the
expected one) is always a cut with `failed_auth`, before any row: it logs what to check
(`team_id` and `account_id`) and not that the next run resumes, since it will not on its own.

| Call | Transport error, timeout or 5xx after retries | 3xx, 400, 409 or another 4xx | 401, 403 before any row | 401, 403 after a row | 404, 410 | 429 | 200 not JSON, of the wrong shape, or with an odd value |
|---|---|---|---|---|---|---|---|
| project read | cut, `failed_unavailable` | raise `VercelAPIError`, `failed_error` | raise `VercelAuthError`, `failed_auth` | (not reachable: the first call of every run) | `unresolved` 1, nothing read or forgotten | cut, `failed_rate_limit` | raise `VercelResponseError`, `failed_response` |
| listing page (fresh pass, continued pass, head) | cut, `failed_unavailable` | raise `VercelAPIError`, `failed_error` | raise `VercelAuthError`, `failed_auth` | cut, `failed_auth` | raise `VercelNotFoundError`, `failed_error` | cut, `failed_rate_limit` | raise `VercelResponseError`, `failed_response` |
| deletion check (`/v13`) | count, `check_errors` | count, `check_errors` | raise `VercelAuthError`, `failed_auth` | cut, `failed_auth` | the deployment is forgotten (tombstone) | cut, `failed_rate_limit` | count, `check_errors` |
| build output (`/v3/.../events`) | count, `build_log_errors` | count, `build_log_errors` | raise `VercelAuthError`, `failed_auth` | cut, `failed_auth` | no output: recorded, never asked again | cut, `failed_rate_limit` | count, `build_log_errors` |

Nothing is forgotten on any failure: the only tombstones are a deployment the listing returns as
`DELETED` and a deletion check answered 404 or 410. A deferred run (`max_requests=0`) is a cut
(`failed_budget`) that makes no request and touches no state. A spent `max_requests` is a cut
(`failed_budget`), a cursor that does not move or a deployment of another project in a listing
is a raise (`failed_response`), and a state of another project, team or version is a raise
before any request (`failed_state`). The table is pinned, cell by cell, by `test_error_matrix`
in `test_vercel_source.py`; the client's retries behind "after retries" are in
`test_vercel_client.py`.

An odd value is a value of the wrong type in a `200`: a time that is not a finite whole number
of milliseconds `datetime` can render (no NaN, no infinity, no `1e20`), a text field that is not
text, an object field that is not an object, or JSON nested deeper than the parser can read. On
the project or a listing it fails fast: the run raises on every sync while Vercel keeps answering
it (a poison pill until the value changes), and nothing after it lands.

`cognee_sync_stats` holds non-negative ints, every key present from the moment the source is
built and reset at the start of each run: `scanned`, `skipped` (unchanged), `deleted`
(tombstones), `unresolved` (1 when the project answered 404 or 410), `check_errors` (deletion
checks that could not be read), `build_log_errors` (build outputs that could not be read),
`failed`, and its reasons.

`failed` means this project's run failed or was cut. Every such failure adds one to `failed`
and one to exactly one reason, so `failed` is always the sum of the `failed_*` keys:

- `failed_rate_limit`, `failed_budget`, `failed_auth`, `failed_unavailable`: a cut, which ends
  the run, so at most 1;
- `failed_response`: a raised run on a `200` that is not JSON or of no documented shape, at
  most 1;
- `failed_error`: a raised run on another error answer (a 3xx, another 4xx, a 404 or 410 on
  the listing), at most 1;
- `failed_state`: a raised run on a state of another project, team or version, at most 1.

Decision (Goran, 2026-11-12): item errors are visible as counters (`check_errors`,
`build_log_errors`) but are never failures of the run. A host blocks retiring every deselected
table of the connection on `failed > 0` (`ingestion.retirement_blocked`), so one deployment of
one project whose check or build output keeps failing would otherwise keep a project the owner
deselected in memory for as long as it failed. The item itself is kept as it was and asked again
in the next pass.

## Limits

Each number is this connector's own choice unless a source is named:

- Retries: a 5xx or no answer is retried twice, after 1 s and 2 s (`_RETRIES = 3` attempts),
  with `check_active` before every attempt and wait. Stated choice, taken from the Asana
  connector's policy; Vercel documents no retry guidance. Nothing else waits.
- Timeout per request: 10 s to connect, 30 s to read (`_TIMEOUT`). Stated choice; Vercel
  documents no latency. A build output that takes longer than 30 s to read counts as failed
  (`build_log_errors`) and costs up to three attempts per pass.
- Page size: 100 (`PAGE_SIZE`). Not documented by Vercel: the spec gives `limit` an example of
  10 and no maximum; the community author observed live that the list answers at most 100.
- Request budget: none by default; at least 2 when set (the project and one page, or a run could
  never move on), or `0` to defer the run: no request, no state read or written (a pass in
  progress stays as it is), no row, and stats `failed` 1 / `failed_budget` 1, a cut. `1` and
  negative values raise `ValueError` when the source is built. Per pass the
  cost is 1 (project) + ceil(window deployments / 100) + one read per known deployment the pass
  did not list + one events call per new failed deployment when build output is on, plus the
  head pages of every run that continues a pass. Vercel documents 1000 deployment-list requests
  and 500 deployment reads per 60 s per user, shared by every token of that user.
- No size cap on build output (a log can be up to 4 MB, roughly a million LLM tokens to
  cognify); turn it on knowingly.
- The window (`lookback_days`) bounds reading only. Its default stays `None`, Goran's decision
  (2026-11-12): any default window would be an invented limit, and a deployment that leaves a
  window is never checked for deletion again. A deployment that left it keeps its document
  and is no longer checked, so a deletion at Vercel after that is not seen. With the default
  (no window) every deployment Vercel lists is read and checked each pass.
- Each sync reads the whole staging table back, so an idle re-sync grows with the project.

### Measured

Measured 2026-11-12 on an Apple M4 Pro (14 cores, 48 GB, macOS 27.2), Python 3.12, against the
fake Vercel API (no network) with the LLM and embeddings mocked (the journeys mock, 512
dimensions), so the LLM time is not in these numbers and the extraction count is the number of
calls a real model would get. One project, 3% of deployments in state ERROR with build output on
(`include_build_logs=True`). Graph on Ladybug; "SQLite" has relational and staging on SQLite and
vectors on LanceDB, "Postgres" has relational, staging and vectors (pgvector) on Postgres 17.

| | 300 deployments, SQLite | 300 deployments, Postgres | 3000 deployments, SQLite |
|---|---|---|---|
| requests, first sync | 15 | 15 | 123 |
| requests, idle re-sync | 5 | 5 | 32 |
| extract (`add`), first sync | 3.3 s | 5.5 s | 28.1 s |
| whole `remember` (add and cognify), first sync | 77 s | 75 s | 1912 s |
| whole `remember`, idle re-sync | 1.9 s | 1.6 s | 47.8 s |
| whole `remember`, 5 deployments edited | 6.2 s | 4.7 s | 46.9 s |
| documents per record (deployments, project, build logs) | 1.0 (311) | 1.0 (311) | 1.0 (3092) |
| chunks per record | 1.0 | 1.0 | 1.0 |
| staging rows | 311 | 311 | 3092 |
| resource state | 22 KB | 22 KB | 220 KB |
| LLM extraction calls: first sync / idle / 5 edited | 311 / 0 / 5 | 311 / 0 / 5 | 3092 / 0 / 5 |

Requests on a first sync are 1 (project) + the listing pages (100 a page, plus one where the
one-millisecond overlap carries a page over) + one events read per failed deployment; an idle
re-sync is the project and the listing only. The 3000 run hit Ladybug's maximum database size with
the default (`KUZU_MAX_DB_SIZE`, 32 GB of address space) during the first cognify and was measured
with 128 GB; the first cognify of 3000 documents is the slow part (32 minutes against 77 seconds
for 300), which is cognee's per-document cost, not the connector's.

### Known limitations

Provider facts not verified against a live account (the live check is postponed):

- whether a Read-scoped Integration token is accepted on `/v9/projects/{id}`,
  `/v7/deployments`, `/v13/deployments/{uid}` and `/v3/deployments/{uid}/events` (Vercel's
  scope table still names older versions);
- the Integration token's lifetime (documented as long-lived, no refresh token);
- the rate-limit response headers (none are read: a 429 cuts the run);
- that `until` is exclusive and that the list answers at most 100 a page (the community author
  checked both live);
- that a deleted deployment is listed with `readyState: DELETED` during its recovery period, and
  that one gone for good answers 404 or 410 on its own read;
- whether a deleted project's deployments are still listed, and whether build output outlives
  its deployment;
- that a team project's `accountId` is the team id (the account guard relies on it);
- whether Vercel ever answers with a UTF-8 byte-order mark: a body that starts with one is
  refused as not JSON (requests' `.json()` accepted it; the client now decodes plain UTF-8);
- that `follow=0` with `limit=-1` always ends the events read; the spec documents `follow` and
  `limit` but not `builds` or `delimiter`, which are not sent.

Open gaps. The ids are stable; each is either tracked for a fix or a decision, as it says:

- **VERCEL-GAP-1.** A run of 100 or more deployments created in one millisecond stops the pass
  (`cursor did not move`, counted as `failed_response`): paging by time cannot step over them,
  so that project's passes never complete and its deletions are no longer detected. Nothing is
  lost or wrongly forgotten. Vercel's daily ceiling is 24000 deployments, so this needs a burst no
  documented flow produces; a fix needs an id cursor from Vercel.
- **VERCEL-GAP-2.** The inspector URL is in the row (`url` is identity) and, by the spec's
  example (`https://vercel.com/acme/nextjs/<id>`), carries the team slug and the project name.
  If Vercel builds it from the current names, a team slug change or a project rename changes
  every deployment's row and re-cognifies the whole project. Not verified live; settled by the
  live check.
- **VERCEL-GAP-3 (Cloud follow-up).** Cloud keeps only the counters it knows; `unresolved`,
  `failed_response`, `failed_error`, `failed_state`, `check_errors` and `build_log_errors` are
  not in its counter set yet, so Cloud drops them until it adds them (the names stay).
- **VERCEL-GAP-4 (closed 2026-11-12).** A build output or a deletion check that always fails
  used to count in `failed` and block the host from retiring deselected projects. It now counts
  in `check_errors` / `build_log_errors` once per pass while its deployment is in the window
  (always, with no window); `failed` stays 0 and retirement goes ahead. What stays: that item
  is asked again every pass and its count stays above 0 until Vercel answers it.
- **VERCEL-GAP-5 (by design: a delay, never a loss).** A run that continues a pass reads again only the head above the newest
  deployment an earlier head (or the pass's first page) saw, plus its first page (and, under a
  budget, only while a request is left for the pass); an edit to a deployment below that, which
  an earlier run of the pass already read, waits for the next pass.
- **VERCEL-GAP-6 (decided 2026-11-11).** A deployment deleted after it left the window keeps its
  document: the window bounds reading, never memory, so leaving it is not evidence of a
  deletion. Its build output keeps its document too. With the default `lookback_days=None`
  there is no window and no such deployment.
- **VERCEL-GAP-7 (by design).** The single read's `softDeletedByRetention` flag is not used: a
  deployment in its recovery period can still be restored, so it is kept until Vercel lists it
  as `DELETED` or answers 404 or 410.
- **VERCEL-GAP-8 (decided 2026-11-11).** The team's display name is not read (it needs the Team read scope), and
  there is no team level in the structure (see "Graph shape").
- **VERCEL-GAP-9 (settled by the live check).** Not covered: the ranking of hits and the answer itself (A1, A14 through a
  real LLM and real embeddings). Covered on the mocked stack: what A1 needs, from the stored
  documents and the graph (`test_vercel_forget.py`), and A14's narrowing, where
  `recall(node_name=["vercel:deployment"])` returns only deployment chunks, of both projects
  (`test_vercel_search.py`).
- **VERCEL-GAP-10 (Cloud follow-up).** The pod gives dispatch units past 40 a budget of
  0. The SDK side is done: `max_requests=0` builds the source and defers the run (no request, no
  state, no row, `failed_budget` 1, a cut), so every unit is settled. The pod must still pass the
  0 through and treat the unit as a cut scope, not as a failure that blocks retiring deselected
  tables. In tree, the generic `ingestion.retirement_blocked(counts)` blocks on any cut unless
  the host passes `cut_scopes` (no in-tree host does on this base; the test
  `test_a_deferred_project_is_a_cut_scope_and_a_deselected_one_is_still_retired` pins both).

Also: `update()` refuses document dlt sources; `forget()` does not remove the staged copy
(below). No layer 3 (integration module, REST routes) and no packaged copy under
`cognee-connectors/` yet.

### Known risks accepted

- **Provider text in prompts.** Commit messages and build output can be written by people
  outside the team (a fork pull request's preview deployment, for example), and they reach
  memory and the answers built from it. Line forging is blocked (a value cannot add a heading or
  a `Label: value` line of its own), but instruction injection through that text is not.
  Accepted by the product owner on 2026-11-11; no code guards it.

## What a search hit carries

Measured on 2026-11-12 with the real pipeline and the mocked LLM and embeddings (one team, web
with 40 deployments of which 3 failed with build output, api with 5): 47 records and 3 build logs
give 50 documents (45 deployments, 2 projects, 3 build logs), every title unique.

| Surface | What it carries | Title | Deployment url |
|---|---|---|---|
| `Data` row (`get_dataset_data`) | `name` `text_<md5>`; `system_metadata`: `title`, `url`, `external_id`, `source`, `table_name`, `structure`; `external_metadata.node_set` | yes | yes |
| `search(..., CHUNKS)` hit | chunk fields: `text`, `document_id`, `document_name` (`text_<md5>`), `chunk_index`, `belongs_to_set`, `score`, provenance (`source_*`) | only as the text's first line (`# <title>`) | no |
| `recall(..., CHUNKS)` item | `text`, `score`, `metadata` with `data_id`, `chunk_id`, `chunk_index`, `document_name` | only in the text | no |
| `recall(..., HYBRID_COMPLETION, only_context=True)` item | the assembled prompt as `text`; `metadata` is empty | inside the prompt text | no |
| `search(..., include_references=True, verbose=True)` | `evidence`: RAG_COMPLETION gives one `EvidenceReference` per chunk (`data_id`, `chunk_id`, `document_name`, `source_ref_key`, `score`); HYBRID_COMPLETION gives none | no | no |
| `search(..., include_references=True, verbose=True)`, `objects_result` chunk payloads (RAG, HYBRID, TEMPORAL; since SDK-1500) | `document_title`, `document_url`, `document_parent` (the project) | yes | yes |

Without references a user sees the title only because the connector writes it as the first line
of the text. With `include_references=True` the retrievers that cite chunks (RAG, hybrid,
temporal) copy the row's title, inspector url and parent into each cited chunk's payload
(re-measured 2026-11-13 on dev 688e454b5: 5 of 5 cited chunks on each). CHUNKS, the evidence
list and recall items still name the row only by `data_id`.
The `node_name` filter works with the prefixed name: `recall(node_name=["vercel:deployment"])`
returned the 45 deployment chunks of both projects and nothing else, `["vercel:project"]` the 2
project chunks, and a project name (`["web"]`) nothing. The probe is not part of the suite; the
filter is pinned by `test_vercel_search.py`.

## Permissions

| Question | Answer | Code (this base) |
|---|---|---|
| Who owns the dataset | the user the sync runs as; a dataset id derives from its name and owner, so two users' `vercel_acme` are two datasets | `cognee/modules/data/methods/create_dataset.py:14,27,30` |
| What the owner gets | read, write, delete and share on the new dataset; an agent's human parent gets the same | `cognee/modules/data/methods/create_authorized_dataset.py:14,36-37,39-53` |
| Who else can read it | a principal with a read grant: the user, the agents the user created, a tenant the user belongs to, a role the user has; only datasets of the user's own tenant count | `cognee/modules/users/permissions/methods/get_all_user_permission_datasets.py:24-53` |
| How it is shared | `POST /v1/permissions/datasets/{principal_id}?permission_name=read` by a user who holds `share` on it | `cognee/api/v1/permissions/routers/get_permissions_router.py:38`; `cognee/modules/users/permissions/methods/authorized_give_permission_on_datasets.py:31-36` |
| What a search checks | the read permission on the dataset, never on a document | `cognee/modules/data/methods/get_authorized_existing_datasets.py:31-45` |
| With backend access control off | no isolation: one shared store for every user | `cognee/context_global_variables.py:98-107` |
| Per-document access | none: every project, deployment and build log of a synced project is readable by everyone with read on the dataset; Vercel's own team roles are not mirrored | the connector writes no ACL (`vercel.py`) |
| Provider side | what the token can read: a personal access token reaches the account or team it was created for; an Integration token the team it was installed on, with the scopes the Integration asks for (Read on Deployment and Project). Every call carries `teamId`, and a project of another account is refused before any row | `vercel.py:578` (`teamId`), `vercel.py:640` (account guard) |
| Identity, SDK | the caller names the dataset and passes `resource_name` (one per user and project) | `vercel_source(resource_name=...)` |
| Identity, host and Cloud | `resource_name` is `vercel_` plus a hash of the owner, the account and the project, so another owner of the same team starts from nothing; one dataset per Vercel team or personal account (`vercel_<account>`, the plan's rule; no layer 3 builds it in tree yet) | `cognee/modules/integrations/ingestion.py:147-159` |

## Privacy

What is read goes to the configured LLM and embedding providers, and anyone with read access to
the target dataset can see it: deployment and commit metadata of the project, commit messages
the commit author's name and GitHub/GitLab/Bitbucket login, and the creator's Vercel username.
Emails are never requested into a row. Build output is off by default; when on, it is stored as Vercel serves it, and Vercel
redacts only secret values of 32 characters or more.

The connector logs no id, team id, URL or token. requests' own urllib3 logs each request line at
DEBUG (`GET /v9/projects/<id>?teamId=<team> ...`), so with DEBUG logging on for `urllib3` the
team and project ids reach the log; the token never does (it is a header). Keep `urllib3` above
DEBUG where those ids must not be logged.

`forget(data_id=...)` and `forget(dataset=...)` remove documents, not the dlt staging copy, so
the next sync builds them again while the deployment exists at Vercel. Erasure goes through
Vercel (delete the deployment) or the selection (deselect the project, or switch build output
off). A disconnect with delete (`forget(dataset=...)` plus
`ingest_dlt_source.drop_dlt_dataset(dataset)`) leaves nothing for the next connect.

## Changes from the community connector

Each fix has a test (`test_vercel_*.py`):

- One table named `resource_name` instead of three per source: a host empties every table it
  did not get back, so three tables were emptied on every sync.
- `merge` with tombstones on evidence instead of `replace`: hosts always pass `merge`, so
  deletions never reached memory, and the 30-day window forgot data Vercel still had.
- `check_active` before every request attempt and retry wait through `dlt_utils.with_retries`,
  instead of after a response.
- `service=` alone is enough; `build_vercel_service(access_token)` is the `_PROVIDERS` builder;
  no `session` alias, no `requests.Session` service.
- Clean cuts (`_CutShort`) on 429, budget, 401/403 and outages instead of raising and losing
  the run; host stats vocabulary (ints only, no per-kind or list-valued keys).
- A pass over the listing that spans runs, so a budget smaller than the listing still gets
  through it, with deletion checks only after a complete pass.
- Build output that cannot be read is counted per deployment instead of cutting or failing the
  run; output that is empty or gone is not asked again; the events read is finite (`follow=0`).
- The single deployment read is checked against its own `/v13` shape, not the list's.
- Typed errors (`VercelSourceError`, `VercelAPIError`, `VercelAuthError`,
  `VercelRateLimitedError`, `VercelNotFoundError`, `VercelResponseError`) that name the
  documented route only and are raised outside any `except`, so no URL, team id or project id
  sits in the exception chain dlt pickles into its trace. The source never raises
  `VercelRateLimitedError` (a 429 is a cut), so it is not an `ingestion.RateLimitedError`.
- No module-level `dlt`/`rest_api` import; a small `requests` client.
- One project id, required, by id only (a name is refused); no empty selection meaning
  "everything"; `resource_name` required.
- Prefixed row ids, state in `dlt.current.resource_state()` with a version key.
- Removed with the one-project design (one source per project, so the host's selection and
  table retire replace them): the projects listing, `_RunProjects`, the
  `parallelize()` override, `forgotten_projects`, `allow_empty`, the empty and missing project
  guards, and the staging read-back of build logs Vercel no longer serves (under `merge` the
  staged row stays). The account guard stays, as a cut.
- No build-log size cap (`max_log_chars` dropped data); every line is read with `limit=-1`.
- Personal (Hobby) accounts: no `teamId`, account guard on the user id.
- Terminal escapes and control characters are removed from every provider string, not only
  logs; commit author emails are never read, and the author's name is rendered next to the
  login (`Name (login)`) instead of replacing it when the login is missing.

## Testing

```bash
uv run pytest cognee/tests/unit/tasks/connectors/test_vercel_source.py \
  cognee/tests/unit/tasks/connectors/test_vercel_client.py \
  cognee/tests/unit/modules/integrations/test_ingestion_contract.py \
  cognee/tests/integration/tasks/connectors/test_vercel_ingestion.py \
  cognee/tests/integration/tasks/connectors/test_vercel_forget.py \
  cognee/tests/integration/tasks/connectors/test_vercel_search.py
```

No Vercel account or token is needed. `cognee/tests/unit/tasks/connectors/fake_vercel.py` is an
in-memory Vercel API shaped by its OpenAPI document: projects with their secrets, a deployments
list paged by an exclusive `until`, per-deployment reads that answer 404 or 410, build events in
both documented shapes, and a 403 for a call without the right `teamId`. The walk is tested on a
plain dict as state, through a real `dlt` pipeline into SQLite, and through the real client with
only the transport faked, and through a real socket: `test_vercel_wire.py` starts a local HTTP
server and checks what it received (the Bearer header, the paths and queries, no redirect
followed, retries, timeouts, a sync through a real dlt pipeline); `test_vercel_forget.py` runs
the real cognee pipeline with the LLM and embeddings mocked and replays the plan's user scenario
end to end.

The suites force SQLite. On 2026-11-12 the three integration files ran once on Postgres 17.11
(pgvector 0.8.7, Docker, on a scratch copy with that forcing lifted): relational, vector
(pgvector, one database per dataset) and dlt staging on Postgres, graph on Ladybug; 30 of 30
pass. Two things only Postgres shows, both about the tests, not the connector: `table_name` is
schema-qualified there (`<dataset>.<table>`), so a test compares the bare name as the host's
retire does; and the staging database of a dataset lives on the server, not in the test's
temporary directory, so `prune` leaves it and a later test that reuses a dataset and resource name
reads the earlier test's rows back unless its staging is dropped (`drop_dlt_dataset`).
