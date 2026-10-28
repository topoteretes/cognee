# Integration sync progress

SDK-managed integrations expose the same progress contract at:

```http
GET /api/v1/integrations/{provider}/sync/progress?account_id=...
Authorization: Bearer <token>
```

Use the account ID of an active connection owned by the authenticated user.
Omitting it selects that user's connection using the existing integration API's
selection rules. A connection with no recorded runs has no latest progress yet.
This endpoint reads status; it does not start or retry a sync.

The response contains `latest`, `activeRuns` and `unknownActivityCount`. A run
includes `state`, `phase`, `startedAt`, `updatedAt`, `endedAt`, `elapsedSeconds`,
`completed`, `total`, `unit`, `counts`, `reason`, `phasePercent`, `phaseEtaSeconds`
and `executionActive`. `executionActive: null` means activity is unconfirmed.

The response separates the latest run from runs confirmed active in this API
process. Nonterminal records from another process or a previous process lifetime
have unconfirmed activity. A persisted `running` state describes the last report;
it is not proof that a worker survived a restart. No timeout invents a failure or
silently starts the job again.

## What the user can see

- The run ID, provider, current phase and state.
- Start, update and finish timestamps and elapsed time.
- Phase-local progress with its unit and total, when the total is known.
- Provider counters such as scanned, skipped, failed or completed repositories.
- A safe reason for waiting, blocking, partial completion or failure.
- A phase-local estimated remaining time when measurable; unknown values are null.

Phases describe actual work: checking, discovering, downloading, processing,
indexing and finalizing. Providers need only emit the phases they can observe.
For example, a DLT `remember()` call combines extraction and indexing, so it is
reported as processing rather than pretending to know an internal phase boundary.

Progress units matter. Finishing one selected folder is not the same as indexing
one file. Source `scanned` counts measure extraction, not successfully indexed
documents. Skips and failures remain visible even when every selected scope has
been attempted. Partial completion is not success.

## Percentages and estimates

A percentage belongs to one phase with a known total. It must not be displayed
as the percentage of the whole sync. Discovery may have no total until provider
pagination finishes; clone size and indexing work may also be unknown. Show the
phase and elapsed time instead of an artificial percentage.

An estimate based on the observed phase throughput is approximate, not a deadline.
Heterogeneous files and folders, rate limits and provider latency can change it.
Do not project that estimate across later phases. Waiting and terminal failures
must not show a countdown or imply that the work completed successfully.
The estimate requires two distinct progress samples and describes the observation
at `updatedAt`; it is not a live countdown while no new progress arrives.

`waiting_for_lock` means another operation holds the dataset lock;
`waiting_for_worker` is reserved for a host that actually queues worker execution.
`rate_limited` indicates an observed provider limit, not a guessed cause of silence.

## Coverage and hosting

Google Drive, Gmail, Google Calendar, Notion and Linear report through the shared
SDK ingestion helpers. GitHub code ingestion and Slack history imports report
through the same progress contract from their own execution paths.

The SDK stores progress in the existing operation ledger. Telemetry failures do
not fail an otherwise valid sync or replace its original error. Progress contains
counts and safe categories, not document text, access tokens or provider exception
messages. Existing provider logs and pipeline errors retain their own policies.
Unchanged observations do not cause another storage write. A background writer
persists the latest changed observation without making each source item wait for
storage. While delivery is busy, intermediate observations coalesce into the latest
snapshot; this is a status record, not a history of every phase transition. Initial
and terminal delivery are awaited, including when an operation is cancelled.
After a worker exits unexpectedly, the last durable observation may lag its last
in-memory update. Report at meaningful provider or processing boundaries.

Cloud hosts and third-party integrations can use the same tracking API and a
progress sink. A sink receives these same coalesced snapshots; it does not schedule work or enforce
resource limits. Cloud must connect these observations to its durable jobs and UI.
Likewise, queued/waiting/blocked states require an actual scheduler or policy
decision; the SDK does not infer them from silence.

This feature does not set a GitHub repository-size limit, allocate larger workers,
or resolve a pod eviction loop. Those need an explicit execution budget and host
policy. It makes the resulting progress and decisions observable consistently.
