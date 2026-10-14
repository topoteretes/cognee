# Typed capture receipts

`POST /api/v1/remember/entry` accepts an optional UUID `client_event_id` for QA
and trace entries. Persist it with the event before sending. Two separate events
must have different IDs even when their content is identical. Omit the field to
retain the existing endpoint behavior. Feedback and skill-run idempotency are
outside this capture contract.

First admission requires an existing dataset and write authorization; capture never
creates a dataset. The server scopes uniqueness
by tenant, authenticated user, resolved dataset UUID, session and event ID. The
SQL cache commits the normalized payload hash, stable receipt and session entry
in one transaction. A duplicate returns the stored receipt; different content
under the same scoped ID returns HTTP 409 with `code=capture_payload_conflict`.
Use a new session ID when switching datasets. This is stricter than legacy typed
entries: the cache uses a (user, session) namespace, so capture requires a verified
session-to-dataset binding on first admission. A mismatch returns HTTP 400 with
`code=capture_session_dataset_mismatch`; binding storage failures are retryable 503s.
`skill_improvement` returns 400 / `capture_skill_improvement_unsupported`.

Retries first look up the receipt by authenticated tenant/user, session, event and
original dataset UUID (or the dataset name recorded at admission). They validate
the payload hash and return the original receipt without touching session state,
including after dataset deletion, rename or permission revocation. This endpoint
returns only that caller's acceptance metadata, never stored entry content.
Incomplete QA indexing is retried only while the original dataset remains writable;
otherwise the original acceptance receipt is returned without a new vector write.
The receipt is not a guarantee that indexing or content remains available.

`/health` advertises `capture_idempotency_v1` from configuration alone when caching is enabled on a SQL
backend (SQLite, Postgres or Turso). The cache's existing create-on-startup schema
mechanism installs the receipt table, including standalone `cache.db` deployments.
Redis, filesystem and tapes caches do not advertise it; requests carrying an event
ID fail explicitly on those backends. Legacy clients remain compatible.

QA vectors use the stable receipt entry ID. An indexing lease serializes ordinary
concurrent retries without holding a database transaction across the embedding
API. Failed indexing releases the lease; a crashed worker's lease expires after
120 seconds, and indexing attempts have a 60-second deadline. A crash after the
vector write can repeat the external embedding call, but upserts the same vector
ID. This is one stored entry/vector, not an exactly-once billing guarantee.
Trace feedback is generated before the cache transaction with a 60-second timeout;
ordinary retries skip it. Concurrent first attempts can repeat the feedback call.
Clients must keep the event until the matching receipt arrives.

Receipts contain a hash and response metadata, never the original capture text.
They outlive session TTL/deletion so delayed retries cannot resurrect removed
entries. They are retained for 90 days from admission and then removed by the throttled
cache sweep (active indexing leases are excluded). Clients must stop retries before
that horizon: deduplication is not guaranteed beyond it. Prune clears receipts and
account deletion removes that user's receipts. A receipt confirms
the original acceptance; it does not assert the entry still exists after deletion.
