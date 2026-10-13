# Typed capture receipts

`POST /api/v1/remember/entry` accepts an optional UUID `client_event_id` for QA
and trace entries. Persist it with the event before sending. Two separate events
must have different IDs even when their content is identical. Omit the field to
retain the existing endpoint behavior. Feedback and skill-run idempotency are
outside this capture contract.

The server authorizes the write dataset on every attempt, then scopes uniqueness
by tenant, authenticated user, resolved dataset UUID, session and event ID. The
SQL cache commits the normalized payload hash, stable receipt and session entry
in one transaction. A duplicate returns the stored receipt; different content
under the same scoped ID returns HTTP 409 with `code=capture_payload_conflict`.
Use a new session ID when switching datasets, matching the cache's session binding.

`/health` advertises `capture_idempotency_v1` when caching is enabled on the SQL
adapter (SQLite, Postgres or Turso). The cache's existing create-on-startup schema
mechanism installs the receipt table, including standalone `cache.db` deployments.
Redis, filesystem and tapes caches do not advertise it; requests carrying an event
ID fail explicitly on those backends. Legacy clients remain compatible.

QA vectors use the stable receipt entry ID. An indexing lease serializes ordinary
concurrent retries without holding a database transaction across the embedding
API. Failed indexing releases the lease; a crashed worker's lease expires after
120 seconds, and indexing attempts have a 60-second deadline. A crash after the
vector write can repeat the external embedding call, but upserts the same vector
ID. This is one stored entry/vector, not an exactly-once billing guarantee.
Clients must keep the event until the matching receipt arrives.

Receipts contain a hash and response metadata, never the original capture text.
They outlive session TTL/deletion so delayed retries cannot resurrect removed
entries. They are not swept with expiring QA/trace content. A receipt confirms
the original acceptance; it does not assert the entry still exists after deletion.
