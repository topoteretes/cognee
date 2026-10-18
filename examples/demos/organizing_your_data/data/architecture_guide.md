# Acme Architecture Guide — Sync Engine

The sync engine replicates data between regions asynchronously. Writes are
acknowledged in the origin region first and propagate to the other regions
with an **eventual-consistency window of up to 60 seconds**.

Sync is therefore *not* real-time: a read in a remote region immediately
after a write may return stale data. Applications that need read-after-write
consistency must pin reads to the origin region.
