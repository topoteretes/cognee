// Selection cap for one upload. Raised 100 -> 200 (CLO-492): the old value was
// a workaround for sending every selected file in ONE multipart request, where
// the backend's per-call concurrent processing was the binding constraint
// (CLO-283). Uploads are now split into bounded batches (see batchFiles.ts), so
// the per-request load no longer scales with the selection, and the cap is only
// about how much a person can reasonably queue at once.
export const MAX_FILES_PER_UPLOAD = 200;

// Files per multipart request. Small enough that one batch completes well
// inside the per-request timeout even on a slow connection, large enough that
// 200 files don't cost 200 round trips.
export const FILES_PER_BATCH = 10;

// Byte ceiling for a single batch, applied before FILES_PER_BATCH — a batch of
// ten 40MB files would otherwise be one 400MB request. A file larger than this
// on its own still goes out alone (never split: the backend needs whole files).
export const MAX_BATCH_BYTES = 25 * 1024 * 1024;

// How many batches are in flight at once. Above ~3 the tenant pod becomes the
// bottleneck and per-request latency climbs, which shows up as a stalled
// progress bar rather than a faster upload.
export const BATCH_CONCURRENCY = 2;

// Per-batch request timeout. A batch is bounded by MAX_BATCH_BYTES, so this is
// a transfer-time budget, not a processing budget — graph building happens
// after the request returns (run_in_background), and is polled separately.
export const BATCH_TIMEOUT_MS = 3 * 60 * 1000;
