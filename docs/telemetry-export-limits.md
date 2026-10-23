# Bounded product telemetry export (SDK-1168)

This extends SDK-1082's single batching sender. It does not enable Dash0 metrics,
change the telemetry endpoint, implement the metadata-only privacy schema
(SDK-1167), or deploy anything to Cloud. Resource limits apply to **all SDK
origins**; the additional event-selection policy applies to `cloud` and
`cloud-*` origins. An HTTP success is a proxy acknowledgment, not proof of
warehouse ingestion.

## Defaults and configuration

Settings are read once when the process creates its sender. Integers are bytes
or units per minute, not unit-suffixed strings. Invalid/zero/negative settings
fall back to finite defaults; inconsistent event/queue/batch sizes fall back as
a group. Use the existing `TELEMETRY_DISABLED` switch to disable collection.

| Environment variable | Default | Maximum accepted setting |
| --- | ---: | ---: |
| `TELEMETRY_EVENT_BYTES` | 16,384 (16 KiB) | 65,536 |
| `TELEMETRY_QUEUE_BYTES` | 1,048,576 (1 MiB) | 16,777,216 |
| `TELEMETRY_BATCH_BYTES` | 65,536 (64 KiB, including JSON framing) | 262,144 |
| `TELEMETRY_EVENTS_PER_MINUTE` | 120 | 1,000,000 |
| `TELEMETRY_BYTES_PER_MINUTE` | 262,144 (256 KiB) | 1,073,741,824 |
| `TELEMETRY_REQUESTS_PER_MINUTE` | 60 | 6,000 |

The count cap is 1,000 queued events; batches contain at most 100 events and
flush after two seconds or when full. Each rate budget is a token bucket with
one minute of initial burst capacity and continuous refill. Event and logical
byte budgets apply on admission. **Separate wire-byte and request budgets also
charge retries and legacy single-event requests.** No budget waits or sleeps on
the product thread. Shutdown/reopening does not refill budgets.

With N processes/replicas, multiply every local budget and memory allowance by
N. Four processes can admit 480 events/minute and spend 1 MiB/minute in request
bodies, with an additional initial burst of those amounts; the request cap is
240/minute. Restarts/forks create fresh buckets. These are neither tenant-wide
limits nor a monthly dollar cap. Shared-process tenants also share budgets.
Fleet-wide quotas, fairness, request/billing overhead, restart amplification
and budget alerts require proxy/platform enforcement (CLO-803).

## Admission, memory and loss

- Cloud events are selected **before** property sanitization and identity work.
  `CLOUD_EVENTS` in `telemetry_limits.py` is the closed volume-policy list:
  pipeline and incremental-update run milestones, operation starts/outcomes,
  and the memory API's existing outcome events. Pipeline item, internal task,
  chunk/token, endpoint polling and unknown events are suppressed. No random
  sampling is used, and existing `pipeline_run_id` properties are preserved.
  Interrupted runs remain detectable when their start reached the warehouse.
- Oversized, non-JSON, cyclic or excessively nested properties are rejected
  before copying/sanitizing them. Encoding visits at most 2,048 values and
  16 nesting levels. Non-finite floats, integers outside 64 bits and arbitrary
  object conversions are rejected. The final complete event must fit too.
  This bounds resource use; it is **not** a content allowlist/privacy guarantee.
- Only serialized immutable bytes enter the queue, so mutation after emission
  cannot grow or change a queued event. The queue holds at most 1 MiB plus the
  bounded overhead of 1,000 byte objects/deque references. New events are
  dropped when full; accepted older events are not evicted.
- One in-flight batch owns encoded event bytes plus a contiguous request body,
  together at most twice the 64 KiB batch cap. One worker/connection performs
  network IO. The sender does not consume response bodies.
- Payload preparation and serialization each have a nonblocking admission
  lock. Concurrent contenders are counted and dropped. Short queue/counter
  bookkeeping uses mutexes; serialization, network IO, capacity waits and retry
  sleeps never hold the queue lock or wait on the product thread.
- Bounded encoding checks string length before escaping. At most one scalar
  temporary per encoder can occupy 12 times the event limit plus framing
  (non-BMP characters), in addition to the bounded output buffer. The normal
  `send_telemetry` path allows one preparation at a time; direct sender users
  can add one concurrent encoder. No recursive closure retains output buffers
  pending cyclic GC. Sanitized objects are bounded by the same input traversal
  limits. Caller-owned input objects, SDK identity work, allocator/TLS/thread
  overhead and the application itself are outside the serialized-byte gauges.
- Saturation can lose either a start or its outcome. Reporting must consult
  loss counters/coverage instead of interpreting every unmatched start as a
  product failure. Delivery is best effort, without disk spooling.

## Failure, compatibility and shutdown

Network errors/timeouts and 5xx have at most four attempts, with the existing
1–4 second exponential jitter, further constrained by request/wire-byte budgets.
429, 413, authentication and other ordinary 4xx are counted and dropped without
retry or per-event fallback. Rejected requests never increment exported events.

Legacy mode requires either explicit deployment configuration
`TELEMETRY_LEGACY_PROXY=true`, or a **422 response with the explicit header**
`X-Cognee-Telemetry-Batch: unsupported`. A bare FastAPI 422 is insufficient.
Legacy requests use the same budgets and stop the batch on their first failure.
Today's old proxy does not advertise this header: deploy batching support first
or explicitly opt into the budgeted legacy mode. Do not infer compatibility
from payload validation, authentication or throttling responses.

`close()` has one request-timeout total waiting budget (the existing timeout is
5 seconds by default). It interrupts backoff, does not retry failures while
closing, and drops leftovers on failure or deadline. No new HTTP request may
start after the deadline. An already-running request cannot be force-cancelled
by the synchronous HTTP client: if it outlives the wait, one daemon worker and
one bounded batch remain, admission stays closed, and no second worker starts.
Once the worker exits, a later emission may reopen the sender. Process exit does
not wait for a daemon worker. Forked children discard inherited queues, locks,
worker state and counters, avoiding duplicate parent-event delivery.

Payload preparation failures never replace a product return value, error or
cancellation. Internal diagnostic failures log a synthetic, redacted exception
instead of the original exception, request URL, response or payload.

## Observing loss without exporting more telemetry

The existing process sender exposes a thread-safe local snapshot:

```python
from cognee.shared import utils

state = utils._telemetry_sender.snapshot()
```

This internal diagnostic hook is available to existing process monitoring; no
new background report, HTTP endpoint or Dash0 metric stream is enabled here.
Counters have fixed names and saturate at 2^63−1. There are no tenant/event-name
labels, payloads, credentials, exception texts or dynamic dictionary keys.

- `accepted_events`, `exported_events`, `exported_bytes` (event bytes, excluding
  array framing); `requests`, `request_bytes` include every HTTP attempt/framing.
- `sampled_events` means deterministic cloud-policy suppression, not a random
  sample suitable for statistical extrapolation.
- `rejected_size`, `rejected_format`; `dropped_queue`, `dropped_rate`,
  `dropped_shutdown`, `dropped_network`, `dropped_http`, `dropped_internal`,
  `dropped_busy`; plus `retries` and `legacy_fallbacks`.
- Queue count/bytes, in-flight bytes (including the assembled body), and peak
  queue/in-flight bytes. Pending in-flight events remain unresolved until their
  request finishes. Exported counts acknowledge 2xx only.

## Measurements and release gates

Measured 2026-10-23, macOS arm64, Python 3.11.15, HTTPX 0.28.1. Five-second
continuous synthetic loads, one and four independent processes; each process
runs each scenario. SDK import is outside the measured incremental footprint.
`tracemalloc` is enabled; RSS is sampled every 10 ms. P99 is the first 1,024
admissions, not a production latency SLA. The unavailable proxy returns 503;
a separate regression holds a request indefinitely to verify the close deadline.

| Processes | Scenario | Attempts per process | HTTP attempts per process | Max traced allocation per process | Max observed RSS growth per process | Max sampled admission p99 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 512-byte padding, default budgets | 579,000 | 3 | 242,170 B | 409,600 B | 0.130 ms |
| 4 | 512-byte padding, default budgets | 467,776–487,088 | 3 | 254,682 B | 589,824 B | 0.256 ms |
| 1 | 15,000-byte padding, default budgets | 64,571 | 6 | 400,451 B | 131,072 B | 0.129 ms |
| 4 | 15,000-byte padding, default budgets | 54,728–57,552 | 6 | 415,007 B | 1,343,488 B | 0.364 ms |
| 1 | Saturated queue, 15,000-byte padding, proxy 503 | 65,001 | 4 | 1,263,931 B | 671,744 B | 0.221 ms |
| 4 | Saturated queue, 15,000-byte padding, proxy 503 | 57,257–58,011 | 4 | 1,264,437 B | 1,900,544 B | 0.247 ms |

The saturation scenario raises admission budgets to 1,000,000 events/minute
and 1,000,000,000 bytes/minute to exercise the queue cap; queue/batch/event caps
stay at their defaults. Maximum observed queued bytes: 1,040,313; in-flight
bytes: 120,621. Maximum measured close time: 1.503 ms with these responsive mock
transports. The stalled-request test separately verifies a 100 ms configured
close deadline, drops queued work and forbids later retries/requests.

These measurements support a small exporter envelope, not an absolute process
RSS guarantee or validation against real tenant workload distributions. Initial
120-event/256-KiB per-minute budgets deliberately sacrifice excess diagnostics.
Review actual admission/drop rates before raising them. The bounded-encoding
regression also runs 2,000 large encodes with cyclic GC disabled.

Reproduce the load measurement (requires the SDK environment plus `psutil`):

```bash
python scripts/benchmarks/telemetry_export.py --processes 1 4 --seconds 5
```

Release sequence:

1. Deploy proxy PR #5 with `firehose:PutRecordBatch`, compatible warehouse
   timestamps and verified single/batch ingestion.
2. Merge SDK-1082 / cognee #5913, then SDK-1168; retarget the stacked PR to `dev`
   when its dependency lands. Publish the SDK with these limits documented.
3. Validate dev traffic, drop counters, run-start/outcome coverage, proxy outage
   behavior and representative concurrent workloads before cloud adoption.
4. Coordinate the metadata schema (SDK-1167), storage retention (CLO-1302), and
   fleet-wide budgets/rollout (CLO-803). Keep the telemetry kill switch available.
   This change does not certify those separate controls or turn them on.
