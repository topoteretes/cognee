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

All origins reserve capacity **inside** those totals for the actual operation/run
names in `MILESTONE_EVENTS`: 10% (rounded up), with a minimum of two events,
two maximum-size framed events in each byte budget, and two requests, capped at
the configured total. Defaults reserve 12 events, 32,772 bytes in each byte
budget and six requests. Detail events cannot consume the reserve; milestones
may consume any remaining capacity. Very small configured budgets can leave
no capacity for detail events. Item errors use detail capacity too.

Batches preserve emission order and contain either milestones or details, so
details cannot spend reserved wire/request budget by sharing a milestone batch.
This can create more requests than a single mixed batch, still within the same
hard request cap. For example, an SDK run start, 240 task events and a run
completion retains both run events while dropping excess task events.

With N processes/replicas, multiply every local budget and memory allowance by
N. Four processes can admit 480 events/minute and spend 1 MiB/minute in request
bodies, with an additional initial burst of those amounts; the request cap is
240/minute. Restarts/forks create fresh buckets. These are neither tenant-wide
limits nor a monthly dollar cap. Shared-process tenants also share budgets.
Fleet-wide quotas, fairness, request/billing overhead, restart amplification
and budget alerts require proxy/platform enforcement (CLO-803).

## Admission, memory and loss

- Cloud events are selected **before** property sanitization and identity work.
  `CLOUD_EVENTS` in `telemetry_limits.py` contains only names wired to emitters
  (checked by a source-level regression): pipeline/incremental run milestones,
  memory API events, search starts/outcomes, and `Pipeline Item Errored`.
  Item errors retain SDK-775's loader/extension context but are detail priority.
  Item starts/completions, internal tasks, chunk/token, **all API endpoint
  invocations** and unknown events are suppressed. This is deliberately not an
  endpoint census: add/cognify/memify are represented by `Pipeline Run` events.
  No random sampling is used; existing `pipeline_run_id` values are preserved.
- Before sanitization, raw properties must fit **four times** the event byte
  cap (64 KiB by default), 2,048 values and 16 nesting levels. This permits a
  32 KiB URL to shrink to a fingerprint without allowing unbounded raw input.
  The final complete event must fit the 16 KiB wire cap. Non-finite floats,
  integers outside 64 bits, cyclic objects and arbitrary object conversions
  are rejected. This is a resource bound, not a content/privacy allowlist.
- Only serialized immutable bytes enter the queue, so later mutation cannot
  change a queued event. The queue holds at most 1 MiB plus the bounded overhead
  of 1,000 entries, byte objects and deque references. Full queues drop new
  detail events. A milestone evicts newest queued details if necessary;
  survivors retain FIFO order. An all-milestone full queue drops the new event.
- One in-flight batch owns encoded event bytes plus a contiguous request body,
  together at most twice the 64 KiB batch cap. One worker/connection performs
  network IO. The sender does not consume response bodies.
- Preparation and serialization each admit **four concurrent calls**, waiting
  at most **25 ms per admission** before counting `dropped_busy`. The normal
  path can therefore wait up to 50 ms for these two slots, plus actual payload/
  identity CPU time and short bookkeeping locks; it never waits for network IO
  or queue capacity. PBKDF2 tracking-ID derivation caches only the latest
  key/salt pair. Warm calls reuse it; key or salt rotation recomputes it. Cold
  concurrent calls may compute the same value independently.
- Each bounded encoder holds an output buffer and up to two overlapping scalar
  representations (JSON text and ASCII bytes), each up to 12 times its input
  limit plus framing. Returning the encoded event can briefly copy the output
  buffer once more. Preparation's raw
  limit is 4 × event bytes; final serialization uses 1 × event bytes. At most
  four raw preparations and four direct sender serializers can run concurrently.
  The normal path shares its four preparation slots with serialization. No
  recursive closure retains buffers pending cyclic GC. Sanitized objects have
  the same traversal limits. Caller-owned input, the single cached identity
  pair, allocator/TLS/thread overhead and the application itself are outside
  the serialized-byte gauges. The byte caps are not an absolute RSS bound.
- Task/item storms cannot spend the milestone reserve, but milestone-only
  overload, invalid payloads, network failure, shutdown and process death can
  still lose a start or outcome. Reporting must consult loss counters/coverage
  instead of treating every unmatched start as a product failure. Delivery is
  best effort, without disk spooling.

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
closing, and drops leftovers on HTTP/network failure or deadline. A detail batch denied
by its budget is dropped independently, so following milestones can still use
their reserved capacity during close. No new HTTP request may
start after the deadline. An already-running request cannot be force-cancelled
by the synchronous HTTP client: if it outlives the wait, one daemon worker and
one bounded batch remain, admission stays closed, and no second worker starts.
Once the worker exits, a later emission may reopen the sender. Process exit does
not wait for a daemon worker. Forked children discard inherited queues, locks,
worker state and counters, avoiding duplicate parent-event delivery. One pair
of process-wide exit/fork hooks uses a weak sender registry; constructing test
senders does not accumulate hooks or retain closed instances.

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
`tracemalloc` is enabled; RSS is sampled every 10 ms. Each load uses a producer
thread (four in the concurrent scenario); producer/thread allocation is included.
P99 samples the first 1,024 admissions per producer, not a production latency SLA.
The concurrent scenario uses real `send_telemetry`, a synthetic LLM key, fixed
synthetic identities and a raw URL that sanitization shrinks. All four producers
start together; zero `dropped_busy` events were observed. Its CPU/scheduling time
is additional to the slot-wait bound; 16 producers across four processes reached
85.822 ms p99 under tracing. Cold key derivations may run concurrently. The
unavailable proxy returns 503; a separate regression holds a request indefinitely
to verify the close deadline.

| Processes | Scenario | Attempts per process | HTTP attempts per process | Max traced allocation per process | Max observed RSS growth per process | Max sampled admission p99 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 512-byte padding, default budgets | 316,213 | 4 | 284,131 B | 1,589,248 B | 0.066 ms |
| 4 | 512-byte padding, default budgets | 165,565–184,892 | 3 | 289,653 B | 1,343,488 B | 0.262 ms |
| 1 | 15,000-byte padding, default budgets | 51,166 | 6 | 415,462 B | 1,818,624 B | 0.958 ms |
| 4 | 15,000-byte padding, default budgets | 25,356–39,592 | 6 | 416,858 B | 3,080,192 B | 1.107 ms |
| 1 | Saturated queue, 15,000-byte padding, proxy 503 | 39,798 | 4 | 1,282,976 B | 2,965,504 B | 0.415 ms |
| 4 | Saturated queue, 15,000-byte padding, proxy 503 | 13,044–30,505 | 4 | 1,280,284 B | 2,916,352 B | 5.352 ms |
| 1 | Four producer threads, 32,000-byte URL, cold key then cached | 13,028 | 3 | 481,795 B | 7,110,656 B | 16.616 ms |
| 4 | Four producer threads, 32,000-byte URL, cold key then cached | 5,560–7,694 | 3–4 | 494,571 B | 5,799,936 B | 85.822 ms |

The saturation scenario raises admission budgets to 1,000,000 events/minute
and 1,000,000,000 bytes/minute to exercise the queue cap; queue/batch/event caps
stay at their defaults. Maximum observed queued bytes: 1,040,313; in-flight
bytes: 130,272. Maximum measured close time: 6.744 ms with these responsive mock
transports. The stalled-request test separately verifies a 100 ms configured
close deadline, drops queued work and forbids later retries/requests.

These measurements support a small exporter envelope, not an absolute process
RSS guarantee or validation against real tenant workload distributions. The
largest observed RSS increase was 7,110,656 bytes per process, including producer threads and
native allocations outside `tracemalloc`; the peak traced allocation was
1,282,976 bytes. Initial 120-event/256-KiB per-minute budgets deliberately
sacrifice excess diagnostics.
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
