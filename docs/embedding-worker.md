# Embedding worker (EMBEDDING_PROVIDER=embedding_worker)

Cognee can send its embeddings to a separate
[embedding worker](https://github.com/topoteretes/embedding-worker) over
RabbitMQ, in the same way the remote GLiNER client sends extraction to a
`gliner_worker` (see "Remote worker" in `CLAUDE.md`). Many cognee processes can
then share one fleet of workers, sized and autoscaled by queue depth, and no
embedding model or tokenizer runtime has to be loaded in the cognee process.

The engine is `cognee/infrastructure/databases/vector/embeddings/EmbeddingWorkerEmbeddingEngine.py`.
Its settings, errors, wire contract and transport are in `embedding_worker/` next
to it. The RabbitMQ RPC code (`cognee/infrastructure/remote_worker/amqp_rpc.py`)
and the retry policy are shared with the GLiNER client.

## Setup

Install the RabbitMQ client:

```bash
pip install "cognee[remote-worker-amqp]"
```

`remote-worker-amqp` is the neutral name of the RabbitMQ client extra shared by
every remote worker client. `gliner-remote-amqp` installs the same dependencies
and keeps working.

Start a worker on the RabbitMQ transport:

```bash
EMBEDDING_WORKER_TRANSPORT=rabbitmq \
EMBEDDING_WORKER_RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/%2f \
EMBEDDING_WORKER_MODEL=bge-small-en-v1.5 \
  embedding-worker
```

Then configure cognee:

```bash
EMBEDDING_PROVIDER=embedding_worker
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5      # informational, also picks the chunking tokenizer
COGNEE_EMBEDDING_WORKER_ENDPOINT=amqp://guest:guest@rabbitmq:5672/%2f
```

The vector size, the token limit and the number of inputs per request come
from the worker (see "Model info" below). Set `EMBEDDING_DIMENSIONS` to pin the
vector size, so that a worker serving a model of another size fails instead of
being adopted.

`%2f` is the default vhost `/`. Both SDKs also read a bare trailing `/`
(`amqp://host:5672/`) as the default vhost `/`.

## Settings

The worker's location is read from the environment only (prefix
`COGNEE_EMBEDDING_WORKER_`), never from an API request. The names and defaults
are the same as the Rust SDK's.

| Setting | Default | Meaning |
|---|---|---|
| `COGNEE_EMBEDDING_WORKER_TRANSPORT` | `amqp` | `amqp` (alias `rabbitmq`). `http` and `grpc` are rejected with a configuration error because they are not supported yet. Use `EMBEDDING_PROVIDER=openai_compatible` to reach the worker over HTTP. |
| `COGNEE_EMBEDDING_WORKER_ENDPOINT` | (required) | `amqp(s)://user:pass@host:5672/vhost`. Credentials are masked in every log line and error. |
| `COGNEE_EMBEDDING_WORKER_AMQP_QUEUE` | `embedding_worker.embed` | The worker's queue (`EMBEDDING_WORKER_RABBITMQ_QUEUE`). |
| `COGNEE_EMBEDDING_WORKER_EXPECTED_MODEL` | unset | If set, the `model` in the worker's info and in every reply must equal this value. A mismatch is fatal. Unset, any model is accepted and logged once. |
| `COGNEE_EMBEDDING_WORKER_MAX_CONCURRENT_REQUESTS` | `16` | Requests in flight to one endpoint, shared by every engine in the process (per event loop). Size it to the fleet. |
| `COGNEE_EMBEDDING_WORKER_CONNECT_TIMEOUT_SECS` | `5` | Broker connect and readiness timeout. |
| `COGNEE_EMBEDDING_WORKER_REQUEST_TIMEOUT_SECS` | `300` | One request, including its time in the queue. It is also the message `expiration`, so the broker drops requests that no client waits for. |

The engine also uses these existing embedding settings:

- `EMBEDDING_DIMENSIONS` is optional. Unset, the engine uses the `dimensions`
  from the worker's info. Set, it must equal the worker's info and every
  reply's `dimensions`; a mismatch is fatal and names both values. With a
  worker that predates info, it is required (a configuration error on first
  contact otherwise).
- `EMBEDDING_BATCH_SIZE` (default 32) is the most inputs per request. With
  the worker's info, requests carry at most
  `min(EMBEDDING_BATCH_SIZE, limits.max_inputs)` inputs (`max_inputs` is 256
  when the info does not state it). A worker without info cannot say its
  limit, so there a value over 256 (the worker's default
  `EMBEDDING_WORKER_MAX_INPUTS`) is a configuration error on first contact.
- `EMBEDDING_MAX_COMPLETION_TOKENS` is optional. Chunks are sized to the
  worker's token budget (see "Token budget" below), lowered to this value when
  it is set and smaller. A larger value is a misconfiguration: it is logged as
  a warning and the worker's budget applies. Without info (an older worker),
  the configured value is used as before: if it is unset, the default cap of
  4096 applies and a warning is logged. An input that is still too long is
  split (see below).
- `EMBEDDING_INPUT_TYPE=query` sends `input_type: "query"` for **every** text,
  documents included: cognee's call sites do not say whether they embed a
  document or a query, so one type is used for both, and a warning is logged
  when the engine is built. Any other value, or no value, sends `"document"`
  for everything. That is correct for symmetric models (pplx-embed). For models
  with query prefixes (BGE, E5), queries are embedded as documents, as they are
  on the OpenAI-compatible route. Do not set `query` unless that is what you
  want for your documents.
- `HUGGINGFACE_TOKENIZER` overrides the tokenizer that sizes chunks. Unset,
  chunks are counted with the worker's tokenizer (its info's `tokenizer`, a
  Hugging Face id) once the info is known, so `EMBEDDING_MODEL` need not be a
  Hugging Face repo id. Until then, and with a worker that does not state its
  tokenizer, the tokenizer resolved from `EMBEDDING_MODEL` is used. If the
  worker's tokenizer cannot be loaded, a warning is logged and the current one
  is kept.
- `EMBEDDING_ENCODING_FORMAT` does not apply. Vectors always travel as base64
  little-endian float32.
- `MOCK_EMBEDDING` and the `EMBEDDING_RATE_LIMIT_*` settings work as they do
  for the other engines. `MOCK_EMBEDDING` needs `EMBEDDING_DIMENSIONS`, because
  mock vectors never ask the worker.

## Model info

After the first successful readiness check the engine asks the worker for its
model info: a message with the AMQP `type` property `info` and body `{}`, sent
like an embed request. The reply is the worker's `GET /v1/info` JSON (`model`,
`dimensions`, `max_input_tokens`, `limits.max_inputs`, and more). It is retried
like any request when the worker answers `unavailable`.

Once a worker has answered, the info is kept for the life of the process (per
event loop and endpoint) and is not asked for again on the 30 s readiness
re-checks: an info message waits in the same queue as the embed requests, so
refreshing it on every check would stall readiness whenever there is a backlog.
A worker replaced by one serving another model or size is still caught, by the
model and `dimensions` check on every reply. The one exception is a latched
outage (below): when it is released, the info is fetched again, because the
worker that came back may be a different one.

A malformed info reply, or an info request that is still `unavailable` after
the retries, fails the readiness check like an unreachable worker (the
latter is latched as an outage).

| Info field | Used for |
|---|---|
| `model` | Checked against `COGNEE_EMBEDDING_WORKER_EXPECTED_MODEL` (fatal on mismatch). |
| `dimensions` | The vector size when `EMBEDDING_DIMENSIONS` is unset; checked against it when set (fatal on mismatch). A worker whose size changes after the engine adopted it is a mismatch too. |
| `max_text_tokens`, `special_tokens`, `max_input_tokens` | `input_limit()`, the token budget chunks are sized by (see "Token budget"). |
| `tokenizer` | Counting chunk tokens, unless `HUGGINGFACE_TOKENIZER` is set. |
| `limits.max_inputs` | Caps the inputs per request. |

`model`, `dimensions` and `max_input_tokens` are required; an info without
them, or with a field of the wrong type, is malformed. The others are
optional: `limits.max_inputs` defaults to 256, and `special_tokens`,
`max_text_tokens` and `tokenizer` are used when present.

### Token budget

Cognee counts chunk tokens without special tokens (`[CLS]`, `[SEP]`, ...) and
without the worker's input-type prefix, while the worker's `max_input_tokens`
counts everything. A chunk sized to `max_input_tokens` would overflow by the
special tokens, be rejected and need splitting: with bge-small, a 512-token
chunk is 514 tokens to the worker and cost three requests. So `input_limit()`
is, in order of preference:

1. `max_text_tokens[input_type]` (BGE: 510 for documents, 502 for queries;
   pplx-embed: 2048), the tokens of text that fit once special tokens and
   prefix are taken out;
2. without it, `max_input_tokens - special_tokens`;
3. without either (an older worker), `max_input_tokens`, with a one-time
   warning that such a chunk may still be split.

`EMBEDDING_MAX_COMPLETION_TOKENS` lowers the result when it is smaller. A
chunk of exactly `input_limit()` tokens, counted with the worker's tokenizer,
embeds in one request.

A worker that predates info parses the message as an embed request and
answers `invalid_request`. The engine then logs one warning and uses the
config, as it did before info existed. Such a worker is asked again on later
readiness checks (without another warning), so an upgraded worker is picked up.

### The vector size before the first contact

Cognee asks for the vector size synchronously (`get_vector_size()`, when a
vector collection is created), sometimes before anything has been embedded.
With `EMBEDDING_DIMENSIONS` set, the configured value is returned at once and
checked against the worker later. Unset, the size is unknown until the worker
has answered, so:

- In the usual flow the size is known before anyone asks: the startup
  connection test (unless `COGNEE_SKIP_CONNECTION_TEST=true`) resolves the
  input limit and embeds a test text, which fetches the info, and then records
  the size in the embedding config. The dataset check compares stored vectors
  with the size it gets from `resolve_vector_size()`, which asks the worker on
  the running event loop.
- If `get_vector_size()` is called first, it fetches the info synchronously, on
  a thread of its own with its own short-lived broker connection and the usual
  retries, and caches the result. The caller and its event loop are blocked for
  one round trip, or up to the retry budget (16–32 s) when no worker is up, in
  which case the call raises `EmbeddingWorkerUnavailableError` and the outage is
  latched as below. This happens at most once per process when it succeeds.

## Wire protocol

Each embed request is the worker's `POST /v1/embed` body, with no AMQP `type`:
`{"inputs": [...], "input_type": "document", "encoding_format": "base64"}`. It
is published to the default exchange with routing key = queue, with these
properties:

- `reply_to` set to `amq.rabbitmq.reply-to` (direct reply-to);
- a fresh `correlation_id`;
- an `x-request-id` header, kept the same across retries;
- `expiration` equal to the request timeout;
- `mandatory`, with publisher confirms on.

An info request is the same, with `type` set to `info` and body `{}`.

`dimensions`, `normalize` and `truncate` are not sent. The worker's defaults
apply: full-size unit vectors, and over-long inputs are rejected rather than
truncated.

Blank texts are sent as `"."` and get a zero vector, as in every engine. The
worker rejects blank inputs.

Token usage comes from `usage.prompt_tokens`.

## Failure behaviour

| Condition | Behaviour |
|---|---|
| No broker, no queue, no consumer on the queue, a request the broker does not route or nacks, a broker under an alarm, or the worker's `unavailable` | Retried 7 times (8 attempts) with jittered backoff from 0.5 s to 8 s, 16–32 s in all. If every attempt fails, the error is **latched**: later calls fail at once after one readiness check without retries. The first check that finds a consumer releases the latch. |
| The broker accepted the request and no reply came within the request timeout | Not resent, because the worker may still be running it. Readiness is checked again: if the worker is gone, the call fails as unavailable. Otherwise it fails with `EmbeddingWorkerTimeoutError`. |
| `input_too_long` | Recovered in one round trip (see "Over-long inputs" below). A text too short to split (fewer than 3 characters) raises `EmbeddingContextWindowTooSmallError`. |
| `invalid_request` | `EmbeddingWorkerRejectedError`. Fails that call. Not retried. |
| `internal`, an unknown code, or a malformed reply | `EmbeddingWorkerRuntimeError`. Fails that call. Not retried. |
| The broker refuses the credentials, or refuses access (`ACCESS_REFUSED`) when the queue is declared or a request is published: the user lacks permissions on the vhost or queue | `EmbeddingWorkerUnauthorizedError`. Fatal and latched; not retried. |
| The info's or a reply's `model` differs from `EXPECTED_MODEL` | `EmbeddingWorkerModelMismatchError`. Fatal and latched. |
| The info's or a reply's `dimensions` differs from `EMBEDDING_DIMENSIONS` (or from the size adopted from the worker) | `EmbeddingWorkerDimensionsMismatchError`, naming both values. Fatal and latched. |
| The worker answers the info request with `invalid_request` (it predates info) | One warning; the config is used. With `EMBEDDING_DIMENSIONS` unset, `EmbeddingWorkerConfigError`, fatal and latched. |
| Missing or malformed settings, `http`/`grpc`, or aio-pika not installed | `EmbeddingWorkerConfigError` when the engine is built (the RabbitMQ transport module is imported then). |
| `EMBEDDING_BATCH_SIZE` over 256 with a worker that has no info | `EmbeddingWorkerConfigError` on first contact. Fatal and latched. |

### Over-long inputs

The worker's `input_too_long` error lists every offending input
(`inputs: [{"index", "chars", "tokens"}]`, `tokens` null for an input over the
character limit). The engine splits exactly those inputs, each text `s` into
the overlapping parts `s[:2t]` and `s[t:]` with `t = len(s) // 3` characters,
and sends the request again once, with each listed input replaced by its two
parts. If that no longer fits one request (`limits.max_inputs`), it goes out as
several requests at once; when one fails, the others are cancelled. A part
that is still too long is listed again in the next reply and split again. The
two parts' vectors are averaged and L2-normalized, so every result is a unit
vector like those the worker returns. Only the accepted requests count
towards the token usage.

A worker that does not list the inputs (it predates the list) gets the batch
bisected instead, until each over-long text is alone in its request and split
as above.

### Connection and readiness

Readiness is a passive declare of the queue that requires at least one
consumer; the first successful check is followed by the info request. It runs
before the first request and is cached for 30 s. Concurrent
callers share one probe. Each event loop and endpoint has one broker
connection, and the connection closes when the loop shuts down. When it has
dropped, the callers that need it share one reconnect (and its outcome)
instead of each waiting for a connect of its own in turn.

The worker delivers at least once. Replies are matched by `correlation_id`, and
a duplicate reply is dropped.

Every error is an `EmbeddingException`, so the API and the pipelines handle it
like any other embedding failure.

## Differences from the Rust client

The Rust SDK (cognee-cloud-rs) speaks the same protocol, reads the same
`COGNEE_EMBEDDING_WORKER_*` settings and follows the same info and
split-and-average rules. These differences remain:

| | Python (this client) | Rust |
|---|---|---|
| Query vs. document | No per-call distinction: every text is sent as `document`, or every text as `query` with `EMBEDDING_INPUT_TYPE=query`. | Each call says whether it embeds documents or a query. |
| An outage that exhausted its retries | Latched: later calls fail after one readiness check without retries, until a check finds a worker again. | Not latched: each call retries on its own. |
| `*_TIMEOUT_SECS` | Fractional seconds accepted (e.g. `2.5`). | Whole seconds. |
| Model info after an outage | Fetched again when a latched outage is released, since the worker that came back may be another one. | Kept once fetched (there is no latch to release); a swapped worker is caught by the per-reply checks only. |

## Tests

- Unit tests: `cognee/tests/unit/infrastructure/test_embedding_worker_*.py`.
  The broker tests in `test_embedding_worker_amqp.py` run only when
  `COGNEE_EMBEDDING_WORKER_AMQP_TEST_URL` is set; the `ACCESS_REFUSED` one also
  needs `COGNEE_EMBEDDING_WORKER_AMQP_TEST_RESTRICTED_URL`, a user without
  permissions on the vhost.
- End to end, against a broker and a real worker: `cognee/tests/e2e/embedding_worker/`
  (see its docstring). It is skipped unless `EMBEDDING_PROVIDER=embedding_worker`
  and `COGNEE_EMBEDDING_WORKER_ENDPOINT` are set.
