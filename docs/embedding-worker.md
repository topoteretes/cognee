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

Install the RabbitMQ client. It is the same extra that the remote GLiNER client
uses:

```bash
pip install "cognee[gliner-remote-amqp]"
```

Start a worker on the RabbitMQ transport:

```bash
EMBEDDING_WORKER_TRANSPORT=rabbitmq \
EMBEDDING_WORKER_RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/ \
EMBEDDING_WORKER_MODEL=bge-small-en-v1.5 \
  embedding-worker
```

Then configure cognee:

```bash
EMBEDDING_PROVIDER=embedding_worker
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5      # informational, also picks the chunking tokenizer
EMBEDDING_DIMENSIONS=384                    # required: the worker model's vector size
EMBEDDING_MAX_COMPLETION_TOKENS=512         # the worker model's max_input_tokens
COGNEE_EMBEDDING_WORKER_ENDPOINT=amqp://guest:guest@rabbitmq:5672/
```

## Settings

The worker's location is read from the environment only (prefix
`COGNEE_EMBEDDING_WORKER_`), never from an API request. The names and defaults
are the same as the Rust SDK's.

| Setting | Default | Meaning |
|---|---|---|
| `COGNEE_EMBEDDING_WORKER_TRANSPORT` | `amqp` | `amqp` (alias `rabbitmq`). `http` and `grpc` are rejected with a configuration error because they are not supported yet. Use `EMBEDDING_PROVIDER=openai_compatible` to reach the worker over HTTP. |
| `COGNEE_EMBEDDING_WORKER_ENDPOINT` | (required) | `amqp(s)://user:pass@host:5672/vhost`. Credentials are masked in every log line and error. |
| `COGNEE_EMBEDDING_WORKER_AMQP_QUEUE` | `embedding_worker.embed` | The worker's queue (`EMBEDDING_WORKER_RABBITMQ_QUEUE`). |
| `COGNEE_EMBEDDING_WORKER_EXPECTED_MODEL` | unset | If set, every reply's `model` must equal this value. A mismatch is fatal. Unset, any model is accepted and logged once. |
| `COGNEE_EMBEDDING_WORKER_MAX_CONCURRENT_REQUESTS` | `16` | Requests in flight to one endpoint, shared by every engine in the process (per event loop). Size it to the fleet. |
| `COGNEE_EMBEDDING_WORKER_CONNECT_TIMEOUT_SECS` | `5` | Broker connect and readiness timeout. |
| `COGNEE_EMBEDDING_WORKER_REQUEST_TIMEOUT_SECS` | `300` | One request, including its time in the queue. It is also the message `expiration`, so the broker drops requests that no client waits for. |

The engine also uses these existing embedding settings:

- `EMBEDDING_DIMENSIONS` must equal the `dimensions` in the worker's replies.
  Otherwise the vectors would not fit the vector index, so a mismatch is fatal.
  It must be set: no model registry knows the worker's model, and the generic
  fallback of 3072 would be wrong.
- `EMBEDDING_BATCH_SIZE` (default 32) is the number of inputs per request. The
  worker accepts up to its `EMBEDDING_WORKER_MAX_INPUTS`, which is 256 by default.
- `EMBEDDING_MAX_COMPLETION_TOKENS` is the chunk-token limit. RabbitMQ has no
  `/readyz`, so the worker's `max_input_tokens` cannot be read. Set this value
  to the worker's limit. If it is unset, the default cap of 4096 applies and a
  warning is logged. An input that is still too long is split (see below).
- `EMBEDDING_INPUT_TYPE=query` sends `input_type: "query"`. Any other value,
  or no value, sends `"document"`. Cognee's call sites do not say whether they
  embed a document or a query, so one type is used for both. That is correct
  for symmetric models (pplx-embed). For models with query prefixes (BGE, E5),
  queries are embedded as documents, as they are on the OpenAI-compatible
  route.
- `HUGGINGFACE_TOKENIZER` overrides the tokenizer that sizes chunks. This is
  useful when `EMBEDDING_MODEL` is not a Hugging Face repo id.
- `EMBEDDING_ENCODING_FORMAT` does not apply. Vectors always travel as base64
  little-endian float32.
- `MOCK_EMBEDDING` and the `EMBEDDING_RATE_LIMIT_*` settings work as they do
  for the other engines.

## Wire protocol

Each request is the worker's `POST /v1/embed` body:
`{"inputs": [...], "input_type": "document", "encoding_format": "base64"}`. It
is published to the default exchange with routing key = queue, with these
properties:

- `reply_to` set to `amq.rabbitmq.reply-to` (direct reply-to);
- a fresh `correlation_id`;
- an `x-request-id` header, kept the same across retries;
- `expiration` equal to the request timeout;
- `mandatory`, with publisher confirms on.

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
| `input_too_long` | A batch is split in halves. A single text is split into two overlapping parts, each two thirds of the text, and their vectors are averaged. This repeats until every part fits, as the OpenAI-compatible engine does for a context-length error. A text too short to split raises `EmbeddingContextWindowTooSmallError`. |
| `invalid_request` | `EmbeddingWorkerRejectedError`. Fails that call. Not retried. |
| `internal`, an unknown code, or a malformed reply | `EmbeddingWorkerRuntimeError`. Fails that call. Not retried. |
| The broker refuses the credentials | `EmbeddingWorkerUnauthorizedError`. Fatal and latched. |
| A reply's `model` differs from `EXPECTED_MODEL` | `EmbeddingWorkerModelMismatchError`. Fatal and latched. |
| A reply's `dimensions` differs from `EMBEDDING_DIMENSIONS` | `EmbeddingWorkerDimensionsMismatchError`. Fatal and latched. |
| Missing or malformed settings, `http`/`grpc`, or aio-pika not installed | `EmbeddingWorkerConfigError` when the engine is built. |

Readiness is a passive declare of the queue that requires at least one
consumer. It runs before the first request and is cached for 30 s. Concurrent
callers share one probe. Each event loop and endpoint has one broker
connection, and the connection closes when the loop shuts down.

The worker delivers at least once. Replies are matched by `correlation_id`, and
a duplicate reply is dropped.

Every error is an `EmbeddingException`, so the API and the pipelines handle it
like any other embedding failure.

## Tests

- Unit tests: `cognee/tests/unit/infrastructure/test_embedding_worker_*.py`.
  The broker tests in `test_embedding_worker_amqp.py` run only when
  `COGNEE_EMBEDDING_WORKER_AMQP_TEST_URL` is set.
- End to end, against a broker and a real worker: `cognee/tests/e2e/embedding_worker/`
  (see its docstring). It is skipped unless `EMBEDDING_PROVIDER=embedding_worker`
  and `COGNEE_EMBEDDING_WORKER_ENDPOINT` are set.
