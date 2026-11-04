"""Embedding engine for cognee's embedding worker (EMBEDDING_PROVIDER=embedding_worker).

Sends ``/v1/embed`` requests to a worker
(https://github.com/topoteretes/embedding-worker) over RabbitMQ, the way the
remote GLiNER client sends extraction (see ``embedding_worker/`` and
``cognee/infrastructure/remote_worker/``):

- Readiness is checked before the first request and then every
  :data:`READINESS_TTL_SECS`; for RabbitMQ it means at least one consumer on the
  queue. The worker publishes no limits over RabbitMQ, so the vector size
  (EMBEDDING_DIMENSIONS) and input limit (EMBEDDING_MAX_COMPLETION_TOKENS) come
  from the config and every reply is checked against them.
- "Unavailable" (no broker, no queue, no consumer, an unrouted or nacked
  request, the worker's ``unavailable``) is retried with :class:`RetryPolicy`
  (8 attempts, 0.5-8 s jittered). A request the broker took but nobody answered
  in COGNEE_EMBEDDING_WORKER_REQUEST_TIMEOUT_SECS is not resent: readiness is
  re-checked, then the call fails.
- ``input_too_long`` splits the request and mean-pools the parts of an input,
  as the other engines do on a provider's context-length error.
- Fatal errors (the outage outliving its retries, refused credentials, the
  wrong model or vector size) are latched, so later calls fail at once instead
  of each waiting out its own retries. A latched outage is released by the
  first readiness check that finds a worker again.
"""

from __future__ import annotations

import asyncio
import math
import os
import time
import uuid
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from typing import Any, TypeVar

import numpy as np

from cognee.infrastructure.databases.exceptions import EmbeddingContextWindowTooSmallError
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import (
    EmbeddingEngine,
    EmbeddingResult,
)
from cognee.infrastructure.databases.vector.embeddings.input_limit import init_input_limit
from cognee.infrastructure.databases.vector.embeddings.utils import (
    handle_embedding_response,
    sanitize_embedding_text_inputs,
)
from cognee.infrastructure.llm.tokenizer.resolver import resolve_embedding_tokenizer
from cognee.infrastructure.remote_worker.retry import RetryPolicy
from cognee.modules.observability.get_observe import get_observe
from cognee.shared.logging_utils import get_logger
from cognee.shared.rate_limiting import embedding_rate_limiter_context_manager

from .embedding_worker.errors import (
    EmbeddingWorkerConfigError,
    EmbeddingWorkerDimensionsMismatchError,
    EmbeddingWorkerError,
    EmbeddingWorkerInputTooLongError,
    EmbeddingWorkerModelMismatchError,
    EmbeddingWorkerTimeoutError,
    EmbeddingWorkerUnavailableError,
)
from .embedding_worker.protocol import (
    EmbeddingWorkerTransport,
    EmbedReply,
    InputType,
    embed_request,
)
from .embedding_worker.settings import EmbeddingWorkerSettings, get_embedding_worker_settings
from .embedding_worker.shared import SharedWorker, shared_worker

logger = get_logger("EmbeddingWorkerEmbeddingEngine")
observe = get_observe()

READINESS_TTL_SECS = 30.0
DEFAULT_RETRY = RetryPolicy()

T = TypeVar("T")


class EmbeddingWorkerEmbeddingEngine(EmbeddingEngine):
    """Embeddings from an embedding worker over RabbitMQ.

    ``transport`` replaces the shared, settings-built one (tests); the engine then
    keeps its own budget and readiness cache.
    """

    input_limit_source = (
        "EMBEDDING_MAX_COMPLETION_TOKENS; the worker's max_input_tokens cannot be read "
        "over RabbitMQ"
    )

    def __init__(
        self,
        model: str | None = None,
        dimensions: int | None = None,
        max_completion_tokens: int | None = None,
        batch_size: int = 32,
        input_type: str | None = None,
        huggingface_tokenizer: str | None = None,
        *,
        settings: EmbeddingWorkerSettings | None = None,
        transport: EmbeddingWorkerTransport | None = None,
        retry: RetryPolicy = DEFAULT_RETRY,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        settings = settings or get_embedding_worker_settings()
        if transport is None:
            settings.require_amqp_endpoint()
        if not dimensions:
            raise EmbeddingWorkerConfigError(
                "EMBEDDING_PROVIDER=embedding_worker needs EMBEDDING_DIMENSIONS.",
                remediation="Set EMBEDDING_DIMENSIONS to the vector size of the worker's model.",
            )
        self.settings = settings
        # Informational: the worker serves one model and ignores any name sent.
        self.model = model or "embedding_worker"
        self.dimensions = dimensions
        self.batch_size = batch_size
        # Call sites do not say whether they embed documents or queries, so one
        # type serves both: "query" only when EMBEDDING_INPUT_TYPE says so.
        self.input_type: InputType = (
            "query" if (input_type or "").strip().lower() == "query" else "document"
        )
        self.tokenizer = resolve_embedding_tokenizer(
            provider="embedding_worker", model=model, huggingface_tokenizer=huggingface_tokenizer
        )
        init_input_limit(self, max_completion_tokens)
        self.mock = os.getenv("MOCK_EMBEDDING", "false").lower() in ("true", "1", "yes")

        self._retry = retry
        self._sleep = sleep
        self._clock = clock
        self._own_worker = (
            SharedWorker(transport, asyncio.Semaphore(settings.max_concurrent_requests))
            if transport is not None
            else None
        )
        self._failure: Exception | None = None
        #: The model the worker reported, once it has.
        self.served_model: str | None = None

    # ------------------------------------------------------------------ public

    @observe(as_type="embeddings")
    async def embed_text(self, text: list[str]) -> EmbeddingResult:
        """One vector per text, in order, with the tokens the worker counted.

        Blank texts are sent as a placeholder and get a zero vector, as in every
        engine (the worker rejects blank inputs). Texts are sent in requests of at
        most EMBEDDING_BATCH_SIZE inputs.
        """
        original_texts = text if isinstance(text, list) else [text]
        sanitized = sanitize_embedding_text_inputs(original_texts)
        if self.mock:
            vectors = [[0.0] * self.dimensions for _ in sanitized]
            return EmbeddingResult(
                vectors=handle_embedding_response(original_texts, vectors, self.dimensions),
                tokens=0,
            )
        if not sanitized:
            return EmbeddingResult(vectors=[], tokens=0)

        groups = [
            sanitized[start : start + self.batch_size]
            for start in range(0, len(sanitized), self.batch_size)
        ]
        try:
            await self._raise_if_failed()
            await self._ready()
            results = await _gather_or_cancel([self._embed_splitting(group) for group in groups])
        except (EmbeddingWorkerError, EmbeddingWorkerConfigError) as error:
            self._latch(error)
            raise
        vectors = [vector for group_vectors, _ in results for vector in group_vectors]
        return EmbeddingResult(
            vectors=handle_embedding_response(original_texts, vectors, self.dimensions),
            tokens=sum(tokens for _, tokens in results),
        )

    def get_vector_size(self) -> int:
        return self.dimensions

    def get_batch_size(self) -> int:
        return self.batch_size

    async def input_limit(self) -> int | None:
        """The configured EMBEDDING_MAX_COMPLETION_TOKENS: RabbitMQ has no readiness
        endpoint that reports the worker's ``max_input_tokens``. None when unset
        (the default cap applies, with a warning to set it); an input over the
        worker's limit is still split on ``input_too_long``."""
        return self.input_cap

    # ------------------------------------------------------------- internals

    async def _worker(self) -> SharedWorker:
        return self._own_worker or await shared_worker(self.settings)

    async def _embed_splitting(self, texts: list[str]) -> tuple[list[list[float]], int]:
        """Embed ``texts``; on ``input_too_long`` split the request in halves, and a
        single over-long text into overlapping thirds whose vectors are averaged,
        as the OpenAI-compatible engine does."""
        try:
            reply = await self._request(texts)
            return reply.vectors, reply.prompt_tokens
        except EmbeddingWorkerInputTooLongError as error:
            if len(texts) > 1:
                middle = math.ceil(len(texts) / 2)
                (left, left_tokens), (right, right_tokens) = await asyncio.gather(
                    self._embed_splitting(texts[:middle]),
                    self._embed_splitting(texts[middle:]),
                )
                return left + right, left_tokens + right_tokens
            whole = texts[0]
            third = len(whole) // 3
            if third == 0:
                raise EmbeddingContextWindowTooSmallError() from error
            # Overlapping parts (two thirds each) keep the context across the cut.
            left_part, right_part = sanitize_embedding_text_inputs(
                [whole[: third * 2], whole[third:]]
            )
            ([left_vector], left_tokens), ([right_vector], right_tokens) = await asyncio.gather(
                self._embed_splitting([left_part]),
                self._embed_splitting([right_part]),
            )
            pooled = (np.array(left_vector) + np.array(right_vector)) / 2
            return [pooled.tolist()], left_tokens + right_tokens

    async def _request(self, texts: list[str]) -> EmbedReply:
        worker = await self._worker()
        payload = embed_request(texts, self.input_type)
        # One id per logical request, kept across its retries, so client and
        # worker logs can be joined.
        request_id = uuid.uuid4().hex

        async def send() -> EmbedReply:
            # The budget is held only while a request is in flight, never across a
            # retry backoff.
            async with embedding_rate_limiter_context_manager(), worker.budget:
                return await worker.transport.embed(payload, request_id=request_id)

        try:
            reply = await self._with_retries(
                send, worker.transport.endpoint_label, f"request {request_id}"
            )
        except EmbeddingWorkerTimeoutError:
            # Only this request is slow, unless the worker is gone: a fresh
            # probe raises the outage in that case.
            await self._ready(fresh=True)
            raise
        self._check_reply(reply, worker.transport.endpoint_label)
        return reply

    def _check_reply(self, reply: EmbedReply, endpoint: str) -> None:
        expected = self.settings.expected_model
        if expected is not None and reply.model != expected:
            raise EmbeddingWorkerModelMismatchError(expected, reply.model)
        if reply.dimensions != self.dimensions:
            raise EmbeddingWorkerDimensionsMismatchError(
                self.dimensions, reply.dimensions, reply.model
            )
        if self.served_model is None:
            self.served_model = reply.model
            logger.info("Embedding worker %s serves %s", endpoint, reply.model)
        elif reply.model != self.served_model:
            logger.warning(
                "Embedding worker replicas at %s serve different models: %s and %s",
                endpoint,
                self.served_model,
                reply.model,
            )
            self.served_model = reply.model

    async def _raise_if_failed(self) -> None:
        failure = self._failure
        if failure is None:
            return
        if not isinstance(failure, EmbeddingWorkerUnavailableError):
            raise failure
        # An outage ends when a worker consumes the queue again: one probe,
        # without retries, decides whether this call fails at once.
        worker = await self._worker()
        try:
            await worker.transport.readiness()
        except EmbeddingWorkerUnavailableError:
            raise failure from None
        worker.ready_until = self._clock() + READINESS_TTL_SECS
        self._failure = None
        logger.info("Embedding worker %s is available again", worker.transport.endpoint_label)

    def _latch(self, error: Exception) -> None:
        if getattr(error, "fatal", False) and self._failure is None:
            self._failure = error
            logger.error(
                "Embedding worker failed; later embedding calls fail at once: %s",
                getattr(error, "message", error),
            )

    async def _ready(self, *, fresh: bool = False) -> None:
        """Probe readiness when the cached answer expired, or ``fresh`` asks for it."""
        worker = await self._worker()
        if fresh or self._clock() >= worker.ready_until:
            await self._probe(worker)

    async def _probe(self, worker: SharedWorker) -> None:
        if worker.probe is None:
            probe = asyncio.ensure_future(self._run_probe(worker))

            def finished(done: asyncio.Future) -> None:
                if worker.probe is done:
                    worker.probe = None
                if not done.cancelled():
                    done.exception()  # retrieved here even when every waiter was cancelled

            probe.add_done_callback(finished)
            worker.probe = probe
        # Shielded: one waiter's cancellation must not cancel the others' probe.
        await asyncio.shield(worker.probe)

    async def _run_probe(self, worker: SharedWorker) -> None:
        await self._with_retries(
            worker.transport.readiness, worker.transport.endpoint_label, "readiness"
        )
        worker.ready_until = self._clock() + READINESS_TTL_SECS

    async def _with_retries(self, call: Callable[[], Awaitable[T]], endpoint: str, what: str) -> T:
        last = self._retry.attempts - 1
        for attempt in range(self._retry.attempts):
            try:
                return await call()
            except EmbeddingWorkerUnavailableError as error:
                if attempt == last:
                    raise
                delay = self._retry.delay(attempt, error.retry_after)
                logger.warning(
                    "Embedding worker %s unavailable during %s (%s); retry %d of %d in %.2fs",
                    endpoint,
                    what,
                    error.message,
                    attempt + 1,
                    last,
                    delay,
                )
                await self._sleep(delay)
        raise AssertionError("unreachable")  # the loop returns or raises


async def _gather_or_cancel(coroutines: Sequence[Coroutine[Any, Any, T]]) -> list[T]:
    """Run all; on the first failure cancel the rest and raise that failure."""
    tasks = [asyncio.ensure_future(coroutine) for coroutine in coroutines]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
