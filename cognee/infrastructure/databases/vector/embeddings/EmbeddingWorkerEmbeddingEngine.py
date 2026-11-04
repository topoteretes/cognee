"""Embedding engine for cognee's embedding worker (EMBEDDING_PROVIDER=embedding_worker).

Sends ``/v1/embed`` requests to a worker
(https://github.com/topoteretes/embedding-worker) over RabbitMQ, the way the
remote GLiNER client sends extraction (see ``embedding_worker/`` and
``cognee/infrastructure/remote_worker/``):

- Readiness is checked before the first request and then every
  :data:`READINESS_TTL_SECS`; for RabbitMQ it means at least one consumer on the
  queue. The first successful probe then asks the worker for its model info (a
  message with AMQP ``type=info``); once a worker has answered with one it is
  kept, and fetched again only when a latched outage is released. The info supplies
  the vector size when EMBEDDING_DIMENSIONS is unset (and is checked against it
  when set), the token limit inputs are sized by and the most inputs one request
  may carry. A worker that predates info answers ``invalid_request``: the engine
  warns once and falls back to the config.
- "Unavailable" (no broker, no queue, no consumer, an unrouted or nacked
  request, the worker's ``unavailable``) is retried with :class:`RetryPolicy`
  (8 attempts, 0.5-8 s jittered). A request the broker took but nobody answered
  in COGNEE_EMBEDDING_WORKER_REQUEST_TIMEOUT_SECS is not resent: readiness is
  re-checked, then the call fails.
- ``input_too_long`` lists every offending input: exactly those are split into
  two overlapping parts and the request is sent again once; the parts' vectors
  are averaged and L2-normalized. (An older worker that does not list them gets
  the request bisected.)
- Fatal errors (the outage outliving its retries, refused credentials, the
  wrong model or vector size) are latched, so later calls fail at once instead
  of each waiting out its own retries. A latched outage is released by the
  first readiness check that finds a worker again.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
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
from cognee.infrastructure.llm.tokenizer.HuggingFace import HuggingFaceTokenizer
from cognee.infrastructure.llm.tokenizer.resolver import resolve_embedding_tokenizer
from cognee.infrastructure.remote_worker.retry import RetryPolicy
from cognee.modules.observability.get_observe import get_observe
from cognee.shared.logging_utils import get_logger
from cognee.shared.rate_limiting import embedding_rate_limiter_context_manager

from .embedding_worker import shared as shared_module
from .embedding_worker.errors import (
    EmbeddingWorkerConfigError,
    EmbeddingWorkerDimensionsMismatchError,
    EmbeddingWorkerError,
    EmbeddingWorkerInputTooLongError,
    EmbeddingWorkerModelMismatchError,
    EmbeddingWorkerRejectedError,
    EmbeddingWorkerTimeoutError,
    EmbeddingWorkerUnavailableError,
)
from .embedding_worker.protocol import (
    EmbeddingWorkerTransport,
    EmbedReply,
    InputType,
    WorkerInfo,
    embed_request,
)
from .embedding_worker.settings import EmbeddingWorkerSettings, get_embedding_worker_settings
from .embedding_worker.shared import SharedWorker, shared_worker

logger = get_logger("EmbeddingWorkerEmbeddingEngine")
observe = get_observe()

READINESS_TTL_SECS = 30.0
DEFAULT_RETRY = RetryPolicy()
#: The most inputs per request a worker without info accepts by default
#: (EMBEDDING_WORKER_MAX_INPUTS).
MAX_INPUTS_WITHOUT_INFO = 256

T = TypeVar("T")


class EmbeddingWorkerEmbeddingEngine(EmbeddingEngine):
    """Embeddings from an embedding worker over RabbitMQ.

    ``transport`` replaces the shared, settings-built one (tests); the engine then
    keeps its own budget and readiness cache.
    """

    input_limit_source = "the embedding worker's max_text_tokens"

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
            # Imported now, not on the first embedding, so a missing aio-pika is
            # a configuration error when the engine is built.
            shared_module.load_transport_module(settings)
        if dimensions is not None and dimensions <= 0:
            raise EmbeddingWorkerConfigError(
                f"EMBEDDING_DIMENSIONS must be a positive vector size, got {dimensions}.",
                remediation="Unset EMBEDDING_DIMENSIONS to use the worker's, or set its model's.",
            )
        self.mock = os.getenv("MOCK_EMBEDDING", "false").lower() in ("true", "1", "yes")
        if self.mock and dimensions is None:
            raise EmbeddingWorkerConfigError(
                "MOCK_EMBEDDING with EMBEDDING_PROVIDER=embedding_worker needs "
                "EMBEDDING_DIMENSIONS: mock vectors never ask the worker for its size.",
                remediation="Set EMBEDDING_DIMENSIONS.",
            )
        self.settings = settings
        # Informational: the worker serves one model and ignores any name sent.
        self.model = model or "embedding_worker"
        #: EMBEDDING_DIMENSIONS, or None to take the worker's size from its info.
        self.configured_dimensions = dimensions
        #: The vector size in use: the configured one, else the worker's once known.
        self.dimensions: int | None = dimensions
        self.batch_size = batch_size
        # Call sites do not say whether they embed documents or queries, so one
        # type serves both: "query" only when EMBEDDING_INPUT_TYPE says so.
        self.input_type: InputType = (
            "query" if (input_type or "").strip().lower() == "query" else "document"
        )
        if self.input_type == "query":
            logger.warning(
                "EMBEDDING_INPUT_TYPE=query: the embedding worker embeds every text as a "
                "query, documents included, because cognee's call sites do not tell the two "
                "apart. Unset it unless the model is symmetric or that is intended."
            )
        self.tokenizer = resolve_embedding_tokenizer(
            provider="embedding_worker", model=model, huggingface_tokenizer=huggingface_tokenizer
        )
        init_input_limit(self, max_completion_tokens)
        # Chunks are counted with the worker's own tokenizer (its info's
        # ``tokenizer``, a Hugging Face id) unless HUGGINGFACE_TOKENIZER says otherwise.
        self._explicit_tokenizer = huggingface_tokenizer or None
        self._tokenizer_id: str | None = (
            (huggingface_tokenizer or model)
            if isinstance(self.tokenizer, HuggingFaceTokenizer)
            else None
        )
        self._tokenizers_failed: set[str] = set()
        self._warned_no_text_budget = False

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
        #: The worker's model info, fetched once after the first readiness probe, if it
        #: reports one.
        self.worker_info: WorkerInfo | None = None

    # ------------------------------------------------------------------ public

    @observe(as_type="embeddings")
    async def embed_text(self, text: list[str]) -> EmbeddingResult:
        """One vector per text, in order, with the tokens the worker counted.

        Blank texts are sent as a placeholder and get a zero vector, as in every
        engine (the worker rejects blank inputs). Texts are sent in requests of at
        most :meth:`get_batch_size` inputs.
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

        try:
            await self._raise_if_failed()
            await self._ready()
            batch_size = self.get_batch_size()
            groups = [
                sanitized[start : start + batch_size]
                for start in range(0, len(sanitized), batch_size)
            ]
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
        """EMBEDDING_DIMENSIONS, else the worker's ``dimensions``.

        With EMBEDDING_DIMENSIONS unset and the worker not asked yet (cognee asks
        for the size before it embeds anything when it creates a collection),
        the info is fetched here, synchronously: on a thread of its own, with its
        own short-lived broker connection and the usual retries, because this
        method is sync and is called from inside the event loop. That blocks the
        caller (and its loop) for one round trip, or up to the retry budget
        (16-32 s) when no worker is up, and happens at most once per process
        unless it fails. Use :meth:`resolve_vector_size` from async code.
        """
        if self.dimensions is not None:
            return self.dimensions
        return self._learn_dimensions_blocking()

    async def resolve_vector_size(self) -> int:
        """:meth:`get_vector_size` without blocking: asks the worker on this loop."""
        if self.dimensions is None:
            await self._ensure_ready()
        return self.dimensions

    def get_batch_size(self) -> int:
        """EMBEDDING_BATCH_SIZE, lowered to the worker's ``limits.max_inputs`` once known."""
        info = self.worker_info
        if info is not None:
            return min(self.batch_size, info.max_inputs)
        return self.batch_size

    async def input_limit(self) -> int | None:
        """The tokens of text one input may have, counted as cognee counts chunks
        (without special tokens): the worker's ``max_text_tokens`` for the input
        type sent, lowered to EMBEDDING_MAX_COMPLETION_TOKENS when that is set and
        smaller. A configured value above the worker's is a misconfiguration:
        warned about, and the worker's budget applies.

        A worker whose info has no ``max_text_tokens`` gives ``max_input_tokens``
        minus its ``special_tokens``, or, without those either, ``max_input_tokens``
        with a one-time warning (a chunk of that size may then be rejected and
        split). Without info (MOCK_EMBEDDING, or a worker that predates it) the
        configured value stands, None when unset."""
        if self.mock:
            return self.input_cap
        await self._ensure_ready()
        info = self.worker_info
        if info is None:
            return self.input_cap
        worker_limit = self._text_budget(info)
        if self.input_cap is None or self.input_cap == worker_limit:
            return worker_limit
        if self.input_cap > worker_limit:
            logger.warning(
                "EMBEDDING_MAX_COMPLETION_TOKENS=%s exceeds the %s tokens of text the embedding "
                "worker's model %r accepts per input; inputs are limited to %s tokens.",
                self.input_cap,
                worker_limit,
                info.model,
                worker_limit,
            )
            return worker_limit
        return self.input_cap

    def _text_budget(self, info: WorkerInfo) -> int:
        """Tokens of text per input, special tokens and prefix taken out."""
        if info.max_text_tokens and self.input_type in info.max_text_tokens:
            return info.max_text_tokens[self.input_type]
        if info.special_tokens is not None:
            return max(1, info.max_input_tokens - info.special_tokens)
        if not self._warned_no_text_budget:
            self._warned_no_text_budget = True
            logger.warning(
                "The embedding worker's info states neither max_text_tokens nor special_tokens "
                "(an older worker); chunks are sized to its max_input_tokens=%s, so a chunk "
                "of that size may overflow by the special tokens and be split.",
                info.max_input_tokens,
            )
        return info.max_input_tokens

    # ------------------------------------------------------------- internals

    async def _worker(self) -> SharedWorker:
        return self._own_worker or await shared_worker(self.settings)

    async def _embed_splitting(self, texts: list[str]) -> tuple[list[list[float]], int]:
        """Embed ``texts``, recovering from ``input_too_long``.

        The worker lists every input that does not fit. Exactly those are split
        into the overlapping parts ``s[:2t]`` and ``s[t:]`` (``t = len(s) // 3``
        characters) and the request is sent once more with each listed input
        replaced by its two parts (in several requests, sent together, if it no
        longer fits one). A part that is still too long is listed again and split
        again. The parts' vectors are averaged and L2-normalized, so the result is
        a unit vector like every other the worker returns. The Rust SDK splits
        the same way. An older worker that lists nothing gets the request
        bisected instead.
        """
        try:
            reply = await self._request(texts)
            return reply.vectors, reply.prompt_tokens
        except EmbeddingWorkerInputTooLongError as error:
            indices = error.indices
            if indices is not None and all(index < len(texts) for index in indices):
                return await self._split_listed(texts, set(indices), error)
            if len(texts) == 1:
                return await self._split_listed(texts, {0}, error)
            middle = math.ceil(len(texts) / 2)
            (left, left_tokens), (right, right_tokens) = await _gather_or_cancel(
                [
                    self._embed_splitting(texts[:middle]),
                    self._embed_splitting(texts[middle:]),
                ]
            )
            return left + right, left_tokens + right_tokens

    async def _split_listed(
        self, texts: list[str], listed: set[int], error: Exception
    ) -> tuple[list[list[float]], int]:
        """Send ``texts`` again with each listed one replaced by its two parts; pool them."""
        expanded: list[str] = []
        layout: list[tuple[int, bool]] = []  # (position in expanded, split)
        for index, text in enumerate(texts):
            layout.append((len(expanded), index in listed))
            if index in listed:
                expanded.extend(split_text(text, error))
            else:
                expanded.append(text)
        vectors, tokens = await self._embed_in_requests(expanded)
        pooled = [
            pool_vectors(vectors[at], vectors[at + 1]) if split else vectors[at]
            for at, split in layout
        ]
        return pooled, tokens

    async def _embed_in_requests(self, texts: list[str]) -> tuple[list[list[float]], int]:
        """``texts`` in as few requests as the worker accepts, sent together; a
        failing request cancels the others."""
        size = max(1, self.get_batch_size())
        if len(texts) <= size:
            return await self._embed_splitting(texts)
        results = await _gather_or_cancel(
            [
                self._embed_splitting(texts[start : start + size])
                for start in range(0, len(texts), size)
            ]
        )
        return (
            [vector for group, _ in results for vector in group],
            sum(tokens for _, tokens in results),
        )

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
        self._check_model(reply.model, endpoint)
        self._check_dimensions(reply.dimensions, reply.model)

    def _check_model(self, model: str, endpoint: str) -> None:
        expected = self.settings.expected_model
        if expected is not None and model != expected:
            raise EmbeddingWorkerModelMismatchError(expected, model)
        if self.served_model is None:
            self.served_model = model
            logger.info("Embedding worker %s serves %s", endpoint, model)
        elif model != self.served_model:
            logger.warning(
                "Embedding worker replicas at %s serve different models: %s and %s",
                endpoint,
                self.served_model,
                model,
            )
            self.served_model = model

    def _check_dimensions(self, served: int, model: str) -> None:
        if self.dimensions is None or served == self.dimensions:
            return
        source = (
            "EMBEDDING_DIMENSIONS"
            if self.configured_dimensions is not None
            else "the vector size first learned from the worker"
        )
        raise EmbeddingWorkerDimensionsMismatchError(self.dimensions, served, model, source)

    def _apply_info(self, worker: SharedWorker) -> None:
        """Check the worker's info against the config and adopt what the config leaves open."""
        info = worker.info
        if info is None:
            if worker.info_unsupported and self.batch_size > MAX_INPUTS_WITHOUT_INFO:
                raise EmbeddingWorkerConfigError(
                    f"EMBEDDING_BATCH_SIZE={self.batch_size} is more inputs per request than an "
                    f"embedding worker accepts by default ({MAX_INPUTS_WITHOUT_INFO}), and the "
                    f"worker on {worker.transport.endpoint_label} does not report its limit.",
                    remediation=f"Set EMBEDDING_BATCH_SIZE to {MAX_INPUTS_WITHOUT_INFO} or less.",
                )
            if worker.info_unsupported and self.dimensions is None:
                raise EmbeddingWorkerConfigError(
                    f"The embedding worker on {worker.transport.endpoint_label} does not report "
                    "its model info (it predates it), so its vector size is unknown.",
                    remediation=(
                        "Set EMBEDDING_DIMENSIONS to the worker model's vector size, or upgrade "
                        "the worker."
                    ),
                )
            self.worker_info = None
            return
        self._check_model(info.model, worker.transport.endpoint_label)
        self._check_dimensions(info.dimensions, info.model)
        if self.dimensions is None:
            self.dimensions = info.dimensions
            logger.info(
                "Embedding worker %s: using its %s-dimensional vectors (EMBEDDING_DIMENSIONS "
                "is unset)",
                worker.transport.endpoint_label,
                info.dimensions,
            )
        if info.max_inputs < self.batch_size and (
            self.worker_info is None or self.worker_info.max_inputs != info.max_inputs
        ):
            logger.info(
                "Embedding worker %s accepts %s inputs per request; sending that many instead "
                "of EMBEDDING_BATCH_SIZE=%s",
                worker.transport.endpoint_label,
                info.max_inputs,
                self.batch_size,
            )
        self.worker_info = info

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
        # The worker that came back may be another one: refresh its info too.
        worker.info = await self._fetch_info(worker)
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

    async def _ensure_ready(self) -> None:
        """Readiness and info, as before a request: a latched failure, or a new
        fatal one, is raised (and latched)."""
        try:
            await self._raise_if_failed()
            await self._ready()
        except (EmbeddingWorkerError, EmbeddingWorkerConfigError) as error:
            self._latch(error)
            raise

    async def _ready(self, *, fresh: bool = False) -> None:
        """Probe readiness (and fetch info) when the cached answer expired, or
        ``fresh`` asks for it; then check the info against this engine."""
        worker = await self._worker()
        if fresh or self._clock() >= worker.ready_until:
            await self._probe(worker)
        self._apply_info(worker)
        repo = self._worker_tokenizer_to_load()
        if repo is not None:
            # Loading may download tokenizer.json: off the event loop.
            await asyncio.to_thread(self._load_worker_tokenizer, repo)

    def _worker_tokenizer_to_load(self) -> str | None:
        """The worker's tokenizer, when chunks should switch to it."""
        info = self.worker_info
        if self._explicit_tokenizer or info is None or not info.tokenizer:
            return None
        if info.tokenizer == self._tokenizer_id or info.tokenizer in self._tokenizers_failed:
            return None
        return info.tokenizer

    def _load_worker_tokenizer(self, repo: str) -> None:
        """Count chunks with the worker's Hugging Face tokenizer (without special
        tokens, as cognee's chunkers count); keep the current one if it cannot load."""
        if repo == self._tokenizer_id:
            return
        try:
            tokenizer = HuggingFaceTokenizer(
                model=repo, max_completion_tokens=self.max_completion_tokens
            )
        except Exception as error:
            self._tokenizers_failed.add(repo)
            logger.warning(
                "Could not load the embedding worker's tokenizer %r (%s); chunks are still "
                "counted with %s. Set HUGGINGFACE_TOKENIZER to a tokenizer matching the "
                "worker's model.",
                repo,
                error,
                type(self.tokenizer).__name__,
                exc_info=True,
            )
            return
        self.tokenizer = tokenizer
        self._tokenizer_id = repo
        logger.info("Counting chunk tokens with the embedding worker's tokenizer %r", repo)

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
        label = worker.transport.endpoint_label
        await self._with_retries(worker.transport.readiness, label, "readiness")
        if worker.info is None:
            # Asked until the worker answers with one, then kept: an info message
            # queues behind the embed backlog, so asking on every probe would
            # stall readiness under load. A swapped worker is still caught by the
            # per-reply checks. A worker that predates info is asked again on
            # later probes (warned about once). A malformed reply, or an outage
            # outliving the retries, fails the probe.
            worker.info = await self._fetch_info(worker)
        worker.ready_until = self._clock() + READINESS_TTL_SECS

    async def _fetch_info(self, worker: SharedWorker) -> WorkerInfo | None:
        """The worker's model info, or None from a worker that predates it."""
        request_id = uuid.uuid4().hex

        async def send() -> WorkerInfo:
            async with worker.budget:
                return await worker.transport.info(request_id=request_id)

        label = worker.transport.endpoint_label
        try:
            info = await self._with_retries(send, label, f"info {request_id}")
        except EmbeddingWorkerRejectedError as error:
            # An older worker reads the info message as an invalid embed request.
            if not worker.info_unsupported:
                logger.warning(
                    "Embedding worker %s does not report its model info (%s); the vector size, "
                    "input limit and batch size come from the EMBEDDING_* config.",
                    label,
                    error.message,
                )
            worker.info_unsupported = True
            return None
        worker.info_unsupported = False
        return info

    def _learn_dimensions_blocking(self) -> int:
        """Fetch the worker's info from sync code; see :meth:`get_vector_size`."""
        if self._failure is not None and not isinstance(
            self._failure, EmbeddingWorkerUnavailableError
        ):
            raise self._failure
        logger.info(
            "EMBEDDING_DIMENSIONS is unset; asking the embedding worker for its vector size"
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(asyncio.run, self._learn_dimensions_on_own_loop()).result()
        return self.dimensions

    async def _learn_dimensions_on_own_loop(self) -> None:
        # The shared transport belongs to the caller's loop, so this loop gets its own.
        own = self._own_worker is None
        transport = (
            shared_module.build_transport(self.settings) if own else self._own_worker.transport
        )
        worker = SharedWorker(transport, asyncio.Semaphore(1))
        try:
            await self._run_probe(worker)
            self._apply_info(worker)
            repo = self._worker_tokenizer_to_load()
            if repo is not None:
                self._load_worker_tokenizer(repo)
        except (EmbeddingWorkerError, EmbeddingWorkerConfigError) as error:
            self._latch(error)
            raise
        finally:
            if own:
                await transport.aclose()

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


def split_text(text: str, error: Exception | None = None) -> list[str]:
    """``text`` as the overlapping parts ``text[:2t]`` and ``text[t:]``, ``t = len(text) // 3``
    characters (each two thirds of it, so the context across the cut is kept). A
    text too short to split raises :class:`EmbeddingContextWindowTooSmallError`."""
    third = len(text) // 3
    if third == 0:
        raise EmbeddingContextWindowTooSmallError() from error
    return sanitize_embedding_text_inputs([text[: third * 2], text[third:]])


def pool_vectors(left: list[float], right: list[float]) -> list[float]:
    """The mean of two vectors, scaled back to unit length (a zero mean stays zero)."""
    mean = (np.asarray(left, dtype=np.float64) + np.asarray(right, dtype=np.float64)) / 2
    norm = float(np.linalg.norm(mean))
    if norm > 0:
        mean /= norm
    return mean.tolist()


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
