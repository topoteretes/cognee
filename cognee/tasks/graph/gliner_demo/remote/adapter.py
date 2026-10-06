"""Transport-neutral client for a remote GLiNER worker.

One :class:`RemoteGlinerAdapter` serves one cognify run. Per worker endpoint and
event loop, adapters share one transport (connection pool), one request budget
(``COGNEE_GLINER_MAX_CONCURRENT_REQUESTS``) and one readiness answer (cached
for :data:`READINESS_TTL_SECS`), so concurrent runs neither open a pool each nor
pile more requests onto a worker than the operator sized it for.

The worker windows long chunks itself (``window_words``), with the same
gliner2 call the local runtime makes, so results come back exactly as local
extraction returns them: one gliner2 result dict per chunk, offsets into it.

Failure scope follows the error class (see :mod:`.errors`): a rejected or
failed request fails its document, and a fatal error (an outage that outlived
the retries, bad credentials, the wrong model, an incompatible worker) is
latched, so every later extraction of the run fails at once instead of each
document waiting out its own retries.
"""

from __future__ import annotations

import asyncio
import time
import uuid
import weakref
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from cognee.modules.observability import new_span
from cognee.shared.logging_utils import get_logger

from ..schema import GlinerSchema
from .errors import (
    GlinerRemoteConfigError,
    GlinerWorkerError,
    GlinerWorkerIncompatibleError,
    GlinerWorkerModelMismatchError,
    GlinerWorkerRejectedError,
    GlinerWorkerRuntimeError,
    GlinerWorkerUnavailableError,
)
from .protocol import (
    WINDOWING_FEATURE,
    WorkerReadiness,
    WorkerReply,
    WorkerTransport,
    redact_url,
)
from .retry import RetryPolicy
from .settings import RemoteGlinerSettings

logger = get_logger("gliner.remote")

READINESS_TTL_SECS = 30.0
# The worker's windowing caps (its domain.py): overlap at most half a window.
MAX_WINDOW_WORDS = 1024
DEFAULT_RETRY = RetryPolicy()
OVERLAP_POLICY = "longest"

T = TypeVar("T")


@dataclass
class _SharedWorker:
    transport: WorkerTransport
    budget: asyncio.Semaphore
    ready_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    readiness: WorkerReadiness | None = None
    ready_until: float = 0.0


# Transports and semaphores belong to one event loop, so the cache is per loop;
# a loop that is gone drops its entries.
_shared: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[tuple, _SharedWorker]] = (
    weakref.WeakKeyDictionary()
)


def build_transport(settings: RemoteGlinerSettings) -> WorkerTransport:
    """Create the transport ``settings.transport`` names; extras are imported lazily."""
    endpoint = settings.require_endpoint()
    api_key = settings.api_key.get_secret_value() if settings.api_key else None
    if settings.transport == "http":
        from .http import HttpWorkerTransport

        return HttpWorkerTransport(
            endpoint,
            api_key=api_key,
            connect_timeout=settings.connect_timeout_secs,
            request_timeout=settings.request_timeout_secs,
            max_response_bytes=settings.max_response_bytes,
        )
    if settings.transport == "grpc":
        from .grpc import GrpcWorkerTransport

        return GrpcWorkerTransport(
            endpoint,
            api_key=api_key,
            connect_timeout=settings.connect_timeout_secs,
            request_timeout=settings.request_timeout_secs,
            max_response_bytes=settings.max_response_bytes,
        )
    if settings.transport == "amqp":
        from .amqp import AmqpWorkerTransport

        return AmqpWorkerTransport(
            endpoint,
            queue=settings.amqp_queue,
            connect_timeout=settings.connect_timeout_secs,
            request_timeout=settings.request_timeout_secs,
            max_response_bytes=settings.max_response_bytes,
        )
    raise GlinerRemoteConfigError(f"Unknown GLiNER transport {settings.transport!r}.")


def _shared_key(settings: RemoteGlinerSettings) -> tuple:
    return (
        settings.transport,
        settings.endpoint,
        settings.amqp_queue,
        settings.api_key.get_secret_value() if settings.api_key else None,
        settings.connect_timeout_secs,
        settings.request_timeout_secs,
        settings.max_response_bytes,
        settings.max_concurrent_requests,
    )


def _shared_worker(settings: RemoteGlinerSettings) -> _SharedWorker:
    per_loop = _shared.setdefault(asyncio.get_running_loop(), {})
    key = _shared_key(settings)
    worker = per_loop.get(key)
    if worker is None:
        worker = _SharedWorker(
            transport=build_transport(settings),
            budget=asyncio.Semaphore(settings.max_concurrent_requests),
        )
        per_loop[key] = worker
    return worker


async def close_shared_workers() -> None:
    """Close the transports this event loop opened (tests, orderly shutdown)."""
    workers = _shared.pop(asyncio.get_running_loop(), {})
    for worker in workers.values():
        await worker.transport.aclose()


async def _gather_or_cancel(coroutines: Sequence[Coroutine[Any, Any, None]]) -> None:
    """Run all; on the first failure cancel the rest and raise that failure."""
    tasks = [asyncio.ensure_future(coroutine) for coroutine in coroutines]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


class RemoteGlinerAdapter:
    """Extraction through a gliner_worker for one cognify run.

    ``transport`` replaces the shared, settings-built one (tests, custom
    transports); the adapter then keeps its own budget and readiness cache.
    """

    def __init__(
        self,
        settings: RemoteGlinerSettings,
        *,
        transport: WorkerTransport | None = None,
        retry: RetryPolicy = DEFAULT_RETRY,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        if transport is None:
            settings.require_endpoint()
        self.settings = settings
        self._retry = retry
        self._sleep = sleep
        self._clock = clock
        self._own_worker = (
            _SharedWorker(transport, asyncio.Semaphore(settings.max_concurrent_requests))
            if transport is not None
            else None
        )
        self._failure: Exception | None = None
        # From the settings, not the transport: it must be printable even when
        # building the transport is what failed.
        self._label = (
            transport.endpoint_label
            if transport is not None
            else redact_url(settings.endpoint or "")
            + (f" queue {settings.amqp_queue}" if settings.transport == "amqp" else "")
        )
        #: The model the worker reported, once it has.
        self.model: str | None = None

    # ------------------------------------------------------------------ public

    async def ensure_ready(self) -> WorkerReadiness:
        """Check (or reuse a recent check) that the worker is up, compatible and
        serving the expected model, before any chunk is sent."""
        self._raise_if_failed()
        try:
            return await self._ready()
        except (GlinerWorkerError, GlinerRemoteConfigError) as error:
            self._latch(error)
            raise

    async def extract_batch(
        self,
        texts: Sequence[str],
        schema: GlinerSchema,
        *,
        threshold: float,
        batch_size: int,
        window_words: int,
        window_overlap_words: int,
    ) -> list[Mapping[str, Any]]:
        """One gliner2 result per text, as the local ``batch_extract_long`` returns."""
        self._raise_if_failed()
        if not texts:
            return []
        if schema.is_empty:
            return [{} for _ in texts]
        self._check_window(window_words, window_overlap_words)

        options = {
            "threshold": threshold,
            "batch_size": batch_size,
            "include_confidence": True,
            "include_spans": True,
            "overlap_policy": OVERLAP_POLICY,
            "window_words": window_words,
            "window_overlap_words": window_overlap_words,
        }
        results: list[Mapping[str, Any]] = [{} for _ in texts]
        # Blank chunks get the empty result the local runtime gives them; the
        # worker would reject the whole request for one of them.
        pending = [(index, text) for index, text in enumerate(texts) if text.strip()]
        groups = [
            pending[start : start + self.settings.inputs_per_request]
            for start in range(0, len(pending), self.settings.inputs_per_request)
        ]

        with new_span("cognee.gliner.remote.extract_batch") as span:
            span.set_attribute("cognee.gliner.remote.endpoint", self.endpoint_label)
            span.set_attribute("cognee.gliner.remote.texts", len(texts))
            span.set_attribute("cognee.gliner.remote.requests", len(groups))
            try:
                readiness = await self._ready()
                self._check_batch_size(readiness, batch_size)
                self._check_text_lengths(readiness, pending)
                in_flight = asyncio.Semaphore(self.settings.max_in_flight_requests)

                async def run(group: list[tuple[int, str]]) -> None:
                    async with in_flight:
                        extractions = await self._request(
                            [text for _, text in group], schema, options, windowed=True
                        )
                    for (index, _), extraction in zip(group, extractions):
                        results[index] = extraction

                await _gather_or_cancel([run(group) for group in groups])
            except (GlinerWorkerError, GlinerRemoteConfigError) as error:
                span.set_attribute("cognee.gliner.remote.error", error.name)
                self._latch(error)
                raise
        return results

    async def extract_once(
        self, text: str, schema: GlinerSchema, *, threshold: float
    ) -> Mapping[str, Any]:
        """One unwindowed, span-less extraction (the label-bank probe), as the
        local ``extract`` returns it."""
        self._raise_if_failed()
        if not text.strip() or schema.is_empty:
            return {}
        options = {
            "threshold": threshold,
            "batch_size": 1,
            "include_confidence": False,
            "include_spans": False,
            "overlap_policy": OVERLAP_POLICY,
        }
        try:
            readiness = await self._ready()
            self._check_text_lengths(readiness, [(0, text)])
            [result] = await self._request([text], schema, options, windowed=False)
        except (GlinerWorkerError, GlinerRemoteConfigError) as error:
            self._latch(error)
            raise
        return result

    @property
    def endpoint_label(self) -> str:
        return self._label

    # ------------------------------------------------------------- internals

    def _worker(self) -> _SharedWorker:
        return self._own_worker or _shared_worker(self.settings)

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            raise self._failure

    def _latch(self, error: Exception) -> None:
        if getattr(error, "fatal", False) and self._failure is None:
            self._failure = error
            logger.error(
                "GLiNER worker %s failed this run: %s",
                self.endpoint_label,
                getattr(error, "message", error),
            )

    async def _ready(self) -> WorkerReadiness:
        worker = self._worker()
        if worker.readiness is None or self._clock() >= worker.ready_until:
            async with worker.ready_lock:
                if worker.readiness is None or self._clock() >= worker.ready_until:
                    readiness = await self._with_retries(
                        worker.transport.readiness, worker.transport.endpoint_label, "readiness"
                    )
                    worker.readiness = readiness
                    worker.ready_until = self._clock() + READINESS_TTL_SECS
        readiness = worker.readiness
        if readiness.model is not None:
            self._record_model(readiness.model)
        if readiness.features is not None and WINDOWING_FEATURE not in readiness.features:
            raise GlinerWorkerIncompatibleError(
                f"The GLiNER worker at {worker.transport.endpoint_label} does not support "
                "windowing long inputs; without it most mentions in a long chunk are lost."
            )
        max_inputs = readiness.limits.max_inputs
        if max_inputs is not None and self.settings.inputs_per_request > max_inputs:
            raise GlinerRemoteConfigError(
                f"COGNEE_GLINER_INPUTS_PER_REQUEST={self.settings.inputs_per_request} exceeds "
                f"the worker's limit of {max_inputs} inputs per request.",
                remediation=f"Set COGNEE_GLINER_INPUTS_PER_REQUEST to {max_inputs} or less.",
            )
        return readiness

    def _check_window(self, window_words: int, window_overlap_words: int) -> None:
        """The worker caps windows so one request cannot multiply its work; a run
        outside the caps would be rejected document by document, so fail it here."""
        if window_words > MAX_WINDOW_WORDS or window_overlap_words > window_words // 2:
            error = GlinerRemoteConfigError(
                f"window_words={window_words} / window_overlap_words={window_overlap_words} "
                "is outside what a GLiNER worker accepts: window_words at most "
                f"{MAX_WINDOW_WORDS}, window_overlap_words at most half of it.",
                remediation="Use the defaults (384 / 64) or values within those bounds.",
            )
            self._latch(error)
            raise error

    @staticmethod
    def _check_batch_size(readiness: WorkerReadiness, batch_size: int) -> None:
        cap = readiness.limits.max_batch_size
        if cap is not None and batch_size > cap:
            raise GlinerRemoteConfigError(
                f"gliner_batch_size={batch_size} exceeds the worker's model batch cap of {cap}.",
                remediation=f"Use a gliner_batch_size of {cap} or less.",
            )

    @staticmethod
    def _check_text_lengths(readiness: WorkerReadiness, pending: list[tuple[int, str]]) -> None:
        cap = readiness.limits.max_text_chars
        if cap is None:
            return
        for index, text in pending:
            if len(text) > cap:
                raise GlinerWorkerRejectedError(
                    f"Chunk {index} has {len(text)} characters; the GLiNER worker accepts at "
                    f"most {cap} per input. Use a smaller chunk_size."
                )

    async def _request(
        self,
        texts: list[str],
        schema: GlinerSchema,
        options: Mapping[str, Any],
        *,
        windowed: bool,
    ) -> list[Mapping[str, Any]]:
        worker = self._worker()
        ids = [str(index) for index in range(len(texts))]
        payload = {
            "inputs": [{"id": id_, "text": text} for id_, text in zip(ids, texts)],
            "schema": {
                "entities": dict(schema.entity_types),
                "relations": dict(schema.relation_types),
            },
            "options": dict(options),
        }
        # One id per logical request, kept across its retries, so client spans
        # and worker logs can be joined.
        request_id = uuid.uuid4().hex
        attempts = 0

        async def send() -> WorkerReply:
            nonlocal attempts
            attempts += 1
            # The budget is held only while a request is in flight, never
            # across a retry backoff.
            async with worker.budget:
                return await worker.transport.extract(payload, request_id=request_id)

        with new_span("cognee.gliner.remote.request") as span:
            span.set_attribute("cognee.gliner.remote.request_id", request_id)
            span.set_attribute("cognee.gliner.remote.inputs", len(texts))
            started = time.perf_counter()
            try:
                reply = await self._with_retries(
                    send, worker.transport.endpoint_label, f"request {request_id}"
                )
            except GlinerWorkerError as error:
                span.set_attribute("cognee.gliner.remote.error", error.name)
                raise
            finally:
                span.set_attribute("cognee.gliner.remote.attempts", attempts)
                span.set_attribute(
                    "cognee.gliner.remote.latency_ms", (time.perf_counter() - started) * 1000
                )
            span.set_attribute("cognee.gliner.remote.model", reply.model)
            if reply.elapsed_ms is not None:
                span.set_attribute("cognee.gliner.remote.worker_elapsed_ms", reply.elapsed_ms)

        extractions = self._match_reply(reply, ids)
        self._record_model(reply.model)
        if windowed and not reply.windowed:
            raise GlinerWorkerIncompatibleError(
                f"The GLiNER worker at {worker.transport.endpoint_label} ignored window_words; "
                "it predates windowing, and most mentions in a long chunk would be lost."
            )
        return extractions

    @staticmethod
    def _match_reply(reply: WorkerReply, ids: list[str]) -> list[Mapping[str, Any]]:
        if len(reply.items) != len(ids):
            raise GlinerWorkerRuntimeError(
                f"The GLiNER worker returned {len(reply.items)} results for {len(ids)} inputs."
            )
        by_id: dict[str, Mapping[str, Any]] = {}
        expected = set(ids)
        for id_, extraction in reply.items:
            if id_ not in expected or id_ in by_id:
                raise GlinerWorkerRuntimeError(
                    f"The GLiNER worker returned an unknown or duplicate input id {id_!r}."
                )
            by_id[id_] = extraction
        return [by_id[id_] for id_ in ids]

    def _record_model(self, model: str) -> None:
        expected = self.settings.expected_model
        if expected is not None and model != expected:
            raise GlinerWorkerModelMismatchError(expected, model)
        if self.model is None:
            self.model = model
            logger.info("GLiNER worker %s serves %s", self.endpoint_label, model)
        elif model != self.model:
            logger.warning(
                "GLiNER worker replicas at %s serve different models: %s and %s",
                self.endpoint_label,
                self.model,
                model,
            )

    async def _with_retries(self, call: Callable[[], Awaitable[T]], endpoint: str, what: str) -> T:
        last = self._retry.attempts - 1
        for attempt in range(self._retry.attempts):
            try:
                return await call()
            except GlinerWorkerUnavailableError as error:
                if attempt == last:
                    raise
                delay = self._retry.delay(attempt, error.retry_after)
                logger.warning(
                    "GLiNER worker %s unavailable during %s (%s); retry %d of %d in %.2fs",
                    endpoint,
                    what,
                    error.message,
                    attempt + 1,
                    last,
                    delay,
                )
                await self._sleep(delay)
        raise AssertionError("unreachable")  # the loop returns or raises
