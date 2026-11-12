import asyncio
from contextlib import contextmanager
from contextvars import ContextVar

from cognee.shared.logging_utils import get_logger

from .EmbeddingEngine import EmbeddingEngine, EmbeddingResult, Vector

logger = get_logger("EmbeddingGateway")


_request_semaphore: ContextVar[asyncio.Semaphore | None] = ContextVar(
    "embedding_request_semaphore", default=None
)


@contextmanager
def embedding_request_scope(semaphore: asyncio.Semaphore):
    """Share a storage step's request budget without holding it during database writes."""
    token = _request_semaphore.set(semaphore)
    try:
        yield
    finally:
        _request_semaphore.reset(token)


class EmbeddingGateway:
    """Gateway to an embedding engine: wraps it, returns its vectors and records
    each call's token usage, the way LLMGateway does for completions.

    Large calls are split into provider-sized requests. Inside a storage step they
    run concurrently under its shared request semaphore. Outside that scope they
    remain sequential, preserving the caller's existing concurrency policy.
    """

    def __init__(self, engine: EmbeddingEngine):
        self._engine = engine

    async def embed_text(self, text: list[str]) -> list[Vector]:
        request_size = self._request_size()
        semaphore = _request_semaphore.get()

        async def request(texts):
            if semaphore is None:
                return await self._engine.embed_text(texts)
            async with semaphore:
                return await self._engine.embed_text(texts)

        if request_size is None or len(text) <= request_size:
            vectors = await request(text)
            await self._record_usage(vectors.tokens)
            return vectors

        batches = [
            text[start : start + request_size] for start in range(0, len(text), request_size)
        ]
        if semaphore is None:
            parts = [await request(batch) for batch in batches]
        else:
            # Preserve input order, and let every sibling finish before propagating
            # failure so no provider request outlives the storage call.
            parts = await asyncio.gather(
                *(request(batch) for batch in batches), return_exceptions=True
            )
            for part in parts:
                if isinstance(part, BaseException):
                    raise part
        vectors = [vector for part in parts for vector in part]
        tokens = sum(part.tokens for part in parts)
        await self._record_usage(tokens)
        return EmbeddingResult(vectors, tokens=tokens)

    def _request_size(self) -> int | None:
        """Texts per request, or None when the engine gives no usable size."""
        get_batch_size = getattr(self._engine, "get_batch_size", None)
        if get_batch_size is None:
            return None
        size = get_batch_size()
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            return None
        return size

    async def _record_usage(self, tokens: int) -> None:
        """Best-effort: usage accounting must never fail the embedding itself."""
        try:
            from cognee.modules.session_lifecycle.usage_tracking import record_embedding_call

            await record_embedding_call(model=self._engine.model, tokens=tokens)
        except Exception:
            logger.debug("Ignoring exception in EmbeddingGateway._record_usage", exc_info=True)

    def __getattr__(self, name: str):
        # Everything except embed_text (vector size, batch size, input limit,
        # model, ...) is the wrapped engine's.
        return getattr(self._engine, name)
