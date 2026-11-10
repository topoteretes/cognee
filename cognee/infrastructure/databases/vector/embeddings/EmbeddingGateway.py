from cognee.shared.logging_utils import get_logger

from .EmbeddingEngine import EmbeddingEngine, EmbeddingResult, Vector

logger = get_logger("EmbeddingGateway")


class EmbeddingGateway:
    """Gateway to an embedding engine: wraps it, returns its vectors and records
    each call's token usage, the way LLMGateway does for completions.

    A call with more texts than the engine sends per request (``get_batch_size``) is
    embedded in consecutive requests of that size, so a caller can hand over a whole
    vector write batch (``VECTOR_WRITE_BATCH_SIZE`` rows) without knowing the
    provider's input limit. The requests of one call run one after another, not
    concurrently: the caller's bound on concurrent calls stays the bound on requests
    in flight, which is what ``EMBEDDING_MAX_CONCURRENT_DATA_POINTS`` promises.
    """

    def __init__(self, engine: EmbeddingEngine):
        self._engine = engine

    async def embed_text(self, text: list[str]) -> list[Vector]:
        request_size = self._request_size()
        if request_size is None or len(text) <= request_size:
            vectors = await self._engine.embed_text(text)
            await self._record_usage(vectors.tokens)
            return vectors

        vectors: list[Vector] = []
        tokens = 0
        for start in range(0, len(text), request_size):
            part = await self._engine.embed_text(text[start : start + request_size])
            vectors.extend(part)
            tokens += part.tokens
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
