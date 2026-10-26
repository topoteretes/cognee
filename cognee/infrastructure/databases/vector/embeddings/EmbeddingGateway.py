from cognee.shared.logging_utils import get_logger

from .EmbeddingEngine import EmbeddingEngine, Vector

logger = get_logger("EmbeddingGateway")


class EmbeddingGateway:
    """Gateway to an embedding engine: wraps it, returns its vectors and records
    each call's token usage, the way LLMGateway does for completions."""

    def __init__(self, engine: EmbeddingEngine):
        self._engine = engine

    async def embed_text(self, text: list[str]) -> list[Vector]:
        vectors = await self._engine.embed_text(text)
        await self._record_usage(vectors.tokens)
        return vectors

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
