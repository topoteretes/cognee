from contextlib import nullcontext

from aiolimiter import AsyncLimiter

from cognee.infrastructure.llm.config import get_llm_config

llm_config = get_llm_config()

llm_rate_limiter = AsyncLimiter(
    llm_config.llm_rate_limit_requests, llm_config.llm_rate_limit_interval
)
# The embedding rate-limiter reads its settings from EmbeddingConfig. It is built
# lazily (on first use) to avoid a circular import at module load time, since the
# embedding engines import this module.
_embedding_rate_limiter = None


def _get_embedding_rate_limiter():
    global _embedding_rate_limiter
    if _embedding_rate_limiter is None:
        from cognee.infrastructure.databases.vector.embeddings.config import (
            get_embedding_config,
        )

        embedding_config = get_embedding_config()
        _embedding_rate_limiter = AsyncLimiter(
            embedding_config.embedding_rate_limit_requests,
            embedding_config.embedding_rate_limit_interval,
        )
    return _embedding_rate_limiter


def llm_rate_limiter_context_manager():
    global llm_rate_limiter
    if llm_config.llm_rate_limit_enabled:
        return llm_rate_limiter
    else:
        #  Return a no-op context manager if rate limiting is disabled
        return nullcontext()


def embedding_rate_limiter_context_manager():
    from cognee.infrastructure.databases.vector.embeddings.config import (
        get_embedding_config,
    )

    if get_embedding_config().embedding_rate_limit_enabled:
        return _get_embedding_rate_limiter()
    else:
        #  Return a no-op context manager if rate limiting is disabled
        return nullcontext()
