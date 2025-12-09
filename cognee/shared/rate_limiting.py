from aiolimiter import AsyncLimiter
from contextlib import nullcontext
from cognee.infrastructure.llm.config import get_llm_config

llm_config = get_llm_config()

llm_rate_limiter = AsyncLimiter(
    llm_config.llm_rate_limit_requests, llm_config.embedding_rate_limit_interval
)
embedding_rate_limiter = AsyncLimiter(
    llm_config.embedding_rate_limit_requests, llm_config.embedding_rate_limit_interval
)


def llm_rate_limiter_context_manager():
    global llm_rate_limiter
    if llm_config.llm_rate_limit_enabled:
        return llm_rate_limiter
    else:
        #  Return a no-op context manager if rate limiting is disabled
        return nullcontext()


def embedding_rate_limiter_context_manager():
    global embedding_rate_limiter
    if llm_config.embedding_rate_limit_enabled:
        return embedding_rate_limiter
    else:
        #  Return a no-op context manager if rate limiting is disabled
        return nullcontext()
