from contextlib import asynccontextmanager, nullcontext

from aiolimiter import AsyncLimiter

from cognee.infrastructure.llm.config import get_llm_config
from cognee.shared.logging_utils import get_logger

logger = get_logger("llm_dispatch")

llm_config = get_llm_config()

llm_rate_limiter = AsyncLimiter(
    llm_config.llm_rate_limit_requests, llm_config.llm_rate_limit_interval
)
# The embedding limiter reads its settings from EmbeddingConfig and is built
# lazily to avoid a circular import at module load time (embedding engines, which
# EmbeddingConfig is reached through, import this module).
_embedding_rate_limiter = None

# Set once AUTO_RATE_LIMIT reacts to a provider rate limit: from then on every
# dispatch is paced by the RPM limiter above, exactly as if
# LLM_RATE_LIMIT_ENABLED had been set.
_auto_rate_limit_activated = False

# Local inference servers never report rate limits — their overload signal is
# a slow client timeout, far too late to react to. With AUTO_RATE_LIMIT they
# are therefore paced from the very first dispatch instead of reactively.
_LOCAL_LLM_PROVIDERS = frozenset({"ollama", "llama_cpp"})
_LOCAL_MODEL_PREFIXES = ("lm_studio/", "hosted_vllm/", "vllm/")
_local_pacing_active: bool | None = None  # resolved lazily on first dispatch


def _local_pacing() -> bool:
    """Pace local LLM servers up front (once, with a warning explaining why)."""
    global _local_pacing_active
    if _local_pacing_active is None:
        provider = (llm_config.llm_provider or "").lower()
        model = (llm_config.llm_model or "").lower()
        is_local = provider in _LOCAL_LLM_PROVIDERS or model.startswith(_LOCAL_MODEL_PREFIXES)
        _local_pacing_active = llm_config.auto_rate_limit and is_local
        if _local_pacing_active:
            logger.warning(
                "Local LLM provider detected (%s / %s) — enabling the RPM limiter "
                "(%d requests per %ds) up front, because local servers cannot absorb "
                "unbounded request floods and never report rate limits themselves. "
                "Tune LLM_RATE_LIMIT_REQUESTS / LLM_RATE_LIMIT_INTERVAL to match your "
                "machine's real throughput, or set AUTO_RATE_LIMIT=false to disable "
                "this behavior.",
                llm_config.llm_provider,
                llm_config.llm_model,
                llm_config.llm_rate_limit_requests,
                llm_config.llm_rate_limit_interval,
            )
    return _local_pacing_active


def _activate_rate_limiter_on_rate_limit(error: BaseException) -> None:
    """Warn and switch on the RPM limiter if the error (or its cause) is a rate limit.

    Wrappers hide the signal — instructor re-raises provider errors inside
    InstructorRetryException — so the cause chain is checked, not just the
    top-level type. openai.RateLimitError covers every client stack: litellm's
    exceptions subclass it, and SDK-direct adapters raise it natively.
    """
    global _auto_rate_limit_activated
    import openai

    seen: set = set()
    node: BaseException | None = error
    while node is not None and id(node) not in seen and len(seen) < 20:
        if isinstance(node, openai.RateLimitError):
            if not _auto_rate_limit_activated:
                _auto_rate_limit_activated = True
                logger.warning(
                    "LLM provider reported a rate limit — enabling the RPM limiter "
                    "(%d requests per %ds) for the rest of this process. Tune with "
                    "LLM_RATE_LIMIT_REQUESTS / LLM_RATE_LIMIT_INTERVAL, or set "
                    "AUTO_RATE_LIMIT=false to opt out.",
                    llm_config.llm_rate_limit_requests,
                    llm_config.llm_rate_limit_interval,
                )
            return
        seen.add(id(node))
        node = node.__cause__ or node.__context__


@asynccontextmanager
async def _governed_llm_dispatch():
    """Gate one LLM dispatch attempt.

    Cloud is unbounded by default; the RPM limiter paces when
    LLM_RATE_LIMIT_ENABLED is set, when AUTO_RATE_LIMIT (default on) reacted
    to a provider rate limit, or from the start for local inference servers.
    Adapters enter this context manager INSIDE their tenacity retry, so
    retried attempts are paced as well.
    """
    pace = llm_config.llm_rate_limit_enabled or _auto_rate_limit_activated or _local_pacing()
    async with llm_rate_limiter if pace else nullcontext():
        try:
            yield
        except Exception as error:
            if llm_config.auto_rate_limit:
                _activate_rate_limiter_on_rate_limit(error)
            raise


def llm_rate_limiter_context_manager():
    return _governed_llm_dispatch()


def embedding_rate_limiter_context_manager():
    global _embedding_rate_limiter
    from cognee.infrastructure.databases.vector.embeddings.config import (
        get_embedding_config,
    )

    embedding_config = get_embedding_config()
    if not embedding_config.embedding_rate_limit_enabled:
        #  Return a no-op context manager if rate limiting is disabled
        return nullcontext()

    if _embedding_rate_limiter is None:
        _embedding_rate_limiter = AsyncLimiter(
            embedding_config.embedding_rate_limit_requests,
            embedding_config.embedding_rate_limit_interval,
        )
    return _embedding_rate_limiter
