from contextlib import asynccontextmanager, nullcontext

from aiolimiter import AsyncLimiter

from cognee.infrastructure.llm.config import get_llm_config
from cognee.shared.logging_utils import get_logger

logger = get_logger("llm_dispatch")

llm_config = get_llm_config()

# The budget is provider-aware: LLMConfig gives local inference servers a
# smaller default LLM_RATE_LIMIT_REQUESTS (see default_local_rate_limit_budget).
llm_rate_limiter = AsyncLimiter(
    llm_config.llm_rate_limit_requests, llm_config.llm_rate_limit_interval
)
# The embedding limiter reads its settings from EmbeddingConfig and is built
# lazily to avoid a circular import at module load time (embedding engines, which
# EmbeddingConfig is reached through, import this module).
_embedding_rate_limiter = None


# Early overload detection: a dispatch that SUCCEEDS but takes this long is
# proof the server's queue is approaching the client timeout cliff (default
# 600s) — react before anything actually fails. Healthy setups complete
# calls in seconds to a couple of minutes and never approach this.
SLOW_COMPLETION_SECONDS = 300.0

# When AUTO_RATE_LIMIT reacts to overload evidence, dispatch is paced by the
# RPM limiter until this deadline; every further piece of evidence extends it.
# Once it lapses, behavior returns to the configured state (unbounded unless
# LLM_RATE_LIMIT_ENABLED is set) — the reaction is never stuck on forever.
AUTO_RATE_LIMIT_COOLDOWN_SECONDS = 300.0

_auto_paced_until = 0.0  # monotonic deadline of the current overload episode


def _fall_back_to_rate_limiter(reason: str) -> None:
    """Pace dispatches with the RPM limiter for a cooldown window.

    Warns once per overload episode; further evidence inside the window
    extends it silently, and a fresh episode after a quiet cooldown warns
    again.
    """
    global _auto_paced_until
    import time

    now = time.monotonic()
    new_episode = now >= _auto_paced_until
    _auto_paced_until = now + AUTO_RATE_LIMIT_COOLDOWN_SECONDS
    if new_episode:
        logger.warning(
            "LLM requests are not being processed fast enough (%s) — enabling "
            "the RPM limiter (%d requests per %ds) for the next %.0fs, extended "
            "while issues persist. Tune LLM_RATE_LIMIT_REQUESTS / "
            "LLM_RATE_LIMIT_INTERVAL to match what your provider or machine can "
            "handle, or set AUTO_RATE_LIMIT=false to opt out.",
            reason,
            llm_config.llm_rate_limit_requests,
            llm_config.llm_rate_limit_interval,
            AUTO_RATE_LIMIT_COOLDOWN_SECONDS,
        )


def _overload_evidence(error: BaseException) -> str | None:
    """Name the overload cause in ``error``'s chain, or None.

    Overload evidence is a rate-limit error (cloud providers), a timeout (how
    overwhelmed local servers usually surface — they never send rate limits),
    or an HTTP 503/529 overload status (Ollama past its queue cap, Anthropic
    overloaded). Wrappers hide the signal — instructor re-raises provider
    errors inside InstructorRetryException — so the cause chain is checked,
    not just the top-level type. The openai base classes cover every client
    stack: litellm's exceptions subclass them, and SDK-direct adapters raise
    them natively.
    """
    import asyncio

    import openai

    overload_types = (
        openai.RateLimitError,
        openai.APITimeoutError,
        asyncio.TimeoutError,
        TimeoutError,
    )
    # Server-overload statuses that arrive as plain HTTP errors: 503 (service
    # unavailable — e.g. Ollama's queue-full reply) and 529 (Anthropic overloaded).
    overload_statuses = frozenset({503, 529})
    seen: set = set()
    node: BaseException | None = error
    while node is not None and id(node) not in seen and len(seen) < 20:
        if isinstance(node, overload_types):
            return type(node).__name__
        if (
            isinstance(node, openai.APIStatusError)
            and getattr(node, "status_code", None) in overload_statuses
        ):
            return f"HTTP {node.status_code}"
        seen.add(id(node))
        node = node.__cause__ or node.__context__
    return None


@asynccontextmanager
async def _governed_llm_dispatch():
    """Gate one LLM dispatch attempt.

    Everything runs unbounded (the fast path). When there is evidence the
    provider cannot keep up — a completion slow enough to prove the queue is
    nearing the client timeout (detected BEFORE anything fails), a rate-limit
    error, an HTTP 503/529, or a timeout — one warning is logged and the RPM
    limiter paces dispatches for a cooldown window, extended while issues
    persist; when it lapses, behavior returns to the configured state.
    LLM_RATE_LIMIT_ENABLED paces from the start, unchanged. Adapters enter
    this context manager INSIDE their tenacity retry, so retried attempts are
    paced as well.
    """
    import time

    pace = llm_config.llm_rate_limit_enabled or time.monotonic() < _auto_paced_until
    async with llm_rate_limiter if pace else nullcontext():
        started = time.monotonic()
        try:
            yield
        except Exception as error:
            if llm_config.auto_rate_limit:
                evidence = _overload_evidence(error)
                if evidence is not None:
                    _fall_back_to_rate_limiter(evidence)
            raise
        else:
            elapsed = time.monotonic() - started
            if llm_config.auto_rate_limit and elapsed >= SLOW_COMPLETION_SECONDS:
                _fall_back_to_rate_limiter(f"completion took {elapsed:.0f}s")


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
