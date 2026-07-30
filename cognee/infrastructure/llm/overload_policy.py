"""Overload detection and reaction policy for LLM dispatch.

Everything runs at full speed until there is evidence the provider cannot
keep up; then dispatches are paced by the RPM limiter for a cooldown window
that extends while evidence keeps arriving and lapses back to the configured
state afterwards. The dispatch seam (``shared.rate_limiting``) consults a
process-wide ``OverloadPolicy`` instance; tests construct their own.
"""

import time

from cognee.shared.logging_utils import get_logger

logger = get_logger("llm_dispatch")

# How long one piece of overload evidence keeps dispatch paced; every further
# piece extends the window. When it lapses, behavior returns to the
# configured state — the reaction is never stuck on forever.
COOLDOWN_SECONDS = 900.0

# Server-overload statuses that arrive as plain HTTP errors: 503 (service
# unavailable — e.g. Ollama's queue-full reply) and 529 (Anthropic overloaded).
_OVERLOAD_STATUS_CODES = frozenset({503, 529})


def overload_evidence(error: BaseException) -> str | None:
    """Name the overload cause in ``error``'s chain, or None.

    Overload evidence is a rate-limit error (cloud providers), a timeout (how
    overwhelmed local servers usually surface — they never send rate limits),
    or an HTTP 503/529 overload status. Wrappers hide the signal — instructor
    re-raises provider errors inside InstructorRetryException — so the cause
    chain is checked, not just the top-level type. The openai base classes
    cover every client stack: litellm's exceptions subclass them, and
    SDK-direct adapters raise them natively.
    """
    import asyncio

    import openai

    overload_types = (
        openai.RateLimitError,
        openai.APITimeoutError,
        asyncio.TimeoutError,
        TimeoutError,
    )
    seen: set = set()
    node: BaseException | None = error
    while node is not None and id(node) not in seen and len(seen) < 20:
        if isinstance(node, overload_types):
            return type(node).__name__
        if (
            isinstance(node, openai.APIStatusError)
            and getattr(node, "status_code", None) in _OVERLOAD_STATUS_CODES
        ):
            return f"HTTP {node.status_code}"
        seen.add(id(node))
        node = node.__cause__ or node.__context__
    return None


class OverloadPolicy:
    """Pace-on-evidence policy with cooldown-episode semantics.

    Warns once per overload episode; further evidence inside the window
    extends it silently, and a fresh episode after a quiet cooldown warns
    again.
    """

    def __init__(self, cooldown_seconds: float = COOLDOWN_SECONDS):
        self.cooldown_seconds = cooldown_seconds
        self._paced_until = 0.0

    def is_paced(self) -> bool:
        """Whether dispatches should currently be paced by the RPM limiter."""
        return time.monotonic() < self._paced_until

    def on_error(self, error: BaseException) -> None:
        """Record a failed dispatch; overload errors count as evidence."""
        evidence = overload_evidence(error)
        if evidence is not None:
            self._react(evidence)

    def _react(self, reason: str) -> None:
        # NOTE: Import inside function to avoid a circular import at module load.
        from cognee.infrastructure.llm.config import get_llm_config

        now = time.monotonic()
        new_episode = now >= self._paced_until
        self._paced_until = now + self.cooldown_seconds
        if new_episode:
            llm_config = get_llm_config()
            logger.warning(
                "Potential RPM issues detected (%s) — slowing down processing to "
                "accommodate: enabling the RPM limiter (%d requests per %ds) for "
                "the next %.0fs, extended while issues persist. Tune "
                "LLM_RATE_LIMIT_REQUESTS / LLM_RATE_LIMIT_INTERVAL to match what "
                "your provider or machine can handle, or set AUTO_RATE_LIMIT=false "
                "to opt out.",
                reason,
                llm_config.llm_rate_limit_requests,
                llm_config.llm_rate_limit_interval,
                self.cooldown_seconds,
            )


# Process-wide default instance consulted by the dispatch seam.
llm_overload_policy = OverloadPolicy()
