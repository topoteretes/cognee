"""Shared tenacity retry policy for LLM structured-output calls.

Used by every structured-output framework — the litellm/instructor adapters and
the BAML integration alike. Each ``acreate_structured_output`` retries until BOTH
floors are satisfied: at least ``LLM_MIN_RETRY_ATTEMPTS`` attempts AND at least
``LLM_MIN_RETRY_SECONDS`` of elapsed wall-clock time.

``&`` builds tenacity's ``stop_all`` predicate, which stops only once *every*
sub-condition holds (``|`` / ``stop_any`` would stop at whichever floor is hit
first). The predicate is stateless — it reads everything off the per-call retry
state — so this single instance is safe to share across every retry decorator.
"""

from tenacity import stop_after_attempt, stop_after_delay

# Minimum number of attempts before the call is allowed to give up.
LLM_MIN_RETRY_ATTEMPTS = 2
# Minimum elapsed seconds before the call is allowed to give up.
LLM_MIN_RETRY_SECONDS = 240

# Stop retrying only once BOTH the attempt floor AND the time floor are met.
llm_retry_stop_condition = stop_after_attempt(LLM_MIN_RETRY_ATTEMPTS) & stop_after_delay(
    LLM_MIN_RETRY_SECONDS
)
