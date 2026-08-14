"""First-run error remediation table.

The failure modes that show up most often on a clean install are:

1. **Invalid API key** — key is set but rejected upstream (401).
2. **Key rejected for permission / quota** — provider accepts the key but
   denies the request (403, or an exhausted-quota / billing error).
3. **Missing API key** — user has not set ``LLM_API_KEY`` at all.
4. **Unreachable custom endpoint** — user pointed ``EMBEDDING_ENDPOINT`` or
   ``LLM_ENDPOINT`` at a URL that resolves but does not respond.
5. **Wrong ontology path** — ``--ontology-file`` argument does not exist.

Each of these otherwise triggers a raw stack trace from deep in the pipeline.
This module lifts a short, prescriptive hint to the top of the terminal so
the user knows the exact env var or flag to fix without reading the trace.

Matching is by case-insensitive substring on the stringified error rather
than exception-type checks: many of these failures reach the CLI wrapped in
``CliCommandException`` with a stringified inner error, so ``isinstance``
would miss the real class. Keeping the match on substrings also lets a new
failure mode land as one table row with no import-graph work. Needles are
anchored on the words that appear in the *raised* message (verified against
the runtime error text), not on env-var spellings a message may not contain.
"""

from __future__ import annotations

from typing import Optional, Tuple

# Ordering matters: the first match wins, so narrower patterns come first.
# An invalid key ("authenticationerror") beats the generic missing-key row.
# Each entry is (needles, hint); the hint fires when any needle is a
# case-insensitive substring of the error message.
_TABLE: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (
        ("authenticationerror", "invalid api key", "incorrect api key"),
        "The LLM provider rejected the API key. Fix: set LLM_API_KEY in "
        "your .env to a valid key for the LLM_PROVIDER you configured "
        "(default provider: openai).",
    ),
    (
        ("permissiondeniederror", "insufficient_quota", "billing"),
        "The LLM provider accepted the key but denied the request. Fix: "
        "confirm the account has active billing and quota, or switch "
        "LLM_PROVIDER/LLM_MODEL to one your account can use.",
    ),
    (
        # Real error: LLMAPIKeyNotSetError("LLM API key is not set.").
        ("llmapikeynotset", "api key is not set", "no api key"),
        "LLM_API_KEY is not set. Fix: copy .env.template to .env and "
        "populate LLM_API_KEY. Cognee defaults to the OpenAI provider "
        "so an OpenAI key is the simplest starting point.",
    ),
    (
        # Real raised messages: "Cannot connect to embedding endpoint. Check
        # EMBEDDING_ENDPOINT." and "Embedding request timed out. Check
        # EMBEDDING_ENDPOINT connectivity."
        ("embedding_endpoint", "cannot connect to embedding", "embedding request timed out"),
        "The configured EMBEDDING_ENDPOINT is not reachable. Fix: verify "
        "the URL, that the host is running, and that the port is open. "
        "Unset EMBEDDING_ENDPOINT to fall back to the provider default.",
    ),
    (
        ("ontology file not found",),
        "The --ontology-file path does not exist. Fix: pass an absolute "
        "path to an .owl / .ttl file, or drop the flag to use the built-in "
        "resolver.",
    ),
)


def find_remediation(message: str) -> Optional[str]:
    """Return the hint for the first matching pattern, or ``None``.

    Deliberately tolerant of a ``None`` or empty message so callers can
    hand off whatever they got from ``str(ex)`` without pre-validation.
    """
    if not message:
        return None
    haystack = message.lower()
    for needles, hint in _TABLE:
        if any(needle in haystack for needle in needles):
            return hint
    return None
