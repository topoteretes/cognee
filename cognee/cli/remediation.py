"""First-run error remediation table.

The four failure modes that show up most often on a clean install are:

1. **Missing / empty API key** — user has not set ``LLM_API_KEY``.
2. **Invalid API key** — key is set but rejected upstream (401 / 403).
3. **Unreachable custom endpoint** — user pointed ``EMBEDDING_ENDPOINT`` or
   ``LLM_ENDPOINT`` at a URL that resolves but does not respond.
4. **Wrong ontology path** — ``--ontology-file`` argument does not exist.

Each of these triggers a raw stack trace from deep in the pipeline today.
This module lifts a short, prescriptive hint to the top of the terminal
so the user knows the exact env var or flag to fix without reading the
trace.

The dispatcher is deliberately pattern-based on the exception ``repr``
rather than exception-type checks: many of these failures surface via
``CliCommandException`` with a stringified inner error, so ``isinstance``
would miss the real class. Keeping the match on substrings also lets a
new failure mode land as one dict entry with no import graph work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class Remediation:
    """A pattern + hint pair.

    ``needles`` is any/all: the hint fires when *any* needle is a
    case-insensitive substring of the error message. Keep needles short
    and stable — full sentence fragments are more brittle across
    ``litellm`` versions than one or two anchor words.
    """

    needles: tuple[str, ...]
    hint: str


# Ordering matters: the first match wins, so put narrower patterns
# ahead of broader ones. Auth failures beat generic BadRequest.
_TABLE: tuple[Remediation, ...] = (
    Remediation(
        needles=("authenticationerror", "invalid api key", "incorrect api key"),
        hint=(
            "The LLM provider rejected the API key. Fix: set LLM_API_KEY in "
            "your .env to a valid key for the LLM_PROVIDER you configured "
            "(default provider: openai)."
        ),
    ),
    Remediation(
        needles=("permissiondeniederror", "insufficient_quota", "billing"),
        hint=(
            "The LLM provider accepted the key but denied the request. Fix: "
            "confirm the account has active billing and quota, or switch "
            "LLM_PROVIDER/LLM_MODEL to one your account can use."
        ),
    ),
    Remediation(
        needles=(
            "llm_api_key",
            "no api key",
        ),
        hint=(
            "LLM_API_KEY is not set. Fix: copy .env.template to .env and "
            "populate LLM_API_KEY. Cognee defaults to the OpenAI provider "
            "so an OpenAI key is the simplest starting point."
        ),
    ),
    Remediation(
        needles=(
            "embedding_endpoint",
            "cannot connect to embedding",
            "embedding endpoint timed out",
        ),
        hint=(
            "The configured EMBEDDING_ENDPOINT is not reachable. Fix: verify "
            "the URL, that the host is running, and that the port is open. "
            "Unset EMBEDDING_ENDPOINT to fall back to the provider default."
        ),
    ),
    Remediation(
        needles=("ontology file not found",),
        hint=(
            "The --ontology-file path does not exist. Fix: pass an absolute "
            "path to an .owl / .ttl file, or drop the flag to use the built-in "
            "resolver."
        ),
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
    for row in _TABLE:
        if _matches(haystack, row.needles):
            return row.hint
    return None


def _matches(haystack: str, needles: Iterable[str]) -> bool:
    """Return True when any needle is a substring of ``haystack``.

    Case handling is caller-owned: pass in an already-lowered haystack.
    """
    return any(needle in haystack for needle in needles)
