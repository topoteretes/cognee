"""Unit tests for the CLI first-run remediation table.

Behaviour under test: for each documented failure mode, the substring
match resolves to the correct actionable hint. The messages here are the
real strings the runtime produces (or the ``str()`` of the real raised
exception), not synthetic stand-ins — a needle that only matches an
invented string would silently downgrade the first-run experience back to
a bare stack trace while the tests stayed green.
"""

import litellm
import pytest

from cognee.cli.remediation import find_remediation
from cognee.infrastructure.llm.exceptions import LLMAPIKeyNotSetError


@pytest.mark.parametrize(
    "message,anchor",
    [
        # Invalid key: litellm raises AuthenticationError; str() carries the
        # class name and the provider's "Incorrect API key" body.
        (
            "litellm.AuthenticationError: OpenAIException - Incorrect API key provided: sk-xx",
            "LLM_API_KEY",
        ),
        # Quota / billing: surfaces as a rate-limit error mentioning insufficient_quota.
        (
            "litellm.RateLimitError: insufficient_quota - You exceeded your current quota",
            "billing and quota",
        ),
        # Missing key: the real error is LLMAPIKeyNotSetError, whose message
        # is "LLM API key is not set." (no underscore).
        (
            "LLMAPIKeyNotSetError: LLM API key is not set. (Status code: 422)",
            ".env.template",
        ),
        # Endpoint unreachable / timed out: the engine's raised messages.
        (
            "Cannot connect to embedding endpoint. Check EMBEDDING_ENDPOINT.",
            "EMBEDDING_ENDPOINT",
        ),
        (
            "Embedding request timed out. Check EMBEDDING_ENDPOINT connectivity.",
            "EMBEDDING_ENDPOINT",
        ),
        (
            "Ontology file not found: /tmp/does-not-exist.owl",
            "--ontology-file",
        ),
    ],
)
def test_match_returns_expected_hint(message: str, anchor: str) -> None:
    hint = find_remediation(message)
    assert hint is not None
    assert anchor in hint


def test_matches_real_missing_key_exception() -> None:
    """The flagship first-run failure: an unset key must resolve to a hint.

    Pins the join between the actual raised exception and the table so a
    needle that drifts away from the real message cannot pass unnoticed.
    """
    hint = find_remediation(str(LLMAPIKeyNotSetError()))
    assert hint is not None
    assert "LLM_API_KEY" in hint


def test_matches_real_authentication_exception() -> None:
    """An invalid key raised by litellm must resolve to the auth hint."""
    err = litellm.exceptions.AuthenticationError(
        message="Incorrect API key provided: sk-xx",
        llm_provider="openai",
        model="text-embedding-3-large",
    )
    hint = find_remediation(str(err))
    assert hint is not None
    assert "rejected the API key" in hint


def test_no_match_returns_none() -> None:
    assert find_remediation("some unrelated error") is None


def test_empty_message_returns_none() -> None:
    assert find_remediation("") is None
    assert find_remediation(None) is None  # type: ignore[arg-type]


def test_auth_wins_over_generic_llm_key_pattern() -> None:
    """Order matters. AuthenticationError should not fall through to the
    missing-key hint just because both patterns overlap the same error.
    """
    hint = find_remediation("AuthenticationError: API key is not set / rejected")
    assert hint is not None
    assert "rejected the API key" in hint
