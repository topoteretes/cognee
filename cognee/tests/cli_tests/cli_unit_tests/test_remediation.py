"""Unit tests for the CLI first-run remediation table.

Behaviour under test: for each documented failure mode, the substring
match resolves to the correct actionable hint. Regressions here would
silently downgrade the first-run experience back to a bare stack trace.
"""

import pytest

from cognee.cli.remediation import find_remediation


@pytest.mark.parametrize(
    "message,anchor",
    [
        (
            "litellm.exceptions.AuthenticationError: invalid api key",
            "LLM_API_KEY",
        ),
        (
            "PermissionDeniedError: insufficient_quota for tier",
            "billing and quota",
        ),
        (
            "ValueError: LLM_API_KEY missing from environment",
            ".env.template",
        ),
        (
            "Cannot connect to embedding endpoint. Check EMBEDDING_ENDPOINT.",
            "EMBEDDING_ENDPOINT",
        ),
        (
            "Embedding endpoint timed out. EMBEDDING_ENDPOINT='http://x'",
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


def test_no_match_returns_none() -> None:
    assert find_remediation("some unrelated error") is None


def test_empty_message_returns_none() -> None:
    assert find_remediation("") is None
    assert find_remediation(None) is None  # type: ignore[arg-type]


def test_auth_wins_over_generic_llm_key_pattern() -> None:
    """Order matters. AuthenticationError should not fall through to the
    "no api key" hint just because both patterns overlap the same error.
    """
    hint = find_remediation("AuthenticationError: LLM_API_KEY rejected")
    assert hint is not None
    assert "rejected the API key" in hint
