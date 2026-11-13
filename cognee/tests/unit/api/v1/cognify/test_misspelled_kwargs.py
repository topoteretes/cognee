"""Misspelled cognify() kwargs are rejected with a "did you mean" hint (#5773).

cognify() forwards unknown kwargs into the extractor LLM call, so we cannot
reject every unknown key. But a one- or two-character typo of a real parameter
(``extactor`` for ``extractor``, ``sumary_method`` for ``summary_method``) would
otherwise be silently forwarded as an extractor option and ignored. We only
flag keys that are close matches to a known parameter name; genuine extractor
options pass through untouched.
"""

import importlib

import pytest

from cognee.api.v1.cognify.cognify import (
    raise_on_close_misspelled_kwargs,
)

cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
serve_state_module = importlib.import_module("cognee.api.v1.serve.state")


def test_empty_kwargs_pass_through():
    raise_on_close_misspelled_kwargs({})


@pytest.mark.parametrize(
    ("typo", "suggestion"),
    [
        ("extactor", "extractor"),
        ("sumary_method", "summary_method"),
        ("summery_method", "summary_method"),
        ("chunks_ize", "chunk_size"),
    ],
)
def test_close_misspelling_raises_with_did_you_mean(typo, suggestion):
    with pytest.raises(TypeError) as exc_info:
        raise_on_close_misspelled_kwargs({typo: "whatever"})
    message = str(exc_info.value)
    assert typo in message
    assert suggestion in message
    assert "did you mean" in message


def test_multiple_typos_are_listed_together():
    with pytest.raises(TypeError) as exc_info:
        raise_on_close_misspelled_kwargs({"extactor": "llm", "sumary_method": "from_extraction"})
    message = str(exc_info.value)
    assert "extactor" in message
    assert "extractor" in message
    assert "sumary_method" in message
    assert "summary_method" in message


def test_genuine_extractor_options_pass_through():
    # These are the sort of keys cognify() intentionally forwards into
    # extract_graph_and_summarize(). They are not close matches to any explicit
    # cognify parameter, so they must NOT raise.
    raise_on_close_misspelled_kwargs(
        {
            "extraction_instructions": "only extract people",
            "system_prompt": "be terse",
            "entity_types": ["PERSON", "ORG"],
        }
    )


def test_unrelated_arbitrary_kwargs_pass_through():
    raise_on_close_misspelled_kwargs({"completely_custom_option": 42})


@pytest.mark.asyncio
async def test_cognify_rejects_typo_before_running_pipeline():
    from unittest.mock import AsyncMock, patch

    with (
        patch.object(serve_state_module, "get_remote_client", return_value=None),
        patch.object(
            cognify_module,
            "ensure_extractor_runtime",
            new_callable=AsyncMock,
        ),
        patch.object(
            cognify_module,
            "run_migrations_and_block",
            new_callable=AsyncMock,
        ),
        pytest.raises(TypeError, match="extactor.*extractor"),
    ):
        await cognify_module.cognify(extactor="gliner_demo")
