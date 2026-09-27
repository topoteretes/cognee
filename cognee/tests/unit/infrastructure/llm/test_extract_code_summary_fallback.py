"""extract_code_summary degrades to the mock summary only on a validation failure.

The fallback used to catch the legacy framework's retry-exhausted exception by type,
which never fires under ``litellm_native`` (the default): that adapter surfaces the last
``pydantic.ValidationError`` / ``json.JSONDecodeError`` once its self-correction loop is
exhausted. The framework-neutral predicate has to fire for both, and for nothing else.
"""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from cognee.infrastructure.llm.exceptions import (
    LLMPaymentRequiredError,
    is_structured_output_validation_error,
)
from cognee.shared.data_models import SummarizedCode

extract_summary_module = importlib.import_module(
    "cognee.infrastructure.llm.extraction.extract_summary"
)


def _validation_error() -> ValidationError:
    try:
        SummarizedCode.model_validate({})
    except ValidationError as error:
        return error
    raise AssertionError("expected an empty payload to fail validation")


class _LegacyRetryExhausted(Exception):
    """Stand-in with the legacy framework's class name; matched by name, not import."""


_LegacyRetryExhausted.__name__ = "InstructorRetryException"


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (_validation_error(), True),
        (json.JSONDecodeError("bad", "{", 0), True),
        (_LegacyRetryExhausted("exhausted"), True),
        (RuntimeError("boom"), False),
        (LLMPaymentRequiredError(), False),
    ],
)
def test_predicate_matches_validation_failures_only(error, expected):
    assert is_structured_output_validation_error(error) is expected


def test_predicate_walks_cause_chain():
    wrapper = RuntimeError("wrapped")
    wrapper.__cause__ = _validation_error()
    assert is_structured_output_validation_error(wrapper) is True


@pytest.mark.asyncio
async def test_code_summary_falls_back_to_mock_on_validation_error(monkeypatch):
    monkeypatch.delenv("MOCK_CODE_SUMMARY", raising=False)
    with patch.object(
        extract_summary_module.LLMGateway,
        "acreate_structured_output",
        AsyncMock(side_effect=_validation_error()),
    ):
        result = await extract_summary_module.extract_code_summary("def f(): pass")

    assert isinstance(result, SummarizedCode)
    assert result.high_level_summary == "Mock code summary"


@pytest.mark.asyncio
async def test_code_summary_propagates_non_validation_errors(monkeypatch):
    monkeypatch.delenv("MOCK_CODE_SUMMARY", raising=False)
    with (
        patch.object(
            extract_summary_module.LLMGateway,
            "acreate_structured_output",
            AsyncMock(side_effect=LLMPaymentRequiredError()),
        ),
        pytest.raises(LLMPaymentRequiredError),
    ):
        await extract_summary_module.extract_code_summary("def f(): pass")


@pytest.mark.asyncio
async def test_code_summary_falls_back_through_real_native_adapter(monkeypatch):
    """End-to-end under litellm_native: the model keeps returning invalid JSON.

    Only ``litellm.acompletion`` is mocked. The adapter's json-fallback loop exhausts
    its self-correction attempts and raises the last validation error, which the
    summary extractor must turn into the mock summary.
    """
    from cognee.infrastructure.llm.structured_output_framework.litellm_native.native_adapter import (
        NativeLiteLLMAdapter,
    )

    monkeypatch.delenv("MOCK_CODE_SUMMARY", raising=False)
    # A model litellm does not know forces the prompted-JSON path, whose
    # self-correction loop is the one that ends in a bare ValidationError.
    adapter = NativeLiteLLMAdapter(
        api_key="test-key", model="custom/unknown-model", max_completion_tokens=1024
    )
    invalid = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"not": "a summary"}'))]
    )
    fake_acompletion = AsyncMock(return_value=invalid)

    gateway_module = importlib.import_module("cognee.infrastructure.llm.LLMGateway")
    native_factory = importlib.import_module(
        "cognee.infrastructure.llm.structured_output_framework.litellm_native.get_native_client"
    )
    config = SimpleNamespace(structured_output_framework="litellm_native")

    with (
        patch("litellm.acompletion", fake_acompletion),
        patch.object(gateway_module, "get_llm_config", return_value=config),
        patch.object(native_factory, "get_native_client", return_value=adapter),
        patch(
            "cognee.infrastructure.llm.structured_output_framework.litellm_native.native_adapter.llm_retry_stop_condition",
            lambda retry_state: True,
        ),
    ):
        result = await extract_summary_module.extract_code_summary("def f(): pass")

    assert result.high_level_summary == "Mock code summary"
    # The self-correction loop ran to exhaustion before the fallback kicked in.
    assert fake_acompletion.call_count >= 3
