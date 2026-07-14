"""Unit tests for the litellm_native structured output framework.

Tests cover:
- Successful structured output via schema-native path
- Validation-error retry with error context injection (JSON-fallback path)
- Rate-limit errors propagating immediately without retry
- Auth errors propagating immediately without retry
- Fallback model activation on content policy violation
- Zero instructor imports in the litellm_native package
- LLMGateway routing to litellm_native when config is set
"""

import ast
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

# ---- Test models ----


class PersonModel(BaseModel):
    """Simple Pydantic model used across tests."""

    name: str
    age: int


# ---- Helpers ----


def _make_mock_response(content: str) -> MagicMock:
    """Build a mock LiteLLM ``ModelResponse`` with the given message content."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


# ---- Tests ----


@pytest.mark.asyncio
async def test_acreate_structured_output_returns_valid_pydantic_object():
    """Schema-native path: mock returns valid JSON, assert result is correct Pydantic instance."""
    from cognee.infrastructure.llm.structured_output_framework.litellm_native.native_adapter import (
        NativeLiteLLMAdapter,
    )

    adapter = NativeLiteLLMAdapter(
        api_key="test-key",
        model="openai/gpt-5-mini",  # schema-native provider
        max_completion_tokens=4096,
    )

    valid_json = json.dumps({"name": "Alice", "age": 30})
    mock_response = _make_mock_response(valid_json)

    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
        result = await adapter.acreate_structured_output(
            text_input="Tell me about Alice",
            system_prompt="Extract person info.",
            response_model=PersonModel,
        )

    assert isinstance(result, PersonModel)
    assert result.name == "Alice"
    assert result.age == 30


@pytest.mark.asyncio
async def test_validation_error_triggers_retry_with_error_context():
    """JSON-fallback path: invalid JSON first, valid JSON second.

    Asserts that the retry happened (two calls total) and the final result
    is a correctly validated Pydantic object.
    """
    from cognee.infrastructure.llm.structured_output_framework.litellm_native.native_adapter import (
        NativeLiteLLMAdapter,
    )

    adapter = NativeLiteLLMAdapter(
        api_key="test-key",
        model="ollama/llama3",  # JSON-fallback provider (not schema-native)
        max_completion_tokens=4096,
    )

    # First call returns invalid JSON (missing 'age'), second returns valid JSON.
    invalid_response = _make_mock_response('{"name": "Bob"}')  # missing 'age' field
    valid_response = _make_mock_response('{"name": "Bob", "age": 25}')

    mock_acompletion = AsyncMock(side_effect=[invalid_response, valid_response])

    with patch("litellm.acompletion", mock_acompletion):
        result = await adapter.acreate_structured_output(
            text_input="Tell me about Bob",
            system_prompt="Extract person info.",
            response_model=PersonModel,
        )

    assert isinstance(result, PersonModel)
    assert result.name == "Bob"
    assert result.age == 25
    # Should have been called twice: first attempt failed, second succeeded.
    assert mock_acompletion.call_count == 2

    # Verify that the second call included the validation error in the user message
    # so the model could self-correct.
    second_call_messages = mock_acompletion.call_args_list[1].kwargs.get(
        "messages", mock_acompletion.call_args_list[1][1].get("messages", [])
    )
    user_message = next((m["content"] for m in second_call_messages if m["role"] == "user"), "")
    assert "failed validation" in user_message.lower() or "validation error" in user_message.lower()


@pytest.mark.asyncio
async def test_rate_limit_error_raises_immediately_without_retry():
    """Rate-limit error must propagate immediately — call count should be 1."""
    import litellm.exceptions

    from cognee.infrastructure.llm.structured_output_framework.litellm_native.native_adapter import (
        NativeLiteLLMAdapter,
    )

    adapter = NativeLiteLLMAdapter(
        api_key="test-key",
        model="openai/gpt-5-mini",
        max_completion_tokens=4096,
    )

    mock_acompletion = AsyncMock(
        side_effect=litellm.exceptions.RateLimitError(
            message="Rate limit exceeded",
            model="openai/gpt-5-mini",
            llm_provider="openai",
        )
    )

    with patch("litellm.acompletion", mock_acompletion):
        with pytest.raises(litellm.exceptions.RateLimitError):
            await adapter.acreate_structured_output(
                text_input="Test input",
                system_prompt="Test prompt",
                response_model=PersonModel,
            )

    # Must have been called exactly once — no retry.
    assert mock_acompletion.call_count == 1


@pytest.mark.asyncio
async def test_auth_error_raises_immediately():
    """Authentication error must propagate immediately — call count should be 1."""
    import litellm.exceptions

    from cognee.infrastructure.llm.structured_output_framework.litellm_native.native_adapter import (
        NativeLiteLLMAdapter,
    )

    adapter = NativeLiteLLMAdapter(
        api_key="bad-key",
        model="openai/gpt-5-mini",
        max_completion_tokens=4096,
    )

    mock_acompletion = AsyncMock(
        side_effect=litellm.exceptions.AuthenticationError(
            message="Invalid API key",
            model="openai/gpt-5-mini",
            llm_provider="openai",
        )
    )

    with patch("litellm.acompletion", mock_acompletion):
        with pytest.raises(litellm.exceptions.AuthenticationError):
            await adapter.acreate_structured_output(
                text_input="Test input",
                system_prompt="Test prompt",
                response_model=PersonModel,
            )

    assert mock_acompletion.call_count == 1


@pytest.mark.asyncio
async def test_fallback_model_used_on_content_policy_error():
    """Primary raises ContentPolicyViolationError, fallback succeeds."""
    from litellm.exceptions import ContentPolicyViolationError

    from cognee.infrastructure.llm.structured_output_framework.litellm_native.native_adapter import (
        NativeLiteLLMAdapter,
    )

    adapter = NativeLiteLLMAdapter(
        api_key="test-key",
        model="openai/gpt-5-mini",
        max_completion_tokens=4096,
        fallback_model="openai/gpt-5",
        fallback_api_key="fallback-key",
        fallback_endpoint="https://fallback.example.com",
    )

    valid_json = json.dumps({"name": "Charlie", "age": 40})
    fallback_response = _make_mock_response(valid_json)

    call_count = 0

    async def side_effect(*args: Any, **kwargs: Any) -> MagicMock:
        nonlocal call_count
        call_count += 1
        # First call (primary model) raises content policy error.
        if call_count == 1:
            raise ContentPolicyViolationError(
                message="Content policy violation",
                model="openai/gpt-5-mini",
                llm_provider="openai",
            )
        # Second call (fallback model) succeeds.
        return fallback_response

    with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=side_effect):
        result = await adapter.acreate_structured_output(
            text_input="Some input",
            system_prompt="Extract info.",
            response_model=PersonModel,
        )

    assert isinstance(result, PersonModel)
    assert result.name == "Charlie"
    assert result.age == 40
    # Primary call + fallback call = 2.
    assert call_count == 2


def test_no_instructor_import_in_litellm_native():
    """AST scan of all .py files under litellm_native/ — zero instructor imports."""
    native_pkg_dir = (
        Path(os.path.dirname(os.path.abspath(__file__))).parent.parent.parent.parent
        / "infrastructure"
        / "llm"
        / "structured_output_framework"
        / "litellm_native"
    )

    assert native_pkg_dir.exists(), f"Could not find litellm_native at {native_pkg_dir}"

    instructor_imports: list[str] = []

    for py_file in native_pkg_dir.rglob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "instructor" or alias.name.startswith("instructor."):
                        instructor_imports.append(
                            f"{py_file.name}:{node.lineno} import {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module and (
                    node.module == "instructor" or node.module.startswith("instructor.")
                ):
                    instructor_imports.append(
                        f"{py_file.name}:{node.lineno} from {node.module} import ..."
                    )

    assert instructor_imports == [], (
        f"Found instructor imports in litellm_native: {instructor_imports}"
    )


@pytest.mark.asyncio
async def test_gateway_routes_to_litellm_native_when_config_set():
    """Patch config to litellm_native, call LLMGateway, assert get_native_client is called."""
    from cognee.infrastructure.llm.LLMGateway import LLMGateway

    mock_adapter = AsyncMock()
    mock_adapter.acreate_structured_output = AsyncMock(
        return_value=PersonModel(name="Diana", age=28)
    )

    mock_get_native_client = MagicMock(return_value=mock_adapter)

    with patch("cognee.infrastructure.llm.LLMGateway.get_llm_config") as mock_config:
        config_instance = MagicMock()
        config_instance.structured_output_framework = "litellm_native"
        mock_config.return_value = config_instance

        with patch(
            "cognee.infrastructure.llm.structured_output_framework.litellm_native.get_native_client.get_native_client",
            mock_get_native_client,
        ):
            # We need to also patch the import inside LLMGateway
            import cognee.infrastructure.llm.structured_output_framework.litellm_native.get_native_client as gnc_module

            original_func = gnc_module.get_native_client

            gnc_module.get_native_client = mock_get_native_client

            try:
                result = await LLMGateway.acreate_structured_output(
                    text_input="Tell me about Diana",
                    system_prompt="Extract person info.",
                    response_model=PersonModel,
                )
            finally:
                gnc_module.get_native_client = original_func

    # Verify get_native_client was called (not get_llm_client).
    mock_get_native_client.assert_called_once()
    mock_adapter.acreate_structured_output.assert_called_once()
    assert isinstance(result, PersonModel)
    assert result.name == "Diana"


class _PaymentRequiredError(Exception):
    """Stand-in for a provider HTTP 402 (payment required / budget exhausted)."""

    status_code = 402


@pytest.mark.asyncio
async def test_budget_exhausted_error_raises_payment_required_without_retry():
    """A budget/402 error surfaces as LLMPaymentRequiredError and is not retried."""
    from cognee.infrastructure.llm.exceptions import LLMPaymentRequiredError
    from cognee.infrastructure.llm.structured_output_framework.litellm_native.native_adapter import (
        NativeLiteLLMAdapter,
    )

    adapter = NativeLiteLLMAdapter(
        api_key="test-key",
        model="openai/gpt-5-mini",
        max_completion_tokens=4096,
    )

    mock_acompletion = AsyncMock(side_effect=_PaymentRequiredError("Payment required"))

    with patch("litellm.acompletion", mock_acompletion):
        with pytest.raises(LLMPaymentRequiredError):
            await adapter.acreate_structured_output(
                text_input="Test input",
                system_prompt="Test prompt",
                response_model=PersonModel,
            )

    # Mapped to an actionable, non-retryable error — called exactly once.
    assert mock_acompletion.call_count == 1


@pytest.mark.asyncio
async def test_schema_native_passes_response_model_as_response_format():
    """Schema-capable models get the Pydantic class straight through as response_format."""
    from cognee.infrastructure.llm.structured_output_framework.litellm_native.native_adapter import (
        NativeLiteLLMAdapter,
    )

    adapter = NativeLiteLLMAdapter(
        api_key="test-key",
        model="openai/gpt-5-mini",  # supports_response_schema is True
        max_completion_tokens=4096,
    )

    mock_acompletion = AsyncMock(
        return_value=_make_mock_response(json.dumps({"name": "Eve", "age": 22}))
    )
    with patch("litellm.acompletion", mock_acompletion):
        await adapter.acreate_structured_output(
            text_input="Tell me about Eve",
            system_prompt="Extract person info.",
            response_model=PersonModel,
        )

    kwargs = mock_acompletion.call_args.kwargs
    assert kwargs["response_format"] is PersonModel


@pytest.mark.asyncio
async def test_json_fallback_uses_json_object_and_injects_schema():
    """Non-schema models get response_format=json_object with the schema in the prompt."""
    from cognee.infrastructure.llm.structured_output_framework.litellm_native.native_adapter import (
        NativeLiteLLMAdapter,
    )

    adapter = NativeLiteLLMAdapter(
        api_key="test-key",
        model="ollama/llama3",  # supports_response_schema is False
        max_completion_tokens=4096,
    )

    mock_acompletion = AsyncMock(
        return_value=_make_mock_response(json.dumps({"name": "Eve", "age": 22}))
    )
    with patch("litellm.acompletion", mock_acompletion):
        await adapter.acreate_structured_output(
            text_input="Tell me about Eve",
            system_prompt="Extract person info.",
            response_model=PersonModel,
        )

    kwargs = mock_acompletion.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    system_message = next(m["content"] for m in kwargs["messages"] if m["role"] == "system")
    assert "schema" in system_message.lower()


@pytest.mark.asyncio
async def test_connection_state_is_call_invariant_across_fallback():
    """The fallback path must not mutate the (shared, cached) adapter's own state.

    Regression guard: each call sends its own model, and the adapter's
    model/api_key/endpoint are unchanged afterwards, so concurrent calls cannot
    bleed the fallback connection params into one another.
    """
    from litellm.exceptions import ContentPolicyViolationError

    from cognee.infrastructure.llm.structured_output_framework.litellm_native.native_adapter import (
        NativeLiteLLMAdapter,
    )

    adapter = NativeLiteLLMAdapter(
        api_key="primary-key",
        model="openai/gpt-5-mini",
        max_completion_tokens=4096,
        endpoint="https://primary.example.com",
        fallback_model="openai/gpt-5",
        fallback_api_key="fallback-key",
        fallback_endpoint="https://fallback.example.com",
    )

    seen_models: list[str] = []

    async def side_effect(*args: Any, **kwargs: Any) -> MagicMock:
        seen_models.append(kwargs.get("model"))
        if len(seen_models) == 1:
            raise ContentPolicyViolationError(
                message="Content policy violation",
                model="openai/gpt-5-mini",
                llm_provider="openai",
            )
        return _make_mock_response(json.dumps({"name": "Zoe", "age": 31}))

    with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=side_effect):
        result = await adapter.acreate_structured_output(
            text_input="Some input",
            system_prompt="Extract info.",
            response_model=PersonModel,
        )

    assert result.name == "Zoe"
    # Primary call used the primary model; fallback call used the fallback model.
    assert seen_models == ["openai/gpt-5-mini", "openai/gpt-5"]
    # Adapter connection state is unchanged after the fallback path.
    assert adapter.model == "openai/gpt-5-mini"
    assert adapter.api_key == "primary-key"
    assert adapter.endpoint == "https://primary.example.com"
