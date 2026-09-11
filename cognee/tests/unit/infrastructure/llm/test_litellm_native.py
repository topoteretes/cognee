"""Unit tests for the litellm_native structured output framework.

Tests cover:
- Successful structured output via the schema-native path
- Validation-error retry with error-context injection (JSON-fallback path)
- Auth errors propagating immediately without retry
- Quota/budget errors mapped to LLMPaymentRequiredError, no retry (#3643)
- asyncio.CancelledError propagating immediately without retry
- Fallback model activation on a content-policy violation
- response_format wiring per path (Pydantic class vs json_object + schema)
- Connection state staying call-invariant across the fallback path
- Zero instructor imports in the litellm_native package
- LLMGateway routing to litellm_native when the config is set
"""

import ast
import asyncio
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
    """With the framework set to litellm_native, the gateway calls get_native_client.

    Resolve the module objects via importlib and patch.object, not dotted strings:
    the package __init__ binds the name ``cognee.infrastructure.llm.LLMGateway`` to
    the *class*, so both a string patch target and ``import ... as`` (which resolves
    by attribute access) can land on the class instead of the submodule depending on
    import order — flaky across test shards. ``import_module`` returns the real module.
    """
    import importlib

    gateway_module = importlib.import_module("cognee.infrastructure.llm.LLMGateway")
    native_factory = importlib.import_module(
        "cognee.infrastructure.llm.structured_output_framework.litellm_native.get_native_client"
    )

    mock_adapter = AsyncMock()
    mock_adapter.acreate_structured_output = AsyncMock(
        return_value=PersonModel(name="Diana", age=28)
    )
    mock_get_native_client = MagicMock(return_value=mock_adapter)

    config_instance = MagicMock()
    config_instance.structured_output_framework = "litellm_native"

    with (
        patch.object(gateway_module, "get_llm_config", return_value=config_instance),
        patch.object(native_factory, "get_native_client", mock_get_native_client),
    ):
        result = await gateway_module.LLMGateway.acreate_structured_output(
            text_input="Tell me about Diana",
            system_prompt="Extract person info.",
            response_model=PersonModel,
        )

    # Routed to get_native_client (not the instructor get_llm_client).
    mock_get_native_client.assert_called_once()
    mock_adapter.acreate_structured_output.assert_called_once()
    assert isinstance(result, PersonModel)
    assert result.name == "Diana"


class _PaymentRequiredError(Exception):
    """Stand-in for a provider HTTP 402 (payment required / budget exhausted)."""

    status_code = 402


@pytest.mark.asyncio
async def test_budget_exhausted_error_raises_payment_required_without_retry():
    """A quota/402 error surfaces as LLMPaymentRequiredError and is not retried (#3643)."""
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

    seen_models: list[str | None] = []

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


@pytest.mark.asyncio
async def test_str_response_model_uses_plain_completion():
    """response_model=str skips structured output and returns the raw content."""
    from cognee.infrastructure.llm.structured_output_framework.litellm_native.native_adapter import (
        NativeLiteLLMAdapter,
    )

    adapter = NativeLiteLLMAdapter(
        api_key="test-key", model="openai/gpt-5-mini", max_completion_tokens=4096
    )

    mock_acompletion = AsyncMock(return_value=_make_mock_response("just some text"))
    with patch("litellm.acompletion", mock_acompletion):
        result = await adapter.acreate_structured_output(
            text_input="say hi",
            system_prompt="be brief",
            response_model=str,
        )

    assert result == "just some text"
    # No schema constraint is sent for a plain-string response.
    assert "response_format" not in mock_acompletion.call_args.kwargs


@pytest.mark.asyncio
async def test_cancellation_is_not_retried():
    """asyncio.CancelledError must propagate immediately, not be retried."""
    from cognee.infrastructure.llm.structured_output_framework.litellm_native.native_adapter import (
        NativeLiteLLMAdapter,
    )

    adapter = NativeLiteLLMAdapter(
        api_key="test-key", model="openai/gpt-5-mini", max_completion_tokens=4096
    )

    mock_acompletion = AsyncMock(side_effect=asyncio.CancelledError())
    with patch("litellm.acompletion", mock_acompletion):
        with pytest.raises(asyncio.CancelledError):
            await adapter.acreate_structured_output(
                text_input="t", system_prompt="s", response_model=PersonModel
            )

    assert mock_acompletion.call_count == 1


# ── provider-qualified model names (CLO-594) ─────────────────────────────────
# cognee stores provider and model separately; litellm routes on a qualified
# model name. The instructor path dispatched per provider, so LLM_PROVIDER=ollama
# with LLM_MODEL=phi4 worked there and 400s on litellm_native with
# "LLM Provider NOT provided". This broke the Ollama nightly the first time it
# ran on dev after litellm_native became the default framework.


class TestQualifyModel:
    """_qualify_model must rescue unroutable names and touch nothing else."""

    @staticmethod
    def _qualify(model, provider):
        from cognee.infrastructure.llm.structured_output_framework.litellm_native.get_native_client import (
            _qualify_model,
        )

        return _qualify_model(model, provider)

    def test_bare_ollama_model_gets_prefixed(self):
        """The exact CI failure: LLM_PROVIDER=ollama, LLM_MODEL=phi4."""
        assert self._qualify("phi4", "ollama") == "ollama/phi4"

    def test_tagged_ollama_model_gets_prefixed(self):
        assert self._qualify("qwen3:latest", "ollama") == "ollama/qwen3:latest"

    def test_already_qualified_is_untouched(self):
        assert self._qualify("ollama/phi4", "ollama") == "ollama/phi4"
        assert self._qualify("openai/gpt-5-mini", "openai") == "openai/gpt-5-mini"

    def test_name_litellm_already_resolves_is_untouched(self):
        """Must not re-route a configuration that works today."""
        assert self._qualify("gpt-4o", "openai") == "gpt-4o"

    def test_unknown_provider_is_untouched(self):
        """generic/llama_cpp have no unambiguous litellm prefix — leave them be."""
        assert self._qualify("some-custom-model", "generic") == "some-custom-model"

    def test_empty_model_is_untouched(self):
        assert self._qualify("", "ollama") == ""

    def test_qualified_name_is_routable_by_litellm(self):
        """End of the chain: the rescued name must actually resolve."""
        import litellm

        qualified = self._qualify("phi4", "ollama")
        assert litellm.get_llm_provider(model=qualified)[1] == "ollama"

    def test_namespaced_ollama_model_gets_prefixed(self):
        """A slash is not a provider: Ollama accepts namespaced names."""
        assert self._qualify("library/phi4", "ollama") == "ollama/library/phi4"

    def test_hugging_face_gguf_path_gets_prefixed(self):
        """The documented way to run a GGUF under Ollama keeps its full path."""
        model = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"
        assert self._qualify(model, "ollama") == f"ollama/{model}"

    def test_namespaced_name_is_routable_by_litellm(self):
        """Same end of the chain, for a name that contains a slash."""
        import litellm

        qualified = self._qualify("library/phi4", "ollama")
        assert litellm.get_llm_provider(model=qualified)[1] == "ollama"


# ── markdown-fenced JSON on the fallback path (CLO-596) ──────────────────────
# The prompted-JSON path hands the model's reply straight to
# model_validate_json. Models on that path routinely wrap the answer in a
# ```json fence: the JSON inside is valid, but pydantic sees a backtick at
# column 1 and rejects it, and the self-correction retry cannot help because a
# model that fences once fences again. Hit while capturing the datasheets cassette.


class TestStripJsonFence:
    @staticmethod
    def _strip(text):
        from cognee.infrastructure.llm.structured_output_framework.litellm_native.native_adapter import (
            _strip_json_fence,
        )

        return _strip_json_fence(text)

    def test_fenced_with_language_tag(self):
        """The exact shape that broke the datasheets capture."""
        assert self._strip('```json\n{"summary": "s"}\n```') == '{"summary": "s"}'

    def test_fenced_without_language_tag(self):
        assert self._strip('```\n{"summary": "s"}\n```') == '{"summary": "s"}'

    def test_surrounding_whitespace(self):
        assert self._strip('  ```json\n{"summary": "s"}\n```  \n') == '{"summary": "s"}'

    def test_plain_json_untouched(self):
        payload = '{"summary": "s"}'
        assert self._strip(payload) == payload

    def test_backticks_inside_a_value_untouched(self):
        """Anchored to the whole payload, so a value containing ``` survives."""
        payload = '{"summary": "use ```json to fence"}'
        assert self._strip(payload) == payload

    def test_fenced_payload_parses_after_stripping(self):
        """End of the chain: the rescued payload must validate."""
        from cognee.shared.data_models import SummarizedContent

        raw = '```json\n{"summary": "a summary", "description": ""}\n```'
        assert SummarizedContent.model_validate_json(self._strip(raw)).summary == "a summary"


# ---- Non-strict demotion + native validation fallback (COG-6271) ----


def _schema_400() -> Exception:
    from litellm.exceptions import BadRequestError

    return BadRequestError(
        message="Invalid schema for response_format 'PersonModel': 'oneOf' is not permitted.",
        model="gpt-5-mini",
        llm_provider="openai",
    )


@pytest.fixture
def _clean_demotions():
    from cognee.infrastructure.llm.structured_output_framework.litellm_native import (
        native_adapter,
    )

    native_adapter.clear_nonstrict_demotions()
    yield
    native_adapter.clear_nonstrict_demotions()


def _schema_adapter():
    from cognee.infrastructure.llm.structured_output_framework.litellm_native.native_adapter import (
        NativeLiteLLMAdapter,
    )

    return NativeLiteLLMAdapter(
        api_key="test-key",
        model="openai/gpt-5-mini",  # supports_response_schema is True
        max_completion_tokens=4096,
    )


@pytest.mark.asyncio
async def test_schema_400_demotes_to_nonstrict_and_is_remembered(_clean_demotions):
    """A strict rejection retries once non-strict; later calls skip the strict attempt."""
    adapter = _schema_adapter()
    valid = json.dumps({"name": "Eve", "age": 22})

    mock_acompletion = AsyncMock(
        side_effect=[_schema_400(), _make_mock_response(valid), _make_mock_response(valid)]
    )
    with patch("litellm.acompletion", mock_acompletion):
        first = await adapter.acreate_structured_output(
            text_input="Tell me about Eve",
            system_prompt="Extract person info.",
            response_model=PersonModel,
        )
        second = await adapter.acreate_structured_output(
            text_input="Tell me about Eve again",
            system_prompt="Extract person info.",
            response_model=PersonModel,
        )

    assert first.name == "Eve" and second.name == "Eve"
    # Call 1: strict (Pydantic class). Call 2: non-strict retry of the same
    # request. Call 3: the next request goes straight to non-strict — the
    # failed strict request is paid once per process, not per call.
    assert mock_acompletion.call_count == 3
    calls = mock_acompletion.call_args_list
    assert calls[0].kwargs["response_format"] is PersonModel
    for call in calls[1:]:
        response_format = call.kwargs["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["strict"] is False
        assert response_format["json_schema"]["name"] == "PersonModel"


@pytest.mark.asyncio
async def test_nonstrict_rejection_falls_back_to_prompted_json(_clean_demotions):
    """If the non-strict retry is also rejected, the prompted-JSON path still answers."""
    adapter = _schema_adapter()
    valid = json.dumps({"name": "Eve", "age": 22})

    mock_acompletion = AsyncMock(
        side_effect=[_schema_400(), _schema_400(), _make_mock_response(valid)]
    )
    with patch("litellm.acompletion", mock_acompletion):
        result = await adapter.acreate_structured_output(
            text_input="Tell me about Eve",
            system_prompt="Extract person info.",
            response_model=PersonModel,
        )

    assert result.name == "Eve"
    assert mock_acompletion.call_count == 3
    assert mock_acompletion.call_args_list[2].kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_native_validation_error_routes_to_json_fallback(_clean_demotions):
    """Invalid native output goes to the self-correcting fallback, not blind tenacity retries."""
    adapter = _schema_adapter()
    valid = json.dumps({"name": "Eve", "age": 22})

    mock_acompletion = AsyncMock(
        side_effect=[_make_mock_response("not json at all"), _make_mock_response(valid)]
    )
    with patch("litellm.acompletion", mock_acompletion):
        result = await adapter.acreate_structured_output(
            text_input="Tell me about Eve",
            system_prompt="Extract person info.",
            response_model=PersonModel,
        )

    assert result.name == "Eve"
    # Exactly two calls: the failed native attempt, then the prompted-JSON
    # fallback — NOT a tenacity re-send of the native request.
    assert mock_acompletion.call_count == 2
    assert mock_acompletion.call_args_list[1].kwargs["response_format"] == {"type": "json_object"}
