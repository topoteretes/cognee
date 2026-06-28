from contextlib import asynccontextmanager
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock
import sys

import pytest
from pydantic import BaseModel


MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "infrastructure/llm/structured_output_framework/litellm_instructor/llm/anthropic/adapter.py"
)


def _load_adapter_module():
    noop_logger = SimpleNamespace(debug=lambda *args, **kwargs: None)

    litellm_module = ModuleType("litellm")
    litellm_module.exceptions = SimpleNamespace(
        NotFoundError=type("NotFoundError", (Exception,), {}),
        AuthenticationError=type("AuthenticationError", (Exception,), {}),
    )

    instructor_module = ModuleType("instructor")
    instructor_module.Mode = lambda value: value
    instructor_module.patch = lambda **kwargs: kwargs

    anthropic_module = ModuleType("anthropic")
    anthropic_module.AsyncAnthropic = lambda api_key: SimpleNamespace(
        messages=SimpleNamespace(create=object())
    )

    generic_adapter_module = ModuleType(
        "cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api.adapter"
    )

    class GenericAPIAdapter:
        def __init__(self, api_key, model, max_completion_tokens, name, **kwargs):
            self.api_key = api_key
            self.model = model
            self.max_completion_tokens = max_completion_tokens
            self.name = name

    generic_adapter_module.GenericAPIAdapter = GenericAPIAdapter

    logging_utils_module = ModuleType("cognee.shared.logging_utils")
    logging_utils_module.get_logger = lambda: noop_logger

    observability_module = ModuleType("cognee.modules.observability.get_observe")
    observability_module.get_observe = lambda: (lambda **kwargs: (lambda fn: fn))

    @asynccontextmanager
    async def noop_rate_limiter():
        yield

    rate_limit_module = ModuleType("cognee.shared.rate_limiting")
    rate_limit_module.llm_rate_limiter_context_manager = noop_rate_limiter

    config_module = ModuleType("cognee.infrastructure.llm.config")
    config_module.get_llm_config = lambda: None

    originals = dict(sys.modules)
    sys.modules.update(
        {
            "litellm": litellm_module,
            "instructor": instructor_module,
            "anthropic": anthropic_module,
            "cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api.adapter": generic_adapter_module,
            "cognee.shared.logging_utils": logging_utils_module,
            "cognee.modules.observability.get_observe": observability_module,
            "cognee.shared.rate_limiting": rate_limit_module,
            "cognee.infrastructure.llm.config": config_module,
        }
    )

    try:
        spec = spec_from_file_location("anthropic_adapter_under_test", MODULE_PATH)
        module = module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.clear()
        sys.modules.update(originals)


class DummyResponse(BaseModel):
    answer: str


@pytest.mark.asyncio
async def test_anthropic_adapter_sends_cached_system_prompt(monkeypatch):
    module = _load_adapter_module()
    aclient = AsyncMock(return_value=DummyResponse(answer="ok"))
    monkeypatch.setattr(module.instructor, "patch", lambda **_: aclient)

    adapter = module.AnthropicAdapter(
        api_key="test-key",
        model="claude-sonnet-4-6",
        max_completion_tokens=4096,
    )

    response = await adapter.acreate_structured_output(
        text_input="Chunk text",
        system_prompt="System instructions",
        response_model=DummyResponse,
    )

    assert response.answer == "ok"
    aclient.assert_awaited_once()
    kwargs = aclient.await_args.kwargs
    assert kwargs["system"] == [
        {
            "type": "text",
            "text": "System instructions",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert kwargs["messages"] == [
        {
            "role": "user",
            "content": "Use the given format to extract information\n"
            "                    from the following input: Chunk text",
        }
    ]


def test_build_cached_system_prompt_omits_empty_prompt():
    module = _load_adapter_module()
    assert module._build_cached_system_prompt("") == []
