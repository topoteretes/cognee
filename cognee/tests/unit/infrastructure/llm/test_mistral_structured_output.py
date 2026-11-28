"""MistralAdapter.acreate_structured_output returns the model instructor parsed.

With ``response_model`` set, instructor validates the answer and returns an instance of
that model, not the raw completion. The adapter used to read ``.choices`` off that
instance, so every structured call on this path raised ``AttributeError``.
"""

import importlib
import sys
from types import ModuleType

import instructor
import litellm
import pytest
from pydantic import BaseModel

ADAPTER_MODULE = (
    "cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.mistral.adapter"
)


class Person(BaseModel):
    name: str


def _import_adapter_module(monkeypatch):
    # The mistral extra is not installed in every test environment, and the adapter
    # only needs the Mistral client class at import time.
    try:
        importlib.import_module("mistralai")
    except ImportError:
        stub = ModuleType("mistralai")
        stub.Mistral = type("Mistral", (), {"__init__": lambda self, **kwargs: None})
        monkeypatch.setitem(sys.modules, "mistralai", stub)
        monkeypatch.delitem(sys.modules, ADAPTER_MODULE, raising=False)
    return importlib.import_module(ADAPTER_MODULE)


@pytest.mark.asyncio
async def test_structured_output_returns_the_parsed_model(monkeypatch):
    adapter_module = _import_adapter_module(monkeypatch)
    adapter_class = adapter_module.MistralAdapter

    async def fake_acompletion(**kwargs):
        # What a Mistral tool call comes back as through litellm.
        return litellm.ModelResponse(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "Person", "arguments": '{"name": "Ada"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        )

    adapter = object.__new__(adapter_class)
    adapter.model = "mistral/mistral-large-latest"
    adapter.llm_args = {}
    adapter.endpoint = None
    adapter.aclient = instructor.from_litellm(
        fake_acompletion, mode=instructor.Mode(adapter_class.default_instructor_mode)
    )

    result = await adapter.acreate_structured_output(
        text_input="Ada wrote the first program.",
        system_prompt="Extract the person.",
        response_model=Person,
    )

    # instructor answers with its own subclass of the response model.
    assert isinstance(result, Person)
    assert result.name == "Ada"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint, llm_args, expected_api_base",
    [
        ("https://proxy.example.test/v1", {}, "https://proxy.example.test/v1"),
        # An api_base set through LLM_ARGS keeps taking precedence.
        (
            "https://proxy.example.test/v1",
            {"api_base": "https://args.example.test/v1"},
            "https://args.example.test/v1",
        ),
        # A blank LLM_ENDPOINT leaves LiteLLM's default Mistral URL in place.
        ("", {}, None),
    ],
)
async def test_structured_output_sends_configured_endpoint(
    monkeypatch, endpoint, llm_args, expected_api_base
):
    adapter_module = _import_adapter_module(monkeypatch)
    calls = []

    async def fake_create(**kwargs):
        calls.append(kwargs)
        return Person(name="Ada")

    adapter = adapter_module.MistralAdapter(
        api_key="test-key",
        model="mistral/mistral-large-latest",
        max_completion_tokens=100,
        endpoint=endpoint,
        llm_args=llm_args,
    )
    monkeypatch.setattr(adapter.aclient.chat.completions, "create", fake_create)

    await adapter.acreate_structured_output(
        text_input="Ada wrote the first program.",
        system_prompt="Extract the person.",
        response_model=Person,
    )

    assert calls[0].get("api_base") == expected_api_base
