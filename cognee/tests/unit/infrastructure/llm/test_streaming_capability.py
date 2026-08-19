"""Which deployments can stream an answer.

The gateway picks an adapter from STRUCTURED_OUTPUT_FRAMEWORK *and*
LLM_PROVIDER, so "does streaming work here" is a property of the pair, not of
any one class. This pins the real dispatch: the earlier regression was streaming
implemented in an adapter no default deployment ever reaches, which no test
noticed because every test constructed that adapter directly.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.bedrock.adapter import (
    BedrockAdapter,
)
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api.adapter import (
    GenericAPIAdapter,
)
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.llama_cpp.adapter import (
    LlamaCppAPIAdapter,
)
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.ollama.adapter import (
    OllamaAPIAdapter,
)
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.openai.adapter import (
    OpenAIAdapter,
)
from cognee.infrastructure.llm.structured_output_framework.litellm_native.native_adapter import (
    NativeLiteLLMAdapter,
)

GATEWAY = "cognee.infrastructure.llm.LLMGateway"


@pytest.mark.parametrize(
    "adapter_class",
    [
        # Reaches the shared streaming path directly...
        GenericAPIAdapter,
        # ...and these inherit it, which is why they are not listed separately:
        # OpenAIAdapter -> GenericAPIAdapter, AzureOpenAIAdapter -> OpenAIAdapter,
        # and the Anthropic / Gemini / Mistral adapters likewise.
        OpenAIAdapter,
        # The default framework's adapter.
        NativeLiteLLMAdapter,
    ],
)
def test_streaming_adapters_declare_support(adapter_class):
    assert adapter_class.supports_answer_streaming is True


@pytest.mark.parametrize("adapter_class", [BedrockAdapter, OllamaAPIAdapter, LlamaCppAPIAdapter])
def test_non_streaming_adapters_declare_no_support(adapter_class):
    """These subclass LLMInterface directly and never route a plain-text answer
    through stream_text_completion. Declaring False is what stops a request
    announcing a stream they cannot produce."""
    assert getattr(adapter_class, "supports_answer_streaming", False) is False


def _config(framework: str):
    return patch(
        f"{GATEWAY}.get_llm_config",
        return_value=SimpleNamespace(structured_output_framework=framework),
    )


def test_the_default_framework_resolves_through_the_native_client():
    """litellm_native is the default, so this is the path an out-of-the-box
    install takes — the one the streaming hook originally missed entirely."""
    with (
        _config("litellm_native"),
        patch(
            "cognee.infrastructure.llm.structured_output_framework.litellm_native"
            ".get_native_client.get_native_client",
            return_value=SimpleNamespace(supports_answer_streaming=True),
        ),
    ):
        assert LLMGateway.supports_answer_streaming() is True


def test_a_non_streaming_client_is_reported_as_such():
    with (
        _config("instructor"),
        patch(
            "cognee.infrastructure.llm.structured_output_framework.litellm_instructor"
            ".llm.get_llm_client.get_llm_client",
            return_value=SimpleNamespace(supports_answer_streaming=False),
        ),
    ):
        assert LLMGateway.supports_answer_streaming() is False


def test_baml_never_streams():
    """The BAML branch bypasses the adapters entirely, so no adapter flag can
    describe it."""
    with _config("baml"):
        assert LLMGateway.supports_answer_streaming() is False


def test_a_client_that_cannot_be_built_does_not_stream():
    """A capability probe must not turn a misconfiguration into a failed
    request; not streaming is the safe direction."""
    with (
        _config("instructor"),
        patch(
            "cognee.infrastructure.llm.structured_output_framework.litellm_instructor"
            ".llm.get_llm_client.get_llm_client",
            side_effect=RuntimeError("no api key"),
        ),
    ):
        assert LLMGateway.supports_answer_streaming() is False
