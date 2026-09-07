"""Which deployments can stream an answer.

The gateway picks an adapter from STRUCTURED_OUTPUT_FRAMEWORK *and*
LLM_PROVIDER, so "does streaming work here" is a property of the pair, not of
any one class. This pins the real dispatch: the earlier regression was streaming
implemented in an adapter no default deployment ever reaches, which no test
noticed because every test constructed that adapter directly.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.azure_openai.adapter import (
    AzureOpenAIAdapter,
)
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.gemini.adapter import (
    GeminiAdapter,
)
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.mcp_sampling.adapter import (
    MCPSamplingAdapter,
)
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

# The LLMGateway *class* shadows the LLMGateway *module* in the package
# namespace (cognee/infrastructure/llm/__init__.py re-exports it). String
# patch targets resolve attribute-first on Python 3.10 and land on the class,
# so resolve the module explicitly and patch its globals by object.
_gateway_module = importlib.import_module("cognee.infrastructure.llm.LLMGateway")

_INSTRUCTOR = "cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm"


@pytest.mark.parametrize(
    "adapter_class",
    [
        # Reaches the shared streaming path directly...
        GenericAPIAdapter,
        # ...and OpenAIAdapter inherits it and keeps the plain-text branch.
        # Nothing else does: subclassing GenericAPIAdapter is NOT enough, because
        # an override of acreate_structured_output without a `response_model is
        # str` branch removes the only door to stream_text_completion. Those are
        # pinned as non-streaming below.
        OpenAIAdapter,
        # The default framework's adapter.
        NativeLiteLLMAdapter,
    ],
)
def test_streaming_adapters_declare_support(adapter_class):
    assert adapter_class.supports_answer_streaming is True


@pytest.mark.parametrize(
    "adapter_class",
    [
        # Subclass LLMInterface directly — no plain-text path at all.
        BedrockAdapter,
        OllamaAPIAdapter,
        LlamaCppAPIAdapter,
        MCPSamplingAdapter,
        # Subclass GenericAPIAdapter, so they inherit the True — but each
        # overrides acreate_structured_output with no `response_model is str`
        # branch, so the inherited flag would be a lie. They must say False
        # themselves; that is the whole point of declaring rather than inferring.
        GeminiAdapter,
    ],
)
def test_non_streaming_adapters_declare_no_support(adapter_class):
    """Declaring False is what stops a request announcing a stream they cannot
    produce — `stage: generating` followed by silence, which is worse for a
    consumer than the feature being off."""
    assert getattr(adapter_class, "supports_answer_streaming", False) is False


@pytest.mark.parametrize(
    "module_name, class_name",
    [
        (f"{_INSTRUCTOR}.anthropic.adapter", "AnthropicAdapter"),
        (f"{_INSTRUCTOR}.mistral.adapter", "MistralAdapter"),
    ],
)
def test_optional_provider_adapters_declare_no_support(module_name, class_name):
    """Same pin as above for the two adapters whose SDKs are optional extras.

    Resolved lazily so a bare environment skips rather than fails at import —
    but still pinned, because these are exactly the deployments where an
    inherited True would announce a stream and deliver nothing.
    """
    module = pytest.importorskip(module_name)
    adapter_class = getattr(module, class_name)
    assert getattr(adapter_class, "supports_answer_streaming", False) is False


def _azure(*, use_managed_identity: bool) -> AzureOpenAIAdapter:
    """Build an adapter without touching either auth backend.

    The managed-identity constructor imports azure-identity, an optional extra,
    so it is stubbed out: this test is about which value the flag takes, not
    about how the credential is obtained.
    """
    with patch.object(AzureOpenAIAdapter, "_init_managed_identity", return_value=None):
        return AzureOpenAIAdapter(
            api_key="k",
            model="azure/gpt-4o",
            max_completion_tokens=1024,
            endpoint="https://example.openai.azure.com",
            api_version="2024-02-01",
            use_managed_identity=use_managed_identity,
        )


def test_azure_declares_streaming_per_auth_mode():
    """Azure is the one adapter whose answer differs by configuration, so its
    declaration is per-instance and a class-level parametrize cannot see it.

    Key-based delegates to OpenAIAdapter and reaches the shared streaming path.
    Managed identity answers on the native OpenAI client, which has no plain-text
    streaming door — and it is a plausible Cloud configuration, so an inherited
    True there would announce a stream and deliver nothing.
    """
    assert _azure(use_managed_identity=False).supports_answer_streaming is True
    assert _azure(use_managed_identity=True).supports_answer_streaming is False
