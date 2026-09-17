"""IMAGE_TRANSCRIBE_MODEL on the legacy ``litellm_instructor`` path.

Every provider whose adapter can actually transcribe an image must route images
at the configured vision model and leave the chat model alone. Adapters that
raise ``NotImplementedError`` for images (bedrock, llama_cpp, mcp_sampling) have
nothing to route and are deliberately absent.

The failure this guards against is silent: an unwired adapter falls back to the
chat model, so a misconfigured deployment sends images to a text-only model
instead of reporting that the setting did nothing.
"""

from unittest.mock import patch

import pytest

import cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client as factory
from cognee.infrastructure.llm.config import LLMConfig

BASE_MODEL = "chat-model-placeholder"

# provider, chat model, IMAGE_TRANSCRIBE_MODEL, expected adapter attribute,
# and the optional SDK the adapter's constructor needs (None when always present).
PROVIDERS = [
    ("openai", "gpt-4o-mini", "gpt-4o", "gpt-4o", None),
    ("custom", "gpt-4o-mini", "gpt-4o", "gpt-4o", None),
    ("ollama", "llama3.1:8b", "llava", "llava", None),
    (
        "gemini",
        "gemini/gemini-2.0-flash",
        "gemini/gemini-2.5-pro",
        "gemini/gemini-2.5-pro",
        None,
    ),
    (
        "mistral",
        "mistral/mistral-small",
        "mistral/pixtral-12b-2409",
        "mistral/pixtral-12b-2409",
        "mistralai",
    ),
    (
        "anthropic",
        "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20241022",
        "anthropic",
    ),
]


def _client(provider: str, model: str, image_model: str):
    """Build the legacy adapter the factory would hand back for this config."""
    config = LLMConfig(
        llm_provider=provider,
        llm_model=model,
        llm_api_key="test-key",
        llm_endpoint="http://local.test/v1",
        image_transcribe_model=image_model,
    )
    factory._get_llm_client_cached.cache_clear()
    with patch.object(factory, "get_llm_context_config", return_value=config):
        return factory.get_llm_client()


@pytest.mark.parametrize(
    ("provider", "model", "image_model", "expected", "requires"),
    PROVIDERS,
    ids=[row[0] for row in PROVIDERS],
)
def test_image_transcribe_model_reaches_every_legacy_adapter(
    provider, model, image_model, expected, requires
):
    if requires:
        pytest.importorskip(requires, reason=f"{provider} adapter requires {requires}")

    adapter = _client(provider, model, image_model)

    assert adapter.image_transcribe_model == expected
    # The chat model must be untouched — this setting only redirects images.
    assert adapter.model != adapter.image_transcribe_model


@pytest.mark.parametrize(
    ("provider", "model", "image_model", "expected", "requires"),
    PROVIDERS,
    ids=[row[0] for row in PROVIDERS],
)
def test_unset_image_model_falls_back_to_the_chat_model(
    provider, model, image_model, expected, requires
):
    """Leaving IMAGE_TRANSCRIBE_MODEL unset must behave exactly as before."""
    if requires:
        pytest.importorskip(requires, reason=f"{provider} adapter requires {requires}")

    adapter = _client(provider, model, "")

    assert adapter.image_transcribe_model == adapter.model


def test_ollama_strips_the_provider_prefix_from_the_image_model():
    """``self.model`` drops an ``ollama/`` prefix, so the image model must too —
    otherwise the Ollama API is asked for a model name it does not know."""
    adapter = _client("ollama", "ollama/llama3.1:8b", "ollama/llava")

    assert adapter.image_transcribe_model == "llava"
