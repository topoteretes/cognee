"""TRANSCRIPTION_MODEL and IMAGE_TRANSCRIBE_MODEL on the legacy ``instructor`` path.

Every provider whose adapter can transcribe audio or an image must route that
media at the configured model and leave the chat model alone. Adapters that
raise ``NotImplementedError`` for both (bedrock, llama_cpp, mcp_sampling) have
nothing to route and are deliberately absent.

The failure this guards against is silent: an unwired adapter falls back to the
chat model (or, for Ollama audio, a hardcoded name), so a misconfigured
deployment sends media to the wrong model instead of reporting that the setting
did nothing.
"""

from unittest.mock import patch

import pytest

import cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client as factory
from cognee.infrastructure.llm.config import LLMConfig

# provider, chat model, IMAGE_TRANSCRIBE_MODEL, and the optional SDK the
# adapter's constructor needs (None when always present).
PROVIDERS = [
    ("openai", "gpt-4o-mini", "gpt-4o", None),
    ("custom", "gpt-4o-mini", "gpt-4o", None),
    ("ollama", "llama3.1:8b", "llava", None),
    ("gemini", "gemini/gemini-2.0-flash", "gemini/gemini-2.5-pro", None),
    ("mistral", "mistral/mistral-small", "mistral/pixtral-12b-2409", "mistralai"),
    ("anthropic", "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "anthropic"),
]
IDS = [row[0] for row in PROVIDERS]


def _client(provider: str, model: str, **overrides):
    """Build the legacy adapter the factory would hand back for this config.

    Fields not overridden keep ``LLMConfig``'s defaults, so ``transcription_model``
    is ``whisper-1`` and ``image_transcribe_model`` is empty unless a test says
    otherwise.
    """
    config = LLMConfig(
        llm_provider=provider,
        llm_model=model,
        llm_api_key="test-key",
        llm_endpoint="http://local.test/v1",
        **overrides,
    )
    factory._get_llm_client_cached.cache_clear()
    with patch.object(factory, "get_llm_context_config", return_value=config):
        return factory.get_llm_client()


def _skip_without(requires, provider):
    if requires:
        pytest.importorskip(requires, reason=f"{provider} adapter requires {requires}")


# ---- images ----


@pytest.mark.parametrize(("provider", "model", "image_model", "requires"), PROVIDERS, ids=IDS)
def test_image_transcribe_model_reaches_every_legacy_adapter(
    provider, model, image_model, requires
):
    _skip_without(requires, provider)

    adapter = _client(provider, model, image_transcribe_model=image_model)

    assert adapter.image_transcribe_model == image_model
    # The chat model must be untouched — this setting only redirects images.
    assert adapter.model != adapter.image_transcribe_model


@pytest.mark.parametrize(("provider", "model", "image_model", "requires"), PROVIDERS, ids=IDS)
def test_unset_image_model_falls_back_to_the_chat_model(provider, model, image_model, requires):
    """Leaving IMAGE_TRANSCRIBE_MODEL unset must behave exactly as before."""
    _skip_without(requires, provider)

    adapter = _client(provider, model, image_transcribe_model="")

    assert adapter.image_transcribe_model == adapter.model


def test_ollama_strips_the_provider_prefix_from_the_image_model():
    """``self.model`` drops an ``ollama/`` prefix, so the image model must too —
    otherwise the Ollama API is asked for a model name it does not know."""
    adapter = _client("ollama", "ollama/llama3.1:8b", image_transcribe_model="ollama/llava")

    assert adapter.image_transcribe_model == "llava"


# ---- audio ----


@pytest.mark.parametrize(("provider", "model", "image_model", "requires"), PROVIDERS, ids=IDS)
def test_transcription_model_reaches_every_legacy_adapter(provider, model, image_model, requires):
    _skip_without(requires, provider)

    adapter = _client(provider, model, transcription_model="speech-model-under-test")

    assert adapter.transcription_model == "speech-model-under-test"
    assert adapter.model != adapter.transcription_model


@pytest.mark.parametrize(("provider", "model", "image_model", "requires"), PROVIDERS, ids=IDS)
def test_default_transcription_model_is_whisper_1_everywhere(
    provider, model, image_model, requires
):
    """The config default must reach every adapter, not only OpenAI and Azure."""
    _skip_without(requires, provider)

    adapter = _client(provider, model)

    assert adapter.transcription_model == "whisper-1"


def test_ollama_strips_the_provider_prefix_from_the_transcription_model():
    adapter = _client("ollama", "llama3.1:8b", transcription_model="ollama/whisper")

    assert adapter.transcription_model == "whisper"
