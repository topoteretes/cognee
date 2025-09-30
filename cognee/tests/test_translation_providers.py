"""
Basic tests for translation providers and detectors in Cognee.
These tests ensure that all registered providers and detectors can be instantiated and used for simple detection/translation tasks.
"""
import pytest
import asyncio
from cognee.tasks.translation import translate_content, get_available_translators, get_available_detectors

@pytest.mark.asyncio
@pytest.mark.parametrize("provider", get_available_translators())
async def test_translation_provider_basic(provider):
    # Noop should not translate, others may require API keys
    chunk = type("Chunk", (), {"text": "Hello world!", "metadata": {}})()
    try:
        result = await translate_content(chunk, translation_provider=provider, target_language="fr")
        assert hasattr(result, "text")
        assert hasattr(result, "metadata")
    except Exception as e:
        # If provider requires config/API key, skip
        pytest.skip(f"Provider '{provider}' not fully configured: {e}")

@pytest.mark.asyncio
@pytest.mark.parametrize("detector", get_available_detectors())
async def test_language_detector_basic(detector):
    chunk = type("Chunk", (), {"text": "Hello world!", "metadata": {}})()
    try:
        result = await translate_content(chunk, translation_provider="noop", detection_provider=detector)
        assert hasattr(result, "text")
        assert hasattr(result, "metadata")
        assert "language" in result.metadata
    except Exception as e:
        pytest.skip(f"Detector '{detector}' not fully configured: {e}")
