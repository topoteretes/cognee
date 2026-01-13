"""
Integration tests for multilingual content translation feature.

Tests the translation module standalone functionality.
"""

import os

import pytest

from cognee.tasks.translation import translate_text
from cognee.tasks.translation.detect_language import detect_language_async


def has_llm_api_key():
    """Check if LLM API key is available"""
    return bool(os.environ.get("LLM_API_KEY"))


@pytest.mark.asyncio
@pytest.mark.skipif(not has_llm_api_key(), reason="No LLM API key available")
async def test_direct_translation_function():
    """Test the translate_text convenience function directly"""
    result = await translate_text(
        text="Hola, ¿cómo estás? Espero que tengas un buen día.",
        target_language="en",
        translation_provider="llm",
    )

    assert result.translated_text is not None
    assert result.translated_text != ""
    assert result.target_language == "en"
    assert result.provider == "llm"


@pytest.mark.asyncio
async def test_language_detection():
    """Test language detection directly"""
    test_texts = [
        ("Hello world, how are you doing today?", "en", False),
        ("Bonjour le monde, comment allez-vous aujourd'hui?", "en", True),
        ("Hola mundo, cómo estás hoy?", "en", True),
        ("This is already in English language", "en", False),
    ]

    for text, target_lang, should_translate in test_texts:
        result = await detect_language_async(text, target_lang)
        assert result.language_code is not None
        assert result.confidence > 0.0
        # Only check requires_translation for high-confidence detections
        if result.confidence > 0.8:
            assert result.requires_translation == should_translate
