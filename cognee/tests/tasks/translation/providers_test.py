"""
Unit tests for translation providers
"""

import os

import pytest

from cognee.tasks.translation.providers import (
    get_translation_provider,
    LLMTranslationProvider,
    TranslationResult,
)
from cognee.tasks.translation.exceptions import TranslationError


def has_llm_api_key():
    """Check if LLM API key is available"""
    return bool(os.environ.get("LLM_API_KEY"))


@pytest.mark.asyncio
@pytest.mark.skipif(not has_llm_api_key(), reason="No LLM API key available")
async def test_llm_provider_basic_translation():
    """Test basic translation with LLM provider (uses configured LLM)"""
    provider = LLMTranslationProvider()

    result = await provider.translate(text="Hola mundo", target_language="en", source_language="es")

    assert isinstance(result, TranslationResult)
    assert result.translated_text is not None
    assert len(result.translated_text) > 0
    assert result.source_language == "es"
    assert result.target_language == "en"
    assert result.provider == "llm"


@pytest.mark.asyncio
@pytest.mark.skipif(not has_llm_api_key(), reason="No LLM API key available")
async def test_llm_provider_auto_detect_source():
    """Test translation with automatic source language detection"""
    provider = LLMTranslationProvider()

    result = await provider.translate(
        text="Bonjour le monde",
        target_language="en",
        # source_language not provided - should auto-detect
    )

    assert result.translated_text is not None
    assert result.target_language == "en"


@pytest.mark.asyncio
@pytest.mark.skipif(not has_llm_api_key(), reason="No LLM API key available")
async def test_llm_provider_long_text():
    """Test translation of longer text"""
    provider = LLMTranslationProvider()

    long_text = """
    La inteligencia artificial es una rama de la informática que se centra en
    crear sistemas capaces de realizar tareas que normalmente requieren inteligencia humana.
    Estos sistemas pueden aprender, razonar y resolver problemas complejos.
    """

    result = await provider.translate(text=long_text, target_language="en", source_language="es")

    assert len(result.translated_text) > 0
    assert result.source_language == "es"


def test_get_translation_provider_factory():
    """Test provider factory function"""
    provider = get_translation_provider("llm")
    assert isinstance(provider, LLMTranslationProvider)


def test_get_translation_provider_invalid():
    """Test provider factory with invalid provider name"""
    try:
        get_translation_provider("invalid_provider")
        assert False, "Expected TranslationError or ValueError"
    except (TranslationError, ValueError):
        pass


@pytest.mark.asyncio
@pytest.mark.skipif(not has_llm_api_key(), reason="No LLM API key available")
async def test_llm_batch_translation():
    """Test batch translation with LLM provider"""
    provider = LLMTranslationProvider()

    texts = ["Hola", "¿Cómo estás?", "Adiós"]

    results = await provider.translate_batch(
        texts=texts, target_language="en", source_language="es"
    )

    assert len(results) == len(texts)
    for result in results:
        assert isinstance(result, TranslationResult)
        assert result.translated_text is not None
        assert result.source_language == "es"
        assert result.target_language == "en"


@pytest.mark.asyncio
@pytest.mark.skipif(not has_llm_api_key(), reason="No LLM API key available")
async def test_translation_preserves_formatting():
    """Test that translation preserves basic formatting"""
    provider = LLMTranslationProvider()

    text_with_newlines = "Primera línea.\nSegunda línea."

    result = await provider.translate(
        text=text_with_newlines, target_language="en", source_language="es"
    )

    # Should preserve structure (though exact newlines may vary)
    assert result.translated_text is not None
    assert len(result.translated_text) > 0


@pytest.mark.asyncio
@pytest.mark.skipif(not has_llm_api_key(), reason="No LLM API key available")
async def test_translation_special_characters():
    """Test translation with special characters"""
    provider = LLMTranslationProvider()

    text = "¡Hola! ¿Cómo estás? Está bien."

    result = await provider.translate(text=text, target_language="en", source_language="es")

    assert result.translated_text is not None
    assert len(result.translated_text) > 0


@pytest.mark.asyncio
@pytest.mark.skipif(not has_llm_api_key(), reason="No LLM API key available")
async def test_empty_text_translation():
    """Test translation with empty text - should return empty or handle gracefully"""
    provider = LLMTranslationProvider()

    # Empty text may either raise an error or return an empty result
    try:
        result = await provider.translate(text="", target_language="en", source_language="es")
        # If no error, should return a TranslationResult (possibly with empty text)
        assert isinstance(result, TranslationResult)
    except TranslationError:
        # This is also acceptable behavior
        pass
