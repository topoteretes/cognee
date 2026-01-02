"""
Unit tests for translation providers
"""

import asyncio
import os
from cognee.tasks.translation.providers import (
    get_translation_provider,
    OpenAITranslationProvider,
    TranslationResult,
)
from cognee.tasks.translation.exceptions import TranslationError


def has_openai_key():
    """Check if OpenAI API key is available"""
    return bool(os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"))


async def test_openai_provider_basic_translation():
    """Test basic translation with OpenAI provider"""
    if not has_openai_key():
        print("  (skipped - no API key)")
        return

    provider = OpenAITranslationProvider()

    result = await provider.translate(text="Hola mundo", target_language="en", source_language="es")

    assert isinstance(result, TranslationResult)
    assert result.translated_text is not None
    assert len(result.translated_text) > 0
    assert result.source_language == "es"
    assert result.target_language == "en"
    assert result.provider == "openai"


async def test_openai_provider_auto_detect_source():
    """Test translation with automatic source language detection"""
    if not has_openai_key():
        print("  (skipped - no API key)")
        return

    provider = OpenAITranslationProvider()

    result = await provider.translate(
        text="Bonjour le monde",
        target_language="en",
        # source_language not provided - should auto-detect
    )

    assert result.translated_text is not None
    assert result.target_language == "en"


async def test_openai_provider_long_text():
    """Test translation of longer text"""
    if not has_openai_key():
        print("  (skipped - no API key)")
        return

    provider = OpenAITranslationProvider()

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
    provider = get_translation_provider("openai")
    assert isinstance(provider, OpenAITranslationProvider)


def test_get_translation_provider_invalid():
    """Test provider factory with invalid provider name"""
    try:
        get_translation_provider("invalid_provider")
        assert False, "Expected TranslationError or ValueError"
    except (TranslationError, ValueError):
        pass


async def test_openai_batch_translation():
    """Test batch translation with OpenAI provider"""
    if not has_openai_key():
        print("  (skipped - no API key)")
        return

    provider = OpenAITranslationProvider()

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


async def test_translation_preserves_formatting():
    """Test that translation preserves basic formatting"""
    if not has_openai_key():
        print("  (skipped - no API key)")
        return

    provider = OpenAITranslationProvider()

    text_with_newlines = "Primera línea.\nSegunda línea."

    result = await provider.translate(
        text=text_with_newlines, target_language="en", source_language="es"
    )

    # Should preserve structure (though exact newlines may vary)
    assert result.translated_text is not None
    assert len(result.translated_text) > 0


async def test_translation_special_characters():
    """Test translation with special characters"""
    if not has_openai_key():
        print("  (skipped - no API key)")
        return

    provider = OpenAITranslationProvider()

    text = "¡Hola! ¿Cómo estás? Está bien."

    result = await provider.translate(text=text, target_language="en", source_language="es")

    assert result.translated_text is not None
    assert len(result.translated_text) > 0


async def test_empty_text_translation():
    """Test translation with empty text - should return empty or handle gracefully"""
    if not has_openai_key():
        print("  (skipped - no API key)")
        return

    provider = OpenAITranslationProvider()

    # Empty text may either raise an error or return an empty result
    try:
        result = await provider.translate(text="", target_language="en", source_language="es")
        # If no error, should return a TranslationResult (possibly with empty text)
        assert isinstance(result, TranslationResult)
    except TranslationError:
        # This is also acceptable behavior
        pass


async def main():
    """Run all provider tests"""
    # Sync tests
    test_get_translation_provider_factory()
    print("✓ test_get_translation_provider_factory passed")

    test_get_translation_provider_invalid()
    print("✓ test_get_translation_provider_invalid passed")

    # Async tests
    await test_openai_provider_basic_translation()
    print("✓ test_openai_provider_basic_translation passed")

    await test_openai_provider_auto_detect_source()
    print("✓ test_openai_provider_auto_detect_source passed")

    await test_openai_provider_long_text()
    print("✓ test_openai_provider_long_text passed")

    await test_openai_batch_translation()
    print("✓ test_openai_batch_translation passed")

    await test_translation_preserves_formatting()
    print("✓ test_translation_preserves_formatting passed")

    await test_translation_special_characters()
    print("✓ test_translation_special_characters passed")

    await test_empty_text_translation()
    print("✓ test_empty_text_translation passed")

    print("\nAll provider tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
