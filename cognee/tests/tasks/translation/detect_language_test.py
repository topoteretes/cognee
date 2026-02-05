"""
Unit tests for language detection functionality
"""

import pytest
from cognee.tasks.translation.detect_language import (
    detect_language_async,
    LanguageDetectionResult,
)
from cognee.tasks.translation.exceptions import LanguageDetectionError


@pytest.mark.asyncio
async def test_detect_english():
    """Test detection of English text"""
    result = await detect_language_async("Hello world, this is a test.", target_language="en")

    assert result.language_code == "en"
    assert result.requires_translation is False
    assert result.confidence > 0.9
    assert result.language_name == "English"


@pytest.mark.asyncio
async def test_detect_spanish():
    """Test detection of Spanish text"""
    result = await detect_language_async("Hola mundo, esta es una prueba.", target_language="en")

    assert result.language_code == "es"
    assert result.requires_translation is True
    assert result.confidence > 0.9
    assert result.language_name == "Spanish"


@pytest.mark.asyncio
async def test_detect_french():
    """Test detection of French text"""
    result = await detect_language_async(
        "Bonjour le monde, ceci est un test.", target_language="en"
    )

    assert result.language_code == "fr"
    assert result.requires_translation is True
    assert result.confidence > 0.9
    assert result.language_name == "French"


@pytest.mark.asyncio
async def test_detect_german():
    """Test detection of German text"""
    result = await detect_language_async("Hallo Welt, das ist ein Test.", target_language="en")

    assert result.language_code == "de"
    assert result.requires_translation is True
    assert result.confidence > 0.9


@pytest.mark.asyncio
async def test_detect_chinese():
    """Test detection of Chinese text"""
    result = await detect_language_async("你好世界，这是一个测试。", target_language="en")

    assert result.language_code.startswith("zh"), f"Expected Chinese, got {result.language_code}"
    assert result.requires_translation is True
    assert result.confidence > 0.9


@pytest.mark.asyncio
async def test_already_target_language():
    """Test when text is already in target language"""
    result = await detect_language_async("This text is already in English.", target_language="en")

    assert result.requires_translation is False


@pytest.mark.asyncio
async def test_short_text():
    """Test detection with very short text"""
    result = await detect_language_async("Hi", target_language="es")

    # Short text may return 'unknown' if langdetect can't reliably detect
    assert result.language_code in ["en", "unknown"]
    assert result.character_count == 2


@pytest.mark.asyncio
async def test_empty_text():
    """Test detection with empty text - returns unknown by default"""
    result = await detect_language_async("", target_language="en")

    # With skip_detection_for_short_text=True (default), returns unknown
    assert result.language_code == "unknown"
    assert result.language_name == "Unknown"
    assert result.confidence == 0.0
    assert result.requires_translation is False
    assert result.character_count == 0


@pytest.mark.asyncio
async def test_confidence_threshold():
    """Test detection respects confidence threshold"""
    result = await detect_language_async(
        "Hello world", target_language="es", confidence_threshold=0.5
    )

    assert result.confidence >= 0.5


@pytest.mark.asyncio
async def test_mixed_language_text():
    """Test detection with mixed language text (predominantly one language)"""
    # Predominantly Spanish with English word
    result = await detect_language_async(
        "La inteligencia artificial es muy importante en technology moderna.", target_language="en"
    )

    assert result.language_code == "es"  # Should detect as Spanish
    assert result.requires_translation is True
