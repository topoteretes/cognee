"""
Unit tests for translate_content task
"""

import os
from uuid import uuid4

import pytest

from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types import TextDocument
from cognee.tasks.translation import translate_content
from cognee.tasks.translation.models import TranslatedContent, LanguageMetadata


def has_llm_api_key():
    """Check if LLM API key is available"""
    return bool(os.environ.get("LLM_API_KEY"))


def create_test_chunk(text: str, chunk_index: int = 0):
    """Helper to create a DocumentChunk with all required fields"""
    # Create a minimal Document for the is_part_of field
    doc = TextDocument(
        id=uuid4(),
        name="test_doc",
        raw_data_location="/tmp/test.txt",
        external_metadata=None,
        mime_type="text/plain",
    )

    return DocumentChunk(
        id=uuid4(),
        text=text,
        chunk_index=chunk_index,
        chunk_size=len(text),
        cut_type="sentence",
        is_part_of=doc,
    )


@pytest.mark.asyncio
@pytest.mark.skipif(not has_llm_api_key(), reason="No LLM API key available")
async def test_translate_content_basic():
    """Test basic content translation"""
    # Create test chunk with Spanish text
    original_text = "Hola mundo, esta es una prueba."
    chunk = create_test_chunk(original_text)

    result = await translate_content(
        data_chunks=[chunk], target_language="en", translation_provider="llm"
    )

    assert len(result) == 1
    # The chunk's text should now be translated (different from original Spanish)
    assert result[0].text != original_text  # Text should be translated to English
    assert result[0].contains is not None

    # Check for TranslatedContent in contains
    has_translated_content = any(isinstance(item, TranslatedContent) for item in result[0].contains)
    assert has_translated_content


@pytest.mark.asyncio
@pytest.mark.skipif(not has_llm_api_key(), reason="No LLM API key available")
async def test_translate_content_preserves_original():
    """Test that original text is preserved"""
    original_text = "Bonjour le monde"
    chunk = create_test_chunk(original_text)

    result = await translate_content(
        data_chunks=[chunk], target_language="en", preserve_original=True
    )

    # Find TranslatedContent in contains
    translated_content = None
    for item in result[0].contains:
        if isinstance(item, TranslatedContent):
            translated_content = item
            break

    assert translated_content is not None
    assert translated_content.original_text == original_text
    assert translated_content.translated_text != original_text


@pytest.mark.asyncio
async def test_translate_content_skip_english():
    """Test skipping translation for English text"""
    # This test doesn't require API call since English text is skipped
    chunk = create_test_chunk("Hello world, this is a test.")

    result = await translate_content(
        data_chunks=[chunk], target_language="en", skip_if_target_language=True
    )

    # Text should remain unchanged
    assert result[0].text == chunk.text

    # Should have LanguageMetadata but not TranslatedContent
    has_language_metadata = any(
        isinstance(item, LanguageMetadata) for item in (result[0].contains or [])
    )
    has_translated_content = any(
        isinstance(item, TranslatedContent) for item in (result[0].contains or [])
    )

    assert has_language_metadata
    assert not has_translated_content


@pytest.mark.asyncio
@pytest.mark.skipif(not has_llm_api_key(), reason="No LLM API key available")
async def test_translate_content_multiple_chunks():
    """Test translation of multiple chunks"""
    # Use longer texts to ensure reliable language detection
    original_texts = [
        "Hola mundo, esta es una prueba de traducción.",
        "Bonjour le monde, ceci est un test de traduction.",
        "Ciao mondo, questo è un test di traduzione.",
    ]
    chunks = [create_test_chunk(text, i) for i, text in enumerate(original_texts)]

    result = await translate_content(data_chunks=chunks, target_language="en")

    assert len(result) == 3
    # Check that at least some chunks were translated
    translated_count = sum(
        1
        for chunk in result
        if any(isinstance(item, TranslatedContent) for item in (chunk.contains or []))
    )
    assert translated_count >= 2  # At least 2 chunks should be translated


@pytest.mark.asyncio
async def test_translate_content_empty_list():
    """Test with empty chunk list"""
    result = await translate_content(data_chunks=[], target_language="en")

    assert result == []


@pytest.mark.asyncio
async def test_translate_content_empty_text():
    """Test with chunk containing empty text"""
    chunk = create_test_chunk("")

    result = await translate_content(data_chunks=[chunk], target_language="en")

    assert len(result) == 1
    assert result[0].text == ""


@pytest.mark.asyncio
@pytest.mark.skipif(not has_llm_api_key(), reason="No LLM API key available")
async def test_translate_content_language_metadata():
    """Test that LanguageMetadata is created correctly"""
    # Use a longer, distinctly Spanish text to ensure reliable detection
    chunk = create_test_chunk(
        "La inteligencia artificial está cambiando el mundo de manera significativa"
    )

    result = await translate_content(data_chunks=[chunk], target_language="en")

    # Find LanguageMetadata
    language_metadata = None
    for item in result[0].contains:
        if isinstance(item, LanguageMetadata):
            language_metadata = item
            break

    assert language_metadata is not None
    # Just check that a language was detected (short texts can be ambiguous)
    assert language_metadata.detected_language is not None
    assert language_metadata.requires_translation is True
    assert language_metadata.language_confidence > 0.0


@pytest.mark.asyncio
@pytest.mark.skipif(not has_llm_api_key(), reason="No LLM API key available")
async def test_translate_content_confidence_threshold():
    """Test with custom confidence threshold"""
    # Use longer text for more reliable detection
    chunk = create_test_chunk("Hola mundo, esta es una frase más larga para mejor detección")

    result = await translate_content(
        data_chunks=[chunk], target_language="en", confidence_threshold=0.5
    )

    assert len(result) == 1


@pytest.mark.asyncio
@pytest.mark.skipif(not has_llm_api_key(), reason="No LLM API key available")
async def test_translate_content_no_preserve_original():
    """Test translation without preserving original"""
    # Use longer text for more reliable detection
    chunk = create_test_chunk("Bonjour le monde, comment allez-vous aujourd'hui")

    result = await translate_content(
        data_chunks=[chunk], target_language="en", preserve_original=False
    )

    # Find TranslatedContent
    translated_content = None
    for item in result[0].contains:
        if isinstance(item, TranslatedContent):
            translated_content = item
            break

    assert translated_content is not None
    assert translated_content.original_text == ""  # Should be empty
