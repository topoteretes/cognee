"""
Integration tests for multilingual content translation feature.

Tests the full cognify pipeline with translation enabled.
"""

import os

import pytest

from cognee import add, cognify, prune, search, SearchType
from cognee.tasks.translation import translate_text
from cognee.tasks.translation.detect_language import detect_language_async


def has_openai_key():
    """Check if OpenAI API key is available"""
    return bool(os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"))


@pytest.mark.asyncio
@pytest.mark.skipif(not has_openai_key(), reason="No OpenAI API key available")
async def test_quick_translation():
    """Quick smoke test for translation feature"""
    await prune.prune_data()
    await prune.prune_system(metadata=True)

    spanish_text = "La inteligencia artificial está transformando el mundo de la tecnología."
    await add(spanish_text, dataset_name="spanish_test")

    result = await cognify(
        datasets=["spanish_test"],
        auto_translate=True,
        target_language="en",
        translation_provider="openai",
    )

    assert result is not None


@pytest.mark.asyncio
@pytest.mark.skipif(not has_openai_key(), reason="No OpenAI API key available")
async def test_translation_basic():
    """Test basic translation functionality with English text"""
    await prune.prune_data()
    await prune.prune_system(metadata=True)

    english_text = "Hello, this is a test document about artificial intelligence."
    await add(english_text, dataset_name="test_english")

    result = await cognify(
        datasets=["test_english"],
        auto_translate=True,
        target_language="en",
        translation_provider="openai",
    )

    assert result is not None

    search_results = await search(
        query_text="What is this document about?",
        query_type=SearchType.SUMMARIES,
    )
    assert search_results is not None


@pytest.mark.asyncio
@pytest.mark.skipif(not has_openai_key(), reason="No OpenAI API key available")
async def test_translation_spanish():
    """Test translation with Spanish text"""
    await prune.prune_data()
    await prune.prune_system(metadata=True)

    spanish_text = """
    La inteligencia artificial es una rama de la informática que se centra en 
    crear sistemas capaces de realizar tareas que normalmente requieren inteligencia humana.
    Estos sistemas pueden aprender, razonar y resolver problemas complejos.
    """

    await add(spanish_text, dataset_name="test_spanish")

    result = await cognify(
        datasets=["test_spanish"],
        auto_translate=True,
        target_language="en",
        translation_provider="openai",
    )

    assert result is not None

    search_results = await search(
        query_text="What is artificial intelligence?",
        query_type=SearchType.SUMMARIES,
    )
    assert search_results is not None


@pytest.mark.asyncio
@pytest.mark.skipif(not has_openai_key(), reason="No OpenAI API key available")
async def test_translation_french():
    """Test translation with French text"""
    await prune.prune_data()
    await prune.prune_system(metadata=True)

    french_text = """
    L'apprentissage automatique est une méthode d'analyse de données qui 
    automatise la construction de modèles analytiques. C'est une branche 
    de l'intelligence artificielle basée sur l'idée que les systèmes peuvent 
    apprendre à partir de données, identifier des modèles et prendre des décisions.
    """

    await add(french_text, dataset_name="test_french")

    result = await cognify(
        datasets=["test_french"],
        auto_translate=True,
        target_language="en",
    )

    assert result is not None

    search_results = await search(
        query_text="What is machine learning?",
        query_type=SearchType.SUMMARIES,
    )
    assert search_results is not None


@pytest.mark.asyncio
@pytest.mark.skipif(not has_openai_key(), reason="No OpenAI API key available")
async def test_translation_disabled():
    """Test that cognify works without translation"""
    await prune.prune_data()
    await prune.prune_system(metadata=True)

    text = "This is a baseline test without translation enabled."
    await add(text, dataset_name="test_baseline")

    result = await cognify(
        datasets=["test_baseline"],
        auto_translate=False,
    )

    assert result is not None


@pytest.mark.asyncio
@pytest.mark.skipif(not has_openai_key(), reason="No OpenAI API key available")
async def test_translation_mixed_languages():
    """Test with multiple documents in different languages"""
    await prune.prune_data()
    await prune.prune_system(metadata=True)

    texts = [
        "Artificial intelligence is transforming the world.",
        "La tecnología está cambiando nuestras vidas.",
        "Les ordinateurs deviennent de plus en plus puissants.",
    ]

    for text in texts:
        await add(text, dataset_name="test_mixed")

    result = await cognify(
        datasets=["test_mixed"],
        auto_translate=True,
        target_language="en",
    )

    assert result is not None

    search_results = await search(
        query_text="What topics are discussed?",
        query_type=SearchType.SUMMARIES,
    )
    assert search_results is not None


@pytest.mark.asyncio
@pytest.mark.skipif(not has_openai_key(), reason="No OpenAI API key available")
async def test_direct_translation_function():
    """Test the translate_text convenience function directly"""
    result = await translate_text(
        text="Hola, ¿cómo estás? Espero que tengas un buen día.",
        target_language="en",
        translation_provider="openai",
    )

    assert result.translated_text is not None
    assert result.translated_text != ""
    assert result.target_language == "en"
    assert result.provider == "openai"


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
