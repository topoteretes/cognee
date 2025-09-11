"""
Unit tests for translation functionality.

Tests cover:
- Translation provider registry and discovery
- Language detection across providers
- Translation functionality 
- Error handling and fallbacks
- Model validation and serialization
"""

import pytest
import asyncio
from unittest.mock import Mock
from typing import Tuple, Optional, Dict
from pydantic import ValidationError

from cognee.tasks.translation.translate_content import (
    translate_content,
    register_translation_provider, 
    get_available_providers,
    TranslationProvider,
    NoOpProvider,
    _get_provider,
)
from cognee.tasks.translation.models import TranslatedContent, LanguageMetadata


class MockDocumentChunk:
    """Mock document chunk for testing."""
    
    def __init__(self, text: str, chunk_id: str = "test_chunk", metadata: Optional[Dict] = None):
        self.text = text
        self.id = chunk_id
        self.chunk_index = chunk_id
        self.metadata = metadata or {}


class MockTranslationProvider:
    """Mock provider for testing custom provider registration."""
    
    async def detect_language(self, text: str) -> Tuple[str, float]:
        if "hola" in text.lower():
            return "es", 0.95
        elif "bonjour" in text.lower():
            return "fr", 0.90
        return "en", 0.85
    
    async def translate(self, text: str, target_language: str) -> Tuple[str, float]:
        if target_language == "en":
            return f"[MOCK TRANSLATED] {text}", 0.88
        return text, 0.0


class TestProviderRegistry:
    """Test translation provider registration and discovery."""
    
    def test_get_available_providers_includes_builtin(self):
        """Test that built-in providers are included in available list."""
        providers = get_available_providers()
        assert "noop" in providers
        assert "langdetect" in providers
        
    def test_register_custom_provider(self):
        """Test custom provider registration.""" 
        register_translation_provider("mock", MockTranslationProvider)
        providers = get_available_providers()
        assert "mock" in providers
        
        # Test provider can be retrieved
        provider = _get_provider("mock")
        assert isinstance(provider, MockTranslationProvider)
        
    def test_provider_name_normalization(self):
        """Test provider names are normalized to lowercase."""
        register_translation_provider("CUSTOM_PROVIDER", MockTranslationProvider)
        providers = get_available_providers()
        assert "custom_provider" in providers
        
        # Should be retrievable with different casing
        provider1 = _get_provider("CUSTOM_PROVIDER")
        provider2 = _get_provider("custom_provider")
        assert type(provider1) == type(provider2)

    def test_unknown_provider_fallback(self):
        """Test unknown providers fall back to NoOp."""
        provider = _get_provider("nonexistent_provider")
        assert isinstance(provider, NoOpProvider)


class TestNoOpProvider:
    """Test NoOp provider functionality."""
    
    @pytest.mark.asyncio
    async def test_detect_language_ascii(self):
        """Test language detection for ASCII text."""
        provider = NoOpProvider()
        lang, conf = await provider.detect_language("Hello world")
        assert lang == "en"
        assert conf == 0.5  # Lower confidence to avoid false positives
        
    @pytest.mark.asyncio
    async def test_detect_language_unicode(self):
        """Test language detection for Unicode text."""
        provider = NoOpProvider()
        lang, conf = await provider.detect_language("Hëllo wörld")
        assert lang == "unknown"
        assert conf == 0.4
        
    @pytest.mark.asyncio 
    async def test_translate_returns_original(self):
        """Test translation returns original text with zero confidence."""
        provider = NoOpProvider()
        text = "Test text"
        translated, conf = await provider.translate(text, "es")
        assert translated == text
        assert conf == 0.0


class TestTranslationModels:
    """Test Pydantic models for translation data."""
    
    def test_translated_content_validation(self):
        """Test TranslatedContent model validation."""
        content = TranslatedContent(
            original_chunk_id="chunk_1",
            original_text="Hello",
            translated_text="Hola", 
            source_language="en",
            target_language="es",
            translation_provider="test",
            confidence_score=0.9
        )
        assert content.original_chunk_id == "chunk_1"
        assert content.confidence_score == 0.9
        
    def test_translated_content_confidence_validation(self):
        """Test confidence score validation bounds."""
        # Valid confidence scores
        TranslatedContent(
            original_chunk_id="test",
            original_text="test", 
            translated_text="test",
            source_language="en",
            confidence_score=0.0
        )
        TranslatedContent(
            original_chunk_id="test", 
            original_text="test",
            translated_text="test", 
            source_language="en",
            confidence_score=1.0
        )
        
        # Invalid confidence scores should raise validation error
        with pytest.raises(ValidationError):
            TranslatedContent(
                original_chunk_id="test",
                original_text="test",
                translated_text="test",
                source_language="en", 
                confidence_score=-0.1
            )
            
        with pytest.raises(ValidationError):
            TranslatedContent(
                original_chunk_id="test",
                original_text="test", 
                translated_text="test",
                source_language="en",
                confidence_score=1.1
            )
            
    def test_language_metadata_validation(self):
        """Test LanguageMetadata model validation."""
        metadata = LanguageMetadata(
            content_id="chunk_1",
            detected_language="es",
            language_confidence=0.95,
            requires_translation=True,
            character_count=100
        )
        assert metadata.content_id == "chunk_1"
        assert metadata.requires_translation is True
        assert metadata.character_count == 100
        
    def test_language_metadata_character_count_validation(self):
        """Test character count cannot be negative."""
        with pytest.raises(ValidationError):
            LanguageMetadata(
                content_id="test",
                detected_language="en",
                character_count=-1
            )


class TestTranslateContentFunction:
    """Test main translate_content function."""
    
    @pytest.mark.asyncio
    async def test_noop_provider_processing(self):
        """Test processing with noop provider."""
        chunks = [
            MockDocumentChunk("Hello world", "chunk_1"),
            MockDocumentChunk("Test content", "chunk_2")
        ]
        
        result = await translate_content(
            chunks, 
            target_language="en",
            translation_provider="noop",
            confidence_threshold=0.8
        )
        
        assert len(result) == 2
        for chunk in result:
            assert "language" in chunk.metadata
            # No translation should occur with noop provider
            assert "translation" not in chunk.metadata
            
    @pytest.mark.asyncio
    async def test_translation_with_custom_provider(self):
        """Test translation with custom registered provider."""
        # Register mock provider
        register_translation_provider("test_provider", MockTranslationProvider)
        
        chunks = [MockDocumentChunk("Hola mundo", "chunk_1")]
        
        result = await translate_content(
            chunks,
            target_language="en", 
            translation_provider="test_provider",
            confidence_threshold=0.8
        )
        
        chunk = result[0]
        assert "language" in chunk.metadata
        assert "translation" in chunk.metadata
        
        # Check language metadata
        lang_meta = chunk.metadata["language"]
        assert lang_meta["detected_language"] == "es"
        assert lang_meta["requires_translation"] is True
        
        # Check translation metadata
        trans_meta = chunk.metadata["translation"] 
        assert trans_meta["original_text"] == "Hola mundo"
        assert "[MOCK TRANSLATED]" in trans_meta["translated_text"]
        assert trans_meta["source_language"] == "es"
        assert trans_meta["target_language"] == "en"
        
        # Check chunk text was updated
        assert "[MOCK TRANSLATED]" in chunk.text
        
    @pytest.mark.asyncio
    async def test_low_confidence_no_translation(self):
        """Test that low confidence detection doesn't trigger translation."""
        register_translation_provider("low_conf", MockTranslationProvider)
        
        chunks = [MockDocumentChunk("Hello world", "chunk_1")]  # English text
        
        result = await translate_content(
            chunks,
            target_language="en",
            translation_provider="low_conf", 
            confidence_threshold=0.9  # High threshold
        )
        
        chunk = result[0]
        assert "language" in chunk.metadata
        # Should not translate due to high threshold and English detection
        assert "translation" not in chunk.metadata
        
    @pytest.mark.asyncio 
    async def test_error_handling_in_detection(self):
        """Test graceful error handling in language detection."""
        class FailingProvider:
            async def detect_language(self, text: str) -> Tuple[str, float]:
                raise Exception("Detection failed")
                
            async def translate(self, text: str, target_language: str) -> Tuple[str, float]:
                return text, 0.0
        
        register_translation_provider("failing", FailingProvider)
        
        chunks = [MockDocumentChunk("Test text", "chunk_1")]
        
        # Should not raise exception, should fallback gracefully
        result = await translate_content(
            chunks,
            translation_provider="failing"
        )
        
        chunk = result[0]
        assert "language" in chunk.metadata
        # Should have unknown language due to detection failure
        lang_meta = chunk.metadata["language"]
        assert lang_meta["detected_language"] == "unknown"
        assert lang_meta["language_confidence"] == 0.0
        
    @pytest.mark.asyncio
    async def test_error_handling_in_translation(self):
        """Test graceful error handling in translation."""
        class PartialProvider:
            async def detect_language(self, text: str) -> Tuple[str, float]:
                return "es", 0.9
                
            async def translate(self, text: str, target_language: str) -> Tuple[str, float]:
                raise Exception("Translation failed")
        
        register_translation_provider("partial", PartialProvider)
        
        chunks = [MockDocumentChunk("Hola", "chunk_1")]
        
        result = await translate_content(
            chunks,
            translation_provider="partial", 
            confidence_threshold=0.8
        )
        
        chunk = result[0]
        # Should have detected Spanish but failed translation
        assert chunk.metadata["language"]["detected_language"] == "es"
        # Should still create translation metadata with original text
        assert "translation" in chunk.metadata
        trans_meta = chunk.metadata["translation"]
        assert trans_meta["translated_text"] == "Hola"  # Original text due to failure
        assert trans_meta["confidence_score"] == 0.0
        
    @pytest.mark.asyncio
    async def test_no_translation_when_same_language(self):
        """Test no translation occurs when source equals target language."""
        register_translation_provider("same_lang", MockTranslationProvider)
        
        chunks = [MockDocumentChunk("Hello world", "chunk_1")]
        
        result = await translate_content(
            chunks,
            target_language="en",  # Same as detected language
            translation_provider="same_lang"
        )
        
        chunk = result[0] 
        assert "language" in chunk.metadata
        # No translation should occur for same language
        assert "translation" not in chunk.metadata
        
    @pytest.mark.asyncio
    async def test_metadata_serialization(self):
        """Test that metadata is properly serialized to dicts."""
        register_translation_provider("serialize_test", MockTranslationProvider)
        
        chunks = [MockDocumentChunk("Hola", "chunk_1")]
        
        result = await translate_content(
            chunks,
            translation_provider="serialize_test",
            confidence_threshold=0.8
        )
        
        chunk = result[0]
        
        # Metadata should be plain dicts, not Pydantic models
        assert isinstance(chunk.metadata["language"], dict)
        if "translation" in chunk.metadata:
            assert isinstance(chunk.metadata["translation"], dict)
            
    def test_model_serialization_compatibility(self):
        """Test that models serialize to JSON-compatible dicts."""
        content = TranslatedContent(
            original_chunk_id="test",
            original_text="Hello",
            translated_text="Hola",
            source_language="en", 
            target_language="es"
        )
        
        # Should serialize to dict
        data = content.model_dump()
        assert isinstance(data, dict)
        assert data["original_chunk_id"] == "test"
        assert "translation_timestamp" in data
        assert "metadata" in data
        
        # Should be JSON serializable
        import json
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed["original_chunk_id"] == "test"


if __name__ == "__main__":
    pytest.main([__file__])
