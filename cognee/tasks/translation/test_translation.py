"""
Unit tests for translation functionality.

Tests cover:
- Translation provider registry and discovery
- Language detection across providers
- Translation functionality 
- Error handling and fallbacks
- Model validation and serialization
"""

import pytest  # type: ignore[import-untyped]
from typing import Tuple, Optional, Dict
from pydantic import ValidationError
import cognee.tasks.translation.translate_content as tr

from cognee.tasks.translation.translate_content import (
    translate_content,
    register_translation_provider, 
    get_available_providers,
    TranslationProvider,
    NoOpProvider,
    _get_provider,
)
from cognee.tasks.translation.models import TranslatedContent, LanguageMetadata


class TestDetectionError(Exception):  # pylint: disable=too-few-public-methods
    """Test exception for detection failures."""


class TestTranslationError(Exception):  # pylint: disable=too-few-public-methods
    """Test exception for translation failures."""


# Ensure registry isolation across tests using public helpers
@pytest.fixture(autouse=True)
def _restore_registry():
<<<<<<< HEAD
=======
    """
    Pytest fixture that snapshots the translation provider registry and restores it after the test.
    
    Use to isolate tests that register or modify providers: the current registry state is captured before the test runs, and always restored when the fixture completes (including on exceptions).
    """
>>>>>>> 9f6b2dca51a936a9de482fc9f3c64934502240b6
    snapshot = tr.snapshot_registry()
    try:
        yield
    finally:
        tr.restore_registry(snapshot)


class MockDocumentChunk:  # pylint: disable=too-few-public-methods
    """Mock document chunk for testing."""
    
    def __init__(self, text: str, chunk_id: str = "test_chunk", metadata: Optional[Dict] = None):
<<<<<<< HEAD
=======
        """
        Initialize a mock document chunk used in tests.
        
        Parameters:
            text (str): Chunk text content.
            chunk_id (str): Identifier for the chunk; also used as chunk_index for tests. Defaults to "test_chunk".
            metadata (Optional[Dict]): Optional mapping of metadata values; defaults to an empty dict.
        """
>>>>>>> 9f6b2dca51a936a9de482fc9f3c64934502240b6
        self.text = text
        self.id = chunk_id
        self.chunk_index = chunk_id
        self.metadata = metadata or {}


class MockTranslationProvider:
    """Mock provider for testing custom provider registration."""
    
    async def detect_language(self, text: str) -> Tuple[str, float]:
<<<<<<< HEAD
=======
        """
        Detect the language of the given text and return an ISO 639-1 language code with a confidence score.
        
        This mock implementation uses simple keyword heuristics: returns ("es", 0.95) if the text contains "hola",
        ("fr", 0.90) if it contains "bonjour", and ("en", 0.85) otherwise.
        
        Parameters:
            text (str): Input text to analyze.
        
        Returns:
            Tuple[str, float]: A tuple of (language_code, confidence) where language_code is an ISO 639-1 code and
            confidence is a float between 0.0 and 1.0 indicating detection confidence.
        """
>>>>>>> 9f6b2dca51a936a9de482fc9f3c64934502240b6
        if "hola" in text.lower():
            return "es", 0.95
        if "bonjour" in text.lower():
            return "fr", 0.90
        return "en", 0.85
    
    async def translate(self, text: str, target_language: str) -> Tuple[str, float]:
<<<<<<< HEAD
=======
        """
        Simulate translating `text` into `target_language` and return a mock translated string with a confidence score.
        
        If `target_language` is "en", returns the input prefixed with "[MOCK TRANSLATED]" and a confidence of 0.88. For any other target language, returns the original `text` and a confidence of 0.0.
        
        Parameters:
            text (str): The text to translate.
            target_language (str): The target language code (e.g., "en").
        
        Returns:
            Tuple[str, float]: A pair of (translated_text, confidence) where confidence is in [0.0, 1.0].
        """
>>>>>>> 9f6b2dca51a936a9de482fc9f3c64934502240b6
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
        assert provider1.__class__ is provider2.__class__

    def test_unknown_provider_raises(self):
        """Test unknown providers raise ValueError."""
        with pytest.raises(ValueError):
            _get_provider("nonexistent_provider")


class TestNoOpProvider:
    """Test NoOp provider functionality."""
    
    @pytest.mark.asyncio
    async def test_detect_language_ascii(self):
        """Test language detection for ASCII text."""
        provider = NoOpProvider()
        lang, conf = await provider.detect_language("Hello world")
        assert lang is None
        assert conf == 0.0
        
    @pytest.mark.asyncio
    async def test_detect_language_unicode(self):
        """Test language detection for Unicode text."""
        provider = NoOpProvider()
        lang, conf = await provider.detect_language("Hëllo wörld")
        assert lang is None
        assert conf == 0.0
        
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
            assert chunk.metadata["language"]["detected_language"] == "unknown"
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
        assert trans_meta["translation_provider"] == "test_provider"
        
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
            async def detect_language(self, _text: str) -> Tuple[str, float]:
<<<<<<< HEAD
                raise TestDetectionError()
                
            async def translate(self, text: str, _target_language: str) -> Tuple[str, float]:
=======
                """
                Simulate a language detection failure by always raising TestDetectionError.
                
                This async method is used in tests to emulate a provider that fails during language detection. It accepts a text string but does not return; it always raises TestDetectionError.
                """
                raise TestDetectionError()
                
            async def translate(self, text: str, _target_language: str) -> Tuple[str, float]:
                """
                Return the input text unchanged and a translation confidence of 0.0.
                
                This no-op translator performs no translation; the supplied target language is ignored.
                
                Parameters:
                    text (str): Source text to "translate".
                    _target_language (str): Target language (ignored).
                
                Returns:
                    Tuple[str, float]: A tuple containing the original text and a confidence score (always 0.0).
                """
>>>>>>> 9f6b2dca51a936a9de482fc9f3c64934502240b6
                return text, 0.0
        
        register_translation_provider("failing", FailingProvider)
        
        chunks = [MockDocumentChunk("Test text", "chunk_1")]
        
        # Disable 'langdetect' fallback to force unknown
        ld = tr._provider_registry.pop("langdetect", None)
        try:
            result = await translate_content(chunks, translation_provider="failing")
        finally:
            if ld is not None:
                tr._provider_registry["langdetect"] = ld
        
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
            async def detect_language(self, _text: str) -> Tuple[str, float]:
<<<<<<< HEAD
                return "es", 0.9
                
            async def translate(self, _text: str, _target_language: str) -> Tuple[str, float]:
=======
                """
                Mock language detection used in tests.
                
                Parameters:
                    _text (str): Input text (ignored by this mock).
                
                Returns:
                    Tuple[str, float]: A fixed detected language code ("es") and confidence (0.9).
                """
                return "es", 0.9
                
            async def translate(self, _text: str, _target_language: str) -> Tuple[str, float]:
                """
                Simulate a failing translation by always raising TestTranslationError.
                
                This async method ignores its inputs and is used in tests to emulate a provider-side failure during translation.
                
                Parameters:
                    _text (str): Unused input text.
                    _target_language (str): Unused target language code.
                
                Raises:
                    TestTranslationError: Always raised to simulate a translation failure.
                """
>>>>>>> 9f6b2dca51a936a9de482fc9f3c64934502240b6
                raise TestTranslationError()
        
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
<<<<<<< HEAD
        """Test that models serialize to JSON-compatible dicts."""
=======
        """
        Verify that a TranslatedContent instance can be dumped to a JSON-serializable dict.
        
        Creates a TranslatedContent with sample fields, calls model_dump(), and asserts:
        - the result is a dict,
        - required fields like `original_chunk_id`, `translation_timestamp`, and `metadata` are present and preserved,
        - the dict can be round-tripped through json.dumps/json.loads without losing `original_chunk_id`.
        """
>>>>>>> 9f6b2dca51a936a9de482fc9f3c64934502240b6
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



