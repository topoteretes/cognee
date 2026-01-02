"""
Unit tests for translation configuration
"""

import os
from typing import get_args
from cognee.tasks.translation.config import (
    get_translation_config,
    TranslationConfig,
    TranslationProviderType,
)


def test_default_translation_config():
    """Test default translation configuration"""
    config = get_translation_config()

    assert isinstance(config, TranslationConfig)
    assert config.translation_provider in ["openai", "google", "azure"]
    assert 0.0 <= config.confidence_threshold <= 1.0


def test_translation_provider_type_literal():
    """Test TranslationProviderType Literal type values"""
    # Get the allowed values from the Literal type
    allowed_values = get_args(TranslationProviderType)

    assert "openai" in allowed_values
    assert "google" in allowed_values
    assert "azure" in allowed_values
    assert len(allowed_values) == 3


def test_confidence_threshold_bounds():
    """Test confidence threshold validation"""
    config = TranslationConfig(translation_provider="openai", confidence_threshold=0.9)

    assert 0.0 <= config.confidence_threshold <= 1.0


def test_multiple_provider_keys():
    """Test configuration with multiple provider API keys"""
    config = TranslationConfig(
        translation_provider="openai",
        google_translate_api_key="google_key",
        azure_translator_key="azure_key",
    )

    assert config.google_translate_api_key == "google_key"
    assert config.azure_translator_key == "azure_key"


if __name__ == "__main__":
    test_default_translation_config()
    print("✓ test_default_translation_config passed")

    test_translation_provider_type_literal()
    print("✓ test_translation_provider_type_literal passed")

    test_confidence_threshold_bounds()
    print("✓ test_confidence_threshold_bounds passed")

    test_multiple_provider_keys()
    print("✓ test_multiple_provider_keys passed")

    print("\nAll config tests passed!")
