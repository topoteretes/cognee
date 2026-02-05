"""
Unit tests for translation configuration
"""

from typing import get_args

from pydantic import ValidationError

from cognee.tasks.translation.config import (
    get_translation_config,
    TranslationConfig,
    TranslationProviderType,
)


def test_default_translation_config():
    """Test default translation configuration"""
    config = get_translation_config()

    assert isinstance(config, TranslationConfig), "Config should be TranslationConfig instance"
    assert config.translation_provider in [
        "llm",
        "google",
        "azure",
    ], f"Invalid provider: {config.translation_provider}"
    assert 0.0 <= config.confidence_threshold <= 1.0, (
        f"Confidence threshold {config.confidence_threshold} out of bounds [0.0, 1.0]"
    )


def test_translation_provider_type_literal():
    """Test TranslationProviderType Literal type values"""
    # Get the allowed values from the Literal type
    allowed_values = get_args(TranslationProviderType)

    assert "llm" in allowed_values, "llm should be an allowed provider"
    assert "google" in allowed_values, "google should be an allowed provider"
    assert "azure" in allowed_values, "azure should be an allowed provider"
    assert len(allowed_values) == 3, f"Expected 3 providers, got {len(allowed_values)}"


def test_confidence_threshold_bounds():
    """Test confidence threshold validation"""
    config = TranslationConfig(translation_provider="llm", confidence_threshold=0.9)

    assert 0.0 <= config.confidence_threshold <= 1.0, (
        f"Confidence threshold {config.confidence_threshold} out of bounds [0.0, 1.0]"
    )


def test_confidence_threshold_validation():
    """Test that invalid confidence thresholds are rejected or clamped"""
    # Test boundary values - these should work
    config_min = TranslationConfig(translation_provider="llm", confidence_threshold=0.0)
    assert config_min.confidence_threshold == 0.0, "Minimum bound (0.0) should be valid"

    config_max = TranslationConfig(translation_provider="llm", confidence_threshold=1.0)
    assert config_max.confidence_threshold == 1.0, "Maximum bound (1.0) should be valid"

    # Test invalid values - these should either raise ValidationError or be clamped
    try:
        config_invalid_low = TranslationConfig(
            translation_provider="llm", confidence_threshold=-0.1
        )
        # If no error, verify it was clamped to valid range
        assert 0.0 <= config_invalid_low.confidence_threshold <= 1.0, (
            f"Invalid low value should be clamped, got {config_invalid_low.confidence_threshold}"
        )
    except ValidationError:
        pass  # Expected validation error

    try:
        config_invalid_high = TranslationConfig(
            translation_provider="llm", confidence_threshold=1.5
        )
        # If no error, verify it was clamped to valid range
        assert 0.0 <= config_invalid_high.confidence_threshold <= 1.0, (
            f"Invalid high value should be clamped, got {config_invalid_high.confidence_threshold}"
        )
    except ValidationError:
        pass  # Expected validation error


def test_multiple_provider_keys():
    """Test configuration with multiple provider API keys"""
    config = TranslationConfig(
        translation_provider="llm",
        google_translate_api_key="google_key",
        azure_translator_key="azure_key",
    )

    assert config.google_translate_api_key == "google_key", "Google API key not set correctly"
    assert config.azure_translator_key == "azure_key", "Azure API key not set correctly"
