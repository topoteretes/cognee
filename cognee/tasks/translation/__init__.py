from .translate_content import (
    translate_content,
    register_translation_provider,
    TranslationProvider,
    validate_provider,
)
from .models import TranslatedContent, LanguageMetadata
from .translation_registry import get_available_providers, get_available_detectors
# Backwards-compatible alias expected by tests and older code
get_available_translators = get_available_providers

__all__ = (
    "get_available_providers",
    "get_available_detectors",
    "get_available_translators",
    "LanguageMetadata",
    "register_translation_provider",
    "translate_content",
    "TranslatedContent",
    "TranslationProvider",
    "validate_provider",
)
