"""
Cognify API - Translation and Cognition Services

Provides endpoints for:
- Submitting content for translation
- Retrieving translation providers
- Language detection and metadata extraction
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os

from cognee.tasks.translation.translate_content import (
    translate_content,
    register_translation_provider,
    get_available_providers,
    TranslationProvider,
    NoOpProvider,
    _get_provider,
)
from cognee.tasks.translation.models import TranslatedContent, LanguageMetadata


router = APIRouter()

# Dependency injection for translation provider
class TranslationProviderError(Exception):
    """Exception raised for errors in the translation provider."""


def _parse_batch_env(var: str, default: int = 10) -> int:
    try:
        return max(1, int(os.getenv(var, str(default))))
    except ValueError:
        return default

DEFAULT_BATCH_SIZE = _parse_batch_env("COGNEE_DEFAULT_BATCH_SIZE", 10)


async def cognify(
    content: str,
    target_language: str,
    translation_provider: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    **kwargs: Any
) -> List[TranslatedContent]:
    """Cognify content - detect language, translate, and extract metadata."""
    
    # Provider initialization and validation
    if translation_provider is not None:
        translation_provider = (translation_provider or "noop").strip().lower()
        # Preflight instantiate to both validate and surface missing deps early
        try:
            from cognee.tasks.translation.translate_content import _get_provider as _preflight_get_provider
            _preflight_get_provider(translation_provider)
        except Exception as e:
            raise TranslationProviderError(f"Provider '{translation_provider}' failed to initialize") from e
    
    # ...existing processing logic...