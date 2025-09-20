from typing import Optional, Tuple
from ..translation_providers_enum import TranslationProvider

class NoopProvider:
    """A no-op translation provider that does not perform detection or translation."""
    async def detect_language(self, _text: str) -> Optional[Tuple[str, float]]:
        """No-op language detection: intentionally performs no detection and always returns None."""
        return None

    async def translate(self, text: str, _target_language: str) -> Optional[Tuple[str, float]]:
        """Return the input text unchanged and a confidence score of 0.0."""
        return text, 0.0
