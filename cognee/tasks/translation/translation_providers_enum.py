from enum import Enum
from typing import Protocol, Optional, Tuple

class TranslationProviderEnum(Enum):
    LLM = "llm"
    GOOGLE = "google"
    AZURE = "azure"
    LANGDETECT = "langdetect"
    NOOP = "noop"

class TranslationProvider(Protocol):
    async def detect_language(self, text: str) -> Optional[Tuple[str, float]]:
        ...
    async def translate(self, text: str, target_language: str) -> Optional[Tuple[str, float]]:
        ...
