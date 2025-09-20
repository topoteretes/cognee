from typing import Optional, Tuple, Any
from ..translation_providers_enum import TranslationProvider
from ..translation_errors import LangDetectError
import logging

logger = logging.getLogger(__name__)

class LangDetectProvider:
    """A provider that uses the 'langdetect' library for offline language detection. This provider does not support translation."""
    _detector: Any = None

    def __init__(self):
        if self._detector is None:
            try:
                from langdetect import DetectorFactory, detect_langs
                from langdetect.lang_detect_exception import LangDetectException
                DetectorFactory.seed = 0
                self._detector = (detect_langs, LangDetectException)
            except ImportError as e:
                raise LangDetectError() from e

    async def detect_language(self, text: str) -> Optional[Tuple[str, float]]:
        """Detect the language of the provided text using the langdetect library."""
        detect_langs, LangDetectException = self._detector
        try:
            langs = detect_langs(text)
            if langs:
                return langs[0].lang, langs[0].prob
        except LangDetectException:
            logger.debug("Langdetect failed (text_len=%d)", len(text) if isinstance(text, str) else -1)
        return None

    async def translate(self, text: str, _target_language: str) -> Optional[Tuple[str, float]]:
        """This provider does not support translation. It returns the original text."""
        return text, 0.0
