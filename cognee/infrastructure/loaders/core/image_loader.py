import asyncio
import os
from functools import lru_cache
from typing import Any

from cognee.infrastructure.files.storage import get_file_storage, get_storage_config
from cognee.infrastructure.files.utils.get_file_metadata import get_file_metadata
from cognee.infrastructure.llm.config import get_llm_config
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.infrastructure.loaders.LoaderInterface import LoaderInterface
from cognee.shared.logging_utils import get_logger

logger = get_logger(__name__)

MAX_OCR_TEXT_LENGTH = 8000


class ImageLoader(LoaderInterface):
    """
    Core image file loader.

    Transcribes images with a vision LLM using an extraction-oriented prompt (disable via
    IMAGE_EXTRACTION_ENABLED=false). When IMAGE_OCR_ENABLED is set, text from a local OCR pass
    (rapidocr-onnxruntime) is appended to the transcription.
    """

    loader_name = "image_loader"

    @property
    def supported_extensions(self) -> list[str]:
        """Supported text file extensions."""
        return [
            "png",
            "dwg",
            "xcf",
            "jpg",
            "jpe",
            "jpeg",
            "jpx",
            "apng",
            "gif",
            "webp",
            "cr2",
            "tif",
            "tiff",
            "bmp",
            "jxr",
            "psd",
            "ico",
            "heic",
            "avif",
        ]

    @property
    def supported_mime_types(self) -> list[str]:
        """Supported MIME types for text content."""
        return [
            "image/png",
            "image/vnd.dwg",
            "image/x-xcf",
            "image/jpeg",
            "image/jpx",
            "image/apng",
            "image/gif",
            "image/webp",
            "image/x-canon-cr2",
            "image/tiff",
            "image/bmp",
            "image/jxr",
            "image/vnd.adobe.photoshop",
            "image/vnd.microsoft.icon",
            "image/heic",
            "image/avif",
        ]

    def can_handle(self, extension: str, mime_type: str) -> bool:
        """Check if file can be handled by this loader."""
        if extension in self.supported_extensions and mime_type in self.supported_mime_types:
            return True

        return False

    async def load(self, file_path: str, **kwargs: Any) -> str:
        """
        Transcribe the image and return the extracted text.

        Args:
            file_path: Path to the image file
            **kwargs: Additional arguments (e.g. persist)

        Returns:
            Path to the stored text file, or the text itself when persist=False

        Raises:
            FileNotFoundError: If the file does not exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "rb") as f:
            file_metadata = await get_file_metadata(f)
        # Name ingested file of current loader based on original file content hash
        storage_file_name = "text_" + file_metadata["content_hash"] + ".txt"

        prompt, max_completion_tokens, reasoning_effort = self._transcription_overrides()
        result = await LLMGateway.transcribe_image(
            file_path,
            prompt=prompt,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
        )
        content = result.choices[0].message.content or ""
        if not content:
            logger.warning(
                f"Empty image transcription for {file_path}; "
                "try raising IMAGE_TRANSCRIPTION_MAX_COMPLETION_TOKENS."
            )

        if self._ocr_enabled():
            ocr_text = await self._extract_ocr_text(file_path)
            if ocr_text:
                content = f"{content}\n\n[OCR extracted text]\n{ocr_text}"

        if not kwargs.get("persist", True):
            return content

        storage_config = get_storage_config()
        data_root_directory = storage_config["data_root_directory"]
        storage = get_file_storage(data_root_directory)

        full_file_path = await storage.store(storage_file_name, content)

        return full_file_path

    def _ocr_enabled(self) -> bool:
        """Whether OCR extraction is enabled via the IMAGE_OCR_ENABLED env flag."""
        return os.getenv("IMAGE_OCR_ENABLED", "false").lower() == "true"

    def _transcription_overrides(self) -> tuple[str | None, int | None, str | None]:
        """Return the configured extraction prompt, token cap, and reasoning effort (on by
        default); set IMAGE_EXTRACTION_ENABLED=false for (None, None, None) to keep the legacy
        caption prompt."""
        if os.getenv("IMAGE_EXTRACTION_ENABLED", "true").lower() == "false":
            return None, None, None
        llm_config = get_llm_config()
        prompt_path = llm_config.image_transcription_prompt_path
        if os.path.isabs(prompt_path):
            base_directory = os.path.dirname(prompt_path)
            prompt_path = os.path.basename(prompt_path)
        else:
            base_directory = None
        prompt = render_prompt(prompt_path, {}, base_directory=base_directory)
        return (
            prompt,
            llm_config.image_transcription_max_completion_tokens,
            llm_config.image_transcription_reasoning_effort,
        )

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_ocr_engine() -> Any:
        """Build the RapidOCR engine once (cached). Requires the rapidocr-onnxruntime dependency."""
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as e:
            raise ImportError(
                "rapidocr-onnxruntime is required for image OCR. "
                'Install with: pip install "cognee[rapidocr]"'
            ) from e

        # Disable the 180-degree angle classifier: it only helps fully upside-down text
        # (rare for screenshots/charts/scans) and otherwise false-flips upright lines.
        return RapidOCR(use_cls=False)

    async def _extract_ocr_text(self, file_path: str) -> str:
        """Run local OCR on the image, returning extracted text (empty string on OCR failure)."""
        engine = self._get_ocr_engine()
        try:
            # RapidOCR is blocking CPU work; offload it so the event loop stays free.
            ocr_result, _ = await asyncio.to_thread(engine, file_path)
        except Exception as e:
            logger.error(f"OCR failed for {file_path}: {e}")
            return ""
        if not ocr_result:
            return ""
        # Each result row is [bounding_box, text, confidence]; keep the recognized text.
        text = "\n".join(line[1] for line in ocr_result).strip()
        if len(text) > MAX_OCR_TEXT_LENGTH:
            text = text[: MAX_OCR_TEXT_LENGTH - 3] + "..."
        return text
