import os
from typing import Any, Optional

from cognee.infrastructure.files.storage import get_file_storage, get_storage_config
from cognee.infrastructure.files.utils.get_file_metadata import get_file_metadata
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.loaders.LoaderInterface import LoaderInterface


class ImageLoader(LoaderInterface):
    """
    Core image file loader that handles basic image file formats.
    Supports optional EXIF metadata extraction and perceptual-hash deduplication
    when the corresponding env vars are enabled.
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
        """
        Check if this loader can handle the given file.

        Args:
            extension: File extension
            mime_type: Optional MIME type

        Returns:
            True if file can be handled, False otherwise
        """
        if extension in self.supported_extensions and mime_type in self.supported_mime_types:
            return True

        return False

    async def load(self, file_path: str, **kwargs: Any) -> str:
        """
        Load and process the image file.

        When IMAGE_EXIF_ENABLED=true, extracts EXIF metadata (timestamp, camera
        make/model, GPS coordinates) from the image and appends it to the
        transcribed content.

        When IMAGE_PERCEPTUAL_HASH_ENABLED=true, computes a perceptual hash
        of the image using PIL. The hash can be used downstream for near-duplicate
        detection; a basic dedup check against previously-ingested hashes is
        performed when this flag is on.

        Args:
            file_path: Path to the file to load
            **kwargs: Additional configuration (unused)

        Returns:
            LoaderResult containing the file content and metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            UnicodeDecodeError: If file cannot be decoded with specified encoding
            OSError: If file cannot be read
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read file for metadata
        with open(file_path, "rb") as f:
            file_metadata = await get_file_metadata(f)

        # Name ingested file of current loader based on original file content hash
        storage_file_name = "text_" + file_metadata["content_hash"] + ".txt"
        content_parts = []

        # --- Step 1: Vision-LLM transcription (existing) ---
        result = await LLMGateway.transcribe_image(file_path)
        transcript = result.choices[0].message.content
        content_parts.append(transcript)

        # --- Step 2: Optional EXIF metadata extraction ---
        if os.environ.get("IMAGE_EXIF_ENABLED", "false").lower() == "true":
            exif_text = self._extract_exif_metadata(file_path)
            if exif_text:
                content_parts.append(f"\n\n[EXIF Metadata]\n{exif_text}")

        # --- Step 3: Optional perceptual-hash dedup ---
        if os.environ.get("IMAGE_PERCEPTUAL_HASH_ENABLED", "false").lower() == "true":
            image_hash = self._compute_perceptual_hash(file_path)
            if image_hash is not None:
                content_parts.append(f"\n[Perceptual Hash: {image_hash}]")
                # Basic in-memory dedup check
                if self._is_duplicate(image_hash):
                    # Still return the content but flag it
                    content_parts.append(
                        "[Duplicate: visually similar to a previously ingested image]"
                    )

        combined_content = "".join(content_parts)

        if not kwargs.get("persist", True):
            return combined_content

        storage_config = get_storage_config()
        data_root_directory = storage_config["data_root_directory"]
        storage = get_file_storage(data_root_directory)

        full_file_path = await storage.store(storage_file_name, combined_content)

        # If dedup is enabled, store the hash for future comparison
        if (
            os.environ.get("IMAGE_PERCEPTUAL_HASH_ENABLED", "false").lower() == "true"
            and image_hash is not None
        ):
            self._store_hash(image_hash)

        return full_file_path

    # ------------------------------------------------------------------
    #  EXIF extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_exif_metadata(file_path: str) -> Optional[str]:
        """
        Extract human-readable EXIF metadata from an image file.

        Returns a formatted string with datetime, camera info, and GPS
        coordinates when available, or None if the image has no EXIF data.
        """
        try:
            from PIL import Image  # ty: ignore[unresolved-import]
            from PIL.ExifTags import TAGS  # ty: ignore[unresolved-import]
        except ImportError:
            return None

        try:
            with Image.open(file_path) as img:
                exif_data = img._getexif()
        except Exception:
            return None

        if exif_data is None:
            return None

        interesting_tags = {
            271: "Camera Make",
            272: "Camera Model",
            36867: "Date Taken",
            37500: "Maker Note",
            34853: "GPS Info",
            33434: "Exposure Time",
            33437: "F Number",
            34855: "ISO Speed",
            37386: "Focal Length",
        }

        lines = []
        for tag_id, tag_name in interesting_tags.items():
            raw_value = exif_data.get(tag_id)
            if raw_value is None:
                continue

            if tag_id == 34853 and isinstance(raw_value, dict):
                # GPS Info – extract coordinates
                gps_text = _format_gps_info(raw_value)
                if gps_text:
                    lines.append(f"{tag_name}: {gps_text}")
            elif tag_id == 36867:
                # DateTimeOriginal – keep as-is
                lines.append(f"{tag_name}: {raw_value}")
            else:
                lines.append(f"{tag_name}: {raw_value}")

        if not lines:
            return None

        return "\n".join(lines)

    # ------------------------------------------------------------------
    #  Perceptual hash helpers
    # ------------------------------------------------------------------

    _seen_hashes: set[str] = set()

    @staticmethod
    def _compute_perceptual_hash(file_path: str) -> Optional[str]:
        """
        Compute a 64-bit perceptual (difference) hash for the image using
        only PIL — no external ``imagehash`` dependency required.

        Returns the hash as a hex string, or None on failure.
        """
        try:
            from PIL import Image  # ty: ignore[unresolved-import]
        except ImportError:
            return None

        try:
            with Image.open(file_path) as img:
                return _dhash(img)
        except Exception:
            return None

    @classmethod
    def _is_duplicate(cls, image_hash: str) -> bool:
        """Check whether this hash has been seen before (in-session)."""
        return image_hash in cls._seen_hashes

    @classmethod
    def _store_hash(cls, image_hash: str) -> None:
        """Record a hash value for future duplicate checks."""
        cls._seen_hashes.add(image_hash)

    @classmethod
    def reset_hashes(cls) -> None:
        """Clear the seen-hash set (mainly for testing)."""
        cls._seen_hashes.clear()


# ------------------------------------------------------------------
#  Module-level helpers (no Pillow/exif dep needed at import time)
# ------------------------------------------------------------------


def _dhash(image, hash_size: int = 8) -> str:
    """
    Difference hash: resize to (hash_size+1 x hash_size), convert to
    grayscale, compare adjacent columns, and pack bits into a hex string.
    """
    from PIL import Image  # ty: ignore[unresolved-import]

    image = image.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = list(image.getdata())
    # pixels now has (hash_size+1) * hash_size entries, row-major
    bits: list[str] = []
    for row in range(hash_size):
        row_start = row * (hash_size + 1)
        for col in range(hash_size):
            left = pixels[row_start + col]
            right = pixels[row_start + col + 1]
            bits.append("1" if left > right else "0")
    # Pack bits into hex
    hex_digits: list[str] = []
    for i in range(0, len(bits), 4):
        nibble = bits[i : i + 4]
        hex_digits.append(f"{int(''.join(nibble), 2):x}")
    return "".join(hex_digits)


def _format_gps_info(gps_dict: dict) -> Optional[str]:
    """Format GPSInfo dict (tag 34853) into human-readable coordinates."""
    try:
        from PIL.ExifTags import GPSTAGS  # ty: ignore[unresolved-import]
    except ImportError:
        return None

    def _to_decimal(values, ref: str) -> Optional[float]:
        """Convert (degrees, minutes, seconds) tuple to decimal degrees."""
        if not values or len(values) < 3:
            return None
        try:
            d, m, s = float(values[0]), float(values[1]), float(values[2])
            decimal = d + m / 60.0 + s / 3600.0
            if ref in ("S", "W"):
                decimal *= -1
            return decimal
        except (TypeError, ValueError):
            return None

    try:
        lat = _to_decimal(gps_dict.get(2), gps_dict.get(1, "N"))  # GPSLatitude, GPSLatitudeRef
        lon = _to_decimal(gps_dict.get(4), gps_dict.get(3, "E"))  # GPSLongitude, GPSLongitudeRef
    except Exception:
        return None

    parts = []
    if lat is not None:
        parts.append(f"{lat:.6f}°{'N' if lat >= 0 else 'S'}")
    if lon is not None:
        parts.append(f"{lon:.6f}°{'E' if lon >= 0 else 'W'}")
    if not parts:
        return None
    return ", ".join(parts)
