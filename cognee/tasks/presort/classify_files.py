"""
File classification for presort.

Deterministic layer: extension-family buckets (aligned with cognee's loader
routing — ``EXTENSION_TO_DOCUMENT_CLASS``) filled onto ``FileRecord.family``.
Opt-in LLM layer (``use_llm=True``): the dormant ``extract_categories``
content taxonomy (``classify_content.txt``) refines text files with a
``content_class`` label.
"""

import asyncio
from typing import List

from cognee.shared.logging_utils import get_logger

from .models import FileRecord

logger = get_logger("presort")

_LLM_CONCURRENCY = 8
_LLM_SAMPLE_CHARS = 4000

_FAMILY_EXTENSIONS = {
    "documents": {
        "pdf",
        "txt",
        "md",
        "rst",
        "doc",
        "docx",
        "odt",
        "rtf",
        "ppt",
        "pptx",
        "odp",
        "epub",
        "pages",
        "key",
    },
    "data": {
        "csv",
        "xls",
        "xlsx",
        "ods",
        "json",
        "xml",
        "yaml",
        "yml",
        "parquet",
        "sqlite",
        "db",
        "numbers",
    },
    "images": {
        "png",
        "jpg",
        "jpeg",
        "gif",
        "webp",
        "tif",
        "tiff",
        "bmp",
        "psd",
        "ico",
        "heic",
        "avif",
        "svg",
        "raw",
    },
    "audio": {"mp3", "wav", "flac", "m4a", "ogg", "aac", "mid", "amr", "aiff"},
    "video": {"mp4", "mov", "avi", "mkv", "webm", "m4v", "wmv", "flv"},
    "archives": {"zip", "rar", "7z", "tar", "gz", "bz2", "xz", "tgz"},
    "installers": {"dmg", "pkg", "exe", "msi", "app", "deb", "rpm", "apk", "appimage"},
}

_EXTENSION_TO_FAMILY = {
    extension: family
    for family, extensions in _FAMILY_EXTENSIONS.items()
    for extension in extensions
}


def classify_family(record: FileRecord) -> str:
    if record.is_code:
        return "code"
    family = _EXTENSION_TO_FAMILY.get(record.extension)
    if family:
        return family
    if record.is_text:
        return "documents"
    return "other"


async def _llm_classify(record: FileRecord) -> None:
    from cognee.infrastructure.llm.extraction.extract_categories import extract_categories
    from cognee.shared.data_models import DefaultContentPrediction

    try:
        with open(record.path, "rb") as file:
            sample = file.read(_LLM_SAMPLE_CHARS * 4).decode("utf-8", errors="replace")
    except OSError as error:
        record.warnings.append(f"could not read sample for classification: {error}")
        return

    prediction = await extract_categories(sample[:_LLM_SAMPLE_CHARS], DefaultContentPrediction)
    label = getattr(prediction, "label", None)
    subclasses = getattr(label, "subclass", None)
    if subclasses:
        first = subclasses[0]
        record.content_class = getattr(first, "value", str(first))
    else:
        record.content_class = getattr(label, "type", None)


async def classify_files(files: List[FileRecord], *, use_llm: bool = False) -> None:
    """Fill ``family`` (always) and ``content_class`` (LLM opt-in); mutates in place."""
    for record in files:
        record.family = classify_family(record)

    if not use_llm:
        return

    semaphore = asyncio.Semaphore(_LLM_CONCURRENCY)

    async def classify(record: FileRecord) -> None:
        async with semaphore:
            try:
                await _llm_classify(record)
            except Exception as error:  # LLM failures must not abort presort
                record.warnings.append(f"LLM classification failed: {error}")
                logger.debug(f"Presort LLM classify failed for {record.path}: {error}")

    await asyncio.gather(
        *(classify(record) for record in files if record.is_text and not record.is_code)
    )
