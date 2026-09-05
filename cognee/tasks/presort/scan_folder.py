"""
Folder scanning for presort: walk a directory, filter junk, and collect cheap
per-file metadata (no content hashing here — that is done lazily by
``detect_duplicates.hash_files`` so unique large files are not read in full).
"""

from pathlib import Path
from typing import List, Tuple

from cognee.infrastructure.files.utils.guess_file_type import guess_file_type
from cognee.infrastructure.files.utils.is_text_content import is_text_content
from cognee.shared.logging_utils import get_logger
from cognee.tasks.code_graph.code_repo import SKIP_DIRS

from .models import FileRecord, JunkFile

logger = get_logger("presort")

# Filenames and extensions that are noise in any folder (OS droppings,
# partial downloads, caches). Matched case-insensitively.
JUNK_FILENAMES = frozenset({".ds_store", "thumbs.db", "desktop.ini", ".localized", "icon\r"})
JUNK_EXTENSIONS = frozenset(
    {"tmp", "temp", "part", "partial", "crdownload", "download", "aria2", "lock", "pyc"}
)

_SNIFF_BYTES = 8192


def _claimable_extensions() -> frozenset:
    """Path extensions any registered loader (code included) claims.

    Keyed on path extensions only — content sniffing reports unknown binaries
    as text, so it cannot answer "would cognee know what to do with this?".
    """
    from cognee.infrastructure.loaders import get_loader_engine
    from cognee.infrastructure.loaders.core.code_loader import SUPPORTED_CODE_EXTENSIONS

    loader_engine = get_loader_engine()
    extensions: set = set(SUPPORTED_CODE_EXTENSIONS)
    for loader_name in loader_engine.get_available_loaders():
        extensions.update(
            ext.lower() for ext in loader_engine.get_loader_info(loader_name)["extensions"]
        )
    return frozenset(extensions)


def _junk_reason(relative_path: Path, size_bytes: int) -> str:
    for part in relative_path.parts[:-1]:
        if part in SKIP_DIRS:
            return f"inside skipped directory {part!r}"
        if part.startswith("."):
            return f"inside hidden directory {part!r}"
    file_name = relative_path.name
    if file_name.lower() in JUNK_FILENAMES:
        return "junk file"
    if file_name.startswith("."):
        return "hidden file"
    extension = relative_path.suffix.lstrip(".").lower()
    if extension in JUNK_EXTENSIONS:
        return f"junk extension .{extension}"
    if size_bytes == 0:
        return "empty file"
    return ""


async def scan_folder(
    root: Path,
    *,
    include_subdirectories: bool = True,
) -> Tuple[List[FileRecord], List[JunkFile]]:
    """Walk `root` and return (kept file records, junk files with reasons)."""
    from cognee.infrastructure.loaders.core.code_loader import SUPPORTED_CODE_EXTENSIONS

    claimable = _claimable_extensions()
    kept: List[FileRecord] = []
    junk: List[JunkFile] = []

    candidates = sorted(root.rglob("*")) if include_subdirectories else sorted(root.iterdir())
    for file_path in candidates:
        try:
            if not file_path.is_file() or file_path.is_symlink():
                continue
            size_bytes = file_path.stat().st_size
        except OSError as error:
            junk.append(JunkFile(path=str(file_path), reason=f"unreadable: {error}"))
            continue

        relative = file_path.relative_to(root)
        reason = _junk_reason(relative, size_bytes)
        if reason:
            junk.append(JunkFile(path=str(file_path), reason=reason))
            continue

        extension = file_path.suffix.lstrip(".").lower()
        record = FileRecord(
            path=str(file_path),
            name=file_path.name,
            extension=extension,
            size_bytes=size_bytes,
            loader_claimed=extension in claimable,
            is_code=extension in SUPPORTED_CODE_EXTENSIONS,
        )

        try:
            with open(file_path, "rb") as file:
                sample = file.read(_SNIFF_BYTES)
                file.seek(0)
                file_type = guess_file_type(file, file_path.name)
            record.mime_type = file_type.mime
            if not record.extension:
                record.extension = file_type.extension
            # A media/archive mime always wins over the byte heuristic (a short
            # sample of a binary format can look ASCII-ish).
            mime = record.mime_type or ""
            non_text_mime = mime.startswith(("image/", "audio/", "video/")) or (
                mime.startswith("application/") and mime != "application/json"
            )
            record.is_text = is_text_content(sample) and not non_text_mime
        except Exception as error:  # sniffing must never abort the scan
            record.warnings.append(f"could not sniff file type: {error}")
            logger.debug(f"Presort sniff failed for {file_path}: {error}")

        kept.append(record)

    return kept, junk
