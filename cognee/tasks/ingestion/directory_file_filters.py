"""Filters applied when a directory is expanded into ingestion candidates.

Directories swept into ingestion (a repo checkout, a project folder) carry
files that are useless or harmful to ingest: compiled artifacts and other
binaries no loader can process, version-control internals, virtualenvs,
caches. These filters run only on directory expansion — a file passed
explicitly is never filtered, on the assumption the caller meant it.
"""

from pathlib import Path
from typing import List, Optional

import pathspec

from cognee.shared.logging_utils import get_logger

logger = get_logger(__name__)

_BINARY_SNIFF_BYTES = 8192

# Version-control internals are never ingestible content.
_ALWAYS_IGNORED_DIRECTORIES = {".git", ".hg", ".svn"}


def loader_supported_extensions() -> set:
    """Lowercase extensions (without the dot) some registered loader can process."""
    from cognee.infrastructure.loaders import get_loader_engine

    engine = get_loader_engine()
    extensions = set()
    for loader_name in engine.get_available_loaders():
        for extension in engine.get_loader_info(loader_name).get("extensions", []):
            extensions.add(extension.lower().lstrip("."))
    return extensions


def _sniffs_binary(file_path: Path) -> bool:
    """A NUL byte in the head of a file marks it binary: no text encoding in
    ingestion use produces one, while native binary formats (executables,
    archives, compiled artifacts) all do. Unreadable files count as binary —
    they could not be ingested either way."""
    try:
        with open(file_path, "rb") as file:
            return b"\0" in file.read(_BINARY_SNIFF_BYTES)
    except OSError as error:
        logger.warning("Skipping unreadable file %s: %s", file_path, error)
        return True


def build_exclusion_spec(
    root: Path,
    respect_gitignore: bool,
    exclude_patterns: Optional[List[str]],
) -> Optional[pathspec.PathSpec]:
    """Combine the root .gitignore (when requested) and user patterns into one
    matcher with gitignore semantics; None when there is nothing to match.

    Only the top-level ``root/.gitignore`` is read — nested .gitignore files
    are not collected.
    """
    patterns: List[str] = []
    if respect_gitignore:
        gitignore = root / ".gitignore"
        if gitignore.is_file():
            patterns.extend(gitignore.read_text(errors="replace").splitlines())
    if exclude_patterns:
        patterns.extend(exclude_patterns)
    if not patterns:
        return None
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def filter_directory_files(
    root: Path,
    file_paths: List[Path],
    respect_gitignore: bool = False,
    exclude_patterns: Optional[List[str]] = None,
) -> List[Path]:
    """Return the files under ``root`` that ingestion should keep.

    Always dropped: files under version-control internals (.git and friends)
    and binary files no registered loader supports (a binary whose extension a
    loader claims — PDFs, images, audio — is kept). Additionally dropped when
    configured: matches of the root .gitignore (``respect_gitignore=True``)
    and of ``exclude_patterns`` (gitignore-style, e.g. ``*.log``, ``.venv/``).
    """
    spec = build_exclusion_spec(root, respect_gitignore, exclude_patterns)
    supported_extensions = loader_supported_extensions()

    kept = []
    skipped_vcs = skipped_excluded = skipped_binary = 0
    for file_path in file_paths:
        relative = file_path.relative_to(root)
        if any(part in _ALWAYS_IGNORED_DIRECTORIES for part in relative.parts):
            skipped_vcs += 1
            continue
        if spec is not None and spec.match_file(relative.as_posix()):
            skipped_excluded += 1
            continue
        extension = file_path.suffix.lower().lstrip(".")
        if extension not in supported_extensions and _sniffs_binary(file_path):
            skipped_binary += 1
            continue
        kept.append(file_path)

    skipped = skipped_vcs + skipped_excluded + skipped_binary
    if skipped:
        logger.info(
            "Directory %s: ingesting %d files, skipped %d "
            "(%d VCS-internal, %d excluded by pattern, %d unsupported binary)",
            root,
            len(kept),
            skipped,
            skipped_vcs,
            skipped_excluded,
            skipped_binary,
        )
    return kept


def filter_s3_keys(
    base_path: str,
    keys: List[str],
    exclude_patterns: Optional[List[str]],
) -> List[str]:
    """Apply ``exclude_patterns`` to S3 keys, matched against the key path
    relative to the listed prefix. Only name-based filtering: .gitignore and
    binary sniffing would require remote reads, so they do not apply to S3.
    """
    if not exclude_patterns:
        return keys
    spec = pathspec.PathSpec.from_lines("gitwildmatch", exclude_patterns)
    base = base_path.removeprefix("s3://").rstrip("/")
    kept = []
    for key in keys:
        relative = key.removeprefix("s3://")
        if relative.startswith(base):
            relative = relative[len(base) :].lstrip("/")
        if not spec.match_file(relative):
            kept.append(key)
    if len(kept) < len(keys):
        logger.info(
            "S3 prefix %s: ingesting %d keys, %d excluded by pattern",
            base_path,
            len(kept),
            len(keys) - len(kept),
        )
    return kept
