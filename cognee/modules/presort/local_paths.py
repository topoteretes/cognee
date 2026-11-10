"""Bound presort scans and report reads to explicitly permitted local roots."""

import os
import tempfile
from pathlib import Path

from cognee.base_config import get_base_config
from cognee.infrastructure.files.utils.local_path_safety import get_allowed_local_file_roots


def get_presort_roots() -> tuple[Path, ...]:
    configured = get_allowed_local_file_roots()
    if configured is not None:
        return configured

    # A scan reads every candidate file. Unlike ordinary ingestion, require a
    # bounded default even when the general local-path allowlist is unset.
    config = get_base_config()
    roots = (
        Path.cwd(),
        tempfile.gettempdir(),
        config.data_root_directory,
        config.system_root_directory,
        config.cache_root_directory,
        config.logs_root_directory,
        config.repos_root_directory,
    )
    return tuple(Path(os.path.realpath(os.path.expanduser(root))) for root in roots)


def resolve_presort_path(path: str | Path, *, must_exist: bool = False) -> Path:
    # Canonicalize symlinks before checking containment. A sibling whose name
    # starts with an allowed root must not pass the directory-boundary check.
    resolved = os.path.realpath(os.path.expanduser(os.fspath(path)))
    for root in get_presort_roots():
        root_str = os.fspath(root)
        if resolved == root_str:
            candidate = Path(root_str)
        elif resolved.startswith(root_str.rstrip(os.sep) + os.sep):
            candidate = Path(resolved)
        else:
            continue
        if must_exist and not candidate.exists():
            raise FileNotFoundError(path)
        return candidate
    raise ValueError("Local file path is outside allowed roots.")
