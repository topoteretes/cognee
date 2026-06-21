"""Filesystem helpers for local skill source ingestion."""

from __future__ import annotations

import os
import tempfile
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterator


def _allowed_base_paths() -> list[str]:
    return [
        os.path.normpath(os.path.realpath(os.getcwd())),
        os.path.normpath(os.path.realpath(tempfile.gettempdir())),
    ]


def _is_under_allowed_base(full_path: str) -> bool:
    for base in _allowed_base_paths():
        if full_path == base or full_path.startswith(
            base if base.endswith(os.sep) else f"{base}{os.sep}"
        ):
            return True
    return False


def _safe_path_string(path: Path) -> str:
    full_path = os.path.normpath(os.path.realpath(os.path.abspath(os.fspath(path))))
    if not _is_under_allowed_base(full_path):
        raise ValueError("Path is outside the current working directory")
    return full_path


def trusted_is_file(path: Path) -> bool:
    return os.path.isfile(_safe_path_string(path))


def trusted_is_dir(path: Path) -> bool:
    return os.path.isdir(_safe_path_string(path))


def trusted_iterdir(path: Path) -> Iterator[Path]:
    full_path = _safe_path_string(path)
    for entry in os.scandir(full_path):
        yield Path(entry.path)


def trusted_rglob(path: Path, pattern: str) -> Iterator[Path]:
    full_path = _safe_path_string(path)
    for root, dirs, files in os.walk(full_path):
        for name in dirs:
            if fnmatch(name, pattern):
                yield Path(root) / name
        for name in files:
            if fnmatch(name, pattern):
                yield Path(root) / name


def trusted_read_text(path: Path, encoding: str = "utf-8", errors: str | None = None) -> str:
    full_path = _safe_path_string(path)
    with open(full_path, encoding=encoding, errors=errors) as file:
        return file.read()
