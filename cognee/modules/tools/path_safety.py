"""Filesystem helpers for local skill source ingestion."""

from __future__ import annotations

import os
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterator


def _safe_path_string(path: Path) -> str:
    base_path = os.path.normpath(os.path.realpath(os.getcwd()))
    full_path = os.path.normpath(os.path.realpath(os.path.abspath(os.fspath(path))))
    base_prefix = base_path if base_path.endswith(os.sep) else f"{base_path}{os.sep}"
    if full_path != base_path and not full_path.startswith(base_prefix):
        raise ValueError("Path is outside the current working directory")
    return full_path


def trusted_is_file(path: Path) -> bool:
    full_path = _safe_path_string(path)
    if not full_path.startswith(os.path.normpath(os.path.realpath(os.getcwd()))):
        raise ValueError("Path is outside the current working directory")
    return os.path.isfile(full_path)


def trusted_is_dir(path: Path) -> bool:
    full_path = _safe_path_string(path)
    if not full_path.startswith(os.path.normpath(os.path.realpath(os.getcwd()))):
        raise ValueError("Path is outside the current working directory")
    return os.path.isdir(full_path)


def trusted_iterdir(path: Path) -> Iterator[Path]:
    full_path = _safe_path_string(path)
    if not full_path.startswith(os.path.normpath(os.path.realpath(os.getcwd()))):
        raise ValueError("Path is outside the current working directory")
    for entry in os.scandir(full_path):
        yield Path(entry.path)


def trusted_rglob(path: Path, pattern: str) -> Iterator[Path]:
    full_path = _safe_path_string(path)
    if not full_path.startswith(os.path.normpath(os.path.realpath(os.getcwd()))):
        raise ValueError("Path is outside the current working directory")
    for root, dirs, files in os.walk(full_path):
        for name in dirs:
            if fnmatch(name, pattern):
                yield Path(root) / name
        for name in files:
            if fnmatch(name, pattern):
                yield Path(root) / name


def trusted_read_text(path: Path, encoding: str = "utf-8", errors: str | None = None) -> str:
    full_path = _safe_path_string(path)
    if not full_path.startswith(os.path.normpath(os.path.realpath(os.getcwd()))):
        raise ValueError("Path is outside the current working directory")
    with open(full_path, encoding=encoding, errors=errors) as file:
        return file.read()
