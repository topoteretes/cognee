"""Regression tests for BaseConfig root-directory validation.

`validate_paths` enforces absolute paths for the root directories via
`ensure_absolute_path` (which raises on a relative path and passes ``s3://`` URLs
through unchanged). ``cache_root_directory`` is a root directory too, but was the
only one of the four not validated, so a relative ``CACHE_ROOT_DIRECTORY`` was
silently accepted and later resolved against the current working directory --
unlike its siblings, which reject the same value with a clear error.

These tests lock in that ``cache_root_directory`` is validated identically.
"""

from pathlib import Path

import pytest

from cognee.base_config import BaseConfig


def _clear_env(monkeypatch):
    monkeypatch.delenv("CACHE_ROOT_DIRECTORY", raising=False)
    monkeypatch.delenv("STORAGE_BACKEND", raising=False)


def test_relative_cache_root_directory_is_rejected(monkeypatch):
    _clear_env(monkeypatch)
    with pytest.raises(ValueError):
        BaseConfig(cache_root_directory="relative_cache_dir")


def test_relative_sibling_directory_is_rejected(monkeypatch):
    # Sanity check that the siblings already reject relative paths, so
    # cache_root_directory must behave the same way.
    _clear_env(monkeypatch)
    with pytest.raises(ValueError):
        BaseConfig(data_root_directory="relative_data_dir")


def test_s3_cache_root_directory_is_preserved(monkeypatch):
    _clear_env(monkeypatch)
    cfg = BaseConfig(cache_root_directory="s3://bucket/cognee/cache")
    assert cfg.cache_root_directory == "s3://bucket/cognee/cache"


def test_absolute_cache_root_directory_is_accepted(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    cfg = BaseConfig(cache_root_directory=str(tmp_path))
    assert Path(cfg.cache_root_directory).is_absolute()
