from pathlib import Path

import pytest

from cognee.infrastructure.files.storage.LocalFileStorage import LocalFileStorage
from cognee.infrastructure.files.utils.local_path_safety import (
    ALLOWED_LOCAL_FILE_ROOTS_ENV,
    resolve_local_path,
)


def test_resolve_local_path_rejects_outside_allowed_roots(monkeypatch, tmp_path: Path):
    allowed_root = tmp_path / "allowed"
    outside_root = tmp_path / "outside"
    allowed_root.mkdir()
    outside_root.mkdir()
    outside_file = outside_root / "secret.txt"
    outside_file.write_text("secret", encoding="utf-8")

    monkeypatch.setenv(ALLOWED_LOCAL_FILE_ROOTS_ENV, str(allowed_root))

    with pytest.raises(ValueError, match="outside allowed roots"):
        resolve_local_path(outside_file, must_exist=True)


@pytest.mark.asyncio
async def test_local_file_storage_rejects_storage_root_outside_allowed_roots(
    monkeypatch, tmp_path: Path
):
    allowed_root = tmp_path / "allowed"
    storage_root = tmp_path / "storage"
    allowed_root.mkdir()
    storage_root.mkdir()
    monkeypatch.setenv(ALLOWED_LOCAL_FILE_ROOTS_ENV, str(allowed_root))

    storage = LocalFileStorage(str(storage_root))

    with pytest.raises(ValueError, match="outside allowed roots"):
        await storage.file_exists("file.txt")


@pytest.mark.asyncio
async def test_local_file_storage_blocks_paths_outside_storage_root(monkeypatch, tmp_path: Path):
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    monkeypatch.setenv(ALLOWED_LOCAL_FILE_ROOTS_ENV, str(tmp_path))

    storage = LocalFileStorage(str(storage_root))

    with pytest.raises(ValueError, match="outside the configured storage root"):
        await storage.file_exists("../secret.txt")
