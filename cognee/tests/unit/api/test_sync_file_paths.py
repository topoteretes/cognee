import importlib
from pathlib import Path

import pytest


class FakeFileStorage:
    def __init__(self, storage_path):
        self.storage_path = Path(storage_path)

    async def get_size(self, file_name):
        return (self.storage_path / file_name).stat().st_size


@pytest.mark.asyncio
async def test_get_file_size_decodes_file_uri_paths(monkeypatch, tmp_path):
    sync_module = importlib.import_module("cognee.api.v1.sync.sync")

    directory = tmp_path / "space dir"
    directory.mkdir()
    file_path = directory / "file name.txt"
    file_path.write_text("hello", encoding="utf-8")

    monkeypatch.setattr(sync_module, "get_file_storage", FakeFileStorage)

    assert await sync_module._get_file_size(file_path.as_uri()) == len("hello")
