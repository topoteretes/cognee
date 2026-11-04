"""The two places a sealed object must be unsealed to a path: loaders and raw download."""

import os

import pytest

from cognee.infrastructure.files.storage import get_file_storage
from cognee.tasks.ingestion import data_item_to_text_file as ditf


class _RecordingLoader:
    loader_name = "recording"

    def __init__(self):
        self.seen = []

    async def load(self, file_path, **kwargs):
        with open(file_path, "rb") as file:
            self.seen.append((file_path, file.read()))
        return "STORAGE_PATH"


class _Engine:
    def __init__(self, loader):
        self.loader = loader

    async def load_file(self, file_path, preferred_loaders=None, **kwargs):
        return await self.loader.load(file_path, **kwargs)

    def get_loader(self, file_path, preferred_loaders=None):
        return self.loader


@pytest.mark.asyncio
async def test_sealed_file_reaches_the_loader_as_a_temporary_plaintext_copy(
    encryption_env, data_root, monkeypatch
):
    storage = get_file_storage(str(data_root))
    await storage.store("report.pdf", b"%PDF-1.4 pretend")
    sealed_path = str(data_root / "report.pdf")

    loader = _RecordingLoader()
    monkeypatch.setattr(ditf, "get_loader_engine", lambda: _Engine(loader))

    result, chosen = await ditf.data_item_to_text_file(sealed_path)

    assert result == "STORAGE_PATH"
    assert chosen is loader
    [(seen_path, seen_bytes)] = loader.seen
    assert seen_bytes == b"%PDF-1.4 pretend"
    assert seen_path != sealed_path
    assert seen_path.endswith(".pdf")  # suffix preserved for loader selection
    assert not os.path.exists(seen_path)  # removed once the loader returned


@pytest.mark.asyncio
async def test_sealed_file_uri_is_also_copied(encryption_env, data_root, monkeypatch):
    storage = get_file_storage(str(data_root))
    uri = await storage.store("notes.txt", "sealed notes")

    loader = _RecordingLoader()
    monkeypatch.setattr(ditf, "get_loader_engine", lambda: _Engine(loader))

    await ditf.data_item_to_text_file(uri)

    [(seen_path, seen_bytes)] = loader.seen
    assert seen_bytes == b"sealed notes"
    assert not seen_path.startswith("file://")


@pytest.mark.asyncio
async def test_plain_file_is_handed_to_the_loader_by_its_own_path(
    encryption_env, data_root, monkeypatch, tmp_path
):
    plain = tmp_path / "outside.txt"
    plain.write_text("not under the data root")

    loader = _RecordingLoader()
    monkeypatch.setattr(ditf, "get_loader_engine", lambda: _Engine(loader))

    await ditf.data_item_to_text_file(str(plain))

    [(seen_path, _)] = loader.seen
    assert seen_path == str(plain)


@pytest.mark.asyncio
async def test_temporary_copy_is_removed_when_the_loader_fails(
    encryption_env, data_root, monkeypatch
):
    storage = get_file_storage(str(data_root))
    await storage.store("bad.txt", "sealed")
    created = []

    class Exploding(_RecordingLoader):
        async def load(self, file_path, **kwargs):
            created.append(file_path)
            raise RuntimeError("loader broke")

    monkeypatch.setattr(ditf, "get_loader_engine", lambda: _Engine(Exploding()))

    with pytest.raises(RuntimeError, match="loader broke"):
        await ditf.data_item_to_text_file(str(data_root / "bad.txt"))

    assert created and not os.path.exists(created[0])
