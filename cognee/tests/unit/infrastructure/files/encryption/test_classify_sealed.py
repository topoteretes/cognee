"""classify() takes the plaintext view of a sealed object.

update() and the ingestion re-read paths classify stored objects through
open_data_file, which hands back a DecryptingReader when the layer is on. A
classifier that only knew the builtin reader classes refused it, so every
update() of a sealed document failed with IngestionError.
"""

from io import BytesIO

import pytest

from cognee.infrastructure.files.storage import get_file_storage
from cognee.infrastructure.files.utils.open_data_file import open_data_file
from cognee.modules.ingestion.classify import classify
from cognee.tasks.ingestion import data_item_to_text_file as ditf


@pytest.mark.asyncio
async def test_classify_describes_a_sealed_object_from_its_plaintext(encryption_env, data_root):
    storage = get_file_storage(str(data_root))
    await storage.store("notes.txt", b"sealed notes, described in the clear")
    sealed = data_root / "notes.txt"
    assert sealed.read_bytes().startswith(b"COGNEEENC")

    expected = classify(
        BytesIO(b"sealed notes, described in the clear"), filename="notes.txt"
    ).get_metadata()
    async with open_data_file(str(sealed)) as file:
        metadata = classify(file).get_metadata()

    assert metadata["name"] == "notes"
    assert metadata["content_hash"] == expected["content_hash"]
    assert metadata["file_size"] == expected["file_size"]
    assert metadata["mime_type"] == expected["mime_type"]


class _StoringLoader:
    """A loader that writes derived text through the storage layer, as real ones do."""

    loader_name = "storing"

    def __init__(self, data_root):
        self.data_root = data_root

    async def load(self, file_path, **kwargs):
        with open(file_path, "rb") as file:
            text = file.read().decode("utf-8")
        storage = get_file_storage(str(self.data_root))
        return await storage.store("derived.txt", f"derived: {text}")


class _Engine:
    def __init__(self, loader):
        self.loader = loader

    async def load_file(self, file_path, preferred_loaders=None, **kwargs):
        return await self.loader.load(file_path, **kwargs)

    def get_loader(self, file_path, preferred_loaders=None):
        return self.loader


@pytest.mark.asyncio
async def test_update_staging_describes_sealed_original_and_text(
    encryption_env, data_root, monkeypatch
):
    """The path update() runs: stage the new content under the data root (sealed),
    load it, then classify both stored objects through open_data_file."""
    from cognee.api.v1.update import incremental

    monkeypatch.setattr(ditf, "get_loader_engine", lambda: _Engine(_StoringLoader(data_root)))

    staged = await incremental._stage_new_content("Updated text for a sealed document.", None)

    original = classify(
        BytesIO(b"Updated text for a sealed document."), filename="x.txt"
    ).get_metadata()
    derived = classify(
        BytesIO(b"derived: Updated text for a sealed document."), filename="derived.txt"
    ).get_metadata()
    assert staged.content_hash == original["content_hash"]
    assert staged.raw_content_hash == derived["content_hash"]
    assert staged.loader_engine == "storing"
    for stored in data_root.iterdir():
        assert stored.read_bytes().startswith(b"COGNEEENC"), stored.name
