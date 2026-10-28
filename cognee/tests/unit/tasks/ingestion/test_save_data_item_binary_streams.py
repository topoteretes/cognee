"""``add()`` accepts the open binary file objects its docstring lists (#6392).

``save_data_item_to_storage_detailed`` used to route only upload objects (those
with a ``.file``), strings and ``DataItem``s, so ``open(path, "rb")`` and
``io.BytesIO`` fell through to "Data type not supported" (415) even though
``classify()`` already handles both.
"""

import importlib
import io

import pytest

from cognee.infrastructure.files.utils.open_data_file import open_data_file
from cognee.modules.ingestion.exceptions import IngestionError
from cognee.tasks.ingestion.refuse_changed_existing_documents import _origins
from cognee.tasks.ingestion.save_data_item_to_storage import save_data_item_to_storage_detailed

# The package __init__ rebinds ``save_data_to_file`` to the function, shadowing
# the module it lives in — import the module explicitly to patch it.
save_module = importlib.import_module("cognee.modules.ingestion.save_data_to_file")


@pytest.fixture
def storage_root(tmp_path, monkeypatch):
    root = tmp_path / "storage"
    monkeypatch.setattr(
        save_module, "get_storage_config", lambda: {"data_root_directory": str(root)}
    )
    return root


@pytest.mark.asyncio
async def test_open_file_object_is_stored(storage_root, tmp_path):
    source = tmp_path / "notes.txt"
    source.write_bytes(b"Ada Lovelace wrote the first program.")

    with open(source, "rb") as file:
        stored = await save_data_item_to_storage_detailed(file)

    assert stored.metadata["name"] == "notes"
    async with open_data_file(stored.file_path) as saved:
        assert saved.read() == b"Ada Lovelace wrote the first program."


@pytest.mark.asyncio
async def test_named_bytesio_is_stored(storage_root):
    stream = io.BytesIO(b"in-memory payload")
    stream.name = "memo.txt"

    stored = await save_data_item_to_storage_detailed(stream)

    assert stored.metadata["name"] == "memo"
    async with open_data_file(stored.file_path) as saved:
        assert saved.read() == b"in-memory payload"


@pytest.mark.asyncio
async def test_nameless_bytesio_gets_classify_error_not_unsupported_type(storage_root):
    with pytest.raises(IngestionError) as error:
        await save_data_item_to_storage_detailed(io.BytesIO(b"no name"))

    assert "filename=" in str(error.value)
    assert "Data type not supported" not in str(error.value)


@pytest.mark.asyncio
async def test_unsupported_type_still_rejected(storage_root):
    with pytest.raises(IngestionError, match="Data type not supported"):
        await save_data_item_to_storage_detailed(object())


@pytest.mark.asyncio
async def test_origins_tracks_open_file_object_like_an_upload(tmp_path):
    source = tmp_path / "report.txt"
    source.write_bytes(b"version one")

    with open(source, "rb") as file:
        by_location, by_name = await _origins(file)
        assert file.tell() == 0

    assert by_location == {}
    assert list(by_name) == [("report", "txt")]
