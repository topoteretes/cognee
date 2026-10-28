"""Regression tests for binary file objects (BufferedReader, BytesIO) in save_data_item_to_storage.

Issue #6392: add() rejects open file objects and BytesIO with IngestionError 415.
"""

import io
from types import SimpleNamespace

import pytest

from cognee.modules.ingestion import StoredFile
from cognee.modules.ingestion.exceptions import IngestionError
from cognee.tasks.ingestion.data_item import DataItem
from cognee.tasks.ingestion.save_data_item_to_storage import (
    save_data_item_to_storage,
    save_data_item_to_storage_detailed,
)


@pytest.mark.asyncio
async def test_save_buffered_reader_to_storage(tmp_path):
    test_file = tmp_path / "notes.txt"
    test_file.write_text("Ada Lovelace wrote the first program.\n", encoding="utf-8")

    with open(test_file, "rb") as f:
        stored = await save_data_item_to_storage_detailed(f)
        assert not f.closed, "Caller-owned file stream must not be closed"
        assert isinstance(stored, StoredFile)
        assert stored.metadata["name"] == "notes"
        assert stored.metadata["extension"] == "txt"
        assert stored.metadata["file_size"] == 38
        assert stored.metadata["content_hash"] != ""

    with open(test_file, "rb") as f:
        path_only = await save_data_item_to_storage(f)
    assert path_only.endswith(("/notes", "/notes.txt"))


@pytest.mark.asyncio
async def test_save_buffered_reader_preserves_stream_position(tmp_path):
    test_file = tmp_path / "seek_test.txt"
    test_file.write_text("0123456789abcdefghij", encoding="utf-8")

    with open(test_file, "rb") as f:
        f.seek(5)
        assert f.tell() == 5
        stored = await save_data_item_to_storage_detailed(f)
        assert not f.closed
        assert f.tell() == 5, f"Expected position 5 preserved, got {f.tell()}"
        assert stored.metadata["file_size"] == 20


@pytest.mark.asyncio
async def test_save_bytesio_to_storage():
    payload = b"Ada Lovelace in-memory content."
    bio = io.BytesIO(payload)
    stored = await save_data_item_to_storage_detailed(bio)

    assert not bio.closed, "Caller-owned BytesIO must not be closed"
    assert isinstance(stored, StoredFile)
    assert stored.metadata["extension"] == "txt"
    assert stored.metadata["file_size"] == len(payload)
    assert stored.metadata["name"].startswith("text_")
    assert stored.metadata["content_hash"] != ""


@pytest.mark.asyncio
async def test_save_bytesio_preserves_stream_position():
    payload = b"0123456789abcdefghij"
    bio = io.BytesIO(payload)
    bio.seek(7)
    assert bio.tell() == 7

    stored = await save_data_item_to_storage_detailed(bio)
    assert not bio.closed
    assert bio.tell() == 7, f"Expected position 7 preserved, got {bio.tell()}"
    assert stored.metadata["file_size"] == len(payload)


@pytest.mark.asyncio
async def test_save_bytesio_with_name_attribute():
    bio = io.BytesIO(b"named stream data")
    bio.name = "custom_report.txt"

    stored = await save_data_item_to_storage_detailed(bio)
    assert stored.metadata["name"] == "custom_report"
    assert stored.metadata["extension"] == "txt"


@pytest.mark.asyncio
async def test_save_raw_bytes_to_storage():
    raw = b"raw binary content without stream wrapper"
    stored = await save_data_item_to_storage_detailed(raw)

    assert isinstance(stored, StoredFile)
    assert stored.metadata["file_size"] == len(raw)
    assert stored.metadata["name"].startswith("text_")


@pytest.mark.asyncio
async def test_save_uploadfile_compatibility():
    upload = SimpleNamespace(file=io.BytesIO(b"upload bytes"), filename="uploaded_doc.pdf")
    stored = await save_data_item_to_storage_detailed(upload)

    assert isinstance(stored, StoredFile)
    assert stored.metadata["name"] == "uploaded_doc"
    assert stored.file_path.endswith("/uploaded_doc.pdf")


@pytest.mark.asyncio
async def test_save_data_item_wrapping_stream(tmp_path):
    test_file = tmp_path / "item_wrapped.txt"
    test_file.write_text("data item wrapped stream", encoding="utf-8")

    with open(test_file, "rb") as f:
        item = DataItem(data=f)
        stored = await save_data_item_to_storage_detailed(item)
        assert stored.metadata["name"] == "item_wrapped"

    bio_item = DataItem(data=io.BytesIO(b"data item wrapped bytesio"))
    stored_bio = await save_data_item_to_storage_detailed(bio_item)
    assert stored_bio.metadata["name"].startswith("text_")


@pytest.mark.asyncio
@pytest.mark.parametrize("unsupported", [12345, object(), ["nested", "list"], {"key": "val"}])
async def test_unsupported_data_types_raise_ingestion_error(unsupported):
    with pytest.raises(IngestionError) as exc_info:
        await save_data_item_to_storage_detailed(unsupported)

    assert "Data type not supported" in str(exc_info.value)
    assert exc_info.value.status_code == 415
