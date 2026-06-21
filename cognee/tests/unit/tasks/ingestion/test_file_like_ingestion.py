import hashlib
import io
from pathlib import Path
from urllib.parse import urlparse

import pytest

from cognee.infrastructure.files.storage.config import file_storage_config
from cognee.modules.ingestion.classify import classify
from cognee.modules.ingestion.data_types import BinaryData
from cognee.tasks.ingestion.save_data_item_to_storage import save_data_item_to_storage


def test_classify_accepts_seekable_file_like_objects():
    stream = io.BytesIO(b"hello")

    classified = classify(stream)

    assert isinstance(classified, BinaryData)
    assert classified.get_metadata()["content_hash"] == hashlib.md5(b"hello").hexdigest()


@pytest.mark.asyncio
async def test_save_data_item_to_storage_accepts_unnamed_file_like_objects(tmp_path):
    token = file_storage_config.set({"data_root_directory": str(tmp_path)})
    try:
        file_uri = await save_data_item_to_storage(io.BytesIO(b"hello"))
    finally:
        file_storage_config.reset(token)

    stored_path = Path(urlparse(file_uri).path)

    assert stored_path.name == f"text_{hashlib.md5(b'hello').hexdigest()}.txt"
    assert stored_path.read_bytes() == b"hello"


@pytest.mark.asyncio
async def test_save_data_item_to_storage_preserves_file_like_name_extension(tmp_path):
    source_path = tmp_path / "notes.md"
    storage_path = tmp_path / "storage"
    source_path.write_bytes(b"# Notes")

    token = file_storage_config.set({"data_root_directory": str(storage_path)})
    try:
        with source_path.open("rb") as stream:
            file_uri = await save_data_item_to_storage(stream)
    finally:
        file_storage_config.reset(token)

    stored_path = Path(urlparse(file_uri).path)

    assert stored_path.name == "notes.md"
    assert stored_path.read_bytes() == b"# Notes"
