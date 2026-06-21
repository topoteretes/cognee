from contextlib import asynccontextmanager
from io import BytesIO

import pytest

import cognee.tasks.ingestion.data_item_to_text_file as data_item_to_text_file_module
from cognee.tasks.ingestion.data_item_to_text_file import pull_from_s3


@pytest.mark.asyncio
async def test_pull_from_s3_reads_source_in_binary_mode(monkeypatch):
    open_modes = []
    destination = BytesIO()

    @asynccontextmanager
    async def fake_open_data_file(file_path, mode="r", **kwargs):
        open_modes.append(mode)
        yield BytesIO(b"\x00\xffcontent")

    monkeypatch.setattr(data_item_to_text_file_module, "open_data_file", fake_open_data_file)

    await pull_from_s3("s3://bucket/file.pdf", destination)

    assert open_modes == ["rb"]
    assert destination.getvalue() == b"\x00\xffcontent"
