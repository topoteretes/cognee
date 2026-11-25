"""``DataItem.literal_text`` is honoured by the storage function every path uses.

``add()``'s incremental pre-save, ``ingest_data`` and ``update()`` all store a
``DataItem`` through ``save_data_item_to_storage_detailed``. It used to unwrap
the item and pass ``.data`` on, dropping the flag, so a ``literal_text`` item
without a pinned ``data_id`` still had its URL fetched or its path read.
"""

import importlib
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from cognee.modules.ingestion import StoredFile
from cognee.tasks.ingestion.data_item import DataItem

storage = importlib.import_module("cognee.tasks.ingestion.save_data_item_to_storage")


def _never(what):
    return AsyncMock(side_effect=AssertionError(f"{what} must not run for literal text"))


@pytest.fixture
def stored_as_text():
    saved = AsyncMock(return_value=StoredFile(file_path="/tmp/text.txt", metadata={}))
    with (
        patch.object(storage, "save_data_to_file_detailed", saved),
        patch.object(storage, "validate_outbound_url", _never("a URL check")),
        patch.object(storage, "fetch_page_content", _never("a URL fetch")),
    ):
        yield saved


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    ["https://example.com/x", "s3://bucket/key", "file:///etc/passwd", "plain words"],
)
async def test_literal_text_is_stored_as_the_text_itself(stored_as_text, content):
    stored = await storage.save_data_item_to_storage_detailed(
        DataItem(data=content, literal_text=True)
    )

    stored_as_text.assert_awaited_once_with(content)
    assert stored.file_path == "/tmp/text.txt"


@pytest.mark.asyncio
async def test_literal_text_naming_an_existing_file_does_not_read_it(stored_as_text, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("do not leak this", encoding="utf-8")

    stored = await storage.save_data_item_to_storage_detailed(
        DataItem(data=str(secret), literal_text=True)
    )

    stored_as_text.assert_awaited_once_with(str(secret))
    assert not stored.file_path.startswith("file://")


@pytest.mark.asyncio
async def test_without_literal_text_a_url_is_still_fetched():
    fetched = AsyncMock(return_value={"https://example.com/x": "<html>page</html>"})
    saved = AsyncMock(return_value=StoredFile(file_path="/tmp/page.html", metadata={}))
    with (
        patch.object(storage, "validate_outbound_url", AsyncMock()),
        patch.object(storage, "fetch_page_content", fetched),
        patch.object(storage, "save_data_to_file_detailed", saved),
    ):
        await storage.save_data_item_to_storage_detailed(DataItem(data="https://example.com/x"))

    fetched.assert_awaited_once_with("https://example.com/x")
    saved.assert_awaited_once_with("<html>page</html>", file_extension="html")


@pytest.mark.asyncio
async def test_literal_text_on_an_upload_stores_the_upload():
    """The flag is about strings; an upload is stored as an upload, not refused."""
    upload = SimpleNamespace(file=BytesIO(b"bytes"), filename="report.txt")
    saved = AsyncMock(return_value=StoredFile(file_path="/tmp/report.txt", metadata={}))
    with patch.object(storage, "save_data_to_file_detailed", saved):
        await storage.save_data_item_to_storage_detailed(DataItem(data=upload, literal_text=True))

    saved.assert_awaited_once_with(upload.file, filename="report.txt")
