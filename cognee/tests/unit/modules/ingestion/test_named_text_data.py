"""Text can be stored under a caller-chosen file name instead of its content hash."""

import hashlib
import sys
from unittest.mock import AsyncMock, patch

import pytest

from cognee.modules.ingestion.classify import classify
from cognee.modules.ingestion.data_types.TextData import TextData, create_text_data
from cognee.tasks.ingestion.data_item import DataItem
from cognee.tasks.ingestion.save_data_item_to_storage import save_data_item_to_storage_detailed

# The package re-exports a function under the module's name, so patch the
# module object itself.
storage_module = sys.modules["cognee.tasks.ingestion.save_data_item_to_storage"]


def test_nameless_text_keeps_the_content_hash_name():
    text = "a small memory"
    expected = "text_" + hashlib.md5(text.encode("utf-8")).hexdigest() + ".txt"
    assert create_text_data(text).get_metadata()["name"] == expected


def test_named_text_is_stored_under_the_given_name():
    data = TextData("a small memory", "2026-09-22T10-14-33Z_session-memory_s1.txt")
    metadata = data.get_metadata()
    assert metadata["name"] == "2026-09-22T10-14-33Z_session-memory_s1.txt"


def test_name_does_not_change_the_content_hash_identity():
    text = "same content"
    assert TextData(text, "one.txt").get_identifier() == TextData(text).get_identifier()


def test_classify_passes_the_filename_to_text_data():
    classified = classify("plain text", "note.txt")
    assert isinstance(classified, TextData)
    assert classified.get_metadata()["name"] == "note.txt"


@pytest.mark.asyncio
async def test_named_data_item_text_is_saved_under_its_name():
    item = DataItem(
        data="Session ID: s1\n\nQuestion: q\n\nAnswer: a\n\n", name="stamp_session-memory_s1.txt"
    )
    with patch.object(storage_module, "save_data_to_file_detailed", new_callable=AsyncMock) as save:
        save.return_value = "stored"
        result = await save_data_item_to_storage_detailed(item)
    save.assert_awaited_once_with(item.data, filename="stamp_session-memory_s1.txt")
    assert result == "stored"


@pytest.mark.asyncio
async def test_unnamed_data_item_falls_through_to_the_payload():
    item = DataItem(data="just text")
    with patch.object(storage_module, "save_data_to_file_detailed", new_callable=AsyncMock) as save:
        save.return_value = "stored"
        await save_data_item_to_storage_detailed(item)
    save.assert_awaited_once_with("just text")
